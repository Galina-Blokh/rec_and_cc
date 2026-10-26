"""Platform-neutral UI glue shared by the macOS (rumps) and Windows (pystray) front-ends.

``Controller`` owns the runtime (monitor, recorder, transcription queue, engine) and turns
engine/queue callbacks (which arrive on worker threads) into events on a thread-safe
``queue.Queue``. The UI drains that queue from its own main-thread timer, so no UI object
is ever touched from a worker thread. Slow things (permission checks, scanning for recent
transcripts) run on short-lived worker threads and publish results under ``self._lock``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import fields
from pathlib import Path

from ..config import CONFIG_PATH, Config, load_config
from ..engine import IDLE, PROMPTING, RECORDING, Engine, EngineState
from ..models import Mode, SessionInfo, TriggerKind

log = logging.getLogger(__name__)

RECENT_COUNT = 10
REFRESH_EVERY_S = 10.0
PERM_KINDS = ("microphone", "system_audio")
PERM_TITLES = {"microphone": "Microphone", "system_audio": "System audio"}
PERM_LABELS = {
    "granted": "✅ granted",
    "denied": "❌ denied",
    "undetermined": "⚠️ not asked",
    "restricted": "⛔ restricted",
    "unknown": "❔ unknown",
}

ONBOARDING_TEXT = (
    "RecAndCC records your calls and videos locally and transcribes them on this Mac.\n\n"
    "To do that it needs:\n"
    '• Microphone access — to record your side of a call ("Me").\n'
    '• System audio access — to record what the other people / the video say ("Others").\n\n'
    "Nothing leaves your computer."
)


# ----------------------------------------------------------------------- helpers
def fmt_elapsed(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def open_path(path: Path | str, text_editor: bool = False) -> None:
    path = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-t", path] if text_editor else ["open", path])
        elif sys.platform == "win32":
            if text_editor:
                subprocess.Popen(["notepad.exe", path])
            else:
                os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        log.exception("could not open %s", path)


def recent_transcripts(output_dir: Path, n: int = RECENT_COUNT) -> list[Path]:
    """Newest ``n`` transcript.md files under <output_dir>/<day>/<session>/."""
    try:
        files = list(Path(output_dir).glob("*/*/transcript.md"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:n]
    except OSError:
        return []


def transcript_label(path: Path) -> str:
    """'2026-10-26 14:03 zoom-us' from .../2026-10-26/140312_zoom-us/transcript.md"""
    day, sess = path.parent.parent.name, path.parent.name
    hhmmss, _, slug = sess.partition("_")
    t = f"{hhmmss[:2]}:{hhmmss[2:4]}" if len(hhmmss) >= 4 and hhmmss[:4].isdigit() else hhmmss
    return f"{day} {t} {slug}".strip()


def untranscribed_sessions(output_dir: Path) -> list[SessionInfo]:
    """Finished sessions never transcribed (e.g. app quit mid-queue), oldest first."""
    out = []
    for f in sorted(Path(output_dir).glob(f"*/*/{SessionInfo.FILENAME}")):
        try:
            s = SessionInfo.load(f.parent)
        except Exception:
            continue
        if s.ended_at is not None and not s.transcribed and s.tracks:
            out.append(s)
    return out


def permissions_module():
    """The mac permissions module, or None (other platforms / not built yet)."""
    if sys.platform != "darwin":
        return None
    try:
        from .. import permissions

        return permissions
    except Exception:
        return None


def permission_status() -> dict[str, str]:
    mod = permissions_module()
    if mod is None:
        return {}
    try:
        return dict(mod.status())
    except Exception:
        log.exception("permission status failed")
        return {}


def missing_permissions(status: dict[str, str], record_mic: str = "calls") -> list[str]:
    """Needed permissions that are not granted (``unknown`` is not treated as missing)."""
    needed = ["system_audio"] + ([] if record_mic == "never" else ["microphone"])
    return [k for k in needed if status.get(k, "unknown") not in ("granted", "unknown")]


def grant_permission(kind: str) -> str | None:
    """Undetermined -> show the system prompt (blocks up to ~60 s, call off the main thread);
    denied/restricted -> open System Settings. Returns the new status if known."""
    mod = permissions_module()
    if mod is None:
        return None
    try:
        st = mod.status().get(kind, "unknown")
        if st in ("denied", "restricted"):
            mod.open_settings(kind)
            return st
        if st != "granted":
            return mod.request(kind)
        return st
    except Exception:
        log.exception("granting %s failed", kind)
        return None


def osascript(script: str, timeout: float | None = None) -> str:
    """Run AppleScript, return stdout ('' on error/cancel)."""
    try:
        r = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        log.exception("osascript failed")
        return ""


def as_str(text: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def mac_notify(title: str, message: str) -> None:
    subprocess.Popen(
        ["osascript", "-e", f"display notification {as_str(message)} with title {as_str(title)}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ----------------------------------------------------------------------- controller
class Controller:
    def __init__(
        self,
        cfg: Config,
        *,
        monitor=None,
        recorder=None,
        transcriber=None,
        tq=None,
        start_threads: bool = True,
    ):
        self.cfg = cfg
        self.events: queue.Queue[tuple] = queue.Queue()
        self._lock = threading.Lock()
        self._perm: dict[str, str] = {}
        self._recent: list[Path] = []
        self._refreshing = False
        self._last_refresh = 0.0
        self._config_mtime = self._mtime()

        if monitor is None:
            from ..detect import create_monitor

            monitor = create_monitor()
        if recorder is None:
            from ..recorder import Recorder

            recorder = Recorder(cfg)
        if tq is None:
            from ..transcribe import Transcriber, TranscriptionQueue

            transcriber = transcriber or Transcriber(cfg.transcription)
            tq = TranscriptionQueue(transcriber, on_done=self._on_transcribed)
        self.transcriber = transcriber
        self.tq = tq
        self.engine = Engine(
            cfg,
            monitor,
            recorder,
            tq,
            on_prompt=self._on_prompt,
            on_notice=lambda t, m: self.events.put(("notice", t, m)),
            on_state_change=lambda st: self.events.put(("state", st)),
            permission_status=permission_status,
        )
        if start_threads:
            self.engine.start()
            threading.Thread(target=self._resubmit_unfinished, daemon=True, name="resubmit").start()
            self.refresh_async(force=True)

    # --- callbacks from worker threads -> events
    def _on_prompt(self, app: str, kind: TriggerKind) -> None:
        st = self.engine.state
        self.events.put(("prompt", app, kind, st.prompt_id))

    def _on_transcribed(self, session: SessionInfo, result) -> None:
        self.events.put(("transcribed", session, result))
        self.refresh_async(force=True)

    def _resubmit_unfinished(self) -> None:
        try:
            for s in untranscribed_sessions(self.cfg.output_dir):
                log.info("re-queueing untranscribed session %s", s.directory)
                self.tq.submit(s)
        except Exception:
            log.exception("scanning for untranscribed sessions failed")

    # --- periodic refresh (permissions, recent transcripts, config file)
    def refresh_async(self, force: bool = False) -> None:
        with self._lock:
            if self._refreshing or (
                not force and time.monotonic() - self._last_refresh < REFRESH_EVERY_S
            ):
                return
            self._refreshing = True
            self._last_refresh = time.monotonic()
        threading.Thread(target=self._refresh, daemon=True, name="ui-refresh").start()

    def _refresh(self) -> None:
        try:
            perm = permission_status()
            recent = recent_transcripts(self.cfg.output_dir)
            with self._lock:
                self._perm, self._recent = perm, recent
            self._maybe_reload_config()
        finally:
            with self._lock:
                self._refreshing = False
        self.events.put(("refreshed",))

    def _mtime(self) -> float:
        try:
            return CONFIG_PATH.stat().st_mtime
        except OSError:
            return 0.0

    def _maybe_reload_config(self) -> None:
        """Apply edits made to config.toml by hand (detection settings + output dir)."""
        m = self._mtime()
        if m == self._config_mtime:
            return
        self._config_mtime = m
        try:
            new = load_config()
        except Exception:
            log.exception("config.toml is invalid; keeping current settings")
            self.events.put(("notice", "Settings not applied", "config.toml could not be parsed"))
            return
        for f in fields(new.detection):
            setattr(self.cfg.detection, f.name, getattr(new.detection, f.name))
        self.cfg.output_dir = new.output_dir
        log.info("config reloaded from %s", CONFIG_PATH)

    @property
    def permissions(self) -> dict[str, str]:
        with self._lock:
            return dict(self._perm)

    @property
    def recent(self) -> list[Path]:
        with self._lock:
            return list(self._recent)

    def missing_permissions(self) -> list[str]:
        return missing_permissions(self.permissions, self.cfg.detection.record_mic)

    def grant_async(self, kind: str) -> None:
        def run():
            grant_permission(kind)
            self.refresh_async(force=True)

        threading.Thread(target=run, daemon=True, name=f"grant-{kind}").start()

    def grant_all_missing(self) -> None:
        """Blocking: request/open settings for each missing permission in turn."""
        for kind in missing_permissions(permission_status(), self.cfg.detection.record_mic):
            grant_permission(kind)
        self.refresh_async(force=True)

    # --- presentation
    def title(self, state: EngineState | None = None) -> str:
        st = state or self.engine.state
        if st.status == RECORDING:
            base = f"🔴 {fmt_elapsed(st.elapsed)}"
        elif st.status == PROMPTING:
            base = "🟠"
        elif st.mode == Mode.OFF:
            base = "⚪"
        elif st.mode == Mode.AUTO:
            base = "🎧"
        else:
            base = "🟡"
        pending = 0
        with contextlib.suppress(Exception):
            pending = int(self.tq.pending())
        if pending:
            base += f" ⏳{pending if pending > 1 else ''}"
        if self.missing_permissions():
            base = "⚠️" + base
        return base

    def status_line(self, state: EngineState | None = None) -> str:
        st = state or self.engine.state
        if st.status == RECORDING:
            what = st.app or "manual recording"
            return f"Recording {what} — {fmt_elapsed(st.elapsed)}"
        if st.status == PROMPTING:
            return f"Waiting for answer: record {st.app}?"
        return f"Idle — {str(st.mode).capitalize()} mode"

    def toggle_recording(self) -> None:
        if self.engine.state.status == RECORDING:
            self.engine.stop()
        else:
            self.engine.start_manual()

    def shutdown(self) -> None:
        with self._lock:
            if getattr(self, "_shut", False):
                return
            self._shut = True
        log.info("shutting down")
        self.engine.shutdown()
        try:
            self.tq.shutdown(wait=False)
        except TypeError:
            self.tq.shutdown()
        except Exception:
            log.exception("transcription queue shutdown failed")
        busy = 0
        with contextlib.suppress(Exception):
            busy = int(self.tq.pending())
        if busy:
            # A session is mid-transcription. Don't pull the pool out from under it (that
            # fails the job) or wait for it (could be minutes): the process exits and the
            # untranscribed session is re-queued on the next start.
            log.info("quitting during a transcription; it will resume on next start")
        elif self.transcriber is not None and hasattr(self.transcriber, "close"):
            with contextlib.suppress(Exception):
                self.transcriber.close()


__all__ = ["Controller", "IDLE", "PROMPTING", "RECORDING", "ONBOARDING_TEXT"]
