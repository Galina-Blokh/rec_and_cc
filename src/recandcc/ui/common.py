"""Platform-neutral UI glue shared by the macOS (rumps) and Windows (pystray) front-ends.

``Controller`` owns the runtime (monitor, recorder, transcription queue, engine) and turns
engine/queue callbacks (which arrive on worker threads) into events on a thread-safe
``queue.Queue``. The UI drains that queue from its own main-thread timer, so no UI object
is ever touched from a worker thread. With ``transcription.live`` each session gets a
``LiveTranscriber`` (fed by the recorder's frame tap) that posts ``live_start`` / ``live`` /
``live_end`` events, and its ``finish()`` replaces the post-session transcription. Slow
things (permission checks, scanning for recent transcripts) run on short-lived worker threads
and publish results under ``self._lock``.

The global shortcut (``cfg.hotkey``, see ``hotkey.py``) calls ``hotkey_pressed()`` on the
listener's thread: it toggles recording through the engine's thread-safe commands and posts a
``notice`` event, so the toast is shown by the UI's main thread.
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
from typing import TYPE_CHECKING

from .. import hotkey as hotkey_mod
from ..cloudsync import cloud_sync_provider, warning_text
from ..config import CONFIG_PATH, LOG_DIR, Config, load_config
from ..engine import IDLE, PROMPTING, RECORDING, Engine, EngineState
from ..models import Mode, SessionInfo, TriggerKind, discard_session
from ..recorder import make_private_dir
from ..retention import purge_old_audio

if TYPE_CHECKING:
    from ..transcribe.live import LiveTranscriber

log = logging.getLogger(__name__)

RECENT_COUNT = 10
HOTKEY_DEBOUNCE_S = 0.5
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


def reveal_logs() -> None:
    """Open the log folder (recandcc.log, faults.log) in Finder / Explorer."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    open_path(LOG_DIR)


def open_path(path: Path | str, text_editor: bool = False) -> None:
    path = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-t", path] if text_editor else ["open", path])
        elif sys.platform == "win32":
            if text_editor:
                subprocess.Popen(["notepad.exe", path])
            else:
                os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        log.exception("could not open %s", path)


def open_output_folder(path: Path) -> str | None:
    """Open the transcripts folder, creating it owner-only (like the session folders) if it is
    missing. Returns an error message for the UI instead of raising if it can't be created."""
    try:
        make_private_dir(path)
    except OSError as e:
        log.warning("could not create transcripts folder %s: %s", path, e)
        return f"{path}: {e.strerror or e}"
    open_path(path)
    return None


def recent_transcripts(output_dir: Path, n: int = RECENT_COUNT) -> list[Path]:
    """Newest ``n`` transcript.md files under <output_dir>/<day>/<session>/."""
    try:
        files = [
            f
            for f in Path(output_dir).glob("*/*/transcript.md")
            if not (f.parent / SessionInfo.DISCARD_MARKER).exists()
        ]
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
            if (f.parent / SessionInfo.DISCARD_MARKER).exists():
                continue  # discarded; purge_discarded_sessions deletes it
            s = SessionInfo.load(f.parent)
        except Exception:
            continue
        if s.ended_at is not None and not s.transcribed and s.tracks:
            out.append(s)
    return out


def purge_discarded_sessions(output_dir: Path) -> int:
    """Retry deleting sessions the user discarded but whose files were in use (session.json
    may be unreadable there); returns how many are still left."""
    markers = sorted(Path(output_dir).glob(f"*/*/{SessionInfo.DISCARD_MARKER}"))
    return sum(not discard_session(m.parent) for m in markers)


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
        self._cloud_warned: set[Path] = set()  # cloud-synced output dirs already warned about
        # global shortcut
        self._hotkey_lock = threading.Lock()
        self._hotkey = None  # HotkeyListener / WinHotkeyListener
        self._hotkey_label = ""  # "⌘⇧R" while the shortcut is (being) registered
        self._last_hotkey = float("-inf")
        self.clock = time.monotonic

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
        self._live = None  # LiveTranscriber of the session being recorded (engine thread)
        self._live_done: dict[str, LiveTranscriber] = {}  # session id -> flushed LiveTranscriber
        self.engine = Engine(
            cfg,
            monitor,
            recorder,
            _LiveAwareQueue(self),
            on_prompt=self._on_prompt,
            on_notice=self.post_notice,
            on_state_change=lambda st: self.events.put(("state", st)),
            permission_status=permission_status,
            on_session_start=self._on_session_start,
            on_session_end=self._on_session_end,
        )
        if start_threads:
            self.engine.start()
            threading.Thread(target=self._resubmit_unfinished, daemon=True, name="resubmit").start()
            threading.Thread(target=self._retention_loop, daemon=True, name="retention").start()
            self.refresh_async(force=True)
            self.apply_hotkey(cfg.hotkey)

    # --- global shortcut
    def apply_hotkey(self, text: str | None, factory=None) -> None:
        """(Re)register the global shortcut; ``""`` disables it. Blocks briefly while the old
        listener shuts down, so call it off the main thread. ``factory`` is for tests."""
        factory = factory or hotkey_mod.create_listener
        with self._hotkey_lock:
            old, self._hotkey, self._hotkey_label = self._hotkey, None, ""
        if old is not None:
            with contextlib.suppress(Exception):
                old.close()
        if getattr(self, "_shut", False):
            return
        try:
            parsed = hotkey_mod.parse_hotkey(text)
        except hotkey_mod.HotkeyError as e:
            log.warning("invalid hotkey %r: %s", text, e)
            self.events.put(("notice", "Shortcut not set", f"{e} — change `hotkey` in settings"))
            return
        if parsed is None:
            log.info("global shortcut disabled")
            return
        with self._hotkey_lock:  # before the listener exists: its on_error may fire at once
            self._hotkey_label = parsed.label()
        try:
            listener = factory(text, self.hotkey_pressed, self._hotkey_failed)
        except Exception:
            log.exception("could not start the global shortcut")
            listener = None
        with self._hotkey_lock:
            self._hotkey = listener
            if listener is None:
                self._hotkey_label = ""

    @property
    def hotkey_label(self) -> str:
        """Shortcut to show in the menu ("⌘⇧R"), "" if disabled or not working."""
        with self._hotkey_lock:
            return self._hotkey_label

    def _hotkey_failed(self, kind: str, message: str) -> None:
        with self._hotkey_lock:
            label, self._hotkey_label = self._hotkey_label, ""
        log.warning("global shortcut %s not available (%s): %s", label, kind, message)
        if kind == "taken":
            text = f"Shortcut {label} is used by another app — change `hotkey` in settings"
        else:
            text = f"Shortcut {label} could not be registered — change `hotkey` in settings"
        self.events.put(("notice", "Shortcut unavailable", text))

    def hotkey_pressed(self) -> None:
        """Global shortcut: stop if recording (any session), else start a manual recording.
        Runs on the listener's thread; presses within HOTKEY_DEBOUNCE_S are ignored."""
        now = self.clock()
        with self._hotkey_lock:
            if now - self._last_hotkey < HOTKEY_DEBOUNCE_S:
                return
            self._last_hotkey = now
            label = self._hotkey_label
        if self.engine.state.status == RECORDING:
            self.engine.stop()
            self.events.put(("notice", "Recording stopped", "Saved; the transcript follows"))
        else:
            self.engine.start_manual()
            title = f"Recording started ({label} to stop)" if label else "Recording started"
            self.events.put(("notice", title, "Manual recording"))

    # --- callbacks from worker threads -> events
    def post_notice(self, title: str, message: str) -> None:
        """Toast from any thread (also the crash notifier, see logging_setup); never blocks."""
        self.events.put(("notice", title, message))

    def _on_prompt(self, app: str, kind: TriggerKind) -> None:
        st = self.engine.state
        self.events.put(("prompt", app, kind, st.prompt_id))

    def _on_transcribed(self, session: SessionInfo, result) -> None:
        self.events.put(("transcribed", session, result))
        self.refresh_async(force=True)

    # --- live transcription (engine thread)
    def _on_session_start(self, session: SessionInfo) -> None:
        if not self.cfg.transcription.live or self.transcriber is None:
            return
        add_listener = getattr(self.engine.recorder, "add_listener", None)
        if add_listener is None:
            return
        from ..transcribe.live import LiveTranscriber, live_title

        sid = session.id
        live = LiveTranscriber(
            self.transcriber,
            session,
            on_segments=lambda segs: self.events.put(("live", sid, segs)),
            on_end=lambda: self.events.put(("live_end", sid)),
        )
        self.events.put(("live_start", sid, live_title(session)))
        try:
            add_listener(live.feed)
        except Exception:
            log.exception("could not attach live transcription")
            live.cancel()
            self.events.put(("live_end", sid))
            return
        self._live = live
        log.info("live transcription on for %s", sid)

    def _on_session_end(self, session: SessionInfo | None, kept: bool) -> None:
        live, self._live = self._live, None
        if live is None:
            return
        if kept and session is not None:
            # Transcribe the tail now (not when the queue gets to it); live_end is posted by
            # the LiveTranscriber once the final transcript is written.
            live.flush(session)
            self._live_done[session.id] = live
        else:
            live.cancel()  # before the engine deletes the directory
            self.events.put(("live_end", live.session.id))

    def _submit(self, session: SessionInfo) -> None:
        live = self._live_done.pop(session.id, None)
        if live is None:
            self.tq.submit(session)
        else:
            self.tq.submit(session, finalize=lambda: live.finish(session))

    def _retention_loop(self) -> None:
        """Delete expired audio at startup and then hourly (audio_retention_days)."""
        while not getattr(self, "_shut", False):
            self.purge_audio_now()
            for _ in range(3600):  # sleep 1 h, but notice shutdown promptly
                if getattr(self, "_shut", False):
                    return
                time.sleep(1)

    def purge_audio_now(self) -> None:
        try:
            purge_old_audio(self.cfg.output_dir, self.cfg.transcription.audio_retention_days)
        except Exception:
            log.exception("audio retention failed")

    def _resubmit_unfinished(self) -> None:
        try:
            for s in untranscribed_sessions(self.cfg.output_dir):
                log.info("re-queueing untranscribed session %s", s.directory)
                self.tq.submit(s)
        except Exception:
            log.exception("scanning for untranscribed sessions failed")
        try:  # after re-queueing, so deleting big WAVs doesn't delay recovery
            purge_discarded_sessions(self.cfg.output_dir)
        except Exception:
            log.exception("deleting discarded sessions failed")

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
            self._check_cloud_sync()
        finally:
            with self._lock:
                self._refreshing = False
        self.events.put(("refreshed",))

    def _check_cloud_sync(self) -> None:
        """Warn once per output folder if it is synced to iCloud / OneDrive / ...: recordings
        would leave the device. Runs on every refresh (worker thread; the check is local and
        cheap), so a folder that becomes synced later is still reported."""
        out = Path(self.cfg.output_dir)
        with self._lock:
            if out in self._cloud_warned:
                return
        provider = cloud_sync_provider(out)
        if provider is None:
            return
        with self._lock:
            if out in self._cloud_warned:
                return
            self._cloud_warned.add(out)
        log.warning(
            "output folder %s is synced to %s; use a local folder to keep it", out, provider
        )
        self.events.put(
            ("notice", "Transcripts folder is cloud-synced", warning_text(out, provider))
        )

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
        if new.hotkey != self.cfg.hotkey:
            self.cfg.hotkey = new.hotkey
            self.apply_hotkey(new.hotkey)
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
        with self._hotkey_lock:
            hk, self._hotkey = self._hotkey, None
        if hk is not None:
            with contextlib.suppress(Exception):
                hk.close()
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


class _LiveAwareQueue:
    """What the Engine submits to: routes sessions that were transcribed live to a fast
    "finalize" job instead of a full transcription."""

    def __init__(self, ctrl: Controller) -> None:
        self._ctrl = ctrl

    def submit(self, session: SessionInfo) -> None:
        self._ctrl._submit(session)


def keep_prompt_text(app: str, kind: TriggerKind, timeout_action: str) -> str:
    """Body of the ask-mode dialog. Recording has already started when it is shown."""
    what = "a call" if kind == TriggerKind.CALL else "audio"
    default = "kept" if timeout_action != "ignore" else "discarded"
    return (
        f"RecAndCC is recording {what} in {app}.\n\n"
        f"Keep the recording and transcribe it?\n(If you don't answer, it will be {default}.)"
    )


# Dialog button label -> Engine.respond() choice.
PROMPT_BUTTONS = (("Discard", "ignore"), ("Always", "always"), ("Keep", "record"))


__all__ = [
    "PROMPT_BUTTONS",
    "keep_prompt_text",
    "Controller",
    "IDLE",
    "PROMPTING",
    "RECORDING",
    "ONBOARDING_TEXT",
]
