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
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .. import hotkey as hotkey_mod
from ..cloudsync import (
    PROBE_FAILED,
    cloud_sync_provider,
    network_volume,
    network_warning_text,
    warning_text,
)
from ..config import (
    CONFIG_PATH,
    LOG_DIR,
    Config,
    ConfigLockTimeout,
    load_config,
    normalize_languages,
    save_keys,
    save_settings,
    validate_settings,
)
from ..engine import IDLE, PROMPTING, RECORDING, Engine, EngineState
from ..models import (
    Mode,
    SessionInfo,
    TriggerKind,
    discard_session,
    recording_lock_held,
    still_writing,
)
from ..recorder import make_private_dir
from ..retention import purge_old_audio, purge_transcribed_audio
from ..sensitivity import PRESETS as SENSITIVITY_PRESETS
from ..sensitivity import apply_preset, current_preset, room_trigger
from ..settings_window import (
    SettingsUnavailable,
    close_settings,
    offered_models,
    raise_settings,
    run_settings,
)

if TYPE_CHECKING:
    from ..transcribe.live import LiveTranscriber

log = logging.getLogger(__name__)


class _ConfigWriteError(Exception):
    """Writing config.toml failed (args: exception name, errno); no path in the message."""


CFG_HOTKEY = object()  # apply_hotkey(): use cfg.hotkey

RECENT_COUNT = 10
SEARCH_LIMIT = 20
HOTKEY_DEBOUNCE_S = 0.5
NET_PROBE_WAIT_S = 2.0  # how long a refresh waits for the network-volume probe
# Menu settings whose effect already happened outside config.toml when their save fails: the
# login item was changed, the captions panel shown/hidden, Off stopped a recording, the room
# mic closed, call-app participant sampling stopped. Putting the old value back would undo the
# click only on paper (autostart, captions) or, worse, resume recording, reopen the mic or read
# participant names the user just turned off. They keep the new value and stay "unsaved"
# (``Controller._unsaved``): a config reload skips them and each refresh retries the save
# until it succeeds.
NO_ROLLBACK_KEYS = frozenset(
    {
        "autostart",
        "live_captions",
        "detection.mode",
        "detection.app_modes",
        "detection.room_listening",
        "transcription.names_from_call_app",
    }
)
UNSAVED_NOTE = "The change stays in effect and is saved as soon as config.toml can be written."
NET_PROBE_RETRY_S = 300.0  # after a hung or failed probe
NET_LOCAL_TTL_S = 600.0  # a folder probed as local is re-probed after this long
NET_PROBE_MAX_THREADS = 4  # stuck probes (one per folder) allowed at once
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


def latest_finished_session(output_dir: Path) -> Path | None:
    """The transcript.md of the session that started last and has a transcript.json (a live
    or still-transcribing one has only transcript.md). Walks <day>/<time>_<app> folders newest
    first by name, not mtime (a re-transcribed old session isn't "latest"), and stops at the
    first hit, so a big or synced library isn't scanned. Unreadable entries are skipped."""

    def names(directory: Path) -> list[Path]:
        try:
            return sorted((d for d in directory.iterdir() if d.is_dir()), reverse=True)
        except OSError:
            return []

    for day in names(Path(output_dir)):
        for session in names(day):
            try:
                if (session / "transcript.json").is_file() and not (
                    session / SessionInfo.DISCARD_MARKER
                ).exists():
                    return session / "transcript.md"
            except OSError:
                continue
    return None


def transcript_label(path: Path) -> str:
    """'2026-10-26 14:03 zoom-us' from .../2026-10-26/140312_zoom-us/transcript.md"""
    day, sess = path.parent.parent.name, path.parent.name
    hhmmss, _, slug = sess.partition("_")
    t = f"{hhmmss[:2]}:{hhmmss[2:4]}" if len(hhmmss) >= 4 and hhmmss[:4].isdigit() else hhmmss
    return f"{day} {t} {slug}".strip()


def search_hit_label(hit) -> str:
    """'2026-10-26 14:03 zoom-us · 12:05 Me: …the [budget] for…' for a search.Hit."""
    text = " ".join(hit.snippet.split())
    return f"{transcript_label(hit.transcript)} · {clock(hit.start)} {hit.speaker}: {text}"


def clock(seconds: float) -> str:
    """'12:05' (minutes:seconds) for an offset into a recording."""
    t = int(seconds)
    return f"{t // 60:02d}:{t % 60:02d}"


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


# A track written this recently may belong to another writer (``recandcc record``, a
# folder synced from another Mac): recovery waits until it has been quiet this long.
RECOVER_QUIET_S = 120.0


def recover_interrupted_sessions(
    output_dir: Path,
    *,
    started_before: datetime,
    busy: Callable[[], Iterable[Path | str]] = tuple,
) -> list[SessionInfo]:
    """Finish sessions a crash, force-quit or power loss cut off (session.json still has no
    ``ended_at``), so ``untranscribed_sessions`` re-queues them; returns those finished.

    ``ended_at`` becomes ``started_at`` plus the longest track's length (``sf.info`` reads
    WAVs whose header was never finalised). Tracks with no frames leave ``tracks`` but stay
    on disk; a session with no frames at all, or with a track that can't be read now (held
    open, evicted by a sync client), is left as it is and retried at the next start.

    Never touched: sessions that began at or after ``started_before`` (this app's start),
    whose WAVs were written less than ``RECOVER_QUIET_S`` before it, that a live process
    records into (``RECORDING_LOCK``), or that ``busy()`` lists (the live session and those
    in flight; asked again just before each write). Blocking I/O: call it on a worker
    thread."""
    import soundfile as sf

    def is_busy(d: Path) -> bool:
        return d.resolve() in {Path(p).resolve() for p in busy()}

    cutoff = started_before.timestamp()
    out = []
    for f in sorted(Path(output_dir).glob(f"*/*/{SessionInfo.FILENAME}")):
        d = f.parent
        try:
            if (d / SessionInfo.DISCARD_MARKER).exists():
                continue
            s = SessionInfo.load(d)
            if s.ended_at is not None or s.transcribed or s.started_at.timestamp() >= cutoff:
                continue
            if is_busy(d) or recording_lock_held(d):
                continue
        except Exception:  # noqa: BLE001 - unreadable: left alone
            continue
        longest, newest, kept = 0.0, float("-inf"), {}
        try:
            for track, name in s.tracks.items():
                p = d / name
                newest = max(newest, p.stat().st_mtime)
                info = sf.info(str(p))
                if info.frames > 0:
                    kept[track] = name
                    longest = max(longest, info.frames / info.samplerate)
        except Exception:  # noqa: BLE001 - unreadable now (or deleted): retried next start
            log.debug("interrupted session %s: a track can't be read; left alone", d.name)
            continue
        if not kept or newest >= cutoff - RECOVER_QUIET_S:  # no audio, or written lately
            continue
        try:
            if is_busy(d):  # it may have become the live session meanwhile
                continue
            s.tracks = kept
            s.ended_at = s.started_at + timedelta(seconds=longest)
            s.update_metrics("recording", {"audio_s": round(longest, 1), "recovered": True})
            s.save()
        except Exception:
            log.exception("could not save recovered session %s", d)
            continue
        log.info("recovered interrupted session %s (%.1fs)", d, longest)
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


DIALOG_TIMEOUT_S = 300  # an abandoned chooser must not keep a flow "open" for ever


def osascript(script: str, timeout: float | None = None) -> str:
    """Run AppleScript, return stdout ('' on error/cancel). The script goes in on stdin, not
    argv, so text in it (search results) isn't visible in the process list."""
    try:
        r = subprocess.run(
            ["osascript", "-"], input=script, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip()
    except Exception:
        log.exception("osascript failed")
        return ""


def as_str(text: str) -> str:
    """Quote a Python string as an AppleScript string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def chooser_script(labels: list[str], prompt: str, button: str) -> str:
    """AppleScript for a list chooser; every text is quoted by ``as_str``."""
    items = ", ".join(map(as_str, labels))
    return (
        f'choose from list {{{items}}} with title "RecAndCC" '
        f"with prompt {as_str(prompt)} OK button name {as_str(button)}"
    )


PLAY_SAMPLE = "Play sample"


def name_dialog_script(label: str, sample: bool = False, answer: str = "") -> str:
    """The name prompt; with `sample` a "Play sample" button too, after which the dialog is
    shown again with the text typed so far (`answer`)."""
    buttons = f'{{"Cancel", {as_str(PLAY_SAMPLE)}, "Save"}}' if sample else '{"Cancel", "Save"}'
    return (
        f"display dialog {as_str(f'Name for {label}:')} default answer {as_str(answer)} "
        f'with title "RecAndCC" buttons {buttons} default button "Save"'
    )


def name_dialog_answer(output: str) -> tuple[bool, str]:
    """(play the sample?, typed text) from a ``name_dialog_script`` result; "" = cancelled."""
    button, _, text = output.partition(", text returned:")
    return button.strip() == f"button returned:{PLAY_SAMPLE}", text.strip()


def ask_speaker_name(label: str, clip, ask: Callable[[str], str], play) -> str:
    """Ask for `label`'s name on a dialog worker thread (``ask`` runs the AppleScript). With a
    `clip` the dialog can play it (``play(clip)`` returns at once) and comes back with the
    text typed so far, as often as the user likes; without one it is the plain prompt.
    Returns the name, "" if cancelled or timed out."""
    answer = ""
    while True:
        replay, answer = name_dialog_answer(
            ask(name_dialog_script(label, clip is not None, answer))
        )
        if not (replay and clip is not None):
            return answer
        play(clip)


# ----------------------------------------------------------------------- controller
def _privatize_new_folder(path: Path) -> None:
    """The folder picker can create a folder (0755 under the usual umask) just before Save.
    An empty folder of ours is made owner-only; a folder with content is never touched."""
    if os.name != "posix":
        return
    try:
        if path.is_dir() and path.stat().st_uid == os.getuid() and not any(path.iterdir()):
            path.chmod(0o700)
    except OSError:
        log.debug("could not make the new transcripts folder private")


# transcription keys a hand edit applies live (models and device need a restart)
RETENTION_KEY = "transcription.audio_retention_days"


def _is_days(value: object) -> bool:
    """A usable ``audio_retention_days``: a whole number (``True`` is not one)."""
    return isinstance(value, int) and not isinstance(value, bool)


# Settings-window fields that are NO_ROLLBACK_KEYS: a successful window save writes them.
WINDOW_KEYS = {"live_captions": "live_captions", "mode": "detection.mode"}
RELOADED_TRANSCRIPTION_KEYS = ("keep_audio", "audio_retention_days", "speaker_names")
MODEL_CHANGE_NOTICE = (
    "Applies after RecAndCC restarts; a model not on disk yet is downloaded once then."
)


class Controller:
    _summaries_lock = threading.Lock()  # creating the SummaryQueue (class-wide: test fakes)

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
        notice = cfg.summary.notice
        if notice:  # load_config(migrate=True) switched or kept an old-default 3B: once
            self.post_notice("Summary model", notice)
        self._lock = threading.Lock()
        self._perm: dict[str, str] = {}
        self._recent: list[Path] = []
        self._export_lock = threading.Lock()
        self._retention_lock = threading.Lock()  # one menu purge at a time
        self._refreshing = False
        self._last_refresh = 0.0
        self._config_mtime = self._mtime()
        self._sync_index_lock = threading.Lock()  # serialises _sync_index (worker threads only)
        self._save_lock = threading.Lock()  # menu writes to config.toml
        # guards _dirty; also spans the reload's snapshot+apply (only _lang_lock nests)
        self._dirty_lock = threading.Lock()
        self._dirty: dict[str, int] = {}  # dotted key -> menu saves not yet written
        # NO_ROLLBACK_KEYS whose save failed: memory differs from config.toml until a save of
        # the key succeeds (retried on every refresh); a reload never overwrites them.
        self._unsaved: set[str] = set()  # guarded by _dirty_lock
        self._settings_lock = threading.Lock()  # one settings window at a time
        self._lang_lock = threading.Lock()  # transcription.languages + its raw spelling
        self._summaries = None  # SummaryQueue, created on the first summary
        self._cloud_warned: set[Path] = set()  # cloud-synced output dirs already warned about
        self._net_probes: dict[Path, threading.Thread] = {}  # one in-flight probe per folder
        self._net_local: dict[Path, float] = {}  # local folder -> when to re-probe (monotonic)
        self._net_retry: dict[Path, float] = {}  # folder -> earliest next probe (monotonic)
        self._net_warned: set[Path] = set()  # network-volume output dirs already warned about
        # global shortcut
        self._hotkey_lock = threading.Lock()
        # Serializes whole apply_hotkey calls (startup and a config reload can overlap), and
        # what is registered, so the same shortcut is never registered twice.
        self._apply_hotkey_lock = threading.Lock()
        self._hotkey_text: str | None = None
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
        if transcriber is not None:  # toasts e.g. a language_models entry that failed to load
            transcriber.on_notice = self.post_notice
        self.tq = tq
        self._live = None  # LiveTranscriber of the session being recorded (engine thread)
        self._live_done: dict[str, LiveTranscriber] = {}  # session id -> flushed LiveTranscriber
        from ..detect.room import RoomMonitor

        # Session folders from start until their transcription is done (or they are dropped):
        # Meeting notes won't rename or delete them meanwhile. Guarded by self._lock.
        self._in_flight: set[Path] = set()
        self._notes_scanning = threading.Lock()
        self.engine = Engine(
            cfg,
            monitor,
            recorder,
            _LiveAwareQueue(self),
            save_config=lambda: self.save_settings(
                "detection.mode", "detection.app_modes", title="Mode"
            ),
            on_prompt=self._on_prompt,
            on_notice=self.post_notice,
            on_state_change=lambda st: self.events.put(("state", st)),
            permission_status=permission_status,
            on_session_start=self._on_session_start,
            on_session_end=self._on_session_end,
            # mic only if room_listening; thresholds follow the outside sensitivity
            room_monitor=RoomMonitor(cfg.detection, thresholds=room_trigger),
        )
        self._started_at = datetime.now().astimezone()  # older sessions: not diarized now
        if start_threads:
            # Standby: nothing records (room mic included) until the user picks a mode, so
            # there is no start-up "mic is on" notice; the toggle's notice still shows.
            self.engine.enter_standby()
            self.engine.start()
            threading.Thread(target=self._resubmit_unfinished, daemon=True, name="resubmit").start()
            threading.Thread(target=self._retention_loop, daemon=True, name="retention").start()
            self._warm_speaker_model()
            self.refresh_async(force=True)
            self.apply_hotkey()

    def _warm_speaker_model(self) -> None:
        """Load the speaker-embedding model off the main thread so the first live session
        doesn't wait for it (US-33). Never downloads; a no-op unless live captions, live labels and
        diarize are all on, or the model isn't on disk."""
        tcfg = self.cfg.transcription
        if not (
            tcfg.live and getattr(tcfg, "live_speaker_labels", False) and tcfg.diarize
        ):  # otherwise no live session would use it: don't hold its memory
            return

        def run() -> None:
            from ..transcribe import diarize

            try:
                diarize.shared_extractor(download=False)
            except Exception:  # noqa: BLE001 - the labeler retries / falls back on its own
                log.debug("speaker model warm-up failed", exc_info=True)

        threading.Thread(target=run, daemon=True, name="speaker-warm").start()

    # --- global shortcut
    def apply_hotkey(self, text: str | None | object = CFG_HOTKEY, factory=None) -> None:
        """(Re)register the global shortcut; ``""`` disables it. By default the configured
        ``cfg.hotkey``, read under the lock, so of two overlapping calls the later one wins and
        the same shortcut is never registered twice. Blocks briefly while the old listener
        shuts down, so call it off the main thread. ``factory`` is for tests."""
        with self._apply_hotkey_lock:
            spec = cast("str | None", self.cfg.hotkey if text is CFG_HOTKEY else text)
            with self._hotkey_lock:
                current = self._hotkey_text if (self._hotkey is not None or not spec) else None
            if current is not None and spec == current and not getattr(self, "_shut", False):
                return  # already registered (or already disabled)
            self._apply_hotkey_locked(spec, factory)

    def _apply_hotkey_locked(self, text: str | None, factory=None) -> None:
        factory = factory or hotkey_mod.create_listener
        with self._hotkey_lock:
            old, self._hotkey, self._hotkey_label = self._hotkey, None, ""
            self._hotkey_text = None
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
            with self._hotkey_lock:
                self._hotkey_text = text or ""
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
            self._hotkey_text = text if listener is not None else None
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
        if (in_flight := getattr(self, "_in_flight", None)) is not None:  # None: test fakes
            with self._lock:
                in_flight.discard(Path(session.directory))
        self.events.put(("transcribed", session, result))
        self.refresh_async(force=True)
        if not isinstance(result, Exception):
            threading.Thread(
                target=self._index_session, args=(session.directory,), daemon=True, name="search"
            ).start()
            if self.cfg.summary.enabled:
                self._summarize_when_labelled(Path(session.directory))

    def _summarize_when_labelled(self, directory: Path) -> None:
        """Summarise once the transcript has its final speaker labels (a slow diarization
        rewrites it later), so action items aren't credited to "Others"."""
        from ..transcribe.summary import mark_pending

        mark_pending(directory)  # owed from now on, even if we quit before diarization ends
        transcriber = getattr(self, "transcriber", None)
        after_labels = getattr(transcriber, "after_labels", None)  # None: test fakes
        if after_labels is None:
            self._summarize(directory)
        else:
            after_labels(directory, lambda: self._summarize(directory))

    def _resubmit_summaries(self) -> None:
        """summary.enabled: queue summaries still owed (cut short by quitting, or stale after
        `name-speaker`): at startup and on idle refreshes. Skips sessions still in flight or
        waiting for speaker labels; SummaryQueue drops repeats of queued ones."""
        if not self.cfg.summary.enabled:
            return
        try:
            from ..transcribe.summary import pending_summaries

            labels_pending = getattr(getattr(self, "transcriber", None), "labels_pending", None)
            with self._lock:
                busy = set(getattr(self, "_in_flight", ()))
            in_flight = {d.resolve() for d in busy}  # resolve() may touch the disk
            for d in pending_summaries(self.cfg.output_dir):
                if d.resolve() in in_flight or (labels_pending and labels_pending(d)):
                    continue
                log.info("queueing the pending summary of %s", d.name)
                self._summarize(d)
        except Exception:
            log.exception("scanning for pending summaries failed")

    def _recording_busy(self) -> bool:
        """Background indexing yields to a recording and to the transcription backlog."""
        tq = getattr(self, "tq", None)
        return (
            self.engine.state.status != IDLE
            or (tq is not None and tq.pending() > 0)
            or bool(getattr(self, "_resubmitting", 0))
        )

    def _summarize(self, directory: Path) -> None:
        """Queue summary.md for a final transcript (local LLM, own background thread)."""
        with self._summaries_lock:  # called from the transcription and diarizer threads
            summaries = getattr(self, "_summaries", None)
            if summaries is None:
                from ..transcribe.summary import SummaryQueue

                summaries = self._summaries = SummaryQueue(
                    self.cfg,
                    notify=self.post_notice,
                    busy=lambda: self.engine.state.status != IDLE,  # never during a recording
                )
        summaries.submit(directory)

    # --- transcript search (worker threads; results arrive as a "search" event)
    def _index_session(self, directory: Path) -> None:
        try:
            from ..search import default_index

            default_index(self.cfg.search.semantic, self._recording_busy).index_session(directory)
        except Exception:  # noqa: BLE001 - search is optional; never break transcription
            log.exception("could not update the search index")

    # --- meeting notes (worker threads; the list arrives as a ("notes", notes) event)
    def notes_async(self) -> None:
        """Scan for Meeting notes off the UI thread; a click while a scan runs is dropped."""
        if not self._notes_scanning.acquire(blocking=False):
            return

        def run() -> None:
            try:
                from .notes import scan_notes

                self.events.put(("notes", scan_notes(self.cfg.output_dir)))
            finally:
                self._notes_scanning.release()

        threading.Thread(target=run, daemon=True, name="notes").start()

    def note_busy(self, directory: Path) -> str | None:
        """Why a session can't be renamed or deleted now, or None: it is being recorded or
        is not through its transcription yet (from session start until the queue is done
        with it, or while startup re-queues unfinished ones). A late diarization or summary
        may still write afterwards: a rename only touches the title, and after a delete
        their writes fail without recreating the folder."""
        live = self.engine.state.session_dir
        with self._lock:
            in_flight = set(self._in_flight)
        try:  # resolve() may touch the disk: outside the lock
            if Path(directory).resolve() in {d.resolve() for d in in_flight}:
                return "It is still being recorded or transcribed; try again when it is done."
            if live and Path(live).resolve() == Path(directory).resolve():
                return "It is still recording."
            info = SessionInfo.load(directory)
        except Exception:  # noqa: BLE001 - unreadable: let the action report it
            return None
        if info.ended_at is None and live:  # unfinished while a session runs: maybe this one
            return "It is still recording."
        if not info.transcribed and (self.tq.pending() > 0 or getattr(self, "_resubmitting", 0)):
            return "It may still be transcribing; try again when the queue is done."
        return None

    def note_action_async(self, directory: Path, action: str, title: str = "") -> None:
        """Rename (``action="rename"``) or delete a session off the UI thread, then post a
        notice and a fresh ``notes`` list."""

        def run() -> None:
            from .notes import delete_notice, delete_session, rename_session, scan_notes

            if action not in ("rename", "delete"):
                log.error("unknown meeting action %r", action)
                return
            if why := self.note_busy(directory):
                self.events.put(("notice", "Can't change this meeting", why))
                return
            try:
                if action == "rename":
                    rename_session(directory, title, self.cfg.output_dir)
                elif action == "delete":
                    result = delete_session(directory, self.cfg.output_dir)
                    self._index_session(directory)  # gone: drops it from the search index
                    self._prune_voices()
                    self.refresh_async(force=True)
                    if notice := delete_notice(result):
                        self.events.put(("notice", *notice))
            except Exception as e:  # noqa: BLE001
                log.error("meeting %s failed for %s: %s", action, directory.name, type(e).__name__)
                self.events.put(("notice", f"Could not {action} the meeting", directory.name))
            with self._notes_scanning:  # one scan at a time, after a running one (fresh list)
                self.events.put(("notes", scan_notes(self.cfg.output_dir)))

        threading.Thread(target=run, daemon=True, name="notes-action").start()

    @staticmethod
    def _prune_voices() -> None:
        """Drop the stored speaker voices of deleted sessions (speaker_names) right away."""
        try:
            from ..transcribe.voices import prune_session_voices

            prune_session_voices()
        except Exception as e:  # noqa: BLE001 - voices are optional
            log.warning("could not prune session voices: %s", type(e).__name__)

    def search_async(self, query: str) -> None:
        """Search the transcripts off the UI thread; posts ``("search", query, hits)``, with
        hits the exception if the search failed."""

        def run() -> None:
            try:
                from ..search import search_transcripts

                hits = search_transcripts(
                    self.cfg.output_dir,
                    query,
                    SEARCH_LIMIT,
                    self.cfg.search.semantic,
                    busy=self._recording_busy,
                )
            except Exception as e:  # noqa: BLE001
                # the type only: an FTS5 syntax error quotes the query
                log.error("transcript search failed: %s", type(e).__name__)
                hits = e
            self.events.put(("search", query, hits))

        threading.Thread(target=run, daemon=True, name="search").start()

    # --- speaker names (opt-in; worker threads; results arrive as events)
    @property
    def speaker_names_enabled(self) -> bool:
        return self.cfg.transcription.speaker_names

    @property
    def sample_allowed(self) -> bool:
        """A speaker sample plays only while nothing records: system capture leaves out only
        our own process, so ``afplay`` would end up in the recording."""
        return self.engine.state.status == IDLE

    def speaker_clip(self, directory: Path, label: str):
        """The label's audio sample (``ui.sample.Clip``) for the name dialog, or None while
        recording or when the audio is gone. Call it on a worker thread (reads the WAV)."""
        if not self.sample_allowed:
            return None
        from . import sample

        return sample.find_clip(directory, label)

    def speakers_async(self, directory: Path) -> None:
        """List the session's speakers that have voice data, with their first lines; posts
        ``("speakers", directory, [(label, lines)])`` (the exception if it failed)."""

        def run() -> None:
            try:
                from ..models import SessionInfo
                from ..transcribe import voices

                # the transcript (maybe cloud-synced) is read outside any lock
                lines = voices.session_speakers(directory)
                known = voices.read_session_voices(SessionInfo.load(directory).id)
                result = [(lbl, ln) for lbl, ln in lines if lbl in known]
            except Exception as e:  # noqa: BLE001
                log.error("could not list the speakers: %s", type(e).__name__)
                result = e
            self.events.put(("speakers", directory, result))

        threading.Thread(target=run, daemon=True, name="speakers").start()

    def name_speaker_async(self, directory: Path, label: str, name: str) -> None:
        """Name a speaker through the same function as ``recandcc name-speaker``; posts
        ``("named", label, name, error | None)``. Names are never logged."""

        def run() -> None:
            error = None
            try:
                if not self.speaker_names_enabled:
                    error = "Voice profiles are off (transcription.speaker_names)."
                else:
                    from ..transcribe import voices

                    voices.name_speaker(Path(directory), label, name)  # holds voices.LOCK
            except Exception as e:  # noqa: BLE001 - always post an outcome
                log.error("could not name a speaker: %s", type(e).__name__)
                # fixed text: the library's messages quote the name, and notices may be logged
                error = (
                    "The name could not be saved: it may be reserved, used by another "
                    "speaker of this session, or the voice data is gone."
                )
            self.events.put(("named", label, name, error))

        threading.Thread(target=run, daemon=True, name="name-speaker").start()

    def forget_voices_async(self) -> None:
        """Delete every voice profile and all stored session voices; posts
        ``("forgot", count | exception)``."""

        def run() -> None:
            try:
                from ..transcribe import voices

                result = voices.forget_all()  # after any naming or finalize in flight
            except Exception as e:  # noqa: BLE001
                log.error("could not delete the voices: %s", type(e).__name__)
                result = e
            self.events.put(("forgot", result))

        threading.Thread(target=run, daemon=True, name="forget-voices").start()

    def export_async(self, transcript: Path | None = None, fmt: str = "both") -> None:
        """Export a session's transcript to DOCX + PDF off the UI thread; posts a ``notice``
        (never the transcript text). One export at a time: a second click is ignored. None
        means "the newest finished session", found on the worker (no file access on the UI
        thread). Only session folders inside ``output_dir`` are exported (no symlinks out)."""
        if not self._export_lock.acquire(blocking=False):
            return

        def run() -> None:
            msg = "Export failed (see the log)"
            try:
                from ..transcribe.export import ExportError, export_session

                skipped: list[str] = []
                try:
                    from .notes import inside

                    target = transcript
                    if target is None:
                        target = latest_finished_session(self.cfg.output_dir)
                    if target is None:
                        raise ExportError("No finished transcripts yet")
                    if not inside(target.parent, self.cfg.output_dir):
                        raise ExportError("That session is outside the transcripts folder")
                    paths = export_session(
                        target.parent, fmt, skipped, template=self.cfg.export.template
                    )
                    if not all(p.is_file() for p in paths):  # e.g. the session was deleted
                        raise ExportError("The session folder changed during the export")
                    names = " and ".join(p.name for p in paths)
                    msg = f"Saved {names} ({transcript_label(target)})"
                    if skipped:
                        msg += f". {skipped[0]}"
                except ExportError as e:
                    msg = str(e)
            except Exception as e:  # noqa: BLE001
                # the type only: a library error can quote the transcript
                log.error("transcript export failed: %s", type(e).__name__)
            finally:  # the notice is queued before the next export may start
                self.events.put(("notice", "Export", msg))
                self._export_lock.release()

        try:
            threading.Thread(target=run, daemon=True, name="export").start()
        except Exception:
            self._export_lock.release()
            raise

    # --- live transcription (engine thread)
    def _on_session_start(self, session: SessionInfo) -> None:
        with self._lock:
            self._in_flight.add(Path(session.directory))
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
        if not kept:
            # Dropped: the engine deletes it. Without a session (the recorder failed to stop)
            # forget every folder but a running one; queued sessions stay busy via note_busy's
            # pending() check.
            live_dir = self.engine.state.session_dir
            with self._lock:
                if session is not None:
                    self._in_flight.discard(Path(session.directory))
                else:
                    self._in_flight = {d for d in self._in_flight if str(d) == live_dir}
        live, self._live = self._live, None
        if live is None:
            return
        # A writer still finishing the WAVs (stalled disk): the flush would diarize partial
        # audio, so the queue transcribes it from the WAVs once they are final instead.
        if kept and session is not None and not still_writing(session.directory):
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
        """Delete expired audio at startup and then hourly (audio_retention_days); sync the
        search index hourly too (half an hour in, clear of the startup sync)."""
        while not getattr(self, "_shut", False):
            try:  # one failure must not end the hourly purge and index sync for good
                self.purge_audio_now()
            except Exception:
                log.exception("audio retention failed")
            for i in range(3600):  # sleep 1 h, but notice shutdown promptly
                if i == 1800:
                    self._sync_index()
                if getattr(self, "_shut", False):
                    return
                time.sleep(1)

    def purge_audio_now(self, days: int | None = None) -> list[Path]:
        """Delete audio older than ``days`` now; the deleted WAVs (none if it failed or was
        skipped). A menu click passes its own saved value: memory may already hold a newer,
        unsaved click. Without ``days`` (startup, the hourly loop) use the in-memory retention,
        but only when config.toml holds that same value: not while a menu save of it is pending
        or pinned unsaved, and not after a failed save left an unsaved click in memory (e.g.
        config.toml was unreadable, so nothing could be restored). A value that was never
        saved must not delete audio; the save's own purge, or the next hourly run, follows."""
        try:
            if days is None:
                with self._dirty_lock:
                    if RETENTION_KEY in self._dirty or RETENTION_KEY in self._unsaved:
                        return []
                days = self.cfg.transcription.audio_retention_days
                if not _is_days(days):  # e.g. a hand-written "7": treated as not on disk
                    log.warning("audio_retention_days is not a whole number; purge skipped")
                    return []
                if days > 0 and self._retention_on_disk() != days:
                    log.info("retention: the current value is not in config.toml; purge skipped")
                    return []
            return purge_old_audio(self.cfg.output_dir, days)
        except Exception:
            log.exception("audio retention failed")
            return []

    def _retention_on_disk(self) -> int | None:
        """``audio_retention_days`` as config.toml holds it, or None when the file is missing
        or unreadable. A plain parse: no lock, no default file, no config warnings."""
        try:
            raw = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            value = raw.get("transcription", {}).get("audio_retention_days", 0)
            return value if _is_days(value) else None
        except Exception:  # missing, mid-edit TOML, unreadable
            return None

    def _purge_leftover_audio(self) -> None:
        """keep_audio = false: delete WAVs left behind by transcribed sessions (e.g. the app
        quit before a slow diarization finished and dropped them, or the delete failed)."""
        if self.cfg.transcription.keep_audio:
            return
        try:
            purge_transcribed_audio(self.cfg.output_dir, ended_before=self._started_at)
        except Exception:
            log.exception("deleting leftover audio failed")

    def _recover_interrupted(self) -> None:
        """Give sessions an earlier run never stopped (crash, force-quit, power loss) an end
        time, so they are re-queued. Never the live one: it started after this controller
        (and is skipped by folder too, in case a room pre-roll moved its start earlier)."""

        def busy() -> set[Path]:
            live = self.engine.state.session_dir  # engine lock taken and released here
            with self._lock:
                out = set(self._in_flight)
            return out | {Path(live)} if live else out

        try:
            recover_interrupted_sessions(
                self.cfg.output_dir, started_before=self._started_at, busy=busy
            )
        except Exception:
            log.exception("recovering interrupted sessions failed")

    def _resubmit_unfinished(self) -> None:
        # Purge first, on this same thread: once a re-queued session is transcribed, its
        # diarization may still need the WAVs, and the purge must not see it as leftover.
        self._resubmitting = True  # note_busy: untranscribed sessions may be about to queue
        try:
            self._purge_leftover_audio()
            self._recover_interrupted()
            for s in untranscribed_sessions(self.cfg.output_dir):
                log.info("re-queueing untranscribed session %s", s.directory)
                self.tq.submit(s)
        except Exception:
            log.exception("scanning for untranscribed sessions failed")
        finally:
            self._resubmitting = False
        self._resubmit_summaries()
        try:  # after re-queueing, so deleting big WAVs doesn't delay recovery
            purge_discarded_sessions(self.cfg.output_dir)
        except Exception:
            log.exception("deleting discarded sessions failed")
        self._sync_index()

    def _sync_index(self) -> None:
        """Drop deleted and discarded sessions from the search index (at start, then hourly),
        so their text doesn't wait for the next search to leave it."""
        try:
            from ..search import default_index

            with self._sync_index_lock:  # read the setting inside: the last runner sees the latest
                idx = default_index(self.cfg.search.semantic, self._recording_busy)
                idx.sync(self.cfg.output_dir)
                idx.release_idle_cache()  # hourly, off the main thread: free an unused cache
        except Exception:  # noqa: BLE001 - search is optional
            log.exception("could not sync the search index")

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
            self._maybe_reload_config()
            self._check_network_volume()
            self._check_cloud_sync()  # before listing output_dir, which may stall on a cloud mount
            perm = permission_status()
            recent = recent_transcripts(self.cfg.output_dir)
            with self._lock:
                self._perm, self._recent = perm, recent
            if self.engine.state.status == IDLE:  # e.g. marked stale by `name-speaker`
                self._resubmit_summaries()
            self._retry_unsaved()  # last: it may wait on the config lock
        finally:
            with self._lock:
                self._refreshing = False
        self.events.put(("refreshed",))

    def _check_network_volume(self) -> None:
        """Warn once per output folder on a network share (SMB/NFS/UNC...): audio and
        transcripts would leave the device. The path is in the notice, never in the log.
        The probe (stat/statfs) can hang forever on a dead hard mount, so it runs on its own
        daemon thread and is waited for briefly; a late answer still posts the notice.
        Local results are cached per path for NET_LOCAL_TTL_S, so a share mounted over that
        path later (or a retargeted symlink) is noticed after the TTL."""
        out = Path(self.cfg.output_dir)
        with self._lock:
            if out in self._net_warned:
                return
            if time.monotonic() < self._net_local.get(out, 0.0):
                return  # known local: no thread, no statfs on every refresh
            prev = self._net_probes.get(out)
            if prev is not None and prev.is_alive():
                return  # this folder's probe is still stuck: don't pile up threads
            if time.monotonic() < self._net_retry.get(out, 0.0):
                return  # a recent probe of this folder hung or failed
            self._net_probes = {p: th for p, th in self._net_probes.items() if th.is_alive()}
            if len(self._net_probes) >= NET_PROBE_MAX_THREADS:
                log.debug("network probe skipped: too many stuck probes; retried next refresh")
                return  # too many stuck shares already
            self._net_probes[out] = t = threading.Thread(
                target=self._probe_network_volume, args=(out,), daemon=True, name="ui-netprobe"
            )
        t.start()
        t.join(NET_PROBE_WAIT_S)
        if t.is_alive():  # hung (dead hard mount): retry only after a while
            with self._lock:
                self._net_retry[out] = time.monotonic() + NET_PROBE_RETRY_S

    def _probe_network_volume(self, out: Path) -> None:
        try:
            if cloud_sync_provider(out) is not None:
                with self._lock:
                    self._net_local[out] = time.monotonic() + NET_LOCAL_TTL_S
                return  # the cloud-sync notice already covers it
            kind = network_volume(out)
        except Exception:  # noqa: BLE001 - best effort
            with self._lock:
                self._net_retry[out] = time.monotonic() + NET_PROBE_RETRY_S
            return
        if kind is PROBE_FAILED:  # errored, not "local": back off like a hung probe
            with self._lock:
                self._net_retry[out] = time.monotonic() + NET_PROBE_RETRY_S
            return
        if kind is None:  # genuinely local (never PROBE_FAILED, which backs off above)
            with self._lock:
                self._net_local[out] = time.monotonic() + NET_LOCAL_TTL_S
            return
        if not isinstance(kind, str):
            return
        with self._lock:
            if out in self._net_warned:
                return
            self._net_warned.add(out)
        log.warning("output folder is on %s; use a local folder to keep it", kind)
        self.events.put(
            ("notice", "Transcripts folder is on the network", network_warning_text(out, kind))
        )

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
        """Apply edits made to config.toml by hand (detection settings, output dir, hotkey,
        languages, keep_audio, audio_retention_days, speaker_names, live_captions, [summary],
        [export]).
        Read and apply are one step under ``_save_lock``: a menu write never interleaves."""
        hotkey_changed = semantic_changed = False
        with self._save_lock:
            m = self._mtime()
            if m == self._config_mtime:
                return
            self._config_mtime = m
            before = self._reloadable(self.cfg)  # to spot menu changes made during the load
            try:
                new = load_config()
            except Exception:
                log.exception("config.toml is invalid; keeping current settings")
                self.post_notice("Settings not applied", "config.toml could not be parsed")
                return
            # Applied under the same lock: a menu or Settings-window save never runs between
            # "file read" and "memory updated", so it can't write half-applied values. A field
            # a menu click changed meanwhile keeps the click (its queued save writes it).
            was_listening = self.cfg.detection.room_listening
            was_semantic = self.cfg.search.semantic
            # _dirty_lock spans snapshot and apply. Menu/tray setters (menu_set, menu_apply,
            # set_mode, toggle_language) mark + set under it too, so their click lands either
            # before the snapshot (seen as dirty) or after the apply (kept), never between.
            # The Settings window applies and writes under _save_lock, which this holds too.
            with self._dirty_lock:
                now = self._reloadable(self.cfg)
                # changed in the menu, save not written yet (or failed: _unsaved)
                dirty = set(self._dirty) | self._unsaved
                if self.engine.standby:  # standby is never saved: the file's mode can't end it
                    dirty.add("detection.mode")
                for key, value in self._reloadable(new).items():
                    if now[key] != before[key] or self._dotted(key) in dirty:
                        continue
                    owner, name = key
                    if owner == "languages":
                        tr = self.cfg.transcription
                        with self._lang_lock:  # both together: save() pairs them
                            if (getattr(tr, "_raw_languages", None), tr.languages) == before[key]:
                                tr._raw_languages = value[0]  # type: ignore[attr-defined]
                                tr.languages = value[1]
                    else:
                        if key == ("top", "hotkey") and value != now[key]:
                            hotkey_changed = True
                        setattr(self._reload_target(owner), name, value)
            # transcription.model is not reloaded (the loaded model stays until a restart), but
            # the Settings window shows what is on disk, a hand edit included
            tr = self.cfg.transcription
            on_disk = new.transcription.model
            tr._pending_model = None if on_disk == tr.model else on_disk  # type: ignore[attr-defined]
            if self.cfg.detection.room_listening and not was_listening:  # turned on by hand
                self.post_notice("Room listening on", ROOM_LISTENING_NOTICE)
            semantic_changed = self.cfg.search.semantic != was_semantic
        if hotkey_changed:  # re-registering the listener may block: outside the lock
            self.apply_hotkey()  # reads cfg.hotkey under the hotkey lock
        if semantic_changed:
            # Apply [search] semantic now, not at the next search: switched off, the embed
            # worker and any model download stop and the vectors are deleted; switched on,
            # embedding starts (while idle). A sync scans the output folder: own thread.
            threading.Thread(target=self._sync_index, daemon=True, name="search-reload").start()
        log.info("config reloaded from %s", CONFIG_PATH)

    @staticmethod
    def _reloadable(cfg) -> dict:
        """Every hot-reloadable setting as {(owner, name): value}, hotkey included."""
        out: dict = {
            ("top", "output_dir"): cfg.output_dir,
            ("top", "live_captions"): cfg.live_captions,
            ("top", "hotkey"): cfg.hotkey,
        }
        for f in fields(cfg.detection):
            out[("detection", f.name)] = getattr(cfg.detection, f.name)
        for f in fields(cfg.summary):
            out[("summary", f.name)] = getattr(cfg.summary, f.name)
        for f in fields(cfg.search):  # [search] semantic: applied at once (_sync_index)
            out[("search", f.name)] = getattr(cfg.search, f.name)
        out[("export", "template")] = cfg.export.template  # read at each export
        for name in RELOADED_TRANSCRIPTION_KEYS:
            out[("transcription", name)] = getattr(cfg.transcription, name)
        tr = cfg.transcription
        out[("languages", "")] = (getattr(tr, "_raw_languages", None), list(tr.languages))
        return out

    @staticmethod
    def _dotted(key: tuple[str, str]) -> str:
        owner, name = key
        if owner == "languages":
            return "transcription.languages"
        return name if owner == "top" else f"{owner}.{name}"

    def _retry_unsaved(self) -> None:
        """Save again the no-rollback keys whose save failed (refresh thread), quietly: the
        failed save already told the user. Memory and config.toml agree once it succeeds."""
        with self._dirty_lock:
            keys = sorted(self._unsaved)
        if keys:
            self.write_settings(*keys, quiet=True)

    def _unpin(self, keys) -> None:
        with self._dirty_lock:
            self._unsaved.difference_update(keys)

    def _mark_dirty(self, keys, delta: int) -> None:
        with self._dirty_lock:
            for k in keys:
                n = self._dirty.get(k, 0) + delta
                if n > 0:
                    self._dirty[k] = n
                else:
                    self._dirty.pop(k, None)

    def _reload_target(self, owner: str):
        return {
            "top": self.cfg,
            "detection": self.cfg.detection,
            "summary": self.cfg.summary,
            "search": self.cfg.search,
            "export": self.cfg.export,
            "transcription": self.cfg.transcription,
        }[owner]

    # --- settings changed from the menu (main thread; saved on a worker thread)
    def toggle_language(self, code: str) -> None:
        """Menu "Languages": check/uncheck ``code`` ("" = Any language). Applies at once: the
        transcriber reads ``cfg.transcription.languages`` for every piece of audio."""
        from .present import LANGUAGES_KEEP_ONE, toggle_languages

        tr = self.cfg.transcription
        key = "transcription.languages"
        # mark + set together under _dirty_lock (a leaf lock; only _lang_lock nests in it)
        new = None
        with self._dirty_lock:
            self._dirty[key] = self._dirty.get(key, 0) + 1
            try:
                with self._lang_lock:
                    new = toggle_languages(tr.languages, getattr(tr, "_raw_languages", None), code)
                    if new is not None:
                        langs = normalize_languages(new)
                        tr._raw_languages = new if new != langs else None  # type: ignore[attr-defined]
                        tr.languages = langs  # a new list: other threads never see it half-built
            except BaseException:
                new = None
                raise
            finally:
                if new is None:  # raised or refused: nothing to save, drop our mark right here
                    n = self._dirty[key] - 1
                    if n > 0:
                        self._dirty[key] = n
                    else:
                        del self._dirty[key]
        if new is None:
            self.post_notice("Languages", LANGUAGES_KEEP_ONE)
            return
        # from here save_settings owns the unmark (exactly once, even if its thread can't start)
        self.save_settings(key, title="Languages", marked=True)

    def toggle_call_app_names(self) -> None:
        """Menu "Use names from the call app" (macOS), from the main thread; never blocks.
        Off is immediate. On needs `speaker_names` and Accessibility trust, checked silently on
        a worker; without trust the UI is asked (a "call_names_confirm" event) to explain and,
        if the user agrees, ``confirm_call_app_names`` opens System Settings > Accessibility:
        the only way the user is ever pointed to that permission."""
        tr = self.cfg.transcription
        if tr.names_from_call_app:
            self._set_call_app_names(False)
        elif not tr.speaker_names:
            self.post_notice(
                "Names from the call app", "Turn on transcription.speaker_names first."
            )
        else:
            threading.Thread(target=self._check_ax_trust, daemon=True, name="ax-trust").start()

    def _check_ax_trust(self) -> None:
        from .. import participants

        if participants.ax_trusted():
            self._set_call_app_names(True)
        else:
            self.events.put(("call_names_confirm",))

    def confirm_call_app_names(self) -> None:
        """The user agreed to the explanation: open the Accessibility pane and switch on."""

        def run() -> None:
            from .. import participants

            participants.open_accessibility_settings()
            self._set_call_app_names(True)

        threading.Thread(target=run, daemon=True, name="ax-settings").start()

    def _set_call_app_names(self, on: bool) -> None:
        self.menu_set("transcription.names_from_call_app", on, title="Names from the call app")

    def menu_set(self, key: str, value: object, *, title: str = "Settings") -> None:
        """A menu/tray setter: set dotted ``key`` in memory and save it (see ``menu_apply``)."""
        owner, _, name = key.rpartition(".")
        target = self.cfg
        for part in filter(None, owner.split(".")):
            target = getattr(target, part)
        self.menu_apply((key,), lambda: setattr(target, name, value), title=title)

    def set_mode(self, mode: Mode | str) -> None:
        """Menu/tray "Mode": mark ``detection.mode`` and switch the engine in one step (see
        ``menu_apply``), so a config reload can't revert the click between set and mark. Lock
        order is ``_dirty_lock`` then the engine's lock, as in the reload."""
        self.menu_apply(
            ("detection.mode",),
            lambda: self.engine.set_mode(mode, save=False),
            title="Mode",
        )

    def menu_apply(
        self,
        keys: tuple[str, ...],
        change: Callable[[], None],
        *,
        title: str = "Settings",
    ) -> None:
        """Mark ``keys`` dirty and run ``change`` (a quick in-memory update) in one step (see
        ``_mark_and_apply``), then queue their save. Never blocks: no file I/O and no
        ``_save_lock`` here."""
        self._mark_and_apply(keys, change)
        self.save_settings(*keys, title=title, marked=True)

    def _mark_and_apply(self, keys: tuple[str, ...], change: Callable[[], None]) -> None:
        """Mark ``keys`` dirty and run ``change`` together under ``_dirty_lock``. A config
        reload holds ``_dirty_lock`` across its snapshot and apply, so the click lands wholly
        before (seen as dirty) or after (kept), and can never be reverted by a hand edit read
        meanwhile. On success the caller owns exactly one mark per key, which its pending save
        drops (``_restore_from_disk`` counts on one mark per pending save); if ``change``
        raises, the mark is dropped here. ``change`` must not take ``_dirty_lock``."""
        with self._dirty_lock:
            for k in keys:
                self._dirty[k] = self._dirty.get(k, 0) + 1
            try:
                change()
            except BaseException:
                for k in keys:
                    n = self._dirty[k] - 1
                    if n > 0:
                        self._dirty[k] = n
                    else:
                        del self._dirty[k]
                raise

    def save_settings(self, *keys: str, title: str = "Settings", marked: bool = False) -> None:
        """Save the in-memory value of each dotted key ("live_captions",
        "transcription.audio_retention_days") to config.toml on a worker thread. Menu
        callbacks use ``menu_set``/``menu_apply`` (which call this) instead of ``cfg.save()``,
        which would overwrite hand edits. ``marked``: the caller already marked ``keys`` dirty
        (before changing memory); from then on this unmarks them exactly once, whether or not
        the thread starts."""
        if not marked:
            self._mark_dirty(keys, 1)  # a config reload before the write must not undo the change

        def run() -> None:
            try:
                self._write_settings(*keys, title=title)
            finally:
                self._mark_dirty(keys, -1)  # one mark per pending save (see _restore_from_disk)

        t = threading.Thread(target=run, name="save-config")
        t.daemon = True
        try:
            t.start()
        except BaseException:
            self._mark_dirty(keys, -1)  # run() will never unmark
            raise

    def write_settings(
        self, *keys: str, title: str = "Settings", quiet: bool = False
    ) -> dict | None:
        """Blocking save of ``keys`` (see ``_write_settings``); marks them as pending so a
        config reload meanwhile keeps the in-memory value. Returns the values written, or
        None if the save failed."""
        self._mark_dirty(keys, 1)
        try:
            return self._write_settings(*keys, title=title, quiet=quiet)
        finally:
            self._mark_dirty(keys, -1)

    def _write_settings(
        self, *keys: str, title: str = "Settings", quiet: bool = False
    ) -> dict | None:
        """Blocking part of ``save_settings``: a read-modify-write of only ``keys``, so a hand
        edit made since the last reload is kept (and picked up by the next refresh). The values
        are read under ``_save_lock``: of two quick menu saves, the last one wins. Returns the
        values written, or None on failure (``quiet``: no notice, for the retry of unsaved
        keys)."""
        with self._save_lock:
            values = {key: self._setting(key) for key in keys}
            edited = self._mtime() != self._config_mtime  # changed on disk, not reloaded yet
            try:
                try:
                    mtime = save_keys(CONFIG_PATH, values)
                except FileNotFoundError:
                    if CONFIG_PATH.exists():  # a write error, not a missing file: keep edits
                        raise
                    self.cfg.save(CONFIG_PATH)  # no file yet: write a fresh one
                    mtime = self._mtime()
                if not edited:
                    self._config_mtime = mtime  # our own write: nothing to reload
            except ValueError:  # invalid TOML (mid-edit?): don't overwrite the user's file
                log.warning("config.toml could not be parsed; %s not saved", ", ".join(keys))
                self._save_failed(
                    values, f"{title} not saved", "config.toml could not be parsed", quiet
                )
            except ConfigLockTimeout:
                log.warning("settings not saved: config lock timed out")  # no traceback
                reason = "config.toml is locked by another process"
                self._save_failed(values, "Settings not saved", reason, quiet)
            except Exception:
                if quiet:
                    log.warning("saving settings failed again")
                else:
                    log.exception("saving settings failed")
                self._save_failed(
                    values, "Settings not saved", "config.toml could not be written", quiet
                )
            else:
                self._unpin(values)
                if quiet:
                    log.info("unsaved settings saved: %s", ", ".join(keys))
                return values
        return None

    def _save_failed(self, tried: dict, heading: str, reason: str, quiet: bool) -> None:
        """A save of ``tried`` failed (caller holds ``_save_lock``): roll back what can be
        rolled back; keep the rest pinned as unsaved, retried by each refresh."""
        self._restore_from_disk(tried)
        pinned = set(tried) & NO_ROLLBACK_KEYS
        with self._dirty_lock:
            self._unsaved |= pinned
        if not quiet:
            self.post_notice(heading, f"{reason}. {UNSAVED_NOTE}" if pinned else reason)

    def _restore_from_disk(self, tried: dict) -> None:
        """A failed write leaves config.toml as it was: put the menu's in-memory value of each
        key in ``tried`` ({dotted key: value this save tried to write}) back to what the file
        says, so the menu doesn't show a change that would silently vanish on restart. The
        menu re-reads cfg on every tick (macOS ``_render``, tray rebuild), so it follows.

        A key is left alone when a newer click owns it: memory no longer holds the value this
        save tried to write, or another save of the key is still pending (each pending save
        holds one ``_dirty`` mark; this one holds its own). That save writes the newer value,
        or restores it if it fails too. Keys in ``NO_ROLLBACK_KEYS`` are never restored.
        Caller holds ``_save_lock``."""
        if not set(tried) - NO_ROLLBACK_KEYS:
            return
        try:
            disk = load_config()  # first: the checks below then sit right before the writes
        except Exception:
            return  # unreadable file: keep memory, the next successful reload corrects it
        with self._dirty_lock:  # a leaf lock: read the counts, nothing nested
            pending = {k: self._dirty.get(k, 0) for k in tried}
        keys = [k for k in tried if k not in NO_ROLLBACK_KEYS and pending[k] <= 1]
        for key in keys:
            if key == "transcription.languages":
                tr, src = self.cfg.transcription, disk.transcription
                with self._lang_lock:  # check and write together: toggle_language takes it too
                    raw = getattr(tr, "_raw_languages", None)
                    if (list(raw) if raw is not None else list(tr.languages)) != tried[key]:
                        continue  # a newer click owns it
                    tr._raw_languages = getattr(src, "_raw_languages", None)  # type: ignore[attr-defined]
                    tr.languages = list(src.languages)
                continue
            if self._setting(key) != tried[key]:  # checked right before the write
                continue  # a newer click owns it
            *path, leaf = key.split(".")
            dst, ref = self.cfg, disk
            for name in path:
                dst, ref = getattr(dst, name), getattr(ref, name)
            setattr(dst, leaf, getattr(ref, leaf))

    def _setting(self, key: str):
        """The TOML value of a dotted config key, as the menu last set it."""
        if key == "transcription.languages":  # keep the user's own spelling ("iw")
            tr = self.cfg.transcription
            with self._lang_lock:
                raw = getattr(tr, "_raw_languages", None)
                return list(raw) if raw is not None else list(tr.languages)
        value = self.cfg
        for name in key.split("."):
            value = getattr(value, name)
        # a copy: the engine thread may add to app_modes while another thread serializes it
        return dict(value) if isinstance(value, dict) else value

    def open_settings_async(self) -> None:
        """Menu "Settings…": the native window runs on a worker thread (it blocks until it is
        closed); saved values are applied and written there, and the UI thread gets a
        ``settings_saved`` event (captions panel) or ``settings_fallback``."""
        if not self._settings_lock.acquire(blocking=False):
            raise_settings()  # already open: bring it to the front
            return

        def run() -> None:
            try:
                try:
                    raw = run_settings(self.cfg)
                except OSError as e:  # temp dir / snapshot file: not the transcripts folder
                    log.warning("could not open settings: %s (errno %s)", type(e).__name__, e.errno)
                    self.post_notice(
                        "Settings not saved", f"couldn't open settings: {type(e).__name__}"
                    )
                    return
                if not raw:  # cancelled, closed, or nothing changed
                    return
                updates = validate_settings(raw, offered_models())
                if "output_dir" in updates:
                    _privatize_new_folder(updates["output_dir"])
                    make_private_dir(updates["output_dir"])
                fields = {k: v for k, v in updates.items() if k != "mode"}
                with self._save_lock:
                    # Fields go live before the write (a menu save in between can't write the
                    # old values back) and are rolled back if the write fails. The mode,
                    # which can stop a recording, only changes once the file is written.
                    tr = self.cfg.transcription
                    old = {
                        "output_dir": self.cfg.output_dir,
                        "live_captions": self.cfg.live_captions,
                        "keep_audio": tr.keep_audio,
                        "speaker_names": tr.speaker_names,
                        "languages": list(tr.languages),
                        "live": tr.live,
                        "model": getattr(tr, "_pending_model", None),
                    }
                    old = {k: v for k, v in old.items() if k in fields}
                    old_raw = getattr(tr, "_raw_languages", None)
                    self.apply_settings(fields)
                    edited = self._mtime() != self._config_mtime  # hand edit not reloaded yet
                    try:
                        try:
                            mtime = save_settings(updates, CONFIG_PATH)
                        except FileNotFoundError:
                            if CONFIG_PATH.exists():  # a write error, not a missing file
                                raise
                            self.cfg.save(CONFIG_PATH)  # no file yet: write a fresh one
                            mtime = save_settings(updates, CONFIG_PATH)
                    except Exception as e:
                        self.apply_settings(old)
                        with self._lang_lock:
                            tr._raw_languages = old_raw  # type: ignore[attr-defined]
                        if isinstance(e, OSError) and not isinstance(e, ConfigLockTimeout):
                            # not the folder: the outer handler would say so
                            raise _ConfigWriteError(type(e).__name__, e.errno) from e
                        raise
                    if not edited:
                        self._config_mtime = mtime  # our own write: nothing to reload
                    self._unpin(WINDOW_KEYS[k] for k in updates if k in WINDOW_KEYS)
                    if "mode" in updates:
                        self.apply_settings({"mode": updates["mode"]})
                if "output_dir" in updates:
                    self.refresh_async(force=True)  # cloud-sync warning for the new folder
                if "model" in updates and updates["model"] != self.cfg.transcription.model:
                    # the loaded model is kept until the app restarts
                    self.post_notice("Model saved", MODEL_CHANGE_NOTICE)
                self.events.put(("settings_saved", updates))
            except SettingsUnavailable as e:
                log.info("native settings unavailable: %s", e)
                self.events.put(("settings_fallback",))
            except ValueError as e:
                log.warning("settings not saved: %s", type(e).__name__)  # no values in logs
                self.post_notice("Settings not saved", str(e))
            except ConfigLockTimeout:
                log.warning("settings not saved: config lock timed out")
                self.post_notice("Settings not saved", "config.toml is locked by another process")
            except _ConfigWriteError as e:
                log.warning("settings not saved: config write %s (errno %s)", *e.args)
                self.post_notice("Settings not saved", "config.toml could not be written")
            except OSError as e:
                log.warning("settings not saved: %s (errno %s)", type(e).__name__, e.errno)
                self.post_notice("Settings not saved", "the transcripts folder can't be created")
            except Exception as e:
                log.error("settings window failed: %s", type(e).__name__)  # no paths in logs
                self.post_notice("Settings not saved", "config.toml could not be written")
            finally:
                self._settings_lock.release()

        threading.Thread(target=run, daemon=True, name="settings-window").start()

    def apply_settings(self, updates: dict) -> None:
        """Make settings live in memory (any thread; no file I/O, the caller saves them)."""
        cfg, tr = self.cfg, self.cfg.transcription
        if "output_dir" in updates:
            cfg.output_dir = updates["output_dir"]
        if "live_captions" in updates:
            cfg.live_captions = updates["live_captions"]
        if "keep_audio" in updates:
            tr.keep_audio = updates["keep_audio"]
        if "speaker_names" in updates:
            tr.speaker_names = updates["speaker_names"]
        if "live" in updates:  # read at each session start
            tr.live = updates["live"]
        if "model" in updates:  # written to config.toml; loaded at the next app start
            pending = None if updates["model"] == tr.model else updates["model"]
            tr._pending_model = pending  # type: ignore[attr-defined]
        if "languages" in updates:
            with self._lang_lock:
                tr._raw_languages = None  # type: ignore[attr-defined]
                tr.languages = list(updates["languages"])
        if "mode" in updates and updates["mode"] != cfg.detection.mode:
            self.engine.set_mode(updates["mode"], save=False)

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

    def set_room_listening(self, on: bool) -> None:
        """Menu toggle: the engine opens or closes the room mic (and stops a room session) on
        its next tick. Turning it on explains the always-on mic indicator and the consent
        question. Saves only this key, like the other toggles (the last toggle wins)."""
        self.menu_set("detection.room_listening", bool(on))
        if on:
            self.post_notice("Room listening on", ROOM_LISTENING_NOTICE)
            if self.permissions.get("microphone") == "denied":
                self.post_notice(
                    "Room listening can't hear anything",
                    "RecAndCC has no microphone access — grant it in the Permissions menu.",
                )

    def set_sensitivity(self, preset: str) -> None:
        """Sensitivity menu: apply a preset from ``sensitivity.PRESETS`` in memory (the engine
        and the room monitor read it on their next tick) and save only these two keys."""
        self.menu_apply(
            ("detection.outside_sensitivity", "detection.inside_sensitivity"),
            lambda: apply_preset(self.cfg.detection, preset),
            title="Sensitivity",
        )

    def set_retention(self, days: int) -> None:
        """ "Delete audio after" menu (both shells): apply ``days`` in memory at once (the menu
        check reads it), then save only this key and purge expired audio on a worker, with
        this click's value. Audio is deleted only once the choice is saved and only if it is
        still the current choice (a quick second click supersedes this one). A failed save is
        rolled back by ``_restore_from_disk`` alone (this save holds the key's only pending
        mark, as ``save_settings`` does); this method only asks the menus to re-render."""
        days = int(days)
        tr = self.cfg.transcription
        # mark + set in one step under _dirty_lock: a reload can't revert the click between
        # them; from here this save holds the key's one pending mark, dropped once in apply()
        self._mark_and_apply((RETENTION_KEY,), lambda: setattr(tr, "audio_retention_days", days))

        def apply() -> None:
            try:
                saved = self._write_settings(RETENTION_KEY, title="Delete audio after")
            finally:
                self._mark_dirty((RETENTION_KEY,), -1)
            if saved is None:  # rolled back (or kept for a newer click): the shells re-render
                self.events.put(("retention_changed",))  # on the UI thread (cached tray menu)
                return
            # purge by the value actually written: an earlier worker may have written a newer
            # click's value, and memory may already hold a later, still unsaved one
            value = saved.get(RETENTION_KEY)
            with self._retention_lock:  # serialises menu purges; the check sees the latest
                if not _is_days(value) or self.cfg.transcription.audio_retention_days != value:
                    return  # a newer click owns the setting now
                if self._retention_on_disk() != value:
                    return  # a later save wrote another value; memory may be an unsaved click
                if n := len({wav.parent for wav in self.purge_audio_now(value) or ()}):
                    # count only: a mis-click is visible, no session names or text
                    what = f"{n} session" + ("s" if n != 1 else "")
                    self.post_notice("Delete audio after", f"Deleted audio of {what}")

        try:
            threading.Thread(target=apply, daemon=True, name="retention-set").start()
        except BaseException:
            self._mark_dirty((RETENTION_KEY,), -1)  # apply() will never unmark
            raise

    def sensitivity_preset(self) -> str | None:
        return current_preset(self.cfg.detection)

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
        close_settings()
        # Wait for an apply_hotkey in progress (a listener it is creating would leak), then
        # close what is registered; later applies see _shut and register nothing.
        got = self._apply_hotkey_lock.acquire(timeout=5)
        try:
            with self._hotkey_lock:
                hk, self._hotkey = self._hotkey, None
            if hk is not None:
                with contextlib.suppress(Exception):
                    hk.close()
        finally:
            if got:
                self._apply_hotkey_lock.release()
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


ROOM_ITEM = "Listen for conversations in the room"
SENSITIVITY_ITEM = "Sensitivity"
SENSITIVITY_HINT = "Room only (apps stay at 50); all but Off gate out noise"
ROOM_LISTENING_NOTICE = (
    "While idle, the microphone stays on (orange dot) to notice conversations near the "
    "laptop, also while the screen is locked, which costs some battery; nothing leaves the "
    "device. Recording people may need their consent: tell them."
)


def keep_prompt_text(app: str, kind: TriggerKind, timeout_action: str) -> str:
    """Body of the ask-mode dialog. Recording has already started when it is shown."""
    what = "a call" if kind == TriggerKind.CALL else "audio"
    where = "near the laptop" if kind == TriggerKind.ROOM else f"in {app}"
    if kind == TriggerKind.ROOM:
        what = "a conversation"
    default = "kept" if timeout_action != "ignore" else "discarded"
    return (
        f"RecAndCC is recording {what} {where}.\n\n"
        f"Keep the recording and transcribe it?\n(If you don't answer, it will be {default}.)"
    )


# Dialog button label -> Engine.respond() choice.
PROMPT_BUTTONS = (("Discard", "ignore"), ("Always", "always"), ("Keep", "record"))


__all__ = [
    "PROMPT_BUTTONS",
    "ROOM_ITEM",
    "SENSITIVITY_HINT",
    "SENSITIVITY_ITEM",
    "SENSITIVITY_PRESETS",
    "ROOM_LISTENING_NOTICE",
    "keep_prompt_text",
    "Controller",
    "IDLE",
    "PROMPTING",
    "RECORDING",
    "ONBOARDING_TEXT",
]
