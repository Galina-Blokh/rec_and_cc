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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .. import hotkey as hotkey_mod
from ..cloudsync import cloud_sync_provider, warning_text
from ..config import (
    CONFIG_PATH,
    LOG_DIR,
    Config,
    load_config,
    normalize_languages,
    save_keys,
    save_settings,
    validate_settings,
)
from ..engine import IDLE, PROMPTING, RECORDING, Engine, EngineState
from ..models import Mode, SessionInfo, TriggerKind, discard_session
from ..recorder import make_private_dir
from ..retention import purge_old_audio, purge_transcribed_audio
from ..settings_window import (
    SettingsUnavailable,
    close_settings,
    raise_settings,
    run_settings,
)

if TYPE_CHECKING:
    from ..transcribe.live import LiveTranscriber

log = logging.getLogger(__name__)

RECENT_COUNT = 10
SEARCH_LIMIT = 20
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
        self._lock = threading.Lock()
        self._perm: dict[str, str] = {}
        self._recent: list[Path] = []
        self._refreshing = False
        self._last_refresh = 0.0
        self._config_mtime = self._mtime()
        self._save_lock = threading.Lock()  # menu writes to config.toml
        self._settings_lock = threading.Lock()  # one settings window at a time
        self._lang_lock = threading.Lock()  # transcription.languages + its raw spelling
        self._summaries = None  # SummaryQueue, created on the first summary
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
            room_monitor=RoomMonitor(cfg.detection),  # mic only if room_listening
        )
        self._started_at = datetime.now().astimezone()  # older sessions: not diarized now
        if start_threads:
            if cfg.detection.room_listening:  # the mic opens right away: say so at every start
                self.post_notice("Room listening on", ROOM_LISTENING_NOTICE)
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
    @staticmethod
    def _index_session(directory: Path) -> None:
        try:
            from ..search import default_index

            default_index().index_session(directory)
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

                hits = search_transcripts(self.cfg.output_dir, query, SEARCH_LIMIT)
            except Exception as e:  # noqa: BLE001
                # the type only: an FTS5 syntax error quotes the query
                log.error("transcript search failed: %s", type(e).__name__)
                hits = e
            self.events.put(("search", query, hits))

        threading.Thread(target=run, daemon=True, name="search").start()

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
        """Delete expired audio at startup and then hourly (audio_retention_days); sync the
        search index hourly too (half an hour in, clear of the startup sync)."""
        while not getattr(self, "_shut", False):
            self.purge_audio_now()
            for i in range(3600):  # sleep 1 h, but notice shutdown promptly
                if i == 1800:
                    self._sync_index()
                if getattr(self, "_shut", False):
                    return
                time.sleep(1)

    def purge_audio_now(self) -> None:
        try:
            purge_old_audio(self.cfg.output_dir, self.cfg.transcription.audio_retention_days)
        except Exception:
            log.exception("audio retention failed")

    def _purge_leftover_audio(self) -> None:
        """keep_audio = false: delete WAVs left behind by transcribed sessions (e.g. the app
        quit before a slow diarization finished and dropped them, or the delete failed)."""
        if self.cfg.transcription.keep_audio:
            return
        try:
            purge_transcribed_audio(self.cfg.output_dir, ended_before=self._started_at)
        except Exception:
            log.exception("deleting leftover audio failed")

    def _resubmit_unfinished(self) -> None:
        # Purge first, on this same thread: once a re-queued session is transcribed, its
        # diarization may still need the WAVs, and the purge must not see it as leftover.
        self._resubmitting = True  # note_busy: untranscribed sessions may be about to queue
        try:
            self._purge_leftover_audio()
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

            default_index().sync(self.cfg.output_dir)
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
            self._check_cloud_sync()  # before listing output_dir, which may stall on a cloud mount
            perm = permission_status()
            recent = recent_transcripts(self.cfg.output_dir)
            with self._lock:
                self._perm, self._recent = perm, recent
            if self.engine.state.status == IDLE:  # e.g. marked stale by `name-speaker`
                self._resubmit_summaries()
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
        """Apply edits made to config.toml by hand (detection settings, output dir, hotkey,
        languages). Under ``_save_lock`` so a menu write never interleaves with the read."""
        with self._save_lock:
            m = self._mtime()
            if m == self._config_mtime:
                return
            self._config_mtime = m
            try:
                new = load_config()
            except Exception:
                log.exception("config.toml is invalid; keeping current settings")
                self.post_notice("Settings not applied", "config.toml could not be parsed")
                return
        was_listening = self.cfg.detection.room_listening
        for f in fields(new.detection):
            setattr(self.cfg.detection, f.name, getattr(new.detection, f.name))
        if self.cfg.detection.room_listening and not was_listening:  # turned on by hand
            self.post_notice("Room listening on", ROOM_LISTENING_NOTICE)
        self.cfg.output_dir = new.output_dir
        tr, new_tr = self.cfg.transcription, new.transcription
        with self._lang_lock:  # both together: save() pairs them
            tr._raw_languages = getattr(new_tr, "_raw_languages", None)  # type: ignore[attr-defined]
            tr.languages = new_tr.languages
        if new.hotkey != self.cfg.hotkey:
            self.cfg.hotkey = new.hotkey
            self.apply_hotkey(new.hotkey)
        log.info("config reloaded from %s", CONFIG_PATH)

    # --- settings changed from the menu (main thread; saved on a worker thread)
    def toggle_language(self, code: str) -> None:
        """Menu "Languages": check/uncheck ``code`` ("" = Any language). Applies at once: the
        transcriber reads ``cfg.transcription.languages`` for every piece of audio."""
        from .present import LANGUAGES_KEEP_ONE, toggle_languages

        tr = self.cfg.transcription
        with self._lang_lock:
            new = toggle_languages(tr.languages, getattr(tr, "_raw_languages", None), code)
            if new is not None:
                langs = normalize_languages(new)
                tr._raw_languages = new if new != langs else None  # type: ignore[attr-defined]
                tr.languages = langs  # a new list: other threads never see it half-built
        if new is None:
            self.post_notice("Languages", LANGUAGES_KEEP_ONE)
            return
        self.save_settings("transcription.languages", title="Languages")

    def save_settings(self, *keys: str, title: str = "Settings") -> None:
        """Save the in-memory value of each dotted key ("live_captions",
        "transcription.audio_retention_days") to config.toml on a worker thread. Menu
        callbacks use this instead of ``cfg.save()``, which would overwrite hand edits."""
        kw = {"title": title}
        t = threading.Thread(target=self.write_settings, args=keys, kwargs=kw, name="save-config")
        t.daemon = True
        t.start()

    def write_settings(self, *keys: str, title: str = "Settings") -> None:
        """Blocking part of ``save_settings``: a read-modify-write of only ``keys``, so a hand
        edit made since the last reload is kept (and picked up by the next refresh). The values
        are read under ``_save_lock``: of two quick menu saves, the last one wins."""
        with self._save_lock:
            values = {key: self._setting(key) for key in keys}
            edited = self._mtime() != self._config_mtime  # changed on disk, not reloaded yet
            try:
                try:
                    mtime = save_keys(CONFIG_PATH, values)
                except FileNotFoundError:
                    self.cfg.save(CONFIG_PATH)  # no file yet: write a fresh one
                    mtime = self._mtime()
                if not edited:
                    self._config_mtime = mtime  # our own write: nothing to reload
            except ValueError:  # invalid TOML (mid-edit?): don't overwrite the user's file
                log.warning("config.toml could not be parsed; %s not saved", ", ".join(keys))
                self.post_notice(f"{title} not saved", "config.toml could not be parsed")
            except Exception:
                log.exception("saving settings failed")
                self.post_notice("Settings not saved", "config.toml could not be written")

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
                raw = run_settings(self.cfg)
                if not raw:  # cancelled, closed, or nothing changed
                    return
                updates = validate_settings(raw)
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
                    }
                    old = {k: v for k, v in old.items() if k in fields}
                    old_raw = getattr(tr, "_raw_languages", None)
                    self.apply_settings(fields)
                    edited = self._mtime() != self._config_mtime  # hand edit not reloaded yet
                    try:
                        try:
                            mtime = save_settings(updates, CONFIG_PATH)
                        except FileNotFoundError:
                            self.cfg.save(CONFIG_PATH)  # no file yet: write a fresh one
                            mtime = save_settings(updates, CONFIG_PATH)
                    except Exception:
                        self.apply_settings(old)
                        with self._lang_lock:
                            tr._raw_languages = old_raw  # type: ignore[attr-defined]
                        raise
                    if not edited:
                        self._config_mtime = mtime  # our own write: nothing to reload
                    if "mode" in updates:
                        self.apply_settings({"mode": updates["mode"]})
                if "output_dir" in updates:
                    self.refresh_async(force=True)  # cloud-sync warning for the new folder
                self.events.put(("settings_saved", updates))
            except SettingsUnavailable as e:
                log.info("native settings unavailable: %s", e)
                self.events.put(("settings_fallback",))
            except ValueError as e:
                log.warning("settings not saved: %s", type(e).__name__)  # no values in logs
                self.post_notice("Settings not saved", str(e))
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
        self.cfg.detection.room_listening = bool(on)
        self.save_settings("detection.room_listening")
        if on:
            self.post_notice("Room listening on", ROOM_LISTENING_NOTICE)
            if self.permissions.get("microphone") == "denied":
                self.post_notice(
                    "Room listening can't hear anything",
                    "RecAndCC has no microphone access — grant it in the Permissions menu.",
                )

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


ROOM_ITEM = "Listen for conversations in the room"
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
    "ROOM_LISTENING_NOTICE",
    "keep_prompt_text",
    "Controller",
    "IDLE",
    "PROMPTING",
    "RECORDING",
    "ONBOARDING_TEXT",
]
