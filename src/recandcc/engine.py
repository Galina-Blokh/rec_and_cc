"""Detection engine: watches audio activity and drives the Recorder.

Threading model
---------------
* One background thread runs ``tick()`` every ``poll_interval_s`` (or immediately when a
  command arrives). ``tick()`` is the *only* place that calls ``recorder.start/stop`` and
  ``queue.submit``, so recorder operations are naturally serialised.
* UI threads call ``start_manual()``, ``stop()``, ``respond()``, ``set_mode()``: these just
  enqueue a command / mutate small state under ``self._lock`` and wake the thread.
* One lock (``self._lock``) guards all state. Nothing blocking (monitor snapshot, recorder
  start/stop, config save, callbacks) runs while it is held.
* ``on_session_start(session)`` / ``on_session_end(session | None, kept)`` (used for live
  transcription) run on the engine thread right after the recorder started / stopped, also
  outside the lock. ``on_session_end`` runs before the session is submitted to the queue (kept)
  or deleted (discarded); ``session`` is None if the recorder dropped it (too short) or failed.
* A discarded session is marked before ``recorder.stop()`` and deleted afterwards via
  ``run_in_background`` (a daemon thread), so big WAVs don't delay detection or the hotkey.
* Time comes from an injectable ``clock`` so tests can drive the state machine with a fake
  clock by calling ``tick()`` directly without starting the thread.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .config import Config, DetectionConfig
from .models import AudioProcess, Mode, SessionInfo, TriggerKind, discard_session, mark_discarded

log = logging.getLogger(__name__)

IDLE, PROMPTING, RECORDING = "idle", "prompting", "recording"


def _run_daemon(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="discard", daemon=True).start()


CHOICES = ("record", "ignore", "always")


# ---------------------------------------------------------------------------- rules
def match_pattern(patterns: Iterable[str], proc: AudioProcess) -> str | None:
    """First pattern that is a case-insensitive substring of bundle_id or name."""
    bid, name = (proc.bundle_id or "").lower(), (proc.name or "").lower()
    for p in patterns:
        pl = p.lower()
        if pl and (pl in bid or pl in name):
            return p
    return None


def classify(
    proc: AudioProcess, det: DetectionConfig, own_pids: Iterable[int] = ()
) -> tuple[str, TriggerKind] | None:
    """-> (app key, kind) or None if the process should be ignored.

    The key is the matched pattern for known apps (so e.g. Chrome and its helper processes
    collapse into one app) or the lower-cased bundle id / name for unknown apps.
    """
    if not (proc.input_active or proc.output_active):
        return None
    if proc.pid and proc.pid in set(own_pids):
        return None
    if match_pattern(det.ignore_apps, proc):
        return None
    if (p := match_pattern(det.call_apps, proc)) is not None:
        return p.lower(), TriggerKind.CALL
    if (p := match_pattern(det.media_apps, proc)) is not None:
        # A browser using the mic is almost certainly a web call (Meet, Teams web, ...).
        return p.lower(), TriggerKind.CALL if proc.input_active else TriggerKind.MEDIA
    key = (proc.bundle_id or proc.name or f"pid{proc.pid}").lower()
    # Unknown app using the mic -> call-like; unknown app only playing -> media.
    return key, TriggerKind.CALL if proc.input_active else TriggerKind.MEDIA


def effective_mode(det: DetectionConfig, key: str, procs: Iterable[AudioProcess] = ()) -> Mode:
    """Global OFF is a master switch; otherwise an ``app_modes`` pattern wins over ``mode``."""
    if Mode(det.mode) == Mode.OFF:
        return Mode.OFF
    procs = list(procs)
    for pattern, m in det.app_modes.items():
        pl = pattern.lower()
        if pl == key or pl in key or any(match_pattern([pattern], p) for p in procs):
            return Mode(m)
    return Mode(det.mode)


def display_name(name: str) -> str:
    """'Google Chrome Helper (Renderer)' -> 'Google Chrome'."""
    return re.sub(r"\s+Helper\b.*$", "", name or "").strip() or name


@dataclass
class AppActivity:
    key: str
    name: str
    kind: TriggerKind
    procs: list[AudioProcess]


def group_activity(
    procs: Iterable[AudioProcess], det: DetectionConfig, own_pids: Iterable[int] = ()
) -> dict[str, AppActivity]:
    own = set(own_pids)
    apps: dict[str, AppActivity] = {}
    for p in procs:
        c = classify(p, det, own)
        if c is None:
            continue
        key, kind = c
        a = apps.get(key)
        if a is None:
            apps[key] = AppActivity(key, display_name(p.name or p.bundle_id or key), kind, [p])
        else:
            a.procs.append(p)
            if kind == TriggerKind.CALL:
                a.kind = TriggerKind.CALL
            if p.name and len(display_name(p.name)) < len(a.name):
                a.name = display_name(p.name)
    return apps


# ---------------------------------------------------------------------------- state
@dataclass(frozen=True)
class EngineState:
    status: str  # idle | prompting | recording
    mode: Mode
    app: str | None = None  # app being prompted for / recorded
    kind: TriggerKind | None = None
    elapsed: float = 0.0  # seconds recording
    manual: bool = False
    prompt_id: int | None = None
    session_dir: str | None = None

    def _key(self):
        return (self.status, self.mode, self.app, self.kind, self.manual, self.prompt_id)


@dataclass
class _Prompt:
    id: int
    key: str
    name: str
    kind: TriggerKind
    deadline: float


@dataclass
class _Rec:
    key: str | None
    name: str | None
    kind: TriggerKind
    started: float
    manual: bool
    session: SessionInfo | None = None
    # Started while asking the user (ask mode): kept or discarded depending on the answer.
    provisional: bool = False


class Engine:
    def __init__(
        self,
        cfg: Config,
        monitor,  # ActivityMonitor
        recorder,  # Recorder-like: start(kind, app), stop(), seconds_since_sound()
        queue=None,  # TranscriptionQueue-like: submit(session)
        *,
        clock: Callable[[], float] = time.monotonic,
        on_state_change: Callable[[EngineState], None] | None = None,
        on_prompt: Callable[[str, TriggerKind], None] | None = None,
        save_config: Callable[[], None] | None = None,
        own_pids: Iterable[int] | None = None,
        on_notice: Callable[[str, str], None] | None = None,
        permission_status: Callable[[], dict[str, str]] | None = None,
        on_session_start: Callable[[SessionInfo], None] | None = None,
        on_session_end: Callable[[SessionInfo | None, bool], None] | None = None,
        run_in_background: Callable[[Callable[[], None]], None] | None = None,
    ):
        self.cfg = cfg
        self.monitor = monitor
        self.recorder = recorder
        self.queue = queue
        self.clock = clock
        self.on_state_change = on_state_change
        self.on_prompt = on_prompt
        # on_notice(title, message): user-facing warning (UI shows a notification).
        self.on_notice = on_notice
        # permission_status() -> {"microphone": ..., "system_audio": ...}; see permissions.py
        self.permission_status = permission_status
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        # run_in_background(fn): slow side work (deleting a discarded session); tests: inline.
        self.run_in_background = run_in_background or _run_daemon
        self._save_config = save_config if save_config is not None else cfg.save
        self.own_pids = set(own_pids) if own_pids is not None else {os.getpid()}

        self._lock = threading.Lock()
        self._first_seen: dict[str, float] = {}
        self._last_seen: dict[str, float] = {}
        self._suppressed: set[str] = set()
        self._commands: list[tuple] = []
        self._prompt: _Prompt | None = None
        self._rec: _Rec | None = None
        self._prompt_seq = 0
        self._last_state_key = None
        self._apps: dict[str, AppActivity] = {}

        self._wake = threading.Event()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_state_key = self._state_locked()._key()  # only report changes

    # ------------------------------------------------------------------ public API
    @property
    def det(self) -> DetectionConfig:
        return self.cfg.detection

    @property
    def state(self) -> EngineState:
        with self._lock:
            return self._state_locked()

    def active_apps(self) -> list[AppActivity]:
        with self._lock:
            return list(self._apps.values())

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop polling, finish any recording (submitting it for transcription)."""
        self._stop_evt.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        with self._lock:
            rec = self._rec
            self._rec, self._prompt = None, None
        if rec is not None:
            self._do_stop()
        try:
            self.monitor.close()
        except Exception:
            log.exception("monitor close failed")
        self._notify()

    def start_manual(self) -> None:
        self._command(("start_manual",))

    def stop(self) -> None:
        self._command(("stop",))

    def respond(self, choice: str, prompt_id: int | None = None) -> None:
        """Answer the current prompt: "record" | "ignore" | "always"."""
        choice = (choice or "").lower()
        if choice not in CHOICES:
            log.warning("unknown prompt response %r (treated as ignore)", choice)
            choice = "ignore"
        self._command(("respond", choice, prompt_id))

    def set_mode(self, mode: Mode | str) -> None:
        mode = Mode(mode)
        with self._lock:
            self.det.mode = mode
            if mode == Mode.OFF:
                if self._prompt is not None:
                    self._suppressed.add(self._prompt.key)
                    self._prompt = None
                if self._rec is not None and not self._rec.manual:
                    self._commands.append(("stop",))
        self._safe_save()
        self._wake.set()
        self._notify()

    # ------------------------------------------------------------------ internals
    def _command(self, cmd: tuple) -> None:
        with self._lock:
            self._commands.append(cmd)
        self._wake.set()

    def _run(self) -> None:
        try:
            while not self._stop_evt.is_set():
                try:
                    self.tick()
                except Exception:
                    log.exception("engine tick failed")
                self._wake.wait(max(0.05, float(self.det.poll_interval_s)))
                self._wake.clear()
        finally:
            # Per-thread resources (Windows COM) must be released by the thread that polled.
            # Only that, so shutdown()'s join budget isn't spent on the full close().
            try:
                self.monitor.release_thread()
            except Exception:
                log.exception("monitor thread release failed")

    def _state_locked(self) -> EngineState:
        mode = Mode(self.det.mode)
        if self._rec is not None:
            r = self._rec
            return EngineState(
                RECORDING,
                mode,
                r.name,
                r.kind,
                max(0.0, self.clock() - r.started),
                r.manual,
                self._prompt.id if self._prompt is not None else None,  # "keep it?" pending
                str(r.session.directory) if r.session else None,
            )
        if self._prompt is not None:
            p = self._prompt
            return EngineState(PROMPTING, mode, p.name, p.kind, prompt_id=p.id)
        return EngineState(IDLE, mode)

    def _notify(self) -> None:
        st = self.state
        if st._key() == self._last_state_key:
            return
        self._last_state_key = st._key()
        if self.on_state_change:
            try:
                self.on_state_change(st)
            except Exception:
                log.exception("on_state_change callback failed")

    def _safe_save(self) -> None:
        try:
            self._save_config()
        except Exception:
            log.exception("saving config failed")

    def tick(self) -> None:
        """One iteration of the state machine. Safe to call directly (tests)."""
        try:
            procs = self.monitor.snapshot()
        except Exception:
            log.exception("activity snapshot failed")
            procs = []
        silence = 0.0
        if self.recorder is not None and getattr(self.recorder, "is_recording", False):
            try:
                silence = float(self.recorder.seconds_since_sound())
            except Exception:
                log.exception("recorder level query failed")
        apps = group_activity(procs, self.det, self.own_pids)
        now = self.clock()

        start_req: tuple[TriggerKind, str | None] | None = None
        stop_req = False
        prompt_evt: tuple[str, TriggerKind] | None = None
        upgrade_req: str | None = None
        discard_req = False
        save = False

        with self._lock:
            det = self.det
            self._apps = apps
            # --- bookkeeping of continuous activity
            for key in apps:
                self._first_seen.setdefault(key, now)
                self._last_seen[key] = now
            for key in list(self._first_seen):
                if key not in apps:
                    del self._first_seen[key]
            for key in list(self._suppressed):
                if (
                    key not in apps
                    and now - self._last_seen.get(key, -1e18) >= det.stop_after_inactive_s
                ):
                    self._suppressed.discard(key)

            # --- commands from the UI
            cmds, self._commands = self._commands, []
            for cmd in cmds:
                if cmd[0] == "start_manual":
                    if self._rec is None:
                        self._prompt = None
                        self._rec = _Rec(None, None, TriggerKind.MANUAL, now, manual=True)
                        start_req = (TriggerKind.MANUAL, None)
                        stop_req = False
                    else:
                        self._rec.manual = True  # pin the running session: no auto-stop
                elif cmd[0] == "stop":
                    if self._rec is not None and start_req is None:
                        if self._rec.key:
                            self._suppressed.add(self._rec.key)
                        self._prompt = None  # explicit stop: keep what was recorded
                        stop_req = True
                    elif self._prompt is not None:
                        self._suppressed.add(self._prompt.key)
                        self._prompt = None
                elif cmd[0] == "respond":
                    _, choice, pid = cmd
                    p = self._prompt
                    if p is None or (pid is not None and pid != p.id):
                        continue
                    self._prompt = None
                    r = self._rec
                    pending = r is not None and r.provisional and r.key == p.key
                    if choice == "ignore":
                        self._suppressed.add(p.key)
                        if pending:
                            discard_req = True
                        continue
                    if choice == "always":
                        det.app_modes[p.key] = Mode.AUTO
                        save = True
                    if pending and r is not None:
                        r.provisional = False  # recording since the prompt appeared: keep it
                    elif r is None:
                        self._rec = _Rec(p.key, p.name, p.kind, now, manual=False)
                        start_req = (p.kind, p.name)

            # --- prompt expiry
            if self._prompt is not None and now >= self._prompt.deadline:
                keep = det.ask_timeout_action != "ignore"
                log.info(
                    "prompt for %s expired: %s the recording",
                    self._prompt.name,
                    "keeping" if keep else "discarding",
                )
                self._suppressed.add(self._prompt.key)
                r = self._rec
                if r is not None and r.provisional and r.key == self._prompt.key:
                    if keep:
                        r.provisional = False
                    else:
                        discard_req = True
                self._prompt = None

            # --- auto-stop
            r = self._rec
            if (
                r is not None
                and r.key is not None  # always set for non-manual sessions
                and not r.manual
                and start_req is None
                and not (stop_req or discard_req)
            ):
                last = max(self._last_seen.get(r.key, r.started), r.started)
                media = r.kind == TriggerKind.MEDIA
                inactive_s = det.media_stop_after_inactive_s if media else det.stop_after_inactive_s
                silence_s = det.media_stop_after_silence_s if media else det.stop_after_silence_s
                if r.key not in apps and now - last >= inactive_s:
                    log.info("stopping: %s inactive for %.0fs", r.name, now - last)
                    stop_req = True
                elif silence >= silence_s:
                    log.info("stopping: silent for %.0fs", silence)
                    self._suppressed.add(r.key)  # app may still hold an audio session
                    stop_req = True

            # Audio ended before the user answered: resolve the prompt by the timeout policy.
            if stop_req and r is not None and r.provisional:
                self._prompt = None
                if det.ask_timeout_action == "ignore":
                    stop_req, discard_req = False, True

            # --- a call started while a media session records (e.g. YouTube was playing when
            # Zoom rang, or the same browser tab now uses the mic): follow the call, so the mic
            # gets recorded and the session doesn't end when the video stops.
            r = self._rec
            if (
                r is not None
                and r.kind == TriggerKind.MEDIA
                and start_req is None
                and not (stop_req or discard_req)
            ):
                for a in apps.values():
                    if a.kind != TriggerKind.CALL:
                        continue
                    if a.key != r.key and (
                        a.key in self._suppressed
                        or now - self._first_seen[a.key] < det.min_active_s
                        or effective_mode(det, a.key, a.procs) == Mode.OFF
                    ):
                        continue
                    log.info("call detected during media recording: now following %s", a.name)
                    r.key, r.name, r.kind = a.key, a.name, TriggerKind.CALL
                    upgrade_req = a.name
                    break

            # --- new triggers
            if self._rec is None and self._prompt is None and start_req is None:
                cands = [
                    a
                    for k, a in apps.items()
                    if k not in self._suppressed and now - self._first_seen[k] >= det.min_active_s
                ]
                cands.sort(key=lambda a: (a.kind != TriggerKind.CALL, self._first_seen[a.key]))
                for a in cands:
                    m = effective_mode(det, a.key, a.procs)
                    if m == Mode.OFF:
                        continue
                    if m == Mode.AUTO:
                        log.info("auto-recording %s (%s)", a.name, a.kind)
                        self._rec = _Rec(a.key, a.name, a.kind, now, manual=False)
                        start_req = (a.kind, a.name)
                    else:
                        # Record right away and ask whether to keep it: an unanswered or
                        # unseen prompt must not lose the start of a meeting.
                        self._prompt_seq += 1
                        self._prompt = _Prompt(
                            self._prompt_seq, a.key, a.name, a.kind, now + det.ask_timeout_s
                        )
                        self._rec = _Rec(a.key, a.name, a.kind, now, manual=False, provisional=True)
                        start_req = (a.kind, a.name)
                        prompt_evt = (a.name, a.kind)
                        log.info("recording %s (%s), asking whether to keep it", a.name, a.kind)
                    break

        # --- side effects, outside the lock
        if save:
            self._safe_save()
        if discard_req:
            self._do_discard()
        if stop_req:
            self._do_stop()
        if start_req is not None:
            self._do_start(*start_req)
        if upgrade_req is not None:
            self._do_upgrade(upgrade_req)
        if prompt_evt is not None and self.on_prompt:
            try:
                self.on_prompt(*prompt_evt)
            except Exception:
                log.exception("on_prompt callback failed")
        self._notify()

    def _notice(self, title: str, message: str) -> None:
        log.warning("%s: %s", title, message)
        if self.on_notice:
            try:
                self.on_notice(title, message)
            except Exception:
                log.exception("on_notice callback failed")

    def _preflight(self) -> None:
        if self.permission_status is None:
            return
        try:
            st = self.permission_status() or {}
        except Exception:
            log.exception("permission status check failed")
            return
        if st.get("system_audio") == "denied":
            self._notice(
                "Recording may be silent",
                "RecAndCC has no system audio access — grant it in the Permissions menu.",
            )
        if st.get("microphone") == "denied":
            self._notice(
                "Your microphone won't be recorded",
                "RecAndCC has no microphone access — grant it in the Permissions menu.",
            )

    def _check_silent_tracks(self) -> None:
        silent = getattr(self.recorder, "silent_tracks", None)
        if silent is None:
            return
        try:
            tracks = list(silent())
        except Exception:
            return
        # A track of pure zeros only means "permission missing" if the permission isn't
        # granted: with nothing playing, the system-audio track is legitimately all zeros.
        perms: dict = {}
        if self.permission_status is not None:
            try:
                perms = self.permission_status() or {}
            except Exception:
                perms = {}
        needed = {"others": "system_audio", "me": "microphone"}
        tracks = [
            t
            for t in tracks
            if perms.get(needed.get(str(t), ""), "unknown") != "granted"
            and (perms or str(t) == "me")  # unknown status: only a dead-silent mic is telling
        ]
        if tracks:
            names = {"others": "system audio", "me": "microphone"}
            what = " and ".join(names.get(str(t), str(t)) for t in tracks)
            self._notice(
                "Recording was completely silent",
                f"The {what} track had no signal at all — the permission is probably "
                "missing. Check the Permissions menu.",
            )

    def _do_start(self, kind: TriggerKind, app: str | None) -> None:
        self._preflight()
        try:
            session = self.recorder.start(kind, app)
        except Exception:
            log.exception("could not start recording")
            with self._lock:
                if self._rec is not None and self._rec.key:
                    self._suppressed.add(self._rec.key)  # don't retry every second
                self._rec = None
            return
        with self._lock:
            if self._rec is not None:
                self._rec.session = session
                self._rec.started = self.clock()
        if self.on_session_start:
            try:
                self.on_session_start(session)
            except Exception:
                log.exception("on_session_start callback failed")

    def _session_end(self, session: SessionInfo | None, kept: bool) -> None:
        if self.on_session_end:
            try:
                self.on_session_end(session, kept)
            except Exception:
                log.exception("on_session_end callback failed")

    def _do_upgrade(self, app: str) -> None:
        upgrade = getattr(self.recorder, "upgrade_to_call", None)
        if upgrade is None:
            return
        try:
            upgrade(app)
        except Exception:
            log.exception("could not add the microphone to the running session")

    def _do_discard(self) -> None:
        """Stop the current session and delete it (the user chose not to keep it)."""
        with self._lock:
            started = self._rec.session if self._rec is not None else None
            self._rec = None
        if started is not None:  # marked first: a crash from here on can't keep it
            mark_discarded(started.directory)
        try:
            session = self.recorder.stop()
        except Exception:
            log.exception("could not stop recording cleanly")
            session = None
        self._session_end(session, False)
        # Also when stop() failed or dropped it as too short: whatever it wrote must go.
        session = session or started
        # Already gone (stop() dropped it as too short): don't schedule a delete, as the next
        # session may reuse the name. Only the engine thread creates folders, so no race here.
        if session is None or not session.directory.exists():
            return
        mark_discarded(session.directory)
        directory = session.directory
        self.run_in_background(lambda: self._delete_discarded(directory))

    def _delete_discarded(self, directory) -> None:
        """Runs off the engine thread: deleting big WAVs mustn't delay detection."""
        try:
            deleted = discard_session(directory)
        except Exception:
            log.exception("could not delete discarded recording %s", directory)
            return
        if deleted:
            log.info("discarded recording %s", directory)
        else:
            self._notice(
                "Recording not fully deleted",
                f"Some files in {directory} are in use; they will be deleted at the next start",
            )

    def _do_stop(self) -> None:
        with self._lock:
            self._rec = None
        try:
            session = self.recorder.stop()
        except Exception:
            log.exception("could not stop recording cleanly")
            self._session_end(None, False)
            return
        self._check_silent_tracks()
        self._session_end(session, session is not None)
        if session is None:
            return
        if self.queue is not None:
            try:
                self.queue.submit(session)
            except Exception:
                log.exception("could not submit session for transcription")
