"""Opt-in: participant names from the call app's visible participant list (macOS Accessibility).

Off by default (``transcription.names_from_call_app``) and only used with ``speaker_names``.
While a call is recorded, a worker thread asks the ``rcc-audiotap participants`` helper every
~15 s for the names in the triggering app's participants panel, if the user has opened it. The
union is kept in the owner-only voices store (not the session folder) and used after
diarization, see ``transcribe/callnames.py``. Names are personal data: never logged above DEBUG.

Accessibility trust is only ever *checked* here (``AXIsProcessTrusted`` via the helper), never
requested: the single place that sends the user to the Accessibility settings is the explicit
menu action, see ``open_accessibility_settings``. Windows: everything here is a no-op.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from collections.abc import Callable

from .config import Config
from .models import AppTarget, SessionInfo

log = logging.getLogger(__name__)

INTERVAL_S = 15.0
FIRST_SAMPLE_S = 3.0  # the panel is usually opened a little after the call starts
HELPER_TIMEOUT_S = 8.0
STOP_JOIN_S = 5.0  # hard cap; a killed helper ends at once
MAX_NAMES = 60
MAX_NAME_LEN = 60
ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

# "Anna (Me)", "Anna (You)", or a bare "You": the user's own entry, who is already `Me`.
_SELF = re.compile(r"\((me|you|host,\s*me|я|אני)\)|^(me|you|я|ты|אני)$", re.IGNORECASE)
_ROLE = re.compile(r"\s*\((host|co-host|guest|organizer|presenter|external)\)\s*$", re.IGNORECASE)
_NOT_A_NAME = re.compile(r"^(participants?|people|attendees?|in this call|\d+)\b", re.IGNORECASE)


def enabled(cfg: Config) -> bool:
    """The feature is on (needs speaker_names too) and this is macOS."""
    tr = cfg.transcription
    return sys.platform == "darwin" and tr.speaker_names and tr.names_from_call_app


def clean_names(raw: list[str]) -> list[str]:
    """Display names from raw list rows: own entry, roles, status text and duplicates dropped."""
    out: list[str] = []
    for item in raw:
        parts = [p.strip() for p in str(item).split("\n") if p.strip()]  # one per text of a row
        if not parts or any(_SELF.search(p) for p in parts):
            continue
        text = _ROLE.sub("", parts[0].split(",")[0]).strip()  # Zoom: "Anna, Computer audio muted"
        if not text or len(text) > MAX_NAME_LEN or _NOT_A_NAME.match(text):
            continue
        if text.casefold() not in {n.casefold() for n in out}:
            out.append(text)
    return out


def _run_helper(
    args: list[str], on_start: Callable[[subprocess.Popen], None] | None = None
) -> dict:
    """Run the helper and parse its last JSON line. ``on_start(proc)`` lets the caller kill it."""
    from .native import find_helper

    # `participants` refuses to run unless this process started it (--parent-pid)
    argv = [str(find_helper()), *args]
    if args[0] == "participants":  # ax-status takes no options
        argv += ["--parent-pid", str(os.getpid())]
    proc = subprocess.Popen(  # noqa: S603 - fixed argv: our own bundled helper, no shell
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if on_start is not None:
        on_start(proc)
    try:
        out, _ = proc.communicate(timeout=HELPER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    lines = [line for line in (out or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"rcc-audiotap {args[0]} printed nothing (exit {proc.returncode})")
    return json.loads(lines[-1])


def ax_trusted(on_start: Callable[[subprocess.Popen], None] | None = None) -> bool:
    """Is Accessibility granted? Silent: the helper uses AXIsProcessTrusted, which never
    prompts. False when unknown or not on macOS."""
    if sys.platform != "darwin":
        return False
    try:
        return bool(_run_helper(["ax-status"], on_start).get("trusted"))
    except Exception:  # noqa: BLE001 - the feature is optional
        log.debug("could not check Accessibility trust", exc_info=True)
        return False


def read_participants(
    target: AppTarget, on_start: Callable[[subprocess.Popen], None] | None = None
) -> list[str] | None:
    """Raw names in the app's visible participant list; None when Accessibility isn't granted
    (or the helper failed), [] when no participant list is visible."""
    args = ["participants"]
    for pid in target.pids:
        args += ["--pid", str(pid)]
    for pattern in target.match:
        args += ["--match", pattern]
    try:
        data = _run_helper(args, on_start)
    except Exception:  # noqa: BLE001
        log.debug("could not read the participant list", exc_info=True)
        return None
    if not data.get("trusted"):
        return None
    return [str(n) for n in data.get("names") or []]


def open_accessibility_settings() -> None:
    """Open System Settings > Privacy & Security > Accessibility. The only thing the app does
    towards getting this permission: the user adds RecAndCC there themselves."""
    if sys.platform != "darwin":
        return
    subprocess.run(  # noqa: S603 - fixed argv (absolute path), constant URL, no shell
        ["/usr/bin/open", ACCESSIBILITY_URL],
        check=False,
    )


class ParticipantSampler:
    """Collects the names of a call's participants on its own thread until ``stop()``."""

    def __init__(
        self,
        target: AppTarget,
        *,
        trusted: Callable[[], bool] | None = None,
        read: Callable[[AppTarget], list[str] | None] | None = None,
        interval: float = INTERVAL_S,
        first_after: float = FIRST_SAMPLE_S,
    ) -> None:
        self._target = target
        self._proc: subprocess.Popen | None = None  # the helper run in flight, if any
        # by default the helper runs are tracked so that signal() can kill an in-flight one
        self._trusted = trusted or (lambda: ax_trusted(self._track))
        self._read = read or (lambda t: read_participants(t, self._track))
        self._interval, self._first = interval, first_after
        self._names: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="participants", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _track(self, proc: subprocess.Popen) -> None:
        with self._lock:
            self._proc = proc
            if self._stop.is_set():  # stopped while the helper was starting
                with contextlib.suppress(OSError):
                    proc.kill()

    def signal(self) -> None:
        """Ask the thread to end and kill the helper run in flight (returns at once);
        ``stop()`` then collects the names."""
        with self._lock:  # same lock as _track: a helper starting right now is killed there
            self._stop.set()
            proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()

    def stop(self) -> list[str]:
        """Stop sampling, wait for the killed helper and the thread (capped), and return every
        stable name seen, in order of first appearance. Call it after ``signal()`` and after
        whatever must not wait for the helper (the recorder flushes its audio first)."""
        self.signal()
        with self._lock:
            proc = self._proc
        if proc is not None:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)  # reaped: no helper outlives the session
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=STOP_JOIN_S)
        with self._lock:
            return list(self._names)

    def _run(self) -> None:
        wait, warned, previous = self._first, False, []
        while not self._stop.wait(wait):
            wait = self._interval
            try:
                # Trust is checked silently at every sample, so granting it mid-call works and
                # nothing is ever read while it is missing.
                if not self._trusted():
                    if not warned:
                        log.info("names from the call app: Accessibility is not granted yet")
                        warned = True
                    continue
                raw = self._read(self._target)
            except Exception:  # noqa: BLE001 - must not kill the thread
                log.debug("participant sampling failed", exc_info=True)
                continue
            if raw is None:
                continue
            current = clean_names(raw)[:MAX_NAMES]
            # Only names seen in two samples in a row are stored (a list that flickers by, like
            # another panel or a half-drawn list, is not), and stored names are kept.
            stable = [n for n in current if n.casefold() in {p.casefold() for p in previous}]
            with self._lock:  # accumulated: names that scroll out of view are kept
                known = {n.casefold() for n in self._names}
                for n in stable:
                    if len(self._names) < MAX_NAMES and n.casefold() not in known:
                        self._names.append(n)
                        known.add(n.casefold())
            previous = current or previous


def start_sampler(cfg: Config, target: AppTarget | None, **kwargs) -> ParticipantSampler | None:
    """A running sampler for a call session, or None when the feature is off, there is no
    triggering app, or this isn't macOS. (Accessibility trust is checked on the sampler's
    own thread, silently.)"""
    if target is None or not enabled(cfg):
        return None
    sampler = ParticipantSampler(target, **kwargs)
    sampler.start()
    return sampler


def forget(session_id: str) -> None:
    """Drop a session's stored names (discarded session, or the option is off)."""
    from .transcribe import callnames

    callnames.forget(session_id)


def store(session: SessionInfo, names: list[str]) -> None:
    """Keep the names in the owner-only voices store (not in the session folder, which may
    sync to a cloud), only when there are any."""
    if not names:
        return
    from .transcribe import voices

    try:
        voices.write_session_names(session, names)
    except OSError:
        log.warning("session %s: could not keep the participant names", session.id)
        return
    log.info("session %s: saw %d participant name(s)", session.id, len(names))
