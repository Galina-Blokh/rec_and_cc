"""Opt-in: participant names from the call app's visible participant list (macOS Accessibility).

Off by default (``transcription.names_from_call_app``) and only used with ``speaker_names``.
While a call is recorded, a worker thread asks the ``rcc-audiotap participants`` helper every
~15 s for the names in the triggering app's participants panel, if the user has opened it. The
union is kept in ``session.json`` (``participants``) and used after diarization, see
``transcribe/callnames.py``. Names are personal data: never logged above DEBUG.

Accessibility trust is only ever *checked* here (``AXIsProcessTrusted`` via the helper), never
requested: the single place that sends the user to the Accessibility settings is the explicit
menu action, see ``open_accessibility_settings``. Windows: everything here is a no-op.
"""

from __future__ import annotations

import json
import logging
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
MAX_NAMES = 60
MAX_NAME_LEN = 60
ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

# "Anna (Me)", "Anna (You)": the user's own entry, who is already `Me`.
_SELF = re.compile(r"\((me|you|host,\s*me|я|אני)\)", re.IGNORECASE)
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
        text = str(item).strip()
        if not text or _SELF.search(text):
            continue
        text = _ROLE.sub("", text.split(",")[0]).strip()  # Zoom: "Anna, Computer audio muted"
        if not text or len(text) > MAX_NAME_LEN or _NOT_A_NAME.match(text):
            continue
        if text.casefold() not in {n.casefold() for n in out}:
            out.append(text)
    return out


def _run_helper(args: list[str]) -> dict:
    from .native import find_helper

    result = subprocess.run(  # noqa: S603 - fixed argv: our own bundled helper, no shell
        [str(find_helper()), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=HELPER_TIMEOUT_S,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"rcc-audiotap {args[0]} printed nothing (exit {result.returncode})")
    return json.loads(lines[-1])


def ax_trusted() -> bool:
    """Is Accessibility granted? Silent: the helper uses AXIsProcessTrusted, which never
    prompts. False when unknown or not on macOS."""
    if sys.platform != "darwin":
        return False
    try:
        return bool(_run_helper(["ax-status"]).get("trusted"))
    except Exception:  # noqa: BLE001 - the feature is optional
        log.debug("could not check Accessibility trust", exc_info=True)
        return False


def read_participants(target: AppTarget) -> list[str] | None:
    """Raw names in the app's visible participant list; None when Accessibility isn't granted
    (or the helper failed), [] when no participant list is visible."""
    args = ["participants"]
    for pid in target.pids:
        args += ["--pid", str(pid)]
    for pattern in target.match:
        args += ["--match", pattern]
    try:
        data = _run_helper(args)
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
        trusted: Callable[[], bool] = ax_trusted,
        read: Callable[[AppTarget], list[str] | None] = read_participants,
        interval: float = INTERVAL_S,
        first_after: float = FIRST_SAMPLE_S,
    ) -> None:
        self._target, self._trusted, self._read = target, trusted, read
        self._interval, self._first = interval, first_after
        self._names: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="participants", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[str]:
        """Stop sampling and return every name seen, in order of first appearance."""
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        with self._lock:
            return list(self._names)

    def _run(self) -> None:
        if not self._trusted():  # checked silently; nothing is ever read without it
            log.info("names from the call app: Accessibility is not granted; not sampling")
            return
        wait = self._first
        while not self._stop.wait(wait):
            wait = self._interval
            try:
                raw = self._read(self._target)
            except Exception:  # noqa: BLE001 - must not kill the thread
                log.debug("participant sampling failed", exc_info=True)
                continue
            if raw is None:
                continue
            with self._lock:
                for name in clean_names(raw):
                    if len(self._names) < MAX_NAMES and name.casefold() not in {
                        n.casefold() for n in self._names
                    }:
                        self._names.append(name)


def start_sampler(cfg: Config, target: AppTarget | None, **kwargs) -> ParticipantSampler | None:
    """A running sampler for a call session, or None when the feature is off, there is no
    triggering app, or this isn't macOS. (Accessibility trust is checked on the sampler's
    own thread, silently.)"""
    if target is None or not enabled(cfg):
        return None
    sampler = ParticipantSampler(target, **kwargs)
    sampler.start()
    return sampler


def store(session: SessionInfo, names: list[str]) -> None:
    """Keep the names with the session (session.json), only when there are any."""
    if names:
        session.participants = list(names)
        log.info("session %s: saw %d participant name(s)", session.id, len(names))
