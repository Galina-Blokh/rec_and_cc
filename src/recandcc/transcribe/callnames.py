"""Names from the call app's participant list for `Speaker N` labels, very conservatively.

The sampler (``participants.py``) keeps the display names it saw in the owner-only voices
store. They are only *assigned* when nothing is left to guess:

* the number of names equals the number of speakers other than `Me` (in a session that started
  as a room recording: the number of speakers on the call's track, see below),
* every speaker that already has a name (a voice-profile match) matches one of the names
  (case-insensitive, by shared word, so "Anna" matches "Anna K."), and
* exactly one `Speaker N` and one name remain, and a self-introduction ("I'm Bob") that
  disagrees with that name vetoes it.

Anything else is only listed by `recandcc speakers`. Assigning only rewrites the transcript
label: it never creates or updates a voice profile (a name read from someone else's window is
not good enough to be remembered as a voice; confirm it with `recandcc name-speaker`).

A room recording that a call joined (``started_as_room``) numbers the people at the laptop on
the mic track too, so the owner can be a `Speaker N`. There only the labels the transcript
marks as from the "others" (system audio) track are candidates and counted; a transcript
written before labels carried their track names nobody (fail closed).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path

from ..models import Segment, SessionInfo, Track
from . import voices
from .intro_hints import transcript_hints
from .merge import ME
from .output import JSON_NAME, write_outputs

log = logging.getLogger(__name__)

_UNKNOWN = re.compile(r"^speaker \d+$", re.IGNORECASE)
_WORD = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if len(w) >= 2}


def same_person(a: str, b: str) -> bool:
    """Two display names share a word ("Anna" and "anna k."). "Bob" and "Robert" do not."""
    return bool(_tokens(a) & _tokens(b))


def _others(labels: list[str]) -> list[str]:
    return [lbl for lbl in labels if lbl != ME]


def free_names(labels: list[str], participants: list[str]) -> list[str]:
    """Participant names no already-named speaker matches."""
    named = [lbl for lbl in _others(labels) if not _UNKNOWN.match(lbl)]
    return [n for n in participants if not any(same_person(lbl, n) for lbl in named)]


def plan(labels: list[str], participants: list[str]) -> dict[str, str]:
    """label -> name to assign: only when it is certain (see the module docstring)."""
    others = _others(labels)
    if not others or len(participants) != len(others):
        return {}
    unnamed = [lbl for lbl in others if _UNKNOWN.match(lbl)]
    free = list(participants)
    for lbl in others:
        if lbl in unnamed:
            continue
        hit = [n for n in free if same_person(lbl, n)]
        if len(hit) != 1:  # a named speaker the panel doesn't show: the panel can't be trusted
            return {}
        free.remove(hit[0])
    return {unnamed[0]: free[0]} if len(unnamed) == 1 and len(free) == 1 else {}


def call_side(directory: Path) -> list[str] | None:
    """Labels (except Me) whose every segment is marked as from the "others" track, in order of
    first appearance; None if any label's track is unknown or mixed (an older transcript)."""
    data = json.loads((Path(directory) / JSON_NAME).read_text(encoding="utf-8"))
    tracks: dict[str, set[str]] = {}
    for s in data.get("segments", []):
        if s["speaker"] != ME:
            tracks.setdefault(s["speaker"], set()).add(s.get("track", ""))
    if any(len(t) != 1 or not t <= {Track.ME, Track.OTHERS} for t in tracks.values()):
        return None
    return [lbl for lbl, t in tracks.items() if t == {Track.OTHERS}]


def forget(session_id: str) -> None:
    """Delete the names kept for a session. Never raises."""
    try:
        voices.delete_session_names(session_id)
    except OSError:
        log.warning("session %s: could not delete the participant names", session_id)


def suggestions(directory: Path) -> list[str]:
    """Participant names a user could still give to unnamed speakers (for `speakers`)."""
    session = SessionInfo.load(directory)
    names = voices.read_session_names(session.id)
    if not names:
        return []
    side = call_side(directory) if session.started_as_room else None
    labels = side if side is not None else [lbl for lbl, _ in voices.session_speakers(directory)]
    return free_names(labels, names)


class _Forgotten(Exception):
    """All voices (and the kept participant names) were deleted since the session began."""


def _rename(directory: Path, session: SessionInfo, label: str, name: str, since: int) -> None:
    """Relabel the transcript files; keep the session's voice (if any) under the new label,
    with no profile. Under NAMING_LOCK (``apply``), like ``voices.name_speaker``."""
    name = voices.check_name(name)
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    if label not in {s.speaker for s in segments}:
        raise ValueError("that speaker was renamed meanwhile")
    if session.started_as_room and {s.track for s in segments if s.speaker == label} != {
        Track.OTHERS
    }:
        raise ValueError("that speaker is not on the call's track")  # never the room/owner
    if any(s.speaker.casefold() == name.casefold() for s in segments):
        raise ValueError("that name is already a speaker of this session")
    if since != voices.forget_generation():  # by another process (this one's waits for us)
        raise _Forgotten
    segments = [replace(s, speaker=name) if s.speaker == label else s for s in segments]
    write_outputs(session, segments, data["duration"], data["model"])
    with voices.store_lock():  # read and write together; a forget since `since` drops it
        stored = voices.read_session_voices(session.id)
        if label in stored:
            new = {(name if k == label else k): v for k, v in stored.items()}
            voices.write_session_voices(
                session,
                {k: v["centroid"] for k, v in new.items()},
                {k: v["profile"] for k, v in new.items() if v["profile"]},
                since=since,
            )


def apply(directory: Path, since: int | None = None) -> bool:
    """Name the one unknown speaker after the one free participant, if that is what the session
    has. True if the transcript was relabelled. Never logs names.

    `since`: the voices' forget generation the session's work began at (default: the one the
    session recorded, else now); nothing is renamed or stored if all voices were deleted
    since. Runs under ``voices.NAMING_LOCK``, so it never interleaves with a naming, a
    finalize or this process's "forget all"; the caller must not hold it."""
    session = SessionInfo.load(directory)
    if since is None:
        since = session.voices_generation
    if since is None:
        since = voices.forget_generation()
    with voices.NAMING_LOCK:
        return _apply(Path(directory), session, since)


def _apply(directory: Path, session: SessionInfo, since: int) -> bool:
    if since != voices.forget_generation():
        return False  # forgotten since the session began: its names went with the voices
    names = voices.read_session_names(session.id)
    if not names:
        return False
    if session.started_as_room:
        # A room session also numbers the mic track, so with two people at the laptop the owner
        # is a "Speaker N" too: only labels marked as from the call's (system audio) track count.
        labels = call_side(directory)
        if labels is None:  # an older transcript without per-track labels: fail closed
            log.info("session %s: speaker tracks unknown; no call-app name", session.id)
            forget(session.id)
            return False
    else:  # only "Others" was diarized: every label but Me is the call's
        labels = [lbl for lbl, _ in voices.session_speakers(directory)]
    for label, name in plan(labels, names).items():
        hint = transcript_hints(directory).get(label)
        if hint is not None and not same_person(hint.name, name):
            log.info("session %s: a self-introduction disagrees; no call-app name", session.id)
            return False
        try:
            _rename(directory, session, label, name, since)
        except _Forgotten:
            log.info("session %s: voices were forgotten; no call-app name", session.id)
            return False
        except (OSError, ValueError):
            log.info("session %s: could not name a speaker from the call app", session.id)
            return False
        voices.delete_session_names(session.id)  # used: nothing more to keep
        log.info("session %s: named a speaker from the call app", session.id)
        return True
    return False
