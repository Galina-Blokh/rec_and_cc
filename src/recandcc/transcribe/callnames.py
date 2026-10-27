"""Names from the call app's participant list for `Speaker N` labels, very conservatively.

The sampler (``participants.py``) keeps the display names it saw in the owner-only voices
store. They are only *assigned* when nothing is left to guess:

* the number of names equals the number of speakers other than `Me`,
* every speaker that already has a name (a voice-profile match) matches one of the names
  (case-insensitive, by shared word, so "Anna" matches "Anna K."), and
* exactly one `Speaker N` and one name remain, and a self-introduction ("I'm Bob") that
  disagrees with that name vetoes it.

Anything else is only listed by `recandcc speakers`. Assigning only rewrites the transcript
label: it never creates or updates a voice profile (a name read from someone else's window is
not good enough to be remembered as a voice; confirm it with `recandcc name-speaker`).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import replace
from pathlib import Path

from ..models import Segment, SessionInfo
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


def suggestions(directory: Path) -> list[str]:
    """Participant names a user could still give to unnamed speakers (for `speakers`)."""
    session = SessionInfo.load(directory)
    names = voices.read_session_names(session.id)
    labels = [lbl for lbl, _ in voices.session_speakers(directory)]
    return free_names(labels, names) if names else []


def _rename(directory: Path, session: SessionInfo, label: str, name: str) -> None:
    """Relabel the transcript files; keep the session's voice (if any) under the new label,
    with no profile."""
    name = voices.check_name(name)
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    if any(s.speaker.casefold() == name.casefold() for s in segments):
        raise ValueError("that name is already a speaker of this session")
    segments = [replace(s, speaker=name) if s.speaker == label else s for s in segments]
    write_outputs(session, segments, data["duration"], data["model"])
    stored = voices.read_session_voices(session.id)
    if label in stored:
        new = {(name if k == label else k): v for k, v in stored.items()}
        voices.write_session_voices(
            session,
            {k: v["centroid"] for k, v in new.items()},
            {k: v["profile"] for k, v in new.items() if v["profile"]},
        )


def apply(directory: Path) -> bool:
    """Name the one unknown speaker after the one free participant, if that is what the session
    has. True if the transcript was relabelled. Never logs names."""
    session = SessionInfo.load(directory)
    names = voices.read_session_names(session.id)
    if not names:
        return False
    labels = [lbl for lbl, _ in voices.session_speakers(directory)]
    for label, name in plan(labels, names).items():
        hint = transcript_hints(directory).get(label)
        if hint is not None and not same_person(hint.name, name):
            log.info("session %s: a self-introduction disagrees; no call-app name", session.id)
            return False
        try:
            _rename(Path(directory), session, label, name)
        except (OSError, ValueError):
            log.info("session %s: could not name a speaker from the call app", session.id)
            return False
        voices.delete_session_names(session.id)  # used: nothing more to keep
        log.info("session %s: named a speaker from the call app", session.id)
        return True
    return False
