"""Names from the call app's participant list for `Speaker N` labels, conservatively.

The sampler (``participants.py``) saves the display names it saw in ``session.participants``.
They are only *assigned* when there is nothing to guess: exactly one unnamed speaker is left
and exactly one participant name is not yet taken. Anything else (several unknown speakers and
several names, since the list doesn't say who is who) is only suggested. A voice-profile match
and ``Me`` always take precedence, because they are never touched. Assigning goes through
``voices.name_speaker``, so profiles and transcripts stay consistent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..models import SessionInfo
from . import voices
from .merge import ME

log = logging.getLogger(__name__)

_UNKNOWN = re.compile(r"^(speaker \d+|others)$", re.IGNORECASE)


def split_candidates(labels: list[str], participants: list[str]) -> tuple[list[str], list[str]]:
    """(labels still unnamed, participant names not yet used by another speaker)."""
    unnamed = [lbl for lbl in labels if lbl != ME and _UNKNOWN.match(lbl)]
    used = {lbl.casefold() for lbl in labels}
    return unnamed, [n for n in participants if n.casefold() not in used]


def plan(labels: list[str], participants: list[str]) -> dict[str, str]:
    """label -> name to assign: only the one-unknown, one-free-name case."""
    unnamed, free = split_candidates(labels, participants)
    return {unnamed[0]: free[0]} if len(unnamed) == 1 and len(free) == 1 else {}


def suggestions(directory: Path) -> list[str]:
    """Participant names a user could still give to unnamed speakers (for `speakers`)."""
    session = SessionInfo.load(directory)
    labels = [lbl for lbl, _ in voices.session_speakers(directory)]
    return split_candidates(labels, session.participants)[1] if session.participants else []


def apply(directory: Path) -> bool:
    """Name the single unknown speaker after the single free participant, if that is what the
    session has. True if a speaker was named. Never logs names."""
    session = SessionInfo.load(directory)
    if not session.participants:
        return False
    labels = [lbl for lbl, _ in voices.session_speakers(directory)]
    for label, name in plan(labels, session.participants).items():
        try:
            voices.name_speaker(directory, label, name)
        except (OSError, ValueError):
            log.info("session %s: could not name a speaker from the call app", session.id)
            return False
        log.info("session %s: named a speaker from the call app", session.id)
        return True
    return False
