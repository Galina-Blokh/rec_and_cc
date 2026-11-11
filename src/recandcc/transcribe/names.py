"""Apply names found in the transcript (self-introductions, and names the others address a
speaker by) after diarization and call-app naming, before summary callbacks, and the checked
names the summary model reads from the conversation (speaker_names, ``apply_names``). Session
labels only: no new voice profile is learned from them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from ..models import Segment, SessionInfo
from . import voices
from .address_names import addressed_names
from .intro_hints import confident_names, introduced_names
from .output import JSON_NAME, write_outputs


def apply_introductions(
    directory: Path,
    *,
    sync_voices: bool = False,
    single_labels: Collection[str] = (),
    since: int | None = None,
) -> dict[str, str]:
    """Save confident names to all transcript formats and rekey existing session voices only
    when voice storage is enabled. A failed write restores the original transcript. No text
    or names are logged; the caller may log the number of assignments."""

    def plan(segments: list[Segment]) -> dict[str, str]:
        mapping = confident_names(segments, single_labels=single_labels)
        claims = introduced_names(segments)  # any introduction vetoes a contradicting address
        return {
            **addressed_names(
                segments,
                single_labels=single_labels,
                skip={*mapping, *claims},
                reserved={*mapping.values(), *(n for names in claims.values() for n in names)},
            ),
            **mapping,
        }

    return _apply(Path(directory), plan, sync_voices=sync_voices, since=since)


def apply_names(
    directory: Path,
    mapping: dict[str, str],
    *,
    checked_on: list[Segment] | None = None,
    sync_voices: bool = False,
    since: int | None = None,
) -> dict[str, str]:
    """Save an already checked label -> name plan (e.g. from speaker_names) the same way.
    ``checked_on``: the segments the plan was checked against; if the transcript changed
    since, nothing is applied. Only labels still in the transcript and names that aren't a
    label yet are applied; returns what was applied."""

    def key(segments: list[Segment]) -> list[tuple]:
        return [(s.start, s.end, s.text, s.speaker) for s in segments]

    def plan(segments: list[Segment]) -> dict[str, str]:
        if checked_on is not None and key(segments) != key(checked_on):
            return {}  # changed while the model was reading it: the evidence may be gone
        labels = {s.speaker for s in segments}
        taken = {label.casefold() for label in labels}
        return {
            label: name
            for label, name in mapping.items()
            if label in labels and name.casefold() not in taken
        }

    return _apply(Path(directory), plan, sync_voices=sync_voices, since=since)


def _apply(
    directory: Path,
    plan: Callable[[list[Segment]], dict[str, str]],
    *,
    sync_voices: bool,
    since: int | None,
) -> dict[str, str]:
    with voices.NAMING_LOCK, voices.store_lock() if sync_voices else nullcontext():
        session = SessionInfo.load(directory)
        data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
        segments = [Segment(**s) for s in data["segments"]]
        mapping = plan(segments)
        if not mapping:
            return {}
        stored = voices.read_session_voices(session.id) if sync_voices else {}
        if sync_voices and since is None:
            since = voices.forget_generation()
        named = [replace(s, speaker=mapping.get(s.speaker, s.speaker)) for s in segments]
        try:
            write_outputs(session, named, data["duration"], data["model"])
            if stored:
                renamed = {mapping.get(label, label): value for label, value in stored.items()}
                voices.write_session_voices(
                    session,
                    {label: value["centroid"] for label, value in renamed.items()},
                    {
                        label: value["profile"]
                        for label, value in renamed.items()
                        if value["profile"]
                    },
                    since=since,
                )
        except BaseException:
            write_outputs(session, segments, data["duration"], data["model"])
            raise
        return mapping
