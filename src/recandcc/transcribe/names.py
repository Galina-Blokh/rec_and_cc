"""Apply names found in the transcript (self-introductions, and names the others address a
speaker by) after diarization and call-app naming, before summary callbacks. Session labels
only: no new voice profile is learned from them.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from ..models import Segment, SessionInfo
from . import voices
from .address_names import addressed_names
from .intro_hints import confident_names
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
    directory = Path(directory)
    with voices.NAMING_LOCK, voices.store_lock() if sync_voices else nullcontext():
        session = SessionInfo.load(directory)
        data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
        segments = [Segment(**s) for s in data["segments"]]
        mapping = confident_names(segments, single_labels=single_labels)
        mapping = {
            **addressed_names(
                segments,
                single_labels=single_labels,
                skip=mapping,
                reserved=mapping.values(),
            ),
            **mapping,
        }
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
