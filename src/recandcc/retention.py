"""Audio retention: delete recordings' audio N days after they were transcribed.

Only ``*.wav`` files of sessions that are fully transcribed are removed; transcripts
(md/srt/json) and session.json are always kept. Sessions still recording or waiting for
transcription are never touched.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

from .models import SessionInfo

log = logging.getLogger("recandcc.retention")

# Choices offered in the menu (days; 0 = never delete).
RETENTION_CHOICES = (0, 1, 7, 30, 90)


def purge_old_audio(output_dir: Path, days: int, now: datetime | None = None) -> list[Path]:
    """Delete audio of transcribed sessions that ended more than ``days`` days ago.
    Returns the deleted files. ``days <= 0`` disables retention."""
    if days <= 0 or not output_dir.is_dir():
        return []
    now = now or datetime.now().astimezone()
    cutoff = now - timedelta(days=days)
    deleted: list[Path] = []
    for meta in output_dir.glob(f"*/*/{SessionInfo.FILENAME}"):
        try:
            session = SessionInfo.load(meta.parent)
        except Exception:
            log.warning("skipping unreadable session %s", meta.parent)
            continue
        if not session.transcribed or session.ended_at is None:
            continue
        ended = session.ended_at
        if ended.tzinfo is None:
            ended = ended.astimezone()
        if ended > cutoff:
            continue
        for wav in session.directory.glob("*.wav"):
            try:
                wav.unlink()
                deleted.append(wav)
            except OSError:
                log.exception("could not delete %s", wav)
    if deleted:
        log.info("retention: deleted %d audio file(s) older than %d day(s)", len(deleted), days)
    return deleted
