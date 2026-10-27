"""Audio retention: delete recordings' audio once it is no longer wanted.

Only ``*.wav`` files of sessions that are fully transcribed are removed; transcripts
(md/srt/json) and session.json are always kept. Sessions still recording or waiting for
transcription are never touched.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from .models import SessionInfo

log = logging.getLogger("recandcc.retention")

# Choices offered in the menu (days; 0 = never delete).
RETENTION_CHOICES = (0, 1, 7, 30, 90)


def _transcribed_sessions(output_dir: Path) -> Iterator[SessionInfo]:
    """Finished, transcribed sessions under ``output_dir`` (unreadable ones are skipped)."""
    for meta in output_dir.glob(f"*/*/{SessionInfo.FILENAME}"):
        if (meta.parent / SessionInfo.DISCARD_MARKER).exists():
            continue  # being deleted whole by purge_discarded_sessions
        try:
            session = SessionInfo.load(meta.parent)
        except Exception:
            log.warning("skipping unreadable session %s", meta.parent)
            continue
        if session.transcribed and session.ended_at is not None:
            yield session


def _unlink_all(paths, deleted: list[Path]) -> None:
    for wav in paths:
        try:
            wav.unlink()
            deleted.append(wav)
        except FileNotFoundError:
            pass
        except OSError as e:  # e.g. a Windows file lock: retried on the next start
            log.warning("could not delete %s: %s", wav, e)


def _track_wavs(session: SessionInfo) -> list[Path]:
    """The session's track WAVs, only if each is a plain ``*.wav`` file directly inside the
    session folder: names come from session.json and must never point anywhere else."""
    out = []
    for name in session.tracks.values():
        name = str(name)
        if not name or Path(name).name != name or "\\" in name or not name.endswith(".wav"):
            log.warning("session %s: ignoring unexpected track name", session.id)
            continue
        path = session.directory / name
        if not path.is_symlink() and not path.exists():
            continue  # already deleted: the usual case, keep startup cheap
        if path.is_symlink() or path.resolve().parent != session.directory.resolve():
            log.warning("session %s: ignoring track outside its folder", session.id)
            continue
        out.append(path)
    return out


def purge_old_audio(output_dir: Path, days: int, now: datetime | None = None) -> list[Path]:
    """Delete audio of transcribed sessions that ended more than ``days`` days ago.
    Returns the deleted files. ``days <= 0`` disables retention."""
    if days <= 0 or not output_dir.is_dir():
        return []
    now = now or datetime.now().astimezone()
    cutoff = now - timedelta(days=days)
    deleted: list[Path] = []
    for session in _transcribed_sessions(output_dir):
        ended = session.ended_at
        if ended.tzinfo is None:
            ended = ended.astimezone()
        if ended <= cutoff:
            _unlink_all(session.directory.glob("*.wav"), deleted)
    if deleted:
        log.info("retention: deleted %d audio file(s) older than %d day(s)", len(deleted), days)
    return deleted


def purge_transcribed_audio(output_dir: Path, ended_before: datetime | None = None) -> list[Path]:
    """``keep_audio = false``: delete the track WAVs of every session already transcribed.

    Normally the transcriber deletes them right after transcription, but when diarization
    outlasts its wait that is deferred to a background callback; if the app quits before
    it runs, the WAVs would stay forever. Run at startup with ``ended_before`` = the app's
    start time, so a session recorded (and maybe still being diarized) in this run keeps
    its audio until the transcriber drops it. Returns the deleted files.
    """
    if not output_dir.is_dir():
        return []
    deleted: list[Path] = []
    for session in _transcribed_sessions(output_dir):
        ended = session.ended_at
        if ended.tzinfo is None:
            ended = ended.astimezone()
        if ended_before is not None and ended >= ended_before:
            continue
        _unlink_all(_track_wavs(session), deleted)
    if deleted:
        log.info("keep_audio = false: deleted %d leftover audio file(s)", len(deleted))
    return deleted
