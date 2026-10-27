"""Shared data types. Every module talks to the others through these."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

# All captured audio is normalised to this format before it leaves a capture backend:
# float32, mono, 16 kHz (what Whisper expects).
SAMPLE_RATE = 16_000

# Whisper languages written right to left (Hebrew, Yiddish, Arabic, Persian, Urdu, Pashto,
# Sindhi). Captions and transcripts render their segments right-aligned / with bidi marks.
RTL_LANGUAGES = frozenset({"he", "yi", "ar", "fa", "ur", "ps", "sd"})

log = logging.getLogger(__name__)

# SessionInfo.save: how often / how long to retry replacing a session.json someone has open.
_REPLACE_TRIES, _REPLACE_WAIT_S = 5, 0.05

# Guards SessionInfo.metrics updates against a concurrent save (several pipeline threads).
_METRICS_LOCK = threading.RLock()
# Serialises session.json writes in this process, so a title change (set_title) can't land
# between a save's read of the on-disk title and its replace.
_SAVE_LOCK = threading.Lock()


class Mode(StrEnum):
    OFF = "off"  # never record
    ASK = "ask"  # offer to record when audio is detected
    AUTO = "auto"  # record automatically when audio is detected


class Track(StrEnum):
    ME = "me"  # microphone
    OTHERS = "others"  # system / loopback audio


class TriggerKind(StrEnum):
    CALL = "call"
    MEDIA = "media"
    MANUAL = "manual"


@dataclass(frozen=True)
class AudioProcess:
    """A process currently doing audio I/O, as reported by an ActivityMonitor."""

    pid: int
    bundle_id: str  # macOS bundle id ("us.zoom.xos") or Windows exe name ("zoom.exe")
    name: str  # human readable name ("zoom.us")
    input_active: bool  # process is capturing from a microphone
    output_active: bool  # process is playing audio


@dataclass(frozen=True)
class AppTarget:
    """The app a session records, for a per-app system-audio tap: its processes' pids plus
    patterns (case-insensitive substrings of bundle id / name) that also catch helper
    processes it starts later, like the patterns detection groups an app's processes by."""

    match: tuple[str, ...] = ()
    pids: tuple[int, ...] = ()


@dataclass
class Segment:
    start: float  # seconds from session start
    end: float
    text: str
    speaker: str  # "Me", "Others" (later: "Speaker 1", ...)
    language: str  # ISO 639-1 code, e.g. "en", "ru", "he"
    overlap: bool = False  # diarization: two other speakers at once (json only, when True)


def is_rtl(language: str, text: str = "") -> bool:
    """True if a segment reads right to left: by its language, or (no language) by the first
    strong-direction character of ``text``."""
    if language:
        return language in RTL_LANGUAGES
    for ch in text:
        kind = unicodedata.bidirectional(ch)
        if kind in ("R", "AL"):
            return True
        if kind == "L":
            return False
    return False


@dataclass
class SessionInfo:
    """One recording. Persisted as <directory>/session.json."""

    id: str
    directory: Path
    started_at: datetime
    trigger_kind: TriggerKind
    trigger_app: str | None = None
    ended_at: datetime | None = None
    # Track -> WAV filename relative to `directory` (16 kHz mono).
    tracks: dict[Track, str] = field(default_factory=dict)
    transcribed: bool = False
    # Pipeline measurements, by stage ("recording", "live", "transcription", ...): plain
    # JSON values. Absent in session.json files written before it existed.
    metrics: dict[str, dict] = field(default_factory=dict)
    # Display title the user gave it in Meeting notes (None: never renamed, "": reset).
    # Changed only by set_title; save() keeps whatever title the file already has.
    title: str | None = None

    FILENAME = "session.json"
    # Present in a folder the user discarded: never transcribe or list it (discard_session).
    DISCARD_MARKER = ".discarded"

    def update_metrics(self, stage: str, values: dict) -> None:
        """Merge `values` into ``metrics[stage]`` (thread-safe; not saved until ``save``)."""
        with _METRICS_LOCK:
            self.metrics[stage] = {**self.metrics.get(stage, {}), **values}

    def save(self) -> None:
        with _METRICS_LOCK:
            data = asdict(self)
        with _SAVE_LOCK:
            # The title on disk wins: a copy loaded before (or after) a rename never undoes it.
            target = self.directory / self.FILENAME
            for attempt in range(_REPLACE_TRIES):
                try:
                    data["title"] = json.loads(target.read_text("utf-8")).get("title")
                    break
                except FileNotFoundError:
                    break  # a new session: its own title
                except (OSError, ValueError, AttributeError):  # locked / half-written (Windows)
                    if attempt + 1 < _REPLACE_TRIES:
                        time.sleep(_REPLACE_WAIT_S)
            self._write(data)

    @classmethod
    def set_title(cls, directory: Path, title: str) -> None:
        """Change only the ``title`` key of a session.json ("" resets it)."""
        d = Path(directory)
        with _SAVE_LOCK:
            data = json.loads((d / cls.FILENAME).read_text(encoding="utf-8"))
            data["title"] = title
            _replace_json(d / cls.FILENAME, json.dumps(data, indent=2, ensure_ascii=False))

    def _write(self, data: dict) -> None:
        data["directory"] = str(self.directory)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        data["tracks"] = {str(k): v for k, v in self.tracks.items()}
        _replace_json(
            self.directory / self.FILENAME, json.dumps(data, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, directory: Path) -> SessionInfo:
        data = json.loads((Path(directory) / cls.FILENAME).read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            directory=Path(directory),
            started_at=datetime.fromisoformat(data["started_at"]),
            trigger_kind=TriggerKind(data["trigger_kind"]),
            trigger_app=data.get("trigger_app"),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            tracks={Track(k): v for k, v in data.get("tracks", {}).items()},
            transcribed=data.get("transcribed", False),
            metrics=data.get("metrics") or {},
            title=t if isinstance(t := data.get("title"), str) else None,
        )


def _replace_json(target: Path, text: str) -> None:
    """Write ``target`` atomically, so a crash mid-write can't leave it truncated; the temp
    name is unique per process and thread (the app and the CLI may both save)."""
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, target)
                return
            except PermissionError:  # Windows: someone (antivirus, a reader) has it open
                if attempt + 1 < _REPLACE_TRIES:
                    time.sleep(_REPLACE_WAIT_S)
        log.warning("%s stays in use; writing it in place", target)
        target.write_text(text, encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def mark_discarded(directory: Path) -> None:
    """Mark a session folder as discarded, so it is never transcribed or listed again: create
    ``SessionInfo.DISCARD_MARKER``, or if that fails rename session.json to it, or as a last
    resort delete session.json. Never raises."""
    d = Path(directory)
    marker, meta = d / SessionInfo.DISCARD_MARKER, d / SessionInfo.FILENAME
    try:
        marker.touch()
        return
    except OSError:
        pass
    try:
        os.replace(meta, marker)
    except FileNotFoundError:
        pass
    except OSError:
        with contextlib.suppress(OSError):
            meta.unlink(missing_ok=True)


def discard_session(directory: Path) -> bool:
    """Delete a session folder the user discarded; True once it is gone.

    The folder is marked (``mark_discarded``) before any audio is touched and the marker is
    deleted last. So if a file can't be deleted (e.g. Windows still locks a WAV) the leftover
    is never transcribed and is deleted again at the next start. Files that vanish meanwhile
    are fine. May take a while (big WAVs): call it off latency-sensitive threads."""
    d = Path(directory)
    marker, meta = d / SessionInfo.DISCARD_MARKER, d / SessionInfo.FILENAME
    failed: list[str] = []
    for _ in range(2):  # a second pass catches files written meanwhile (e.g. a racing purge)
        mark_discarded(d)  # every pass runs marked
        failed = []
        try:
            entries = list(d.iterdir())
        except FileNotFoundError:
            return True
        except OSError as e:
            failed.append(str(e))
            break
        # Audio and transcripts first, then session.json, the marker last.
        for p in sorted(entries, key=lambda p: (p == marker, p == meta)):
            if p == marker and failed:
                break
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except OSError as e:
                failed.append(f"{p.name}: {e.strerror or e}")
        if not failed:
            try:
                d.rmdir()
                return True
            except FileNotFoundError:
                return True
            except OSError as e:
                failed.append(str(e))
    with contextlib.suppress(OSError):
        if not d.exists():
            return True
    log.warning("could not delete discarded session %s: %s", d, "; ".join(failed))
    return False
