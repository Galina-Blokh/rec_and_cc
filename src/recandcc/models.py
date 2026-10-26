"""Shared data types. Every module talks to the others through these."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

# All captured audio is normalised to this format before it leaves a capture backend:
# float32, mono, 16 kHz (what Whisper expects).
SAMPLE_RATE = 16_000


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


@dataclass
class Segment:
    start: float  # seconds from session start
    end: float
    text: str
    speaker: str  # "Me", "Others" (later: "Speaker 1", ...)
    language: str  # ISO 639-1 code, e.g. "en", "ru"


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

    FILENAME = "session.json"

    def save(self) -> None:
        data = asdict(self)
        data["directory"] = str(self.directory)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        data["tracks"] = {str(k): v for k, v in self.tracks.items()}
        (self.directory / self.FILENAME).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
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
        )
