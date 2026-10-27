"""Loading track audio and cutting it into speech chunks for Whisper."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from ..models import SAMPLE_RATE, Track

log = logging.getLogger("recandcc.transcribe")

MAX_CHUNK_S = 30.0  # Whisper's context window
MAX_MERGE_GAP_S = 2.0  # a longer pause always starts a new chunk (and a new language decision)


@dataclass(frozen=True)
class Chunk:
    """A piece of one track's audio, with its absolute position in the session."""

    track: Track
    start: float  # seconds from session start
    end: float
    language: str | None = None  # decided before transcription (None = detect per chunk)
    confidence: float = 1.0  # language confidence among the allowed languages

    @property
    def duration(self) -> float:
        return self.end - self.start


def load_track(path: Path) -> np.ndarray:
    """Read a WAV as float32 mono SAMPLE_RATE (down-mixes / resamples if it isn't already)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != SAMPLE_RATE:
        log.warning("%s is %d Hz, resampling to %d Hz", path.name, sr, SAMPLE_RATE)
        n = int(round(len(audio) * SAMPLE_RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(
            np.float32
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def find_speech(audio: np.ndarray) -> list[tuple[float, float]]:
    """Silero VAD speech regions in seconds. Empty for silence / music-only audio."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if len(audio) < SAMPLE_RATE // 4:
        return []
    opts = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=250,
        max_speech_duration_s=MAX_CHUNK_S - 1.0,
        min_silence_duration_ms=500,
        speech_pad_ms=200,
    )
    stamps = get_speech_timestamps(audio, opts, sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def merge_regions(
    regions: list[tuple[float, float]],
    max_chunk_s: float = MAX_CHUNK_S,
    max_gap_s: float = MAX_MERGE_GAP_S,
    labels: list[str] | None = None,
) -> list[tuple[float, float]]:
    """Greedily join consecutive speech regions into chunks of at most `max_chunk_s`.

    A new chunk starts when adding the next region would exceed `max_chunk_s`, when the pause
    before it is longer than `max_gap_s`, or (if `labels` is given, one per region) when its
    label (language) differs from the previous region's. A single region longer than
    `max_chunk_s` is split into equal pieces.
    """
    return [(s, e) for s, e, _ in merge_labelled(regions, labels, max_chunk_s, max_gap_s)]


def merge_labelled(
    regions: list[tuple[float, float]],
    labels: Sequence[str | None] | None = None,
    max_chunk_s: float = MAX_CHUNK_S,
    max_gap_s: float = MAX_MERGE_GAP_S,
) -> list[tuple[float, float, str | None]]:
    """Like merge_regions, but returns (start, end, label) triples."""
    items = sorted(zip(regions, labels or [None] * len(regions), strict=True))
    out: list[tuple[float, float, str | None]] = []
    for (start, end), label in items:
        if end <= start:
            continue
        if (
            out
            and out[-1][2] == label
            and start - out[-1][1] <= max_gap_s
            and end - out[-1][0] <= max_chunk_s
        ):
            out[-1] = (out[-1][0], max(out[-1][1], end), label)
            continue
        n = int(np.ceil((end - start) / max_chunk_s))
        step = (end - start) / n
        out.extend((start + i * step, start + (i + 1) * step, label) for i in range(n))
    return out


def slice_chunk(audio: np.ndarray, chunk: Chunk) -> np.ndarray:
    return audio[int(chunk.start * SAMPLE_RATE) : int(chunk.end * SAMPLE_RATE)]
