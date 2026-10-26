"""Whisper transcription of a recorded session: VAD chunks of both tracks in parallel.

Pipeline per session:
  1. VAD each track (Silero) -> speech regions.
  2. Language per region with a small language-ID model (cheap; the main model would cost a
     full encoder pass per call), restricted to cfg.languages; short uncertain regions inherit
     their neighbour's language.
  3. Merge consecutive same-language regions into chunks <= 30 s.
  4. Transcribe all chunks of both tracks concurrently on one shared WhisperModel.
  5. Merge tracks, drop mic echo, write transcript.{md,json,srt}.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import TranscriptionConfig
from ..models import SAMPLE_RATE, Segment, SessionInfo, Track
from .chunking import Chunk, find_speech, load_track, merge_labelled, slice_chunk
from .merge import ME, OTHERS, language_confidence, merge_tracks, smooth_languages
from .output import write_outputs

log = logging.getLogger("recandcc.transcribe")

ProgressCallback = Callable[[float], None]
SPEAKERS = {Track.ME: ME, Track.OTHERS: OTHERS}

LANGUAGE_ID_MODEL = "base"  # small multilingual model used only for per-region language ID
MIN_LANGUAGE_CONFIDENCE = 0.75  # below this a chunk's language is re-checked with the main model


def auto_parallelism(workers: int = 0, cores: int | None = None) -> tuple[int, int]:
    """(num_workers, cpu_threads per worker). `workers` > 0 forces the worker count.

    Benchmarked on an M5 Max (18 cores), large-v3-turbo: 3-4 workers beat 1 worker x 18 threads
    by ~1.5-2x; more workers than ~cores/4 gives no further gain.
    """
    cores = cores or os.cpu_count() or 4
    n = workers if workers > 0 else max(1, min(4, cores // 4))
    return n, max(1, cores // n)


class Transcriber:
    """Owns one shared WhisperModel (loaded lazily, reused across sessions) and a thread pool.

    `transcribe_session` may be called from any thread; calls are serialised so that a session
    gets the whole pool, while its chunks (from both tracks) are transcribed concurrently.
    `language_model`: model used for per-region language ID; None = detect once per chunk with
    the main model (slower: doubles the encoder work).
    """

    def __init__(
        self, cfg: TranscriptionConfig, language_model: str | None = LANGUAGE_ID_MODEL
    ) -> None:
        self.cfg = cfg
        self.num_workers, self.cpu_threads = auto_parallelism(cfg.workers)
        self.language_model = language_model if language_model != cfg.model else None
        self._model = None
        self._lid_model = None
        self._pool: ThreadPoolExecutor | None = None
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()

    # -- models ------------------------------------------------------------------------------

    def _load(self, name: str):
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        model = WhisperModel(
            name,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            num_workers=self.num_workers,
            cpu_threads=self.cpu_threads,
        )
        log.info(
            "loaded whisper model %s (%s, %s, workers=%d, threads=%d) in %.1fs",
            name,
            self.cfg.device,
            self.cfg.compute_type,
            self.num_workers,
            self.cpu_threads,
            time.monotonic() - t0,
        )
        return model

    def _ensure_loaded(self):
        """Thread-safe lazy init of the model(s) and the worker pool."""
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    if self.language_model and self._needs_detection():
                        self._lid_model = self._load(self.language_model)
                    model = self._load(self.cfg.model)
                    self._pool = ThreadPoolExecutor(self.num_workers, thread_name_prefix="whisper")
                    self._model = model
        return self._model

    def _needs_detection(self) -> bool:
        return len(self.cfg.languages) != 1

    def close(self) -> None:
        """Release the models and pool. The Transcriber reloads lazily if used again."""
        with self._init_lock:
            if self._pool:
                self._pool.shutdown(wait=True)
            self._pool, self._model, self._lid_model = None, None, None

    # -- steps -------------------------------------------------------------------------------

    def _detect(self, model, piece: np.ndarray) -> tuple[str, float]:
        _, _, probs = model.detect_language(audio=piece)
        return language_confidence(probs, self.cfg.languages)

    def _region_languages(
        self, audio: np.ndarray, regions: list[tuple[float, float]]
    ) -> list[tuple[str | None, float]]:
        """(language, confidence) per region, detected concurrently on the pool."""
        if not self._needs_detection():
            return [(self.cfg.languages[0], 1.0)] * len(regions)
        if self._lid_model is None:
            return [(None, 0.0)] * len(regions)  # decided per chunk by the main model
        det = list(
            self._pool.map(
                lambda r: self._detect(
                    self._lid_model, audio[int(r[0] * SAMPLE_RATE) : int(r[1] * SAMPLE_RATE)]
                ),
                regions,
            )
        )
        langs = smooth_languages(regions, det, MIN_LANGUAGE_CONFIDENCE)
        # an inherited language counts as "just confident enough"
        return [
            (lg, c if lg == d else MIN_LANGUAGE_CONFIDENCE)
            for lg, (d, c) in zip(langs, det, strict=True)
        ]

    @staticmethod
    def _build_chunks(
        track: Track, regions: list[tuple[float, float]], det: list[tuple[str | None, float]]
    ) -> list[Chunk]:
        chunks = []
        for start, end, lang in merge_labelled(regions, [d[0] for d in det]):
            inside = [
                (e - s, c)
                for (s, e), (_, c) in zip(regions, det, strict=True)
                if s < end and e > start
            ]
            w = sum(d for d, _ in inside) or 1.0
            chunks.append(Chunk(track, start, end, lang, sum(d * c for d, c in inside) / w))
        return chunks

    def _transcribe_chunk(self, chunk: Chunk, audio: np.ndarray) -> list[Segment]:
        piece = slice_chunk(audio, chunk)
        lang = chunk.language
        if lang is None or chunk.confidence < MIN_LANGUAGE_CONFIDENCE:
            lang, conf = self._detect(self._model, piece)
            log.debug(
                "chunk %s %.1fs: main-model language %s (%.2f, was %s)",
                chunk.track,
                chunk.start,
                lang,
                conf,
                chunk.language,
            )
        return self.transcribe_audio(
            piece, chunk.start, SPEAKERS[chunk.track], lang, max_end=chunk.end
        )

    # -- building blocks (also used by LiveTranscriber) --------------------------------------

    def ensure_loaded(self) -> None:
        """Load the model(s) and create the worker pool if needed (thread-safe, blocking)."""
        self._ensure_loaded()

    def submit(self, fn: Callable, *args) -> Future:
        """Run ``fn(*args)`` on the shared worker pool (loads the model first if needed)."""
        self._ensure_loaded()
        return self._pool.submit(fn, *args)

    @property
    def has_language_model(self) -> bool:
        """True if a separate small language-ID model is used (main-model re-check possible)."""
        self._ensure_loaded()
        return self._lid_model is not None

    def detect_language(self, audio: np.ndarray, main: bool = False) -> tuple[str, float]:
        """(language, confidence) among cfg.languages. Uses the small language-ID model unless
        ``main`` is set or there is none. A single configured language is returned as is."""
        if not self._needs_detection():
            return self.cfg.languages[0], 1.0
        self._ensure_loaded()
        model = self._model if main or self._lid_model is None else self._lid_model
        return self._detect(model, audio)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        offset: float,
        speaker: str,
        language: str,
        *,
        beam_size: int = 5,
        max_end: float | None = None,
    ) -> list[Segment]:
        """Transcribe one piece of audio (<= 30 s) in a fixed language. Segment times are
        ``offset`` + position in the piece, clipped to ``max_end``. Blocking; call it from a
        pool worker (see ``submit``) to run pieces in parallel."""
        self._ensure_loaded()
        segments, _ = self._model.transcribe(
            audio,
            language=language,
            vad_filter=False,
            beam_size=beam_size,
            condition_on_previous_text=False,
        )
        end_cap = max_end if max_end is not None else offset + len(audio) / SAMPLE_RATE
        out = []
        for s in segments:
            text = s.text.strip()
            if text:
                out.append(
                    Segment(
                        start=round(offset + s.start, 3),
                        end=round(min(offset + s.end, end_cap), 3),
                        text=text,
                        speaker=speaker,
                        language=language,
                    )
                )
        return out

    # -- public ------------------------------------------------------------------------------

    def transcribe_session(
        self, session: SessionInfo, progress: ProgressCallback | None = None
    ) -> list[Segment]:
        """Transcribe all tracks, write transcript.{md,json,srt}, mark the session transcribed.

        `progress` receives 0.0..1.0 (fraction of speech audio done); it may be called from a
        worker thread. Raises on failure (nothing is marked transcribed or deleted then).
        """
        with self._run_lock:
            return self._run(session, progress)

    def _run(self, session: SessionInfo, progress: ProgressCallback | None) -> list[Segment]:
        report = progress or (lambda _f: None)
        report(0.0)
        audio: dict[Track, np.ndarray] = {}
        for track, name in session.tracks.items():
            path = Path(session.directory) / name
            if not path.exists():
                log.warning("session %s: track file %s missing, skipped", session.id, path)
                continue
            audio[track] = load_track(path)
        duration = max((len(a) for a in audio.values()), default=0) / SAMPLE_RATE
        if session.ended_at:
            duration = max(duration, (session.ended_at - session.started_at).total_seconds())

        segments: list[Segment] = []
        if audio:
            self._ensure_loaded()
            t0 = time.monotonic()
            pool = self._pool
            tracks = list(audio)
            regions = dict(
                zip(tracks, pool.map(lambda t: find_speech(audio[t]), tracks), strict=True)
            )
            per_track_chunks = [
                self._build_chunks(t, regions[t], self._region_languages(audio[t], regions[t]))
                for t in tracks
            ]
            t_prep = time.monotonic() - t0
            # longest first -> better load balance across workers
            chunks = sorted((c for cl in per_track_chunks for c in cl), key=lambda c: -c.duration)
            total = sum(c.duration for c in chunks) or 1.0
            done, lock = 0.0, threading.Lock()

            def work(c: Chunk) -> list[Segment]:
                nonlocal done
                res = self._transcribe_chunk(c, audio[c.track])
                with lock:
                    done += c.duration
                    report(min(done / total, 0.999))
                return res

            results = list(pool.map(work, chunks))
            per_track: dict[Track, list[Segment]] = {Track.ME: [], Track.OTHERS: []}
            for c, segs in zip(chunks, results, strict=True):
                per_track.setdefault(c.track, []).extend(segs)
            segments = merge_tracks(
                per_track[Track.ME], per_track[Track.OTHERS], dedupe=self.cfg.dedupe_echo
            )
            elapsed = time.monotonic() - t0
            log.info(
                "session %s: %d chunks (%.1fs speech) -> %d segments; %.1fs audio in "
                "%.1fs (vad+lang %.1fs), RTF %.3f",
                session.id,
                len(chunks),
                total,
                len(segments),
                duration,
                elapsed,
                t_prep,
                elapsed / duration if duration else 0.0,
            )

        self.write_final(session, segments, duration)
        report(1.0)
        return segments

    def write_final(self, session: SessionInfo, segments: list[Segment], duration: float) -> None:
        """Write transcript.{md,json,srt}, mark the session transcribed, honour keep_audio."""
        write_outputs(session, segments, duration, self.cfg.model)
        session.transcribed = True
        session.save()
        if not self.cfg.keep_audio:
            for name in session.tracks.values():
                (Path(session.directory) / name).unlink(missing_ok=True)


def transcribe_directory(path: Path, cfg: TranscriptionConfig) -> list[Segment]:
    """Transcribe the session stored in `path` (must contain session.json)."""
    session = SessionInfo.load(Path(path))
    t = Transcriber(cfg)
    try:
        return t.transcribe_session(session)
    finally:
        t.close()
