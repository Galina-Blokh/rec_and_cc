"""Whisper transcription of a recorded session: VAD chunks of both tracks in parallel.

Pipeline per session:
  1. VAD each track (Silero) -> speech regions.
  2. Language per region with a small language-ID model (cheap; the main model would cost a
     full encoder pass per call), restricted to cfg.languages; short uncertain regions inherit
     their neighbour's language.
  3. Merge consecutive same-language regions into chunks <= 30 s.
  4. Transcribe all chunks of both tracks concurrently on one shared WhisperModel.
  5. Merge tracks, drop mic echo, write transcript.{md,json,srt}.
  6. Speaker diarization of the "Others" track (diarize.py) runs on its own CPU threads from
     the start, alongside steps 1-4; "Others" segments become "Speaker 1", "Speaker 2", ...
     (the "Me" track in a room session, both tracks in a room session a call upgraded).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..config import TranscriptionConfig
from ..metrics import log_session_metrics, rounded
from ..models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind
from . import callnames, voices
from .chunking import Chunk, find_speech, load_track, merge_labelled, slice_chunk
from .diarize import Diarizer, Turn, label_speakers, model_path
from .gpu import MlxWhisper, hf_cache_folder, mlx_word_timestamps, use_gpu
from .merge import (
    ME,
    OTHERS,
    language_confidence,
    merge_tracks,
    smooth_languages,
    split_on_speaker_change,
)
from .output import write_outputs
from .revisions import (
    UnpinnedModelError,
    check_download_allowed,
    ct2_repo,
    faster_whisper_revision,
)

log = logging.getLogger("recandcc.transcribe")

ProgressCallback = Callable[[float], None]
SPEAKERS = {Track.ME: ME, Track.OTHERS: OTHERS}

# A language_models entry that failed to load is retried after this long (an unpinned repo,
# which is refused by config, is not retried).
ROUTE_RETRY_S = 300.0
LANGUAGE_NAMES = {"he": "Hebrew", "en": "English", "ru": "Russian"}  # for notices


def diarized_tracks(session: SessionInfo) -> tuple[Track, ...]:
    """The tracks whose speakers get numbered: "Others" (the remote side of a call), "Me" in
    a room session (everyone in the room is on the mic), both in a room session a call
    upgraded later (the room on "Me", the call on "Others")."""
    if session.trigger_kind == TriggerKind.ROOM:
        return (Track.ME,)
    return (Track.ME, Track.OTHERS) if session.started_as_room else (Track.OTHERS,)


def _gather(futures: dict[Track, Future]) -> Future:
    """One future for several ``Diarizer.submit`` futures: track -> turns (None if that one
    failed or was cancelled), with their timings summed into ``timing`` before it is set."""
    out: Future = Future()
    timing: dict = {}
    out.timing = timing  # pyright: ignore[reportAttributeAccessIssue]
    left, lock = [len(futures)], threading.Lock()

    def done(_f: Future) -> None:
        with lock:
            left[0] -= 1
            if left[0]:
                return
        result: dict[Track, Any] = {}
        for track, f in futures.items():
            for key, value in getattr(f, "timing", {}).items():
                timing[key] = round(timing.get(key, 0.0) + value, 2)
            result[track] = None if f.cancelled() or f.exception() else f.result()
        out.set_result(result)

    for f in futures.values():
        f.add_done_callback(done)
    return out


LANGUAGE_ID_MODEL = "base"  # small multilingual model used only for per-region language ID
MIN_LANGUAGE_CONFIDENCE = 0.75  # below this a chunk's language is re-checked with the main model
# How long writing the final transcript waits for speaker diarization. If it takes longer, the
# transcript is written with "Others" right away and rewritten with speaker labels when ready.
DIARIZE_WAIT_S = 5.0
# A summary waiting this long for a diarization that never ends is dropped from memory; its
# .summary-pending marker gets it summarised (with "Others") by the app's idle re-scan.
LABEL_WAIT_MAX_S = 3600.0


def _dir_key(directory) -> str:
    """A session folder as a dict key, the same for every spelling (no disk access)."""
    return os.path.normcase(os.path.abspath(directory))


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
        # Main model: a faster-whisper WhisperModel (CPU/CUDA) or, on Apple silicon, an
        # MlxWhisper running on the GPU (see gpu.py). The small language-ID model stays on the
        # CPU, so CPU workers and the GPU run at the same time.
        self.gpu = use_gpu(cfg)
        self._model = None
        self._lid_model = None
        self._pool: ThreadPoolExecutor | None = None
        self._init_lock = threading.Lock()
        self._run_lock = threading.Lock()
        # Word-timestamp failures in the current session (reset per session in _run); after
        # the first one the session's remaining chunks are transcribed without words.
        self._word_failures = 0
        self._word_failures_lock = threading.Lock()
        self._diarizer: Diarizer | None = None
        # cfg.language_models: model name -> loaded model (under _route_lock, on first use or
        # prefetched). Until it is loaded, or after a failed load until _route_retry[name]
        # (clock time; inf = never), its language uses the main model.
        self._routed: dict[str, Any] = {}
        self._route_retry: dict[str, float] = {}
        self._route_noticed: set[str] = set()
        self._route_lock = threading.Lock()  # held for a whole routed load (or download)
        # Guards storing a routed model against close(), which never waits for _route_lock.
        self._store_lock = threading.Lock()
        self.clock = time.monotonic
        # (title, message) toast from any thread, e.g. the controller's post_notice
        self.on_notice: Callable[[str, str], None] | None = None
        self.diarize_wait_s = DIARIZE_WAIT_S
        # model -> load seconds, not yet reported in a session's metrics (the session that
        # waited for the load gets them)
        self._load_times: dict[str, float] = {}
        # session folder -> (since, callbacks waiting for a slow diarization's relabelled
        # transcript), see after_labels; under _init_lock
        self._label_waiters: dict[str, tuple[float, list[Callable[[], None]]]] = {}

    # -- models ------------------------------------------------------------------------------

    def _model_dir(self, name: str) -> tuple[Path, str]:
        """Local folder of faster-whisper model ``name`` and the fix to log if it is broken.

        Offline first: once downloaded, a model resolves without contacting Hugging Face. A
        missing or partial snapshot (LocalEntryNotFoundError, incl. IncompleteSnapshotError,
        both FileNotFoundErrors) is downloaded at its pinned commit (revisions.py); a snapshot
        that is present but unreadable or corrupt is raised without going online.
        """
        if Path(name).is_dir():  # the user's own model folder: never download over it
            return Path(name), "fix or replace that folder"
        from faster_whisper import utils

        rev = faster_whisper_revision(name)
        cache = hf_cache_folder(getattr(utils, "_MODELS", {}).get(name, name))
        remedy = f"delete {cache} to download the model again"
        try:
            path = utils.download_model(name, local_files_only=True, revision=rev)
        except FileNotFoundError:
            check_download_allowed(name, rev, self.cfg.allow_unpinned_models)
            log.info("model %s not on disk yet; downloading it once", name)
            path = utils.download_model(name, revision=rev)
        except Exception:
            log.warning("model %s failed to load from the local cache; %s", name, remedy)
            raise
        return Path(path), remedy

    def _load(self, name: str):
        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        path, remedy = self._model_dir(name)
        # Without tokenizer.json faster-whisper fetches a tokenizer from Hugging Face (even
        # with local_files_only), so treat such a folder as corrupt instead.
        if not (path / "tokenizer.json").is_file():
            raise RuntimeError(f"model {name}: tokenizer.json missing in {path}; {remedy}")
        try:
            # A local folder: WhisperModel loads it without any network access.
            model = WhisperModel(
                str(path),
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
                num_workers=self.num_workers,
                cpu_threads=self.cpu_threads,
            )
        except Exception:
            log.warning("model %s failed to load from %s; %s", name, path, remedy)
            raise
        elapsed = time.monotonic() - t0
        self._load_times[name] = rounded(elapsed, 2)
        log.info(
            "loaded whisper model %s (%s, %s, workers=%d, threads=%d) in %.1fs",
            name,
            self.cfg.device,
            self.cfg.compute_type,
            self.num_workers,
            self.cpu_threads,
            elapsed,
        )
        return model

    def _ensure_loaded(self):
        """Thread-safe lazy init of the model(s) and the worker pool."""
        if self._model is None:
            with self._init_lock:
                if self._model is None:
                    lid_name = self.language_model
                    if lid_name and self._needs_detection() and self._lid_model is None:
                        self._lid_model = self._load(lid_name)  # kept if the main load fails
                    if self.gpu:
                        t0 = time.monotonic()
                        model = MlxWhisper(
                            self.cfg.model, allow_unpinned=self.cfg.allow_unpinned_models
                        )
                        try:
                            model.load()
                        except BaseException:
                            model.close()  # e.g. an unpinned model refused: free its thread
                            raise
                        self._load_times[f"{self.cfg.model} (gpu)"] = rounded(
                            time.monotonic() - t0, 2
                        )
                    else:
                        model = self._load(self.cfg.model)
                    self._pool = ThreadPoolExecutor(self.num_workers, thread_name_prefix="whisper")
                    self._model = model
        return self._model

    def _model_for(self, language: str, wait: bool = True):
        """The model that transcribes ``language``: its cfg.language_models entry or the main
        model. ``wait``: block while the routed model loads; else (live) the main model is
        used until a background load has finished."""
        main = self._ensure_loaded()
        name = self.cfg.language_models.get(language)
        if not name or name == self.cfg.model:
            return main
        return self._routed_model(name, language, main, wait) or main

    def _routed_model(self, name: str, language: str, main, wait: bool):
        """The loaded routed model ``name``, loading it if due; None = use ``main`` for now."""
        model = self._routed.get(name)
        if model is not None or self.clock() < self._route_retry.get(name, 0.0):
            return model
        if not wait:
            if not self._route_lock.locked():  # not being loaded yet: load it off this thread
                threading.Thread(
                    target=self._routed_model,
                    args=(name, language, main, True),
                    name="whisper-prefetch",
                    daemon=True,
                ).start()
            return None
        with self._route_lock:  # other languages keep transcribing while this one loads
            if self._model is not main:  # closed meanwhile
                return None
            if name not in self._routed and self.clock() >= self._route_retry.get(name, 0.0):
                model = self._load_routed(name, language, main)
                if model is not None:
                    with self._store_lock:
                        closed = self._model is not main
                        if not closed:
                            self._routed[name] = model
                    if closed:  # close() ran during the load: drop the late result
                        if isinstance(model, MlxWhisper):
                            model.close()
                        return None
            return self._routed.get(name)

    def start_prefetch(self) -> None:
        """Load the routed models of the configured languages in the background, so the first
        region in such a language doesn't wait for a load (or the one-time download). Needs
        the main model loaded; never blocks."""
        main, langs = self._model, self.cfg.languages
        if main is None:
            return
        for language, name in self.cfg.language_models.items():
            if name != self.cfg.model and (not langs or language in langs):
                self._routed_model(name, language, main, wait=False)

    def _load_routed(self, name: str, language: str, main):
        if self._model is not main:  # closed meanwhile: don't start a download now
            return None
        try:
            if isinstance(main, MlxWhisper):
                t0 = time.monotonic()
                model = MlxWhisper(name, allow_unpinned=self.cfg.allow_unpinned_models, share=main)
                model.load()
                self._load_times[f"{name} (gpu)"] = rounded(time.monotonic() - t0, 2)
                return model
            return self._load(ct2_repo(name))
        except Exception as e:  # noqa: BLE001 - the main model still transcribes this language
            if self._model is not main:  # closed meanwhile (e.g. its GPU thread shut down)
                return None
            permanent = isinstance(e, UnpinnedModelError)
            self._route_retry[name] = float("inf") if permanent else self.clock() + ROUTE_RETRY_S
            log.warning(
                "model %s for language %s failed to load (%s); using %s for it%s",
                name,
                language,
                type(e).__name__,
                self.cfg.model,
                "" if permanent else f", retrying in {ROUTE_RETRY_S / 60:.0f} min",
            )
            if self.on_notice is not None and name not in self._route_noticed:
                self._route_noticed.add(name)
                lang = LANGUAGE_NAMES.get(language, language)
                self.on_notice(f"{lang} model unavailable", f"Using the default model for {lang}")
            return None

    def _require_pool(self) -> ThreadPoolExecutor:
        """The worker pool; only valid after _ensure_loaded() and before close()."""
        if self._pool is None:
            raise RuntimeError("Transcriber is closed")
        return self._pool

    def _needs_detection(self) -> bool:
        return len(self.cfg.languages) != 1

    # `cfg.languages` can be replaced from the menu while a transcription runs, so each step
    # reads it once into a local (never len() then [0] on two different lists).

    def close(self) -> None:
        """Release the models and pool. The Transcriber reloads lazily if used again."""
        with self._init_lock:
            main = self._model
            if self._pool:
                self._pool.shutdown(wait=True)
            # Not _route_lock: a routed load (or download) may hold it for minutes, and close()
            # runs on the UI thread at Quit. A load that finishes later sees the transcriber
            # closed and drops its result.
            with self._store_lock:
                routed, self._routed, self._route_retry = self._routed, {}, {}
                self._model = None
            for model in routed.values():
                if isinstance(model, MlxWhisper):
                    model.close()  # before the main model, whose GPU thread it shares
            if isinstance(main, MlxWhisper):
                # Waits only for GPU jobs already queued, e.g. a routed weight load (seconds);
                # a routed download runs on its own thread (MlxWhisper.load) and never here.
                main.close()
            self._pool, self._lid_model = None, None
            diarizer, self._diarizer = self._diarizer, None
        if diarizer is not None:
            diarizer.close(wait=True)  # lets a pending transcript relabel finish (seconds)

    # -- steps -------------------------------------------------------------------------------

    def _detect(
        self, model, piece: np.ndarray, langs: list[str] | None = None
    ) -> tuple[str, float]:
        langs = self.cfg.languages if langs is None else langs
        if isinstance(model, MlxWhisper):
            return model.detect_language(piece, langs)
        _, _, probs = model.detect_language(audio=piece)
        return language_confidence(probs, langs)

    def _region_languages(
        self, audio: np.ndarray, regions: list[tuple[float, float]]
    ) -> list[tuple[str | None, float]]:
        """(language, confidence) per region, detected concurrently on the pool."""
        langs = self.cfg.languages
        if len(langs) == 1:
            return [(langs[0], 1.0)] * len(regions)
        if self._lid_model is None:
            return [(None, 0.0)] * len(regions)  # decided per chunk by the main model
        det = list(
            self._require_pool().map(
                lambda r: self._detect(
                    self._lid_model, audio[int(r[0] * SAMPLE_RATE) : int(r[1] * SAMPLE_RATE)], langs
                ),
                regions,
            )
        )
        smoothed = smooth_languages(regions, det, MIN_LANGUAGE_CONFIDENCE)
        # an inherited language counts as "just confident enough"
        return [
            (lg, c if lg == d else MIN_LANGUAGE_CONFIDENCE)
            for lg, (d, c) in zip(smoothed, det, strict=True)
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

    def _transcribe_chunk(
        self, chunk: Chunk, audio: np.ndarray, word_tracks: tuple[Track, ...] = ()
    ) -> list[Segment]:
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
        speaker = SPEAKERS[chunk.track]
        # decided once per session (see _word_tracks), so every chunk of it agrees
        words = chunk.track in word_tracks
        return self.transcribe_audio(
            piece, chunk.start, speaker, lang, max_end=chunk.end, words=words
        )

    # -- building blocks (also used by LiveTranscriber) --------------------------------------

    def ensure_loaded(self) -> None:
        """Load the model(s) and create the worker pool if needed (thread-safe, blocking)."""
        self._ensure_loaded()

    def submit(self, fn: Callable, *args) -> Future:
        """Run ``fn(*args)`` on the shared worker pool (loads the model first if needed)."""
        self._ensure_loaded()
        return self._require_pool().submit(fn, *args)

    @property
    def has_language_model(self) -> bool:
        """True if a separate small language-ID model is used (main-model re-check possible)."""
        self._ensure_loaded()
        return self._lid_model is not None

    def detect_language(self, audio: np.ndarray, main: bool = False) -> tuple[str, float]:
        """(language, confidence) among cfg.languages. Uses the small language-ID model unless
        ``main`` is set or there is none. A single configured language is returned as is."""
        langs = self.cfg.languages
        if len(langs) == 1:
            return langs[0], 1.0
        self._ensure_loaded()
        model = self._model if main or self._lid_model is None else self._lid_model
        return self._detect(model, audio, langs)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        offset: float,
        speaker: str,
        language: str,
        *,
        beam_size: int = 5,
        max_end: float | None = None,
        wait_for_model: bool = True,
        words: bool = False,
    ) -> list[Segment]:
        """Transcribe one piece of audio (<= 30 s) in a fixed language. Segment times are
        ``offset`` + position in the piece, clipped to ``max_end``. Blocking; call it from a
        pool worker (see ``submit``) to run pieces in parallel. ``language`` picks the model
        (cfg.language_models); ``wait_for_model=False`` uses the main model while that one is
        still loading (live). ``words``: keep word timings in ``Segment.words`` (on MLX only
        when ``mlx_word_timestamps()``: mlx-whisper 0.4.3 with our numpy aligner)."""
        model = self._model_for(language, wait_for_model)
        if words and self._word_failures:
            words = False  # circuit breaker: this session's word pass already failed
        failure = None
        try:
            pieces = self._pieces(model, audio, language, beam_size, words)
        except Exception as e:
            if not words:
                raise
            failure = type(e).__name__
            log.debug("word timestamp failure", exc_info=True)
        if failure is not None:
            # Retried outside the except block, so the failed call (and its frames) are gone.
            # One odd chunk must not fail the session: keep it whole (no words, never split).
            self._note_word_failure(type(model).__name__, failure, offset)
            pieces = self._pieces(model, audio, language, beam_size, False)
        end_cap = max_end if max_end is not None else offset + len(audio) / SAMPLE_RATE
        out = []
        for start, end, raw, ws in pieces:
            text = raw.strip()
            if text:
                # Both backends' words as (start, end, " word"), in session time.
                kept = [
                    (round(offset + a, 3), round(min(offset + b, end_cap), 3), w)
                    for a, b, w in ws
                    if offset + a < end_cap
                ]
                out.append(
                    Segment(
                        start=round(offset + start, 3),
                        end=round(min(offset + end, end_cap), 3),
                        text=text,
                        speaker=speaker,
                        language=language,
                        # words were cut off: the text is whole, so never split it
                        words=kept if len(kept) == len(ws) else [],
                    )
                )
        return out

    def _note_word_failure(self, backend: str, error: str, offset: float) -> None:
        """Count, and warn once per session (no transcript text); chunks already running
        in parallel may fail too, and only log at DEBUG."""
        with self._word_failures_lock:
            self._word_failures += 1
            first = self._word_failures == 1
        (log.warning if first else log.debug)(
            "word timestamps failed (%s, %s) for the chunk at %.1f s; kept without words, "
            "and off for the rest of this session",
            backend,
            error,
            offset,
        )

    @staticmethod
    def _pieces(
        model: Any, audio: np.ndarray, language: str, beam_size: int, words: bool
    ) -> list[tuple[float, float, str, list[tuple[float, float, str]]]]:
        """(start, end, text, [(start, end, " word")]) per segment, relative to ``audio``."""
        if isinstance(model, MlxWhisper):
            if words and mlx_word_timestamps():
                return model.transcribe(audio, language, words=True)
            return [(a, b, t, []) for a, b, t in model.transcribe(audio, language)]
        segments, _ = model.transcribe(
            audio,
            language=language,
            vad_filter=False,
            beam_size=beam_size,
            condition_on_previous_text=False,
            word_timestamps=words,
        )
        try:  # faster-whisper decodes lazily: consume it here, and always close it
            return [
                (
                    s.start,
                    s.end,
                    s.text,
                    [(w.start, w.end, w.word) for w in getattr(s, "words", None) or []],
                )
                for s in segments
            ]
        finally:
            close = getattr(segments, "close", None)
            if close is not None:
                close()

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

    def _word_tracks(self, session: SessionInfo) -> tuple[Track, ...]:
        """Tracks whose chunks get word timestamps: the diarized ones, only if the split is
        on and the speaker model is on disk. One model-present check per session, so a model
        download finishing mid-run cannot make early and later chunks differ."""
        if not (self.cfg.diarize and self.cfg.split_on_speaker_change):
            return ()
        if model_path(download=False) is None:
            return ()
        return diarized_tracks(session)

    def _run(self, session: SessionInfo, progress: ProgressCallback | None) -> list[Segment]:
        report = progress or (lambda _f: None)
        report(0.0)
        self._word_failures = 0
        diarization = self.start_diarization(session)  # CPU threads, overlaps with Whisper
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
            self.start_prefetch()  # routed models load while VAD and language ID run
            t0 = time.monotonic()
            pool = self._require_pool()
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

            word_tracks = self._word_tracks(session)  # once: not per chunk

            def work(c: Chunk) -> list[Segment]:
                nonlocal done
                res = self._transcribe_chunk(c, audio[c.track], word_tracks)
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
            session.update_metrics(
                "transcription",
                {
                    "audio_s": rounded(duration, 1),
                    "elapsed_s": rounded(elapsed, 2),
                    "rtf": rounded(elapsed / duration if duration else 0.0),
                    "chunks": len(chunks),
                },
            )
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

        segments = self.write_final(session, segments, duration, diarization)
        log_session_metrics(session)
        report(1.0)
        return segments

    # -- final transcript + speaker diarization ----------------------------------------------

    def start_diarization(self, session: SessionInfo) -> Future | None:
        """Start diarizing the session's ``diarized_tracks`` in the background (CPU threads).
        The future's result is the turns of the one track, or track -> turns when there are
        two. None if disabled (cfg.diarize) or none of those tracks is on disk."""
        if not self.cfg.diarize:
            return None
        paths = {
            track: Path(session.directory) / session.tracks[track]
            for track in diarized_tracks(session)
            if track in session.tracks
        }
        paths = {track: path for track, path in paths.items() if path.exists()}
        if not paths:
            return None
        with self._init_lock:
            if self._diarizer is None:
                self._diarizer = Diarizer(
                    min_speakers=self.cfg.min_speakers,
                    max_speakers=self.cfg.max_speakers,
                    segmentation=self.cfg.diarize_overlap,
                )
            diarizer = self._diarizer
        futures: dict[Track, Future] = {}
        for track, path in paths.items():
            timing: dict = {}
            futures[track] = diarizer.submit(path, timing)
            # Ad-hoc attribute, read back with getattr. Filled in before the result is set
            # (see Diarizer._safe).
            futures[track].timing = timing  # pyright: ignore[reportAttributeAccessIssue]
        if len(diarized_tracks(session)) > 1:  # keyed by track, even if one file is missing
            return _gather(futures)
        return next(iter(futures.values()))

    def write_final(
        self,
        session: SessionInfo,
        segments: list[Segment],
        duration: float,
        diarization: Future | None = None,
    ) -> list[Segment]:
        """Write transcript.{md,json,srt}, mark the session transcribed, honour keep_audio.

        `diarization`: a ``start_diarization`` future (started now if None). If its speaker
        turns are ready within ``diarize_wait_s``, "Others" segments are relabelled before
        writing; otherwise the transcript is written as is and rewritten with the labels in the
        background as soon as they are ready (WAVs are deleted only after that). Returns the
        segments written now.
        """
        if diarization is None:
            diarization = self.start_diarization(session)
        with self._init_lock:
            loads, self._load_times = self._load_times, {}
        if loads:
            session.update_metrics("models", {"load_s": loads})
        if diarization is not None:
            try:
                turns = diarization.result(timeout=self.diarize_wait_s)
            except TimeoutError:
                log.info("session %s: speaker labels follow when diarization ends", session.id)
                self._write(session, segments, duration)
                with self._init_lock:  # before the callback, which may run right away
                    self._label_waiters[_dir_key(session.directory)] = (self.clock(), [])
                diarization.add_done_callback(
                    lambda f: self._relabel_later(session, segments, duration, f)
                )
                return segments
            except Exception:  # noqa: BLE001 - cancelled; diarization is optional
                turns = None
            self._diarization_metrics(session, diarization)
            segments = self._labelled(session, segments, turns)
        self._write(session, segments, duration)
        self._name_from_call_app(session)
        self._drop_audio(session)
        return segments

    def _name_from_call_app(self, session: SessionInfo) -> None:
        """Opt-in: name the one unknown speaker after the one free call-app participant."""
        if not (self.cfg.speaker_names and self.cfg.names_from_call_app):
            callnames.forget(session.id)  # switched off since the call: don't keep the names
            return
        try:
            callnames.apply(Path(session.directory))
        except Exception as e:  # noqa: BLE001 - optional; the numbered label stays
            # The exception text or traceback could quote a name: type at WARNING, rest at DEBUG
            log.warning(
                "session %s: could not use call-app names (%s)", session.id, type(e).__name__
            )
            log.debug("call-app names failure", exc_info=True)

    def after_labels(self, directory: Path, fn: Callable[[], None]) -> None:
        """Run ``fn`` once the session's transcript has its final speaker labels: now, or
        (on the diarizer thread) after a slow diarization has rewritten it."""
        with self._init_lock:
            if self._waiting_for_labels(directory):
                self._label_waiters[_dir_key(directory)][1].append(fn)
                return
        fn()

    def labels_pending(self, directory: Path) -> bool:
        """True while a slow diarization will still rewrite this session's transcript."""
        with self._init_lock:
            return self._waiting_for_labels(directory)

    def _waiting_for_labels(self, directory: Path) -> bool:  # under _init_lock
        now = self.clock()
        for key, (since, _fns) in list(self._label_waiters.items()):
            if now - since > LABEL_WAIT_MAX_S:  # never finished (hung): stop waiting
                del self._label_waiters[key]
        return _dir_key(directory) in self._label_waiters

    def _labels_final(self, session: SessionInfo) -> None:
        with self._init_lock:
            waiters = self._label_waiters.pop(_dir_key(session.directory), (0.0, []))[1]
        for fn in waiters:
            try:
                fn()
            except Exception:  # noqa: BLE001 - runs on the diarizer thread
                log.exception("session %s: a speaker-label callback failed", session.id)

    def _write(self, session: SessionInfo, segments: list[Segment], duration: float) -> None:
        write_outputs(session, segments, duration, self.cfg.model)
        session.transcribed = True
        session.save()

    @staticmethod
    def _diarization_metrics(session: SessionInfo, future: Future) -> None:
        timing = getattr(future, "timing", None)
        if timing:
            session.update_metrics("diarization", dict(timing))

    def _drop_audio(self, session: SessionInfo) -> None:
        if not self.cfg.keep_audio:
            for name in session.tracks.values():
                (Path(session.directory) / name).unlink(missing_ok=True)

    def _labelled(
        self,
        session: SessionInfo,
        segments: list[Segment],
        turns: list[Turn] | dict[Track, list[Turn] | None] | None,
    ) -> list[Segment]:
        """Number (or name) the speakers of each diarized track independently; a later
        track's numbers continue after the earlier ones' ("Speaker 3" after 1 and 2)."""
        by_track = turns if isinstance(turns, dict) else {diarized_tracks(session)[0]: turns}
        out, n_labels, n_named = list(segments), 0, 0
        speakers: dict[str, np.ndarray] = {}
        for track, track_turns in by_track.items():
            centroids = getattr(track_turns, "centroids", {}) if self.cfg.speaker_names else {}
            names: dict[int, str] = {}
            if track_turns and centroids:
                try:
                    names = voices.match_profiles(centroids, voices.load_profiles())
                except Exception:  # noqa: BLE001 - names are optional; numbers still work
                    log.exception("session %s: could not match voice profiles", session.id)
            taken = {s.speaker for s in out} - {ME, OTHERS}
            # One person, one name: a profile already given on an earlier track numbers this
            # track's cluster instead (and its stored voice can't overwrite the first one).
            folded = {t.casefold() for t in taken}
            names = {k: n for k, n in names.items() if n.casefold() not in folded}
            if track_turns and self.cfg.split_on_speaker_change:
                # this track's segments still carry "Me"/"Others": split them at speaker changes
                out = split_on_speaker_change(out, track_turns, SPEAKERS[track])
            out, labels = label_speakers(
                out, track_turns or [], SPEAKERS[track], names=names, reserved=taken
            )
            # A lone speaker in a room session keeps "Me", which is never renamed: no voice.
            speakers |= {
                lbl: centroids[k] for k, lbl in labels.items() if k in centroids and lbl != ME
            }
            n_labels += len(labels)
            n_named += len(names.keys() & labels.keys())
        try:  # opt-in only, in the private voices folder: lets `name-speaker` name them later
            if speakers and Path(session.directory).exists():
                voices.write_session_voices(session, speakers)
            else:  # never leave voices of an older transcript of this session behind
                voices.delete_session_voices(session.id)
            voices.prune_session_voices()  # of sessions whose folder was deleted
        except OSError:
            log.exception("session %s: could not update speaker voices", session.id)
        if not any(by_track.values()):
            return segments
        log.info(  # counts only: names are personal data
            "session %s: %d speaker(s), %d recognized by voice",
            session.id,
            n_labels,
            n_named,
        )
        return out

    def _relabel_later(
        self, session: SessionInfo, segments: list[Segment], duration: float, future: Future
    ) -> None:
        """Done-callback of a slow diarization: rewrite the transcript with speaker labels."""
        try:
            turns = None if future.cancelled() else future.result()
            labelled = self._labelled(session, segments, turns)
            if labelled != segments and Path(session.directory).exists():
                write_outputs(session, labelled, duration, self.cfg.model)
                log.info("session %s: transcript rewritten with speaker labels", session.id)
            if Path(session.directory).exists():
                self._name_from_call_app(session)
            if getattr(future, "timing", None) and Path(session.directory).exists():
                self._diarization_metrics(session, future)
                session.save()
                log_session_metrics(session)
        except Exception:  # noqa: BLE001 - runs on the diarizer thread; must not raise
            log.exception("session %s: could not add speaker labels", session.id)
        finally:
            if Path(session.directory).exists():
                self._drop_audio(session)
            self._labels_final(session)


def transcribe_directory(path: Path, cfg: TranscriptionConfig) -> list[Segment]:
    """Transcribe the session stored in `path` (must contain session.json)."""
    session = SessionInfo.load(Path(path))
    t = Transcriber(cfg)
    try:
        return t.transcribe_session(session)
    finally:
        t.close()
