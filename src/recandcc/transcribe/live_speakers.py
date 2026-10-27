"""Live speaker labels: "Others" captions become "Speaker N" while recording.

Display only: the final transcript keeps the offline diarization (``diarize.py``). Uses the
embedding model ``diarize.py`` already pins, and only if it is on disk: this module never
downloads anything.

Threading: ``SpeakerLabeler.submit`` is called from a transcription pool thread. It puts the
audio on a small bounded queue (dropped when full, never blocking the queue) and one dedicated
thread computes the embeddings; requests whose deadline passed are skipped. The clustering
(``SpeakerClusterer``) is not thread-safe: the live worker calls it in time order.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import numpy as np

from ..models import SAMPLE_RATE
from . import diarize

log = logging.getLogger("recandcc.transcribe.live_speakers")

SIMILARITY = 0.5  # cosine similarity to a centroid at or above which it is the same speaker
MAX_SPEAKERS = 8  # beyond this a new voice joins the closest speaker
MIN_AUDIO_S = 1.0  # shorter segments make noisy embeddings (like diarize.MIN_UNIT_S)
QUEUE_SIZE = 8  # pending embedding requests; more are dropped
INHERIT_GAP_S = 1.5  # a short segment inherits the previous label only within this gap
CLOSE_JOIN_S = 0.5

Embedder = Callable[[np.ndarray], np.ndarray | None]

# One CAM++ extractor per process (loading takes longer than the live deadline), created lazily
# on a labeler thread. A failed load is remembered so it is not retried for every segment.
_extractor_lock = threading.Lock()
_extractor: Any = None
_extractor_failed = False
_compute_lock = threading.Lock()  # an extractor is not shared between concurrent computes


def _reset_extractor_cache() -> None:  # tests
    global _extractor, _extractor_failed
    with _extractor_lock:
        _extractor, _extractor_failed = None, False


def _get_extractor() -> Any:
    """The shared extractor, or None: model not on disk (checked again next time, it may be
    downloaded later) or it failed to load (remembered)."""
    global _extractor, _extractor_failed
    with _extractor_lock:
        if _extractor is not None or _extractor_failed:
            return _extractor
        path = diarize.model_path(download=False)
        if path is None:
            return None
        try:
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1)
            _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        except Exception as e:  # noqa: BLE001 - ImportError (Windows) or a bad model
            _extractor_failed = True
            log.debug("live speaker labels off: cannot load the speaker model: %s", e)
        return _extractor


class SpeakerClusterer:
    """Online clustering: nearest running centroid by cosine similarity, else a new speaker."""

    def __init__(self, threshold: float = SIMILARITY, max_speakers: int = MAX_SPEAKERS) -> None:
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._sums: list[np.ndarray] = []  # sum of unit embeddings; its direction = the mean

    def __len__(self) -> int:
        return len(self._sums)

    def assign(self, embedding: np.ndarray) -> int:
        """Speaker number (1-based, by first appearance) of `embedding`; updates the centroid."""
        v = np.asarray(embedding, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("unusable embedding")
        v = v / norm
        best, sim = -1, -1.0
        for i, s in enumerate(self._sums):
            c = float(np.dot(v, s) / np.linalg.norm(s))
            if c > sim:
                best, sim = i, c
        if best < 0 or (sim < self.threshold and len(self._sums) < self.max_speakers):
            self._sums.append(v)
            return len(self._sums)
        self._sums[best] = self._sums[best] + v  # running mean (up to scale)
        return best + 1


def sherpa_embedder() -> Embedder:
    """Embedder on the shared sherpa-onnx extractor; None (logged once at DEBUG) when the model
    isn't on disk or can't be loaded. Meant to run on the labeler thread."""
    warned = False

    def embed(audio: np.ndarray) -> np.ndarray | None:
        nonlocal warned
        extractor = _get_extractor()
        if extractor is None:
            if not warned:
                warned = True
                log.debug("live speaker labels off: speaker model not available")
            return None
        with _compute_lock:
            stream = extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            stream.input_finished()
            return np.asarray(extractor.compute(stream), dtype=np.float32)

    return embed


def _no_embedder(audio: np.ndarray) -> np.ndarray | None:
    return None


class SpeakerLabeler:
    """Embeds audio on its own thread behind a bounded queue and clusters the results."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        queue_size: int = QUEUE_SIZE,
        clusterer: SpeakerClusterer | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._preload = embedder is None
        self._embedder: Embedder = embedder or sherpa_embedder()
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._stop = threading.Event()
        self.clock = clock
        self.clusterer = clusterer or SpeakerClusterer()
        self.dropped = 0
        self._last = ""
        self._last_end = float("-inf")  # end of the previous labelled segment

    def _start_locked(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="live-speakers", daemon=True)
            self._thread.start()

    def warm(self) -> None:
        """Start the thread now; with the default embedder it loads the model before the first
        request, so the first captions are not late. Never blocks."""
        with self._lock:
            if not self._closed:
                self._start_locked()

    def submit(self, audio: np.ndarray, deadline: float | None = None) -> Future | None:
        """Queue `audio` for embedding; the future resolves to the embedding or None. A request
        still queued after `deadline` (on `clock`) is skipped, resolving to None. Never blocks;
        returns None if too short, closed or the queue is full."""
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return None
        fut: Future = Future()
        with self._lock:
            if self._closed:
                return None
            self._start_locked()
            try:
                self._q.put_nowait((audio, fut, deadline))
            except queue.Full:
                self.dropped += 1
                return None
        return fut

    def _run(self) -> None:
        if self._preload:
            with contextlib.suppress(Exception):
                _get_extractor()
        while not self._stop.is_set():
            try:
                audio, fut, deadline = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            result = None
            try:
                if deadline is None or self.clock() < deadline:  # else nobody waits for it
                    result = self._embedder(audio)
            except BaseException as e:  # noqa: BLE001 - labels are optional; keep the thread alive
                log.debug("live speaker embedding failed: %s", e)
            if not fut.done():
                fut.set_result(result)

    def label(
        self, embedding: np.ndarray | None, start: float = 0.0, end: float | None = None
    ) -> str:
        """Display label for a segment: "Speaker N", or "Others" until a voice is known.
        Without an embedding a segment keeps the previous label (most often the same person);
        a short one (< 1 s, too short to embed) only if the previous segment ended within
        INHERIT_GAP_S, as in a quick exchange the speaker may well have changed."""
        end = start if end is None else end
        if embedding is not None:
            with contextlib.suppress(ValueError):
                self._last = f"Speaker {self.clusterer.assign(embedding)}"
        elif end - start < MIN_AUDIO_S and start - self._last_end > INHERIT_GAP_S:
            return "Others"
        self._last_end = max(self._last_end, end)
        return self._last or "Others"

    def close(self) -> None:
        with self._lock:
            self._closed = True
            thread = self._thread
        self._stop.set()  # the thread checks it between jobs, even with a full queue
        self._embedder = _no_embedder  # release the extractor reference
        while True:  # fail what is still queued so nobody waits on it
            try:
                _, fut, _ = self._q.get_nowait()
            except queue.Empty:
                break
            fut.set_result(None)
        # The thread is a daemon and may be inside a long embedding; wait only briefly so
        # close() never stalls its caller (the live worker thread), then let it finish alone.
        if thread is not None and thread is not threading.current_thread():
            thread.join(CLOSE_JOIN_S)


__all__ = ["SpeakerClusterer", "SpeakerLabeler", "sherpa_embedder"]
