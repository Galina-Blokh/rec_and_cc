"""Live speaker labels: "Others" captions become "Speaker N" while recording.

Display only: the final transcript keeps the offline diarization (``diarize.py``). Uses the
embedding model ``diarize.py`` already pins, and only if it is on disk: this module never
downloads anything.

Threading: ``SpeakerLabeler.embed`` is called from a transcription pool thread. It puts the
audio on a small bounded queue (dropped when full, never blocking the queue) and one dedicated
thread computes the embeddings. The clustering (``SpeakerClusterer``) is not thread-safe: the
live worker calls it in time order.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
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
RESULT_TIMEOUT_S = 10.0

Embedder = Callable[[np.ndarray], np.ndarray | None]


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
    """Embedder on the pinned sherpa-onnx model; returns None (and logs once at DEBUG) when
    the model isn't on disk. Loaded lazily, on the labeler thread."""
    state: dict[str, Any] = {}

    def embed(audio: np.ndarray) -> np.ndarray | None:
        if "extractor" not in state:
            path = diarize.model_path(download=False)
            if path is None:
                if not state.get("warned"):
                    state["warned"] = True
                    log.debug("live speaker labels off: speaker model not on disk")
                return None
            import sherpa_onnx

            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1)
            state["extractor"] = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        extractor = state["extractor"]
        stream = extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        return np.asarray(extractor.compute(stream), dtype=np.float32)

    return embed


class SpeakerLabeler:
    """Embeds audio on its own thread behind a bounded queue and clusters the results."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        queue_size: int = QUEUE_SIZE,
        clusterer: SpeakerClusterer | None = None,
    ) -> None:
        self._embedder = embedder or sherpa_embedder()
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        self.clusterer = clusterer or SpeakerClusterer()
        self.dropped = 0
        self._last = ""

    def submit(self, audio: np.ndarray) -> Future | None:
        """Queue `audio` for embedding. Never blocks; None if too short, closed or queue full."""
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return None
        fut: Future = Future()
        with self._lock:
            if self._closed:
                return None
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="live-speakers", daemon=True)
                self._thread.start()
            try:
                self._q.put_nowait((audio, fut))
            except queue.Full:
                self.dropped += 1
                return None
        return fut

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            audio, fut = item
            try:
                fut.set_result(self._embedder(audio))
            except Exception as e:  # noqa: BLE001 - labels are optional
                log.debug("live speaker embedding failed: %s", e)
                fut.set_result(None)

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Embedding of `audio` or None (too short, model missing, queue full, timeout).
        Waits for the labeler thread, so call it from a pool thread, never the capture
        callback or the UI."""
        fut = self.submit(audio)
        if fut is None:
            return None
        try:
            return fut.result(RESULT_TIMEOUT_S)
        except Exception:  # noqa: BLE001 - timeout
            return None

    def label(self, embedding: np.ndarray | None) -> str:
        """Display label for a segment: "Speaker N", or "Others" until a voice is known. A
        segment without an embedding keeps the previous label (most often the same person)."""
        if embedding is not None:
            with contextlib.suppress(ValueError):
                self._last = f"Speaker {self.clusterer.assign(embedding)}"
        return self._last or "Others"

    def close(self) -> None:
        with self._lock:
            self._closed = True
            started = self._thread is not None
        if started:
            with contextlib.suppress(queue.Full):  # daemon thread: ends with the process
                self._q.put_nowait(None)


__all__ = ["SpeakerClusterer", "SpeakerLabeler", "sherpa_embedder"]
