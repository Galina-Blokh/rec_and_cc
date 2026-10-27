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
from concurrent.futures import Future, InvalidStateError
from pathlib import Path
from typing import Any

import numpy as np

from ..models import SAMPLE_RATE
from . import diarize

log = logging.getLogger("recandcc.transcribe.live_speakers")

SIMILARITY = 0.5  # cosine similarity to a centroid at or above which it is the same speaker
MAX_SPEAKERS = 8  # beyond this a new voice joins the closest speaker
MIN_AUDIO_S = 1.0  # shorter segments make noisy embeddings (like diarize.MIN_UNIT_S)
QUEUE_SIZE = 8  # pending embedding requests; more are dropped
INHERIT_GAP_S = 1.5  # no-embedding segments inherit the previous label only within this gap
CLOSE_JOIN_S = 0.5
LOAD_GRACE_S = 1.5  # most a cold model load may extend the request deadlines

Embedder = Callable[[np.ndarray], np.ndarray | None]

# The CAM++ extractor is shared with the offline Diarizer (``diarize.shared_extractor``) and
# loaded lazily on a labeler thread (loading takes longer than the live deadline). A failed
# load is remembered (with the model's path and mtime) so it is not retried for every segment,
# but is retried once the model file changes. ``compute`` on separate streams is thread-safe,
# so no lock is held across it: a compute orphaned by close() cannot block the next session.
_failed_lock = threading.Lock()
_extractor_failed: tuple[str, float] | None = None  # (path, mtime) of the model that failed


def _reset_extractor_cache() -> None:  # tests
    global _extractor_failed
    with _failed_lock:
        _extractor_failed = None
    diarize.reset_shared_extractor()


def _model_key(path: Any) -> tuple[str, float]:
    try:
        return str(path), Path(path).stat().st_mtime
    except OSError:
        return str(path), 0.0


def _get_extractor() -> Any:
    """The shared extractor, or None: model not on disk (checked again next time, it may be
    downloaded later) or it failed to load (remembered until the model file changes)."""
    global _extractor_failed
    path = diarize.model_path(download=False)
    if path is None:
        return None
    key = _model_key(path)
    with _failed_lock:
        if _extractor_failed == key:
            if diarize.peek_shared_extractor() is None:
                return None
            _extractor_failed = None  # loaded since (e.g. by the final diarization)
    try:
        return diarize.shared_extractor(download=False)
    except Exception as e:  # noqa: BLE001 - ImportError (Windows) or a bad model
        with _failed_lock:
            _extractor_failed = key
        log.debug("live speaker labels off: cannot load the speaker model: %s", e)
        return None


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
        self._ready_at: float | None = None  # clock() when the extractor load ended
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

    def effective_deadline(self, submitted: float, deadline: float) -> float:
        """`deadline` of a request submitted at `submitted`, pushed back by the time the
        extractor load took (at most LOAD_GRACE_S), so a cold load is not counted against it.
        While the load is still running: the full grace."""
        ready = self._ready_at
        if ready is None:
            return deadline + LOAD_GRACE_S
        return deadline + min(max(0.0, ready - submitted), LOAD_GRACE_S)

    def submit(self, audio: np.ndarray, deadline: float | None = None) -> Future | None:
        """Queue `audio` for embedding; the future resolves to the embedding or None. A request
        still queued after `deadline` (on `clock`) is skipped, resolving to None. Never blocks;
        returns None if too short, closed or the queue is full."""
        if len(audio) < MIN_AUDIO_S * SAMPLE_RATE:
            return None
        win = int(diarize.WINDOW_S * SAMPLE_RATE)
        if len(audio) > win:  # the central window is enough to tell voices apart, and cheap
            lo = (len(audio) - win) // 2
            audio = np.array(audio[lo : lo + win])
        fut: Future = Future()
        with self._lock:
            if self._closed:
                return None
            self._start_locked()
            try:
                self._q.put_nowait((audio, fut, self.clock(), deadline))
            except queue.Full:
                self.dropped += 1
                return None
        return fut

    def _run(self) -> None:
        try:
            if self._preload:
                with contextlib.suppress(Exception):
                    _get_extractor()
        finally:
            self._ready_at = self.clock()
        while not self._stop.is_set():
            try:
                audio, fut, submitted, deadline = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            result = None
            try:
                if deadline is None or self.clock() < self.effective_deadline(
                    submitted, deadline
                ):  # else nobody waits for it
                    result = self._embedder(audio)
            except BaseException as e:  # noqa: BLE001 - labels are optional; keep the thread alive
                log.debug("live speaker embedding failed: %s", e)
            with contextlib.suppress(InvalidStateError):  # close() may have resolved it
                fut.set_result(result)

    def label(
        self, embedding: np.ndarray | None, start: float = 0.0, end: float | None = None
    ) -> str:
        """Display label for a segment: "Speaker N", or "Others" until a voice is known.
        Without a usable embedding (too short, NaN, queue full, late, no model) a segment keeps
        the previous label only if the previous segment ended within INHERIT_GAP_S; after a
        longer gap the speaker may well have changed, so it shows "Others"."""
        end = start if end is None else end
        usable = False
        if embedding is not None:
            try:
                self._last = f"Speaker {self.clusterer.assign(embedding)}"
                usable = True
            except ValueError:  # e.g. NaN: treated like a missing embedding
                pass
        if not usable and start - self._last_end > INHERIT_GAP_S:
            return "Others"
        self._last_end = max(self._last_end, end)
        return self._last or "Others"

    def close(self) -> None:
        with self._lock:
            self._closed = True
            thread = self._thread
        self._stop.set()  # the thread checks it between jobs, even with a full queue
        # the extractor itself stays in the module cache (shared, slow to load) for the next session
        self._embedder = _no_embedder
        while True:  # fail what is still queued so nobody waits on it
            try:
                _, fut, _, _ = self._q.get_nowait()
            except queue.Empty:
                break
            with contextlib.suppress(InvalidStateError):
                fut.set_result(None)
        # The thread is a daemon and may be inside a long embedding; wait only briefly so
        # close() never stalls its caller (the live worker thread), then let it finish alone.
        if thread is not None and thread is not threading.current_thread():
            thread.join(CLOSE_JOIN_S)


__all__ = ["SpeakerClusterer", "SpeakerLabeler", "sherpa_embedder"]
