"""Room listening: notice a conversation near the laptop from the microphone alone.

Opt-in (``[detection] room_listening``) and only running while the engine wants it: the mic
is closed while room listening is off and while a session records (the recorder opens its
own mic stream). The source's frame callback only copies the frames into a queue; a worker
thread runs Silero VAD on ``BLOCK_S`` blocks and sets ``active`` once speech totals
``speech_s`` within the last ``window_s`` of audio. No audio is kept or logged: only the
``active`` flag changes, at DEBUG.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from collections.abc import Callable

import numpy as np

from ..capture.base import AudioSource
from ..models import SAMPLE_RATE

log = logging.getLogger(__name__)

BLOCK_S = 1.0  # VAD runs on blocks this long
MAX_QUEUED = 100  # frame batches (~10 s of 0.1 s mic blocks); newer audio is dropped beyond

SpeechFn = Callable[[np.ndarray], float]  # audio block -> seconds of speech in it


def silero_speech_s(block: np.ndarray) -> float:
    """Seconds of speech in `block` according to Silero VAD (bundled with faster-whisper)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=0)
    stamps = get_speech_timestamps(block, opts, sampling_rate=SAMPLE_RATE)
    return sum(s["end"] - s["start"] for s in stamps) / SAMPLE_RATE


class SpeechWindow:
    """Speech seconds per block over a sliding window of audio time (pure logic)."""

    def __init__(self, speech_s: float, window_s: float):
        self.speech_s, self.window_s = float(speech_s), float(window_s)
        self._t = 0.0
        self._blocks: deque[tuple[float, float]] = deque()  # (block end time, speech s)

    def add(self, duration_s: float, speech_s: float) -> bool:
        """Account one block; returns whether the window now counts as active."""
        self._t += duration_s
        self._blocks.append((self._t, max(0.0, min(speech_s, duration_s))))
        while self._blocks and self._blocks[0][0] <= self._t - self.window_s + 1e-9:
            self._blocks.popleft()
        return self.active

    @property
    def active(self) -> bool:
        return sum(s for _, s in self._blocks) >= self.speech_s - 1e-9


class RoomMonitor:
    """Mic stream + VAD worker. ``start``/``stop`` are called by the engine thread only."""

    def __init__(
        self,
        source_factory: Callable[[], AudioSource] | None = None,
        speech_fn: SpeechFn = silero_speech_s,
    ):
        self._factory = source_factory
        self._speech_fn = speech_fn
        self._source: AudioSource | None = None
        self._worker: threading.Thread | None = None
        self._q: queue.Queue | None = None
        self._stop_evt = threading.Event()
        self.active = False  # sustained speech in the window (read by the engine)

    @property
    def running(self) -> bool:
        return self._source is not None

    def start(self, speech_s: float, window_s: float) -> None:
        """Open the mic and start the VAD worker. Raises if the mic can't be opened."""
        if self._source is not None:
            return
        factory = self._factory
        if factory is None:
            from ..capture import create_mic_source

            factory = create_mic_source
        q: queue.Queue = queue.Queue()
        stop_evt = self._stop_evt = threading.Event()
        self.active = False

        def on_frames(frames: np.ndarray) -> None:  # capture thread: copy and enqueue only
            if q.qsize() < MAX_QUEUED:
                q.put_nowait(np.array(frames, dtype=np.float32, copy=True).reshape(-1))

        worker = threading.Thread(
            target=self._work,
            args=(q, stop_evt, SpeechWindow(speech_s, window_s)),
            name="room-vad",
            daemon=True,
        )
        worker.start()
        try:
            source = factory()
            source.start(on_frames)
        except Exception:
            stop_evt.set()
            q.put_nowait(None)
            raise
        self._source, self._worker, self._q = source, worker, q
        log.info("room listening: microphone open")

    def stop(self) -> None:
        """Close the mic and stop the worker. Idempotent."""
        source, worker, q = self._source, self._worker, self._q
        self._source = self._worker = self._q = None
        self._stop_evt.set()
        if source is not None:
            try:
                source.stop()
            except Exception:
                log.exception("room listening: closing the microphone failed")
        if q is not None:
            q.put_nowait(None)
        if worker is not None:
            worker.join(timeout=5)
        self.active = False
        if source is not None:
            log.info("room listening: microphone closed")

    def _work(self, q: queue.Queue, stop_evt: threading.Event, window: SpeechWindow) -> None:
        size = int(BLOCK_S * SAMPLE_RATE)
        pending = np.zeros(0, dtype=np.float32)
        failed = False
        while (frames := q.get()) is not None:
            pending = np.concatenate([pending, frames])
            while pending.size >= size and not stop_evt.is_set():
                block, pending = pending[:size], pending[size:]
                try:
                    speech = float(self._speech_fn(block))
                except Exception:
                    if not failed:  # once per stream, not every second
                        log.exception("room listening: VAD failed; counting it as silence")
                    failed, speech = True, 0.0
                active = window.add(BLOCK_S, speech)
                if stop_evt.is_set():
                    return
                if active != self.active:
                    log.debug("room listening: speech active=%s", active)
                self.active = active
