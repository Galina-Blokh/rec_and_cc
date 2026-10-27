"""Room listening: an always-on microphone monitor that notices a conversation nearby.

Data flow::

    mic source thread --_on_frames()--> pre-roll ring buffer (last ``room_preroll_s``)
                          |               + session taps (the room session's mic track)
                          v
                     SimpleQueue --> room-vad worker: Silero VAD per 32 ms window
                                       -> speech seconds per step, in a sliding window

``_on_frames`` runs on the capture thread and only copies, appends and enqueues. Silero runs
on the ``room-vad`` thread (about 2 ms per 0.5 s of audio). The engine polls ``triggered()``
and ``seconds_since_speech()``. ``session_source()`` returns an ``AudioSource`` that replays
the pre-roll and then continues with the live frames of the same stream, so a room session's
mic track has no gap or overlap and keeps the sentence that triggered it.

Times are measured in audio samples (deterministic in tests); if the mic stops delivering,
the wall-clock stall is added to ``seconds_since_speech()``.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from queue import SimpleQueue

import numpy as np

from ..capture.base import AudioSource, FrameCallback
from ..config import DetectionConfig
from ..models import SAMPLE_RATE

log = logging.getLogger(__name__)

WINDOW = 512  # Silero frame: 32 ms at 16 kHz
STEP = 16 * WINDOW  # VAD runs on ~0.5 s of new audio at a time ...
# ... with ~1 s of the audio before it: Silero needs it to settle (without it, half of a
# steady voice is missed at step boundaries).
HISTORY = 32 * WINDOW
STALL_GRACE_S = 1.0  # frames this late still count as "on time"

# vad(audio) -> one speech probability per WINDOW samples (len(audio) is a WINDOW multiple).
VadFn = Callable[[np.ndarray], np.ndarray]
SourceFactory = Callable[[], AudioSource]


def silero_vad(audio: np.ndarray) -> np.ndarray:
    from faster_whisper.vad import get_vad_model  # lazy: heavy import

    return np.asarray(get_vad_model()(audio), dtype=np.float32).reshape(-1)


class _Tap(AudioSource):
    """The room session's mic: pre-roll first, then the monitor's live frames."""

    name = "room"

    def __init__(self, monitor: RoomMonitor):
        self.monitor = monitor
        self._cb: FrameCallback | None = None

    def start(self, on_frames: FrameCallback) -> None:
        self._cb = on_frames
        self.monitor._attach(on_frames)

    def stop(self) -> None:
        cb, self._cb = self._cb, None
        if cb is not None:
            self.monitor._detach(cb)


class RoomMonitor:
    """Keeps the mic open and tracks how much speech it heard recently. Thread-safe.

    ``start``/``pause``/``resume``/``close`` open or close the mic and may block briefly:
    call them from the engine thread, not the UI thread."""

    def __init__(
        self,
        det: DetectionConfig,
        source_factory: SourceFactory | None = None,
        *,
        vad: VadFn | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if source_factory is None:
            from ..capture import create_mic_source

            source_factory = create_mic_source
        self.det = det
        self.source_factory = source_factory
        self.vad = vad or silero_vad
        self.clock = clock
        self._lock = threading.Lock()  # everything below except the worker's own buffers
        self._source: AudioSource | None = None
        self._paused = False
        self._gen = 0  # bumped on every (re)start/stop: frames of an older stream are dropped
        self._ring: deque[np.ndarray] = deque()
        self._ring_size = 0
        self._taps: tuple[FrameCallback, ...] = ()
        self._steps: deque[tuple[int, int]] = deque()  # (end sample, speech samples)
        self._processed = 0  # samples analysed by the VAD (per stream)
        self._last_speech = 0  # sample index where speech was last heard
        self._last_step_clock = clock()
        self._q: SimpleQueue = SimpleQueue()
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._source is not None

    @property
    def paused(self) -> bool:
        return self._paused

    def start(self) -> bool:
        """Open the mic and start listening (no-op while running or paused). False if the
        mic can't be opened (logged)."""
        with self._lock:
            if self._source is not None or self._paused:
                return self._source is not None
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name="room-vad", daemon=True)
                self._worker.start()
            self._reset_locked()
            gen = self._gen
        try:
            src = self.source_factory()
            src.start(lambda frames: self._on_frames(gen, frames))
        except Exception:
            log.exception("room listening: could not open the microphone")
            return False
        with self._lock:
            if gen == self._gen and not self._paused:
                self._source, src = src, None
        if src is not None:  # paused or closed meanwhile
            self._stop_source(src)
            return False
        log.info("room listening started")
        return True

    def pause(self) -> None:
        """Close the mic (the orange indicator goes off) until ``resume()``."""
        with self._lock:
            self._paused = True
        self._close_source()
        log.info("room listening paused")

    def resume(self) -> bool:
        with self._lock:
            self._paused = False
        return self.start()

    def close(self) -> None:
        self._close_source()
        with self._lock:
            worker, self._worker = self._worker, None
        if worker is not None:
            self._q.put(None)
            worker.join(timeout=5)

    def _close_source(self) -> None:
        with self._lock:
            src, self._source = self._source, None
            self._reset_locked()
        if src is not None:
            self._stop_source(src)

    @staticmethod
    def _stop_source(src: AudioSource) -> None:
        try:
            src.stop()
        except Exception:
            log.exception("room listening: closing the microphone failed")

    def _reset_locked(self) -> None:
        self._gen += 1
        self._ring.clear()
        self._ring_size = 0
        self._steps.clear()
        self._processed = self._last_speech = 0
        self._last_step_clock = self.clock()
        self._q.put(self._gen)  # tells the worker to drop its partial buffer

    # ------------------------------------------------------------------ queries
    def speech_in_window(self) -> float:
        """Seconds of speech heard in the last ``room_window_s`` seconds of audio."""
        with self._lock:
            return self._speech_locked() / SAMPLE_RATE

    def _speech_locked(self) -> int:
        lo = self._processed - self.det.room_window_s * SAMPLE_RATE
        return sum(n for end, n in self._steps if end > lo)

    def triggered(self) -> bool:
        """At least ``room_min_speech_s`` of speech within the window (and listening)."""
        with self._lock:
            return (
                self._source is not None
                and self._speech_locked() >= self.det.room_min_speech_s * SAMPLE_RATE
            )

    def seconds_since_speech(self) -> float:
        """Audio seconds since the last speech (or since listening started), plus the time
        the mic has stopped delivering frames, if it has."""
        with self._lock:
            audio_s = (self._processed - self._last_speech) / SAMPLE_RATE
            stall = self.clock() - self._last_step_clock - STALL_GRACE_S
        return audio_s + max(0.0, stall)

    def session_source(self) -> AudioSource:
        """An ``AudioSource`` for the room session's mic track: the pre-roll (last
        ``max(room_preroll_s, room_window_s)`` seconds), then the live frames. When it is
        stopped the pre-roll and speech history are cleared, so the next session neither
        repeats this one's audio nor triggers on it."""
        return _Tap(self)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until every frame queued so far has been analysed (tests, shutdown)."""
        done = threading.Event()
        self._q.put(done)
        return done.wait(timeout)

    # ------------------------------------------------------------------ taps
    def _attach(self, cb: FrameCallback) -> None:
        with self._lock:
            for block in self._ring:
                try:
                    cb(block)
                except Exception:
                    log.exception("room session tap failed")
            self._taps = (*self._taps, cb)

    def _detach(self, cb: FrameCallback) -> None:
        with self._lock:
            self._taps = tuple(c for c in self._taps if c is not cb)
            self._ring.clear()
            self._ring_size = 0
            self._steps.clear()
            self._last_speech = self._processed

    # ------------------------------------------------------------------ capture thread
    def _on_frames(self, gen: int, frames: np.ndarray) -> None:
        block = np.asarray(frames, dtype=np.float32).reshape(-1).copy()
        if not block.size:
            return
        keep = max(self.det.room_preroll_s, self.det.room_window_s) * SAMPLE_RATE
        with self._lock:
            if gen != self._gen:
                return
            self._ring.append(block)
            self._ring_size += block.size
            while self._ring and self._ring_size - self._ring[0].size >= keep:
                self._ring_size -= self._ring.popleft().size
            for cb in self._taps:
                try:
                    cb(block)
                except Exception:
                    log.exception("room session tap failed")
        self._q.put((gen, block))

    # ------------------------------------------------------------------ VAD worker
    def _run(self) -> None:
        gen = -1
        pending: list[np.ndarray] = []
        size = 0
        ctx = np.zeros(0, np.float32)  # the HISTORY samples before `pending`
        while True:
            item = self._q.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            if isinstance(item, int):  # stream reset
                gen, pending, size, ctx = item, [], 0, np.zeros(0, np.float32)
                continue
            g, block = item
            if g != gen:
                continue
            pending.append(block)
            size += block.size
            if size < STEP:
                continue
            audio = np.concatenate(pending)
            n = (audio.size // WINDOW) * WINDOW
            chunk, rest = audio[:n], audio[n:]
            pending, size = ([rest], rest.size) if rest.size else ([], 0)
            try:
                probs = self.vad(np.concatenate([ctx, chunk]))[-(n // WINDOW) :]
            except Exception:
                log.exception("room listening: VAD failed")
                probs = np.zeros(n // WINDOW, np.float32)
            ctx = np.concatenate([ctx, chunk])[-HISTORY:]
            speech = probs >= self.det.room_vad_threshold
            self._add_step(gen, n, speech)

    def _add_step(self, gen: int, n: int, speech: np.ndarray) -> None:
        with self._lock:
            if gen != self._gen:
                return
            start = self._processed
            self._processed += n
            count = int(np.count_nonzero(speech))
            if count:
                self._steps.append((self._processed, count * WINDOW))
                last = int(np.flatnonzero(speech)[-1])
                self._last_speech = start + (last + 1) * WINDOW
            lo = self._processed - self.det.room_window_s * SAMPLE_RATE
            while self._steps and self._steps[0][0] <= lo:
                self._steps.popleft()
            self._last_step_clock = self.clock()
