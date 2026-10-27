"""Room listening: a microphone monitor that notices a conversation near the laptop.

Opt-in (``[detection] room_listening``); the engine thread opens and closes the mic. The ring
buffer, the session tap and the VAD worker come from #58. Data flow::

    mic source thread --_on_frames()--> pre-roll ring buffer (last max(room_preroll_s,
                          |               room_window_s)) + session taps (the room
                          v               session's mic track)
                     SimpleQueue --> room-vad worker: Silero VAD per 32 ms window
                                       -> speech samples per ~0.5 s step, sliding window

``_on_frames`` runs on the capture thread and only copies, appends and enqueues; taps only
enqueue too (the recorder's track writer). Silero runs on the ``room-vad`` thread. The
engine polls ``triggered()`` and ``seconds_since_speech()``. ``session_source()`` is an
``AudioSource`` that replays the pre-roll and then continues with the live frames of the same
stream, sample-exact, so a room session keeps the sentence that triggered it and the mic is
never opened twice. No audio is logged; it lives only in memory until a session records it.

Speech is counted in audio samples (deterministic in tests); if the mic stops delivering
frames, the wall-clock stall is added to ``seconds_since_speech()``.
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
    """The room session's mic: pre-roll first, then the monitor's live frames, in order.

    Live frames that arrive while the pre-roll is being replayed are held (the capture
    thread only appends them) and delivered right after it."""

    name = "room"

    def __init__(self, monitor: RoomMonitor):
        self.monitor = monitor
        self._cb: FrameCallback | None = None
        self._lock = threading.Lock()
        self._held: list[np.ndarray] | None = None  # not None while replaying

    def start(self, on_frames: FrameCallback) -> None:
        self._cb = on_frames
        with self._lock:
            self._held = []
        backlog = self.monitor._attach(self._live)
        while True:
            for block in backlog:
                self._deliver(block)
            with self._lock:
                backlog = self._held or []
                self._held = [] if backlog else None  # caught up: deliver live frames
            if not backlog:
                return

    def _live(self, block: np.ndarray) -> None:  # capture thread
        with self._lock:
            if self._held is not None:
                self._held.append(block)
                return
        self._deliver(block)

    def _deliver(self, block: np.ndarray) -> None:
        cb = self._cb
        if cb is None:
            return
        try:
            cb(block)
        except Exception:
            log.exception("room session tap failed")

    def stop(self) -> None:
        if self._cb is not None:
            self.monitor._detach(self._live)
            self._cb = None


class RoomMonitor:
    """Keeps the mic open and tracks how much speech it heard recently. Thread-safe.

    ``start``/``close_mic``/``close`` open or close the mic and may block briefly: the engine
    calls them from its own thread, never under its lock. Constructing it opens nothing."""

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
        self._stream = 0  # bumped per mic (re)open/close: frames of an older stream are dropped
        self._gen = 0  # bumped per stream and per session end: older VAD work is dropped
        self._ring: deque[np.ndarray] = deque()
        self._ring_size = 0
        self._taps: tuple[FrameCallback, ...] = ()
        self._steps: deque[tuple[int, int]] = deque()  # (end sample, speech samples)
        self._processed = 0  # samples analysed by the VAD in this generation
        self._last_speech = 0  # sample index where speech was last heard
        self._last_frame_clock = clock()
        self._q: SimpleQueue = SimpleQueue()
        self._worker: threading.Thread | None = None
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._source is not None

    def start(self) -> bool:
        """Open the mic and start listening (no-op while running). False if the mic can't be
        opened (logged)."""
        with self._lock:
            if self._closed:
                return False
            if self._source is not None:
                return True
            if self._worker is None or not self._worker.is_alive():
                # A fresh queue per worker: a worker that outlived close() keeps its own.
                self._q = SimpleQueue()
                self._worker = threading.Thread(
                    target=self._run, args=(self._q,), name="room-vad", daemon=True
                )
                self._worker.start()
            self._stream += 1
            self._reset_locked()
            stream = self._stream
        try:
            src = self.source_factory()
            src.start(lambda frames: self._on_frames(stream, frames))
        except Exception:
            log.exception("room listening: could not open the microphone")
            return False
        with self._lock:
            if stream == self._stream:
                self._source, src = src, None
        if src is not None:  # closed meanwhile
            self._stop_source(src)
            return False
        log.info("room listening: microphone open")
        return True

    def reset(self) -> None:
        """Forget the pre-roll and the speech heard so far (an app is doing audio)."""
        with self._lock:
            if self._ring or self._steps or self._processed:
                self._reset_locked()

    def close_mic(self) -> None:
        """Close the mic (the orange indicator goes off); ``start()`` reopens it."""
        with self._lock:
            src, self._source = self._source, None
            self._stream += 1
            self._reset_locked()
        if src is not None:
            self._stop_source(src)
            log.info("room listening: microphone closed")

    def close(self) -> None:
        """Close the mic and end the worker (shutdown); ``start()`` won't reopen it."""
        with self._lock:
            self._closed = True
        self.close_mic()
        with self._lock:
            worker, q = self._worker, self._q
        if worker is not None:
            q.put(None)
            worker.join(timeout=5)

    @staticmethod
    def _stop_source(src: AudioSource) -> None:
        try:
            src.stop()
        except Exception:
            log.exception("room listening: closing the microphone failed")

    def _reset_locked(self) -> None:
        """Forget the pre-roll and the speech history (VAD work still queued is dropped)."""
        self._gen += 1
        self._ring.clear()
        self._ring_size = 0
        self._steps.clear()
        self._processed = self._last_speech = 0
        self._last_frame_clock = self.clock()
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
        """At least ``room_speech_s`` of speech within the window (and listening)."""
        with self._lock:
            return (
                self._source is not None
                and self._speech_locked() >= self.det.room_speech_s * SAMPLE_RATE
            )

    def seconds_since_speech(self) -> float:
        """Audio seconds since the last speech (or since listening started), plus the time
        the mic has stopped delivering frames, if it has."""
        with self._lock:
            audio_s = (self._processed - self._last_speech) / SAMPLE_RATE
            stall = self.clock() - self._last_frame_clock - STALL_GRACE_S
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
    def _attach(self, cb: FrameCallback) -> list[np.ndarray]:
        """Register ``cb`` for live frames; returns the pre-roll to replay before them."""
        with self._lock:
            self._taps = (*self._taps, cb)
            return list(self._ring)

    def _detach(self, cb: FrameCallback) -> None:
        with self._lock:
            self._taps = tuple(c for c in self._taps if c is not cb)
            self._reset_locked()

    # ------------------------------------------------------------------ capture thread
    def _on_frames(self, stream: int, frames: np.ndarray) -> None:
        block = np.asarray(frames, dtype=np.float32).reshape(-1).copy()
        if not block.size:
            return
        keep = max(self.det.room_preroll_s, self.det.room_window_s) * SAMPLE_RATE
        with self._lock:
            if stream != self._stream:
                return
            self._last_frame_clock = self.clock()
            self._ring.append(block)
            self._ring_size += block.size
            while self._ring and self._ring_size - self._ring[0].size >= keep:
                self._ring_size -= self._ring.popleft().size
            taps, gen, q = self._taps, self._gen, self._q
        for cb in taps:
            try:
                cb(block)
            except Exception:
                log.exception("room session tap failed")
        q.put((gen, block))

    # ------------------------------------------------------------------ VAD worker
    def _run(self, q: SimpleQueue) -> None:
        gen = -1
        pending: list[np.ndarray] = []
        size = 0
        ctx = np.zeros(0, np.float32)  # the HISTORY samples before `pending`
        failed = False
        while True:
            item = q.get()
            if item is None:
                return
            try:
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                if isinstance(item, int):  # reset: new stream or a session ended
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
                    if not failed:  # once per worker, not every 0.5 s
                        log.exception("room listening: VAD failed; counting it as silence")
                    failed, probs = True, np.zeros(n // WINDOW, np.float32)
                ctx = np.concatenate([ctx, chunk])[-HISTORY:]
                self._add_step(gen, n, np.asarray(probs) >= self.det.room_vad_threshold)
            except Exception:  # never let one bad step end listening
                log.exception("room listening: speech detection step failed")
                pending, size = [], 0

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
