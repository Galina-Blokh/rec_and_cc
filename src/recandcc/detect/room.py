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

With an outside sensitivity set (``thresholds``), a step only counts if its level is
``margin_db`` above the room's noise floor (``NoiseFloor``: a low percentile of recent
levels, relearnt each time the mic opens), so a fan or café babble doesn't keep triggering.
The gate is per ~0.5 s step (its loud windows against the floor), not per 32 ms window, so
the dips between syllables still count; steps below it skip the VAD.

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
# Noise floor: this percentile of the per-window levels of the last FLOOR_HISTORY_S of audio.
# A conversation is at the floor over half the time (pauses between turns, gaps between
# syllables), two or more people chattering much less, so the 35th percentile sits under a
# conversation but on top of babble. It falls fast (a few quiet seconds) and rises slowly.
# With less than FLOOR_MIN_S of history nothing counts.
FLOOR_PERCENTILE, FLOOR_HISTORY_S, FLOOR_MIN_S = 35, 20.0, 3.0
STEP_PERCENTILE = 90  # a step's loudness: this percentile of its window levels
# Level shift (walked into a café, woke up somewhere else): when even the quieter moments of
# the last SHIFT_S are louder than floor + margin/2, the floor restarts from that audio.
# A conversation always dips back to the floor within a few seconds; babble doesn't.
SHIFT_PERCENTILE, SHIFT_S = 30, 3.0
GAP_S = 5.0  # a wall-clock gap between frame batches this long (sleep/wake) clears the floor
FLOOR_RESET = "floor-reset"  # queue marker from the capture thread to the VAD worker

# vad(audio) -> one speech probability per WINDOW samples (len(audio) is a WINDOW multiple).
VadFn = Callable[[np.ndarray], np.ndarray]
SourceFactory = Callable[[], AudioSource]
# det -> (speech_s, window_s, margin_db or None): the effective trigger (sensitivity.py).
Thresholds = Callable[[DetectionConfig], tuple[float, float, float | None]]


def plain_thresholds(det: DetectionConfig) -> tuple[float, float, float | None]:
    """room_speech_s within room_window_s; speech longer than the window is clamped to it."""
    window = float(det.room_window_s)
    return min(float(det.room_speech_s), window), window, None


def window_levels(audio: np.ndarray) -> np.ndarray:
    """RMS level in dBFS of each WINDOW of `audio` (a WINDOW multiple); -120 for silence."""
    w = audio.reshape(-1, WINDOW).astype(np.float64)
    return 20.0 * np.log10(np.maximum(np.sqrt(np.mean(w * w, axis=1)), 1e-6))


class NoiseFloor:
    """Background level of the room, independent of the VAD (thread-safe)."""

    def __init__(self, history_s: float = FLOOR_HISTORY_S, min_s: float = FLOOR_MIN_S):
        per_s = SAMPLE_RATE / WINDOW
        self._levels: deque[float] = deque(maxlen=int(history_s * per_s))
        self._recent: deque[float] = deque(maxlen=int(SHIFT_S * per_s))
        self._min = int(min_s * per_s)
        self._lock = threading.Lock()

    @property
    def db(self) -> float | None:
        """The floor, or None while there is too little history."""
        with self._lock:
            if len(self._levels) < self._min:
                return None
            return float(np.percentile(np.fromiter(self._levels, float), FLOOR_PERCENTILE))

    def loud(self, levels: np.ndarray, margin_db: float) -> bool:
        """Whether a step (its window levels) is margin_db above the floor; False without
        a floor yet."""
        floor = self.db
        if floor is None or not len(levels):
            return False
        return float(np.percentile(levels, STEP_PERCENTILE)) >= floor + margin_db - 1e-9

    def add(self, levels: np.ndarray, margin_db: float | None = None) -> bool:
        """Account a step's window levels. With a margin, returns True if the room's level
        shifted up (the floor then restarts from the last SHIFT_S of audio)."""
        with self._lock:
            vals = [float(v) for v in levels]
            self._levels.extend(vals)
            self._recent.extend(vals)
            if margin_db is None or len(self._levels) < self._min:
                return False
            if len(self._recent) < (self._recent.maxlen or 0):
                return False
            floor = float(np.percentile(np.fromiter(self._levels, float), FLOOR_PERCENTILE))
            quiet = float(np.percentile(np.fromiter(self._recent, float), SHIFT_PERCENTILE))
            # Trade-off (accepted): a long stretch of continuous talk in a quiet room can look
            # like a louder place too. The trigger still works (the talk has already counted),
            # but for a few seconds after a shift soft speech counts as background.
            if quiet <= floor + margin_db / 2:
                return False
            self._levels.clear()
            self._levels.extend(self._recent)
            return True

    def clear(self) -> None:
        with self._lock:
            self._levels.clear()
            self._recent.clear()


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
        self.preroll_s = monitor.preroll_s()  # what start() replays (the recorder asks)
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
        thresholds: Thresholds = plain_thresholds,
    ):
        if source_factory is None:
            from ..capture import create_mic_source

            source_factory = create_mic_source
        self.det = det
        self.source_factory = source_factory
        self.vad = vad or silero_vad
        self.clock = clock
        self.thresholds = thresholds  # read on every step: a preset change applies at once
        self.floor = NoiseFloor()  # VAD worker; cleared whenever the mic (re)opens
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
            self.floor.clear()  # maybe another place now: warm up again
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
        lo = self._processed - self.thresholds(self.det)[1] * SAMPLE_RATE
        return sum(n for end, n in self._steps if end > lo)

    def triggered(self) -> bool:
        """At least ``room_speech_s`` of speech within the window (and listening)."""
        speech_s = self.thresholds(self.det)[0]
        with self._lock:
            return self._source is not None and self._speech_locked() >= speech_s * SAMPLE_RATE

    def seconds_since_speech(self) -> float:
        """Audio seconds since the last speech (or since listening started), plus the time
        the mic has stopped delivering frames, if it has."""
        with self._lock:
            audio_s = (self._processed - self._last_speech) / SAMPLE_RATE
            stall = self.clock() - self._last_frame_clock - STALL_GRACE_S
        return audio_s + max(0.0, stall)

    def preroll_s(self) -> float:
        """Seconds of audio kept before a trigger: room_preroll_s, or the effective window
        if that is longer (low outside sensitivities widen it)."""
        return max(float(self.det.room_preroll_s), float(self.thresholds(self.det)[1]))

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
        keep = self.preroll_s() * SAMPLE_RATE
        with self._lock:
            if stream != self._stream:
                return
            now = self.clock()
            gap = now - self._last_frame_clock > GAP_S  # slept, or the stream stalled
            self._last_frame_clock = now
            self._ring.append(block)
            self._ring_size += block.size
            while self._ring and self._ring_size - self._ring[0].size >= keep:
                self._ring_size -= self._ring.popleft().size
            taps, gen, q = self._taps, self._gen, self._q
        if gap:
            q.put(FLOOR_RESET)  # maybe another place now: the worker relearns the floor
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
                if isinstance(item, str):  # FLOOR_RESET
                    self.floor.clear()
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
                with self._lock:
                    in_session = bool(self._taps)
                # The gate decides what *starts* a session. While a room session records this
                # stream, plain VAD decides when the talk has ended (room_stop_after_silence_s)
                # and the floor is frozen, so a long lecture can't become "the background".
                margin = None if in_session else self.thresholds(self.det)[2]
                loud = None
                if margin is not None:
                    levels = window_levels(chunk)
                    loud = self.floor.loud(levels, margin)
                    if self.floor.add(levels, margin):
                        loud = False  # the place got louder: this step is background
                probs = np.zeros(n // WINDOW, np.float32)
                if loud is None or loud:  # a step at the floor: skip the VAD
                    try:
                        probs = self.vad(np.concatenate([ctx, chunk]))[-(n // WINDOW) :]
                    except Exception:
                        if not failed:  # once per worker, not every 0.5 s
                            log.exception("room listening: VAD failed; counting it as silence")
                        failed = True
                ctx = np.concatenate([ctx, chunk])[-HISTORY:]
                speech = np.asarray(probs) >= self.det.room_vad_threshold
                self._add_step(gen, n, speech)
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
            lo = self._processed - self.thresholds(self.det)[1] * SAMPLE_RATE
            while self._steps and self._steps[0][0] <= lo:
                self._steps.popleft()
