"""Microphone capture via sounddevice (PortAudio). Works on macOS and Windows.

The PortAudio callback only copies the block into a queue; a worker thread does the
downmix + resample to 16 kHz mono float32 and invokes the FrameCallback, so the
real-time audio thread never does more than a memcpy.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from ..models import SAMPLE_RATE
from .base import AudioSource, FrameCallback

log = logging.getLogger(__name__)


class StreamResampler:
    """Streaming mono resampler (numpy only, no scipy).

    Downsampling applies a windowed-sinc low-pass FIR first (anti-aliasing), then linear
    interpolation at the target rate. State (filter history, fractional read position) is
    carried across chunks so block boundaries produce no clicks.
    """

    def __init__(self, src_rate: float, dst_rate: float = SAMPLE_RATE, taps: int = 63):
        self.src_rate = float(src_rate)
        self.dst_rate = float(dst_rate)
        self.step = self.src_rate / self.dst_rate
        self.passthrough = abs(self.src_rate - self.dst_rate) < 1e-6
        self._fir: np.ndarray | None = None
        self._hist = np.zeros(0, dtype=np.float32)
        if self.src_rate > self.dst_rate and not self.passthrough:
            cutoff = 0.45 * self.dst_rate / self.src_rate  # cycles/sample, a bit below Nyquist
            n = np.arange(taps) - (taps - 1) / 2
            h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
            self._fir = (h / h.sum()).astype(np.float32)
            self._hist = np.zeros(taps - 1, dtype=np.float32)
        self._prev: np.ndarray | None = None  # last input sample of the previous chunk
        self._pos = 0.0  # position of the next output sample, relative to buffer index 0

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if self.passthrough or x.size == 0:
            return x
        if self._fir is not None:
            ext = np.concatenate([self._hist, x])
            x = np.convolve(ext, self._fir, mode="valid").astype(np.float32)
            self._hist = ext[-(self._fir.size - 1) :]
        buf = x if self._prev is None else np.concatenate([self._prev, x])
        last = buf.size - 1
        if last < self._pos:
            self._prev = buf[-1:]
            self._pos -= last
            return np.zeros(0, dtype=np.float32)
        n = int(np.floor((last - self._pos) / self.step)) + 1
        t = self._pos + np.arange(n) * self.step
        out = np.interp(t, np.arange(buf.size), buf).astype(np.float32)
        self._pos = self._pos + n * self.step - last
        self._prev = buf[-1:]
        return out


def to_mono(block: np.ndarray) -> np.ndarray:
    """(frames, channels) or (frames,) -> (frames,) float32."""
    block = np.asarray(block, dtype=np.float32)
    if block.ndim == 2:
        return block.mean(axis=1) if block.shape[1] > 1 else block[:, 0]
    return block.reshape(-1)


class MicSource(AudioSource):
    """Default (or given) input device -> 16 kHz mono float32 frames."""

    name = "mic"

    def __init__(self, device: int | str | None = None, blocksize_s: float = 0.1):
        self.device = device
        self.blocksize_s = blocksize_s
        self._stream = None
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    def _pick_rate(self, sd) -> float:
        try:
            sd.check_input_settings(
                device=self.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            return float(SAMPLE_RATE)
        except Exception:
            info = sd.query_devices(self.device, "input")
            return float(info["default_samplerate"])

    def start(self, on_frames: FrameCallback) -> None:
        import sounddevice as sd

        with self._lock:
            if self._stream is not None:
                return
            rate = self._pick_rate(sd)
            resampler = StreamResampler(rate, SAMPLE_RATE)
            q = self._q = queue.Queue()

            def pa_callback(indata, frames, time_info, status):  # PortAudio thread
                if status:
                    log.debug("mic status: %s", status)
                q.put(indata.copy())

            def worker():
                while True:
                    block = q.get()
                    if block is None:
                        break
                    out = resampler.process(to_mono(block))
                    if out.size:
                        try:
                            on_frames(out)
                        except Exception:
                            log.exception("mic frame callback failed")

            self._worker = threading.Thread(target=worker, name="mic-resample", daemon=True)
            self._worker.start()
            stream = sd.InputStream(
                device=self.device,
                samplerate=rate,
                channels=1,
                dtype="float32",
                blocksize=int(rate * self.blocksize_s),
                callback=pa_callback,
            )
            try:
                stream.start()
            except Exception:
                q.put(None)
                stream.close()
                raise
            self._stream = stream
            log.info("mic started (device=%s, rate=%.0f)", self.device, rate)

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            worker, self._worker = self._worker, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                log.exception("closing mic stream failed")
        if worker is not None:
            self._q.put(None)
            worker.join(timeout=5)
