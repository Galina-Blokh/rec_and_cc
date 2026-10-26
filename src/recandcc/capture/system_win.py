"""Windows system-audio capture: WASAPI loopback of the default output via PyAudioWPatch.

UNTESTED (written without access to a Windows machine).

WASAPI loopback delivers *no packets* while nothing is playing. To keep others.wav
time-aligned with me.wav, the worker thread inserts zeros whenever the stream has
fallen behind wall-clock time by more than GAP_FILL_S.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from ..models import SAMPLE_RATE
from .base import AudioSource, FrameCallback
from .mic import StreamResampler

log = logging.getLogger(__name__)

GAP_FILL_S = 0.25


class WinLoopbackSource(AudioSource):
    name = "system"

    def __init__(self, blocksize_s: float = 0.1):
        self.blocksize_s = blocksize_s
        self._pa = None
        self._stream = None
        self._q: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _find_loopback(pa, pyaudio):
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for lb in pa.get_loopback_device_info_generator():
            if speakers["name"] in lb["name"]:
                return lb
        raise RuntimeError(f"No WASAPI loopback device found for {speakers['name']!r}")

    def start(self, on_frames: FrameCallback) -> None:
        import pyaudiowpatch as pyaudio

        with self._lock:
            if self._stream is not None:
                return
            pa = pyaudio.PyAudio()
            try:
                dev = self._find_loopback(pa, pyaudio)
                channels = max(1, int(dev["maxInputChannels"]))
                rate = int(dev["defaultSampleRate"])
                resampler = StreamResampler(rate, SAMPLE_RATE)
                q = self._q = queue.Queue()
                stop_flag = threading.Event()

                def pa_callback(in_data, frame_count, time_info, status):
                    q.put(in_data)  # bytes; conversion happens on the worker thread
                    return (None, pyaudio.paContinue)

                def worker():
                    t0 = time.monotonic()
                    emitted = 0
                    while not stop_flag.is_set():
                        try:
                            data = q.get(timeout=0.1)
                        except queue.Empty:
                            data = b""
                        if data is None:
                            break
                        if data:
                            block = np.frombuffer(data, dtype=np.float32)
                            block = block.reshape(-1, channels).mean(axis=1)
                            out = resampler.process(block)
                        else:
                            expected = int((time.monotonic() - t0) * SAMPLE_RATE)
                            gap = expected - emitted
                            if gap < GAP_FILL_S * SAMPLE_RATE:
                                continue
                            out = np.zeros(gap, dtype=np.float32)
                        if out.size:
                            emitted += out.size
                            try:
                                on_frames(out)
                            except Exception:
                                log.exception("loopback frame callback failed")

                self._stop_flag = stop_flag
                self._worker = threading.Thread(
                    target=worker, name="loopback-resample", daemon=True
                )
                self._worker.start()
                self._stream = pa.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=dev["index"],
                    frames_per_buffer=int(rate * self.blocksize_s),
                    stream_callback=pa_callback,
                )
                self._stream.start_stream()
                self._pa = pa
                log.info("loopback started: %s (%d ch, %d Hz)", dev["name"], channels, rate)
            except Exception:
                self._q.put(None)
                pa.terminate()
                self._stream = None
                raise

    def stop(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
            pa, self._pa = self._pa, None
            worker, self._worker = self._worker, None
        if stream is not None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                log.exception("closing loopback stream failed")
        if worker is not None:
            self._q.put(None)
            worker.join(timeout=5)
        if pa is not None:
            pa.terminate()
