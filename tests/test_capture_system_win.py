"""WASAPI loopback capture with a fake PyAudioWPatch (runs anywhere).

Stories: US-08, US-18.
"""

from __future__ import annotations

import sys
import threading
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

from recandcc.capture import system_win
from recandcc.capture.system_win import WinLoopbackSource
from recandcc.models import SAMPLE_RATE

SPEAKERS = {"index": 3, "name": "Speakers (Realtek)", "maxInputChannels": 0}
LOOPBACK = dict(
    index=7, name="Speakers (Realtek) [Loopback]", maxInputChannels=2, defaultSampleRate=48000.0
)


def push(stream, block: np.ndarray):
    """What PortAudio does on its callback thread: deliver interleaved float32 bytes."""
    data = np.ascontiguousarray(block, dtype=np.float32).tobytes()
    assert stream.kw["stream_callback"](data, len(block), {}, 0) == (None, "continue")


class FakePyAudio:
    def __init__(self, env):
        self.env, self.terminated = env, False
        env.instances.append(self)

    def get_host_api_info_by_type(self, api):
        assert api == "wasapi"
        return {"defaultOutputDevice": self.env.speakers["index"]}

    def get_device_info_by_index(self, i):
        assert i == self.env.speakers["index"]
        return self.env.speakers

    def get_loopback_device_info_generator(self):
        yield from self.env.loopbacks

    def open(self, **kwargs):
        if self.env.open_error:
            raise self.env.open_error
        self.env.stream = stream = MagicMock(kw=kwargs)
        for method in self.env.failing:
            getattr(stream, method).side_effect = OSError("device unplugged")
        return stream

    def terminate(self):
        self.terminated = True


@pytest.fixture
def pa(monkeypatch):
    others = {"index": 5, "name": "Headset [Loopback]"}  # skipped: not the default output
    env = types.SimpleNamespace(speakers=SPEAKERS, loopbacks=[others, LOOPBACK], instances=[])
    env.stream = env.open_error = None
    env.failing = ()
    mod = types.ModuleType("pyaudiowpatch")
    mod.paWASAPI, mod.paFloat32, mod.paContinue = "wasapi", "float32", "continue"
    mod.PyAudio = lambda: FakePyAudio(env)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", mod)
    return env


class Sink:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.cond = threading.Condition()

    def __call__(self, frames):
        with self.cond:
            self.frames.append(frames)
            self.cond.notify_all()

    def wait_samples(self, n, timeout=5.0):
        with self.cond:
            assert self.cond.wait_for(lambda: self.total >= n, timeout), self.total
        return np.concatenate(self.frames)

    @property
    def total(self):
        return sum(f.size for f in self.frames)


# ------------------------------------------------------------------ frames
def test_stereo_48k_is_downmixed_and_resampled(pa):
    src, sink = WinLoopbackSource(blocksize_s=0.1), Sink()
    src.start(sink)
    try:
        kw = pa.stream.kw
        assert (kw["format"], kw["channels"], kw["rate"]) == ("float32", 2, 48000)
        assert (kw["input"], kw["input_device_index"], kw["frames_per_buffer"]) == (True, 7, 4800)
        pa.stream.start_stream.assert_called_once()
        block = np.empty((4800, 2), dtype=np.float32)
        block[:, 0], block[:, 1] = 0.2, 0.6  # L/R differ: mono is their mean
        for _ in range(10):  # 1 s of audio
            push(pa.stream, block)
        out = sink.wait_samples(SAMPLE_RATE - 100)
    finally:
        src.stop()
    assert all(f.dtype == np.float32 and f.ndim == 1 for f in sink.frames)
    assert abs(out.size - SAMPLE_RATE) <= 2  # 48 kHz -> 16 kHz
    np.testing.assert_allclose(out[200:], 0.4, atol=1e-3)  # after the FIR warms up


def test_mono_16k_passes_through_and_survives_callback_errors(pa, caplog):
    # default output already is a loopback device
    pa.speakers = {**LOOPBACK, "isLoopbackDevice": True, "maxInputChannels": 0}
    pa.speakers["defaultSampleRate"] = SAMPLE_RATE
    sink = Sink()

    def on_frames(frames):
        sink(frames)
        if len(sink.frames) == 1:
            raise ValueError("consumer bug")  # logged; the worker keeps going

    src = WinLoopbackSource()
    src.start(on_frames)
    try:
        assert pa.stream.kw["channels"] == 1  # 0 reported channels is clamped to 1
        sig = np.linspace(-1, 1, 1600, dtype=np.float32)
        push(pa.stream, sig[:400].reshape(-1, 1))
        push(pa.stream, sig[400:].reshape(-1, 1))
        out = sink.wait_samples(1600)
    finally:
        src.stop()
    np.testing.assert_array_equal(out, sig)
    assert "loopback frame callback failed" in caplog.text


class StepClock:
    """Fake monotonic clock that counts reads, so the test can wait for worker polls."""

    def __init__(self):
        self.now, self.reads = 100.0, 0
        self.cond = threading.Condition()

    def __call__(self):
        with self.cond:
            self.reads += 1
            self.cond.notify_all()
            return self.now

    def advance_and_wait(self, dt, polls=2):
        """Advance, then wait until a worker poll has surely seen the new time."""
        with self.cond:
            self.now += dt
            target = self.reads + polls  # the next read sees `now`; the one after ends it
            assert self.cond.wait_for(lambda: self.reads >= target, 5)


def test_silence_gaps_are_zero_filled(pa, monkeypatch):
    """Loopback sends nothing while nothing plays; the worker pads to wall-clock time."""
    clock = StepClock()
    monkeypatch.setattr(system_win, "time", types.SimpleNamespace(monotonic=clock))
    src, sink = WinLoopbackSource(), Sink()
    src.start(sink)
    try:
        clock.advance_and_wait(0.0)  # t0 read and an empty poll: no gap, nothing emitted
        assert sink.total == 0
        clock.advance_and_wait(0.2)  # below GAP_FILL_S: still nothing
        assert sink.total == 0
        clock.advance_and_wait(0.8)  # 1 s behind wall clock
        out = sink.wait_samples(SAMPLE_RATE)
    finally:
        src.stop()
    assert out.size == SAMPLE_RATE and out.dtype == np.float32 and not out.any()


# ------------------------------------------------------------------ lifecycle
def test_start_is_idempotent_and_stop_releases_everything(pa):
    src = WinLoopbackSource()
    src.stop()  # stop before start is a no-op
    assert pa.instances == []
    src.start(Sink())
    stream, worker = pa.stream, src._worker
    src.start(Sink())  # already running: no second device
    assert len(pa.instances) == 1 and pa.stream is stream
    src.stop()
    stream.stop_stream.assert_called_once()
    stream.close.assert_called_once()
    assert pa.instances[0].terminated
    assert not worker.is_alive()
    src.stop()  # second stop is a no-op
    src.start(Sink())  # and it can be restarted
    assert len(pa.instances) == 2 and pa.stream is not stream
    src.stop()


@pytest.mark.parametrize(
    "failing, message",
    [
        (("stop_stream",), "stopping loopback stream failed"),
        (("close",), "closing loopback stream failed"),
        (("stop_stream", "close"), "closing loopback stream failed"),
    ],
)
def test_stream_errors_on_stop_are_logged_and_cleanup_continues(pa, caplog, failing, message):
    pa.failing = failing
    src = WinLoopbackSource()
    src.start(Sink())
    worker = src._worker
    src.stop()  # does not raise
    assert message in caplog.text
    pa.stream.close.assert_called_once()  # attempted even when stop_stream() raised
    assert pa.instances[0].terminated and not worker.is_alive()


def test_open_failure_cleans_up_and_reraises(pa):
    pa.open_error = OSError("Invalid sample rate")
    src = WinLoopbackSource()
    with pytest.raises(OSError, match="Invalid sample rate"):
        src.start(Sink())
    assert pa.instances[0].terminated and src._stream is None
    src._worker.join(timeout=5)
    assert not src._worker.is_alive()  # the worker was told to exit
    pa.open_error = None
    pa.loopbacks = []
    with pytest.raises(RuntimeError, match="No WASAPI loopback device"):
        src.start(Sink())
    assert pa.instances[1].terminated
