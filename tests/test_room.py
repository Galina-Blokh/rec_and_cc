"""Room listening monitor (detect/room.py): real Silero VAD on synthetic speech-like bursts,
no real microphone.

Stories: US-22.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from recandcc.capture.base import AudioSource
from recandcc.config import Config, DetectionConfig, load_config
from recandcc.detect.room import WINDOW, RoomMonitor
from recandcc.models import SAMPLE_RATE

SR = SAMPLE_RATE


def burst(seconds: float, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    """Amplitude-modulated tone: Silero VAD flags it as speech."""
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (1 + np.sin(2 * np.pi * 4 * t)) / 2).astype(
        np.float32
    )


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


class ManualSource(AudioSource):
    """The test thread plays the capture thread: ``push`` calls the frame callback."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.cb = None
        self.stopped = False

    def start(self, on_frames):
        if self.fail:
            raise RuntimeError("no microphone")
        self.cb = on_frames

    def stop(self):
        self.stopped = True

    def push(self, signal: np.ndarray, block: int = 1600) -> None:
        assert self.cb is not None
        for i in range(0, len(signal), block):
            self.cb(signal[i : i + block])


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def make(det: DetectionConfig | None = None, vad=None, fail: bool = False, clock=None):
    sources: list[ManualSource] = []

    def factory():
        sources.append(ManualSource(fail=fail))
        return sources[-1]

    det = det or DetectionConfig(room_listening=True)
    mon = RoomMonitor(det, factory, vad=vad, clock=clock or Clock())
    return mon, sources


def feed(mon: RoomMonitor, src: ManualSource, signal: np.ndarray) -> None:
    src.push(signal)
    assert mon.wait_idle()


def test_room_listening_is_off_by_default(tmp_path):
    assert DetectionConfig().room_listening is False
    assert Config().detection.room_listening is False
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "auto"\n', encoding="utf-8")
    assert load_config(path).detection.room_listening is False
    assert load_config(tmp_path / "new.toml").detection.room_listening is False


def test_conversation_triggers_but_silence_and_short_utterances_do_not():
    mon, sources = make()
    try:
        assert mon.start() and mon.running
        (src,) = sources
        feed(mon, src, silence(12))
        assert not mon.triggered() and mon.speech_in_window() == 0
        feed(mon, src, burst(1.5))  # a short remark of my own
        feed(mon, src, silence(3))
        assert not mon.triggered()
        feed(mon, src, burst(5))
        assert mon.triggered()
        assert 4.0 <= mon.speech_in_window() <= 7.0
    finally:
        mon.close()


def test_speech_outside_the_window_expires():
    mon, sources = make()
    try:
        mon.start()
        (src,) = sources
        feed(mon, src, burst(3))
        feed(mon, src, silence(11))
        feed(mon, src, burst(3))
        assert not mon.triggered()  # 3 s + 3 s, but never 4 s within 10 s
        feed(mon, src, burst(2))
        assert mon.triggered()
    finally:
        mon.close()


def test_seconds_since_speech_counts_audio_and_adds_a_stalled_mic():
    clock = Clock()
    mon, sources = make(clock=clock)
    try:
        mon.start()
        (src,) = sources
        feed(mon, src, burst(3))
        feed(mon, src, silence(5))
        assert 4.0 <= mon.seconds_since_speech() <= 5.5
        clock.t += 30  # the mic stopped delivering frames
        assert 33.0 <= mon.seconds_since_speech() <= 35.5
    finally:
        mon.close()


def test_session_source_replays_the_preroll_then_continues_without_gap():
    det = DetectionConfig(room_listening=True, room_preroll_s=4.0, room_window_s=2.0)
    mon, sources = make(det, vad=lambda a: np.ones(a.size // WINDOW, np.float32))
    try:
        mon.start()
        (src,) = sources
        ramp = np.arange(10 * SR, dtype=np.float32) / (10 * SR)  # every sample is unique
        src.push(ramp[: 6 * SR])
        got: list[np.ndarray] = []
        tap = mon.session_source()
        tap.start(got.append)
        src.push(ramp[6 * SR :])
        audio = np.concatenate(got)
        first = int(round(float(audio[0]) * 10 * SR))
        assert 6 * SR - first >= 4 * SR - 1600  # >= pre-roll (minus one capture block)
        assert 6 * SR - first <= 4 * SR + 1600
        np.testing.assert_array_equal(audio, ramp[first:])  # contiguous to the end
        tap.stop()
        assert mon.wait_idle()
        assert not mon.triggered()  # the next session doesn't start from this one's speech
        src.push(ramp[:SR])
        assert sum(map(len, got)) == audio.size  # detached: nothing more delivered
        got.clear()
        tap2 = mon.session_source()
        tap2.start(got.append)
        assert sum(map(len, got)) == SR  # pre-roll holds only audio after the last session
        tap2.stop()
    finally:
        mon.close()


def test_pause_closes_the_mic_and_resume_reopens_it():
    mon, sources = make(vad=lambda a: np.ones(a.size // WINDOW, np.float32))
    try:
        mon.start()
        old = sources[0]
        feed(mon, old, burst(5))
        assert mon.triggered()
        mon.pause()
        assert old.stopped and not mon.running and mon.paused
        assert not mon.triggered() and mon.speech_in_window() == 0
        assert mon.start() is False  # paused: start() doesn't reopen it
        old.push(burst(5))  # late frames of the closed stream are dropped
        assert mon.wait_idle() and mon.speech_in_window() == 0
        assert mon.resume() and mon.running and len(sources) == 2
        feed(mon, sources[1], burst(5))
        assert mon.triggered()
    finally:
        mon.close()


def test_a_missing_microphone_is_reported_not_raised():
    mon, _ = make(fail=True)
    try:
        assert mon.start() is False
        assert not mon.running and not mon.triggered()
    finally:
        mon.close()


def test_vad_runs_off_the_capture_thread_and_never_blocks_it():
    gate = threading.Event()
    threads: list[str] = []
    calls = 0

    def slow_vad(audio):
        nonlocal calls
        threads.append(threading.current_thread().name)
        calls += 1
        if calls == 2:
            raise RuntimeError("model failed")  # one failing step must not stop the worker
        gate.wait(5)
        return np.ones(audio.size // WINDOW, np.float32)

    mon, sources = make(vad=slow_vad)
    try:
        mon.start()
        t0 = time.monotonic()
        sources[0].push(burst(10))  # the VAD is stuck on the first step meanwhile
        assert time.monotonic() - t0 < 1.0
        gate.set()
        assert mon.wait_idle()
        assert set(threads) == {"room-vad"}
        assert mon.triggered()
    finally:
        mon.close()
