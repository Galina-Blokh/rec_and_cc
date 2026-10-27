"""Room listening (detect/room.py) on synthetic audio, no real microphone: a deterministic
energy "VAD", plus one check with the real Silero VAD on speech-like bursts; the pre-roll in
a real Recorder session; mic-track diarization; the menu toggle. Monitor tests from #58.

Stories: US-22.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

import numpy as np
import soundfile as sf

from recandcc.capture.base import AudioSource
from recandcc.config import Config, DetectionConfig, load_config
from recandcc.detect.room import WINDOW, RoomMonitor, silero_vad
from recandcc.models import SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.recorder import Recorder, should_record_mic

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


def energy_vad(audio: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for Silero: a window is speech if it isn't silent."""
    w = audio.reshape(-1, WINDOW)
    return (np.abs(w).max(axis=1) > 0.01).astype(np.float32)


def make(det: DetectionConfig | None = None, vad=energy_vad, fail: bool = False, clock=None):
    sources: list[ManualSource] = []

    def factory():
        sources.append(ManualSource(fail=fail))
        return sources[-1]

    det = det or DetectionConfig(room_listening=True, room_speech_s=4.0, room_window_s=10.0)
    mon = RoomMonitor(det, factory, vad=vad, clock=clock or Clock())
    return mon, sources


def feed(mon: RoomMonitor, src: ManualSource, signal: np.ndarray) -> None:
    src.push(signal)
    assert mon.wait_idle()


def test_room_listening_is_off_by_default_and_needs_the_mic(tmp_path):
    assert DetectionConfig().room_listening is False
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "auto"\n', encoding="utf-8")
    assert load_config(path).detection.room_listening is False
    assert should_record_mic("calls", TriggerKind.ROOM)
    assert not should_record_mic("never", TriggerKind.ROOM)
    mon, sources = make()
    assert not mon.running and sources == []  # constructing opens nothing


def test_conversation_triggers_but_silence_and_short_utterances_do_not():
    silero_vad(silence(1)[: 2 * WINDOW])  # load the model before timing anything
    mon, sources = make(vad=silero_vad)
    try:
        assert mon.start() and mon.running
        (src,) = sources
        feed(mon, src, silence(12))
        assert not mon.triggered() and mon.speech_in_window() == 0
        feed(mon, src, burst(1.5))  # a short remark of my own
        feed(mon, src, silence(3))
        assert not mon.triggered()
        feed(mon, src, burst(6))
        assert mon.triggered()
        assert mon.speech_in_window() >= 4.0
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
        assert abs(mon.seconds_since_speech() - 5.0) < 0.6  # up to one 0.5 s step pending
        clock.t += 0.9  # frames a bit late (VAD or capture jitter): no stall yet
        assert mon.seconds_since_speech() < 5.6
        clock.t += 30  # the mic stopped delivering frames
        assert 34.0 < mon.seconds_since_speech() < 35.6
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


def test_live_frames_during_the_preroll_replay_follow_it_in_order():
    det = DetectionConfig(room_listening=True, room_preroll_s=2.0, room_window_s=1.0)
    mon, sources = make(det)
    try:
        mon.start()
        (src,) = sources
        ramp = np.arange(4 * SR, dtype=np.float32) / (4 * SR)
        src.push(ramp[: 2 * SR])
        got: list[np.ndarray] = []

        def slow_writer(block):  # a live frame arrives during the first replayed block
            if not got:
                src.push(ramp[2 * SR : 3 * SR])
            got.append(block)

        tap = mon.session_source()
        tap.start(slow_writer)
        src.push(ramp[3 * SR :])
        audio = np.concatenate(got)
        first = int(round(float(audio[0]) * 4 * SR))
        np.testing.assert_array_equal(audio, ramp[first:])
        tap.stop()
    finally:
        mon.close()


def test_close_mic_closes_it_and_start_reopens_a_fresh_stream():
    mon, sources = make(vad=lambda a: np.ones(a.size // WINDOW, np.float32))
    try:
        mon.start()
        old = sources[0]
        feed(mon, old, burst(5))
        assert mon.triggered()
        mon.close_mic()
        assert old.stopped and not mon.running
        assert not mon.triggered() and mon.speech_in_window() == 0
        old.push(burst(5))  # late frames of the closed stream are dropped
        assert mon.wait_idle() and mon.speech_in_window() == 0
        assert mon.start() and mon.running and len(sources) == 2
        feed(mon, sources[1], burst(5))
        assert mon.triggered()
    finally:
        mon.close()


def test_reset_forgets_the_preroll_and_speech_and_close_is_final():
    mon, sources = make()
    mon.start()
    feed(mon, sources[0], burst(6))
    assert mon.triggered() and mon._ring_size
    mon.reset()  # an app is doing audio: nothing it played may reach a room session
    assert mon.wait_idle() and not mon.triggered() and mon._ring_size == 0
    got: list[np.ndarray] = []
    tap = mon.session_source()
    tap.start(got.append)
    assert got == []
    tap.stop()
    mon.close()
    assert mon.start() is False and not mon.running  # a late engine tick can't reopen it


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


def test_the_preroll_stays_bounded_and_close_ends_the_worker():
    det = DetectionConfig(room_listening=True, room_preroll_s=3.0, room_window_s=2.0)
    mon, sources = make(det)
    mon.start()
    feed(mon, sources[0], burst(60))
    assert 3 * SR <= mon._ring_size <= 3 * SR + 1600  # pre-roll + one capture block
    worker = mon._worker
    mon.close()
    assert worker is not None and not worker.is_alive()


def test_speech_queued_before_a_session_ends_does_not_retrigger():
    gate = threading.Event()

    def gated_vad(audio):
        gate.wait(5)
        return energy_vad(audio)

    mon, sources = make(vad=gated_vad)
    try:
        mon.start()
        (src,) = sources
        tap = mon.session_source()
        tap.start(lambda block: None)
        src.push(burst(8))  # still queued for the VAD when the session ends
        tap.stop()
        gate.set()
        assert mon.wait_idle()
        assert not mon.triggered() and mon.speech_in_window() == 0
    finally:
        mon.close()


def test_room_session_keeps_the_preroll_and_pads_the_other_track(tmp_path):
    cfg = Config(output_dir=tmp_path)
    cfg.detection.room_preroll_s = 4.0
    mon, sources = make(cfg.detection)
    system = ManualSource()
    rec = Recorder(cfg, system_factory=lambda: system, min_session_s=0)
    try:
        mon.start()
        (mic,) = sources
        ramp = np.arange(6 * SR, dtype=np.float32) / (8 * SR)  # unique, < 1 for the WAV
        mic.push(ramp[: 4 * SR])
        before = datetime.now().astimezone()
        session = rec.start(TriggerKind.ROOM, "Room", mic_source=mon.session_source)
        assert 3.9 < (before - session.started_at).total_seconds() < 4.5  # incl. pre-roll
        heard: list[tuple] = []  # live transcription attaches after start(): whole pre-roll
        rec.add_listener(lambda track, start, block: heard.append((track, start, block.size)))
        mic.push(ramp[4 * SR :])
        system.push(np.full(SR, 0.25, np.float32))
        out = rec.stop()
    finally:
        mon.close()
    assert out is not None and out.trigger_kind == TriggerKind.ROOM
    me, _ = sf.read(session.directory / "me.wav", dtype="float32")
    others, _ = sf.read(session.directory / "others.wav", dtype="float32")
    # The mic track starts with the 4 s before the trigger and continues without a gap ...
    assert len(me) == 6 * SR
    np.testing.assert_allclose(me, ramp, atol=1e-4)
    # ... and the system track starts with that much silence, so both stay aligned.
    assert len(others) == 5 * SR and not others[: 4 * SR].any() and others[4 * SR :].all()
    assert SessionInfo.load(session.directory).trigger_kind == TriggerKind.ROOM
    started = SessionInfo.load(session.directory).started_at  # when the audio begins
    assert abs((out.started_at - started).total_seconds()) < 1e-3
    me_heard = [(start, n) for track, start, n in heard if track == Track.ME]
    assert me_heard[0][0] == 0 and sum(n for _, n in me_heard) == 6 * SR


def test_room_sessions_diarize_the_mic_track():
    from pathlib import Path

    from recandcc.transcribe.engine import diarized_track

    def session(kind):
        return SessionInfo("s", Path("."), datetime.now(), kind)

    assert diarized_track(session(TriggerKind.ROOM)) == Track.ME
    assert diarized_track(session(TriggerKind.CALL)) == Track.OTHERS


def test_menu_toggle_saves_warns_and_engine_owns_a_closed_room_monitor(tmp_path, monkeypatch):
    from recandcc.ui.common import ROOM_LISTENING_NOTICE, Controller, keep_prompt_text

    class Monitor:
        def snapshot(self):
            return []

    class Q:
        def submit(self, *_a, **_k):
            pass

    cfg = Config(output_dir=tmp_path)
    saved = threading.Event()
    monkeypatch.setattr(cfg, "save", lambda *a, **k: saved.set())
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    assert isinstance(ctrl.engine.room, RoomMonitor) and not ctrl.engine.room.running
    ctrl._perm = {"microphone": "denied"}
    ctrl.set_room_listening(True)
    assert cfg.detection.room_listening and saved.wait(2)
    assert ctrl.events.get_nowait() == ("notice", "Room listening on", ROOM_LISTENING_NOTICE)
    assert ctrl.events.get_nowait()[1] == "Room listening can't hear anything"
    ctrl.set_room_listening(False)
    assert not cfg.detection.room_listening and ctrl.events.empty()  # no notice when off
    text = keep_prompt_text("Room", TriggerKind.ROOM, "record")
    assert "a conversation near the laptop" in text
