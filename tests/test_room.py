"""Room listening detector: sliding speech window, mic stream handling, VAD off the capture
thread (fake source and fake VAD; no models, no real mic).

Stories: US-22.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from recandcc.config import DetectionConfig
from recandcc.detect.room import BLOCK_S, RoomMonitor, SpeechWindow
from recandcc.models import SAMPLE_RATE, TriggerKind
from recandcc.recorder import should_record_mic


class FakeSource:
    def __init__(self, fail=False):
        self.fail = fail
        self.on_frames = None
        self.stopped = False

    def start(self, on_frames):
        if self.fail:
            raise OSError("no mic")
        self.on_frames = on_frames

    def stop(self):
        self.stopped = True


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.01)
    return cond()


def test_speech_window_needs_enough_speech_within_the_window():
    w = SpeechWindow(speech_s=6, window_s=15)
    assert not any(w.add(1.0, 1.0) for _ in range(5))  # 5 s of speech: not yet
    assert w.add(1.0, 1.0)  # 6 s
    for _ in range(9):  # quiet: 6 s still inside the 15 s window
        assert w.add(1.0, 0.0)
    assert not w.add(1.0, 0.0)  # the first speech second slid out


def test_speech_window_scattered_short_utterances_do_not_trigger():
    w = SpeechWindow(speech_s=6, window_s=15)
    for _ in range(60):  # 0.3 s of speech every second: ~4.5 s per window
        assert not w.add(1.0, 0.3)
    assert not w.add(1.0, 5.0)  # a block never counts more than its length


def test_monitor_runs_vad_off_the_capture_thread_and_reports_speech():
    src = FakeSource()
    calls: list[str] = []

    def vad(block):
        assert block.dtype == np.float32 and block.size == int(BLOCK_S * SAMPLE_RATE)
        calls.append(threading.current_thread().name)
        return 1.0

    mon = RoomMonitor(lambda: src, speech_fn=vad)
    assert not mon.running and src.on_frames is None  # constructing opens nothing
    mon.start(speech_s=2, window_s=5)
    assert mon.running and src.on_frames is not None
    frames = np.zeros(1600, dtype=np.float32)
    for _ in range(25):  # 2.5 s in 0.1 s batches
        src.on_frames(frames)
    frames[:] = 1.0  # the callback copied the frames: later changes don't leak in
    assert wait_for(lambda: mon.active)
    assert calls and set(calls) == {"room-vad"}
    mon.stop()
    assert src.stopped and not mon.running and not mon.active
    mon.stop()  # idempotent


def test_monitor_vad_failure_counts_as_silence():
    src = FakeSource()

    def vad(block):
        raise RuntimeError("model missing")

    mon = RoomMonitor(lambda: src, speech_fn=vad)
    mon.start(speech_s=1, window_s=5)
    for _ in range(30):
        src.on_frames(np.zeros(1600, dtype=np.float32))
    time.sleep(0.1)
    assert not mon.active
    mon.stop()


def test_monitor_mic_failure_raises_and_leaves_nothing_running():
    mon = RoomMonitor(lambda: FakeSource(fail=True), speech_fn=lambda b: 1.0)
    with pytest.raises(OSError):
        mon.start(speech_s=1, window_s=5)
    assert not mon.running


def test_room_sessions_record_the_mic_unless_never_and_default_off():
    assert DetectionConfig().room_listening is False
    assert should_record_mic("calls", TriggerKind.ROOM)
    assert not should_record_mic("never", TriggerKind.ROOM)


def test_menu_toggle_saves_warns_and_engine_owns_a_closed_room_monitor(tmp_path, monkeypatch):
    from recandcc.config import Config
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
    ctrl.set_room_listening(True)
    assert cfg.detection.room_listening and saved.wait(2)
    assert ctrl.events.get_nowait() == ("notice", "Room listening on", ROOM_LISTENING_NOTICE)
    ctrl.set_room_listening(False)
    assert not cfg.detection.room_listening and ctrl.events.empty()  # no notice when off
    text = keep_prompt_text("Room", TriggerKind.ROOM, "record")
    assert "a conversation near the laptop" in text
