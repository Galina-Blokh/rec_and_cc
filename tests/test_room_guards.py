"""Room listening guards: a NaN/inf sample from a glitching mic must not deafen the noise-floor
gate, and a zero or negative room threshold must not start a session without speech (fake mic
source, energy VAD; no models, no real mic). Follow-up to #102.

Stories: US-22.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from recandcc.capture.base import AudioSource
from recandcc.config import DetectionConfig, load_config
from recandcc.detect.room import WINDOW, NoiseFloor, RoomMonitor, window_levels
from recandcc.models import SAMPLE_RATE
from recandcc.sensitivity import room_trigger

SR = SAMPLE_RATE
PER_S = int(SR / WINDOW)


class Source(AudioSource):
    def __init__(self):
        self.cb = None

    def start(self, on_frames):
        self.cb = on_frames

    def stop(self):
        self.cb = None


def energy_vad(audio: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(audio).reshape(-1, WINDOW)
    return (np.abs(w).max(axis=1) > 0.01).astype(np.float32)


def quiet(seconds: float, seed: int = 0) -> np.ndarray:
    """A quiet room at about -60 dBFS."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.001).astype(np.float32)


def voice(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def monitor(det: DetectionConfig) -> tuple[RoomMonitor, Source]:
    sources: list[Source] = []

    def factory():
        sources.append(Source())
        return sources[-1]

    mon = RoomMonitor(det, factory, vad=energy_vad, clock=lambda: 1000.0, thresholds=room_trigger)
    assert mon.start()
    return mon, sources[0]


def push(mon: RoomMonitor, src: Source, signal: np.ndarray, block: int = 1600) -> None:
    assert src.cb is not None
    for i in range(0, len(signal), block):
        src.cb(signal[i : i + block])
    assert mon.wait_idle()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_non_finite_level_does_not_poison_the_floor(bad):
    nf = NoiseFloor()
    assert not nf.add(np.full(PER_S * 30, -60.0), 10)
    step = np.full(PER_S // 2, -60.0)
    step[3] = bad
    assert not nf.add(step, 10)  # not a level shift ...
    assert nf.db == pytest.approx(-60.0)  # ... and the floor stays a number
    assert nf.loud(np.full(PER_S // 2, -6.0), 10)  # a voice right after is loud


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_window_levels_are_finite_for_a_glitching_device(bad):
    audio = np.full(2 * WINDOW, 0.1, np.float32)
    audio[5] = bad
    levels = window_levels(audio)
    assert np.isfinite(levels).all()
    assert levels[1] == pytest.approx(-20.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_speech_right_after_a_bad_sample_is_heard(bad):
    det = DetectionConfig(room_listening=True, room_speech_s=1.0, room_window_s=10.0)
    det.outside_sensitivity = 50
    mon, src = monitor(det)
    try:
        push(mon, src, quiet(10))  # the floor settles at about -60 dB
        floor = mon.floor.db
        assert floor is not None and floor < -50
        glitch = quiet(0.5, seed=1)
        glitch[100] = bad
        push(mon, src, glitch)
        assert mon.floor.db is not None and np.isfinite(mon.floor.db)
        push(mon, src, voice(1.5))  # before the fix: deaf for about 3 s
        assert mon.triggered()
    finally:
        mon.close()


@pytest.mark.parametrize("key", ["room_speech_s", "room_window_s"])
@pytest.mark.parametrize("value", ["0", "0.0", "-3", "nan"])
@pytest.mark.parametrize("outside", [None, 50])
def test_non_positive_room_thresholds_fall_back_with_one_warning(
    tmp_path, caplog, key, value, outside
):
    path = tmp_path / "config.toml"
    extra = f"outside_sensitivity = {outside}\n" if outside is not None else ""
    path.write_text(f"[detection]\n{key} = {value}\n{extra}", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        det = load_config(path).detection
    assert getattr(det, key) == getattr(DetectionConfig(), key)  # the default
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and key in warnings[0].getMessage()


def test_both_thresholds_bad_give_a_single_warning(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text("[detection]\nroom_speech_s = 0\nroom_window_s = -1\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        det = load_config(path).detection
    defaults = DetectionConfig()
    assert (det.room_speech_s, det.room_window_s) == (
        defaults.room_speech_s,
        defaults.room_window_s,
    )
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_positive_thresholds_load_without_a_warning(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text("[detection]\nroom_speech_s = 2\nroom_window_s = 8\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        det = load_config(path).detection
    assert (det.room_speech_s, det.room_window_s) == (2, 8)
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize("outside", [None, 50, 100])
@pytest.mark.parametrize(("speech_s", "window_s"), [(0.0, 10.0), (0.0, 0.0), (4.0, 0.0)])
def test_a_zero_threshold_never_triggers_without_speech(outside, speech_s, window_s):
    det = DetectionConfig(room_listening=True, room_speech_s=speech_s, room_window_s=window_s)
    det.outside_sensitivity = outside
    mon, src = monitor(det)
    try:
        assert not mon.triggered()  # the mic just opened: nothing heard
        push(mon, src, np.zeros(2 * SR, np.float32))
        assert not mon.triggered()  # silence is not speech
    finally:
        mon.close()


def test_a_zero_speech_threshold_still_triggers_on_speech():
    det = DetectionConfig(room_listening=True, room_speech_s=0.0, room_window_s=10.0)
    mon, src = monitor(det)
    try:
        push(mon, src, voice(1))
        assert mon.triggered()
    finally:
        mon.close()
