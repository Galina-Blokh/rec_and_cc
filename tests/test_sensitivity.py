"""Sensitivity: 0-100 settings -> effective thresholds, presets, the room noise-floor gate
(pure functions, fake mic source and fake VAD; no models, no real mic).

Stories: US-23.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from recandcc.config import Config, DetectionConfig, load_config
from recandcc.detect.room import NoiseFloor, RoomMonitor, level_db
from recandcc.sensitivity import (
    PRESETS,
    apply_preset,
    current_preset,
    effective_min_active_s,
    room_trigger,
    scale,
)


def test_unset_leaves_every_threshold_as_configured():
    det = DetectionConfig(min_active_s=3, room_speech_s=6, room_window_s=15)
    assert det.outside_sensitivity is None and det.inside_sensitivity is None  # off by default
    assert scale(None) == 1.0
    assert effective_min_active_s(det) == 3
    assert room_trigger(det) == (6.0, 15.0, None)


@pytest.mark.parametrize(
    ("s", "factor"),
    [(0, 4.0), (25, 2.0), (50, 1.0), (75, 0.5), (100, 0.25), (-20, 4.0), (300, 0.25)],
)
def test_scale_edges_and_clamping(s, factor):
    assert scale(s) == pytest.approx(factor)


def test_inside_maps_to_min_active_s():
    det = DetectionConfig(min_active_s=3)
    det.inside_sensitivity = 100
    assert effective_min_active_s(det) == pytest.approx(0.75)
    det.inside_sensitivity = 0
    assert effective_min_active_s(det) == pytest.approx(12.0)


def test_outside_maps_to_speech_and_margin_and_never_exceeds_the_window():
    det = DetectionConfig(room_speech_s=6, room_window_s=15)
    det.outside_sensitivity = 100
    assert room_trigger(det) == pytest.approx((1.5, 15.0, 3.0))
    det.outside_sensitivity = 50
    assert room_trigger(det) == pytest.approx((6.0, 15.0, 9.0))
    det.outside_sensitivity = 0
    assert room_trigger(det) == pytest.approx((15.0, 15.0, 15.0))  # 24 s capped at the window


def test_presets_apply_match_and_persist(tmp_path):
    cfg = Config()
    assert current_preset(cfg.detection) == "Off"
    for name in PRESETS:
        apply_preset(cfg.detection, name)
        assert current_preset(cfg.detection) == name
    apply_preset(cfg.detection, "Noisy place")
    cfg.save(tmp_path / "c.toml")
    det = load_config(tmp_path / "c.toml").detection
    assert (det.outside_sensitivity, det.inside_sensitivity) == (25, 50)
    apply_preset(cfg.detection, "Off")  # unset values are omitted (TOML has no null)
    cfg.save(tmp_path / "c.toml")
    assert "sensitivity" not in (tmp_path / "c.toml").read_text()
    assert current_preset(load_config(tmp_path / "c.toml").detection) == "Off"
    cfg.detection.outside_sensitivity = 60
    assert current_preset(cfg.detection) is None  # hand-edited values: no preset ticked


def test_noise_floor_needs_the_margin_above_recent_levels():
    nf = NoiseFloor(margin_db=6)
    assert nf.loud_enough(-40)  # no history yet
    for _ in range(20):
        nf.loud_enough(-40)  # a steady fan
    assert not nf.loud_enough(-37)
    assert nf.loud_enough(-30)  # someone talking over it
    assert level_db(np.zeros(10, dtype=np.float32)) == -120.0
    assert level_db(np.full(10, 0.1, dtype=np.float32)) == pytest.approx(-20.0)


class FakeSource:
    def start(self, on_frames):
        self.on_frames = on_frames

    def stop(self):
        pass


def test_room_monitor_skips_vad_for_steady_noise():
    src = FakeSource()
    calls = []

    def vad(block):
        calls.append(level_db(block))
        return 1.0

    mon = RoomMonitor(lambda: src, speech_fn=vad)
    mon.start(speech_s=2, window_s=5, margin_db=6)
    noise = np.full(1600, 0.01, dtype=np.float32)  # constant -40 dBFS
    for _ in range(50):  # 5 s
        src.on_frames(noise)
    loud = np.full(1600, 0.1, dtype=np.float32)  # -20 dBFS: someone talking
    for _ in range(20):  # 2 s
        src.on_frames(loud)
    deadline = time.monotonic() + 3
    while not mon.active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert mon.active
    # VAD ran on the first block (no floor yet) and the loud ones, never on the steady noise.
    assert calls == pytest.approx([-40.0] + [-20.0] * 2, abs=0.01)
    mon.stop()
