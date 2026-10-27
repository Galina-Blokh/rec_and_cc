"""Sensitivity: 0-100 settings -> effective thresholds, presets, the room noise-floor gate
(pure functions, fake mic source and fake VAD; no models, no real mic).

Stories: US-23.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from recandcc.config import Config, DetectionConfig, load_config
from recandcc.detect.room import WINDOW, NoiseFloor, RoomMonitor, level_db
from recandcc.models import SAMPLE_RATE
from recandcc.sensitivity import (
    PRESETS,
    apply_preset,
    current_preset,
    effective_min_active_s,
    room_trigger,
    scale,
    valid,
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


def test_outside_maps_to_speech_and_margin_and_widens_the_window():
    det = DetectionConfig(room_speech_s=6, room_window_s=15)
    det.outside_sensitivity = 100
    assert room_trigger(det) == pytest.approx((1.5, 15.0, 3.0))
    det.outside_sensitivity = 50
    assert room_trigger(det) == pytest.approx((6.0, 15.0, 9.0))
    det.outside_sensitivity = 25
    assert room_trigger(det) == pytest.approx((12.0, 20.0, 12.0))  # at most 60% of the window
    det.outside_sensitivity = 0
    assert room_trigger(det) == pytest.approx((24.0, 40.0, 15.0))


@pytest.mark.parametrize("bad", ["normal", True, float("nan"), [50]])
def test_non_numeric_values_count_as_unset(bad):
    assert valid(bad) is None and scale(bad) == 1.0
    det = DetectionConfig(min_active_s=3, outside_sensitivity=bad, inside_sensitivity=bad)
    assert effective_min_active_s(det) == 3 and room_trigger(det)[2] is None


def test_load_config_drops_non_numeric_sensitivity(tmp_path, caplog):
    path = tmp_path / "c.toml"
    path.write_text('[detection]\ninside_sensitivity = "normal"\noutside_sensitivity = true\n')
    det = load_config(path).detection
    assert det.inside_sensitivity is None and det.outside_sensitivity is None
    assert "inside_sensitivity" in caplog.text and "outside_sensitivity" in caplog.text
    path.write_text("[detection]\ninside_sensitivity = 80\n")
    assert load_config(path).detection.inside_sensitivity == 80


def test_concurrent_saves_never_tear_the_file(tmp_path):
    path = tmp_path / "c.toml"
    cfg = Config()

    def spam(i):
        for j in range(20):
            cfg.detection.inside_sensitivity = i * 20 + j
            cfg.save(path)

    threads = [threading.Thread(target=spam, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert load_config(path).detection.inside_sensitivity is not None
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]  # no temp files left


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


def test_noise_floor_learns_fast_down_and_slowly_up():
    nf = NoiseFloor()
    assert nf.loud_enough(-90, 9)  # no floor yet: everything counts
    nf.learn(-40)
    assert not nf.loud_enough(-35, 9) and nf.loud_enough(-31, 9)
    nf.learn(-60)
    assert nf.db == pytest.approx(-43)  # falls 3 dB per step
    for _ in range(4):
        nf.learn(0)
    assert nf.db == pytest.approx(-41)  # rises 0.5 dB per step
    assert level_db(np.zeros(10, np.float32)) == -120.0
    assert level_db(np.full(10, 0.1, np.float32)) == pytest.approx(-20.0)


# --- RoomMonitor with an outside sensitivity: synthetic audio, an energy "VAD" ---------------
SR = SAMPLE_RATE


def tone(seconds: float, amp: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class Source:
    def start(self, on_frames):
        self.cb = on_frames

    def stop(self):
        pass


def monitor(outside: float, calls: list | None = None):
    det = DetectionConfig(room_listening=True, outside_sensitivity=outside)

    def vad(audio):  # a window is speech if it isn't near-silent
        if calls is not None:
            calls.append(audio.size)
        return (np.abs(audio.reshape(-1, WINDOW)).max(axis=1) > 0.01).astype(np.float32)

    src = Source()
    mon = RoomMonitor(det, lambda: src, vad=vad, clock=lambda: 0.0, thresholds=room_trigger)
    assert mon.start()

    def feed(signal):
        for i in range(0, len(signal), 1600):
            src.cb(signal[i : i + 1600])
        assert mon.wait_idle()

    return mon, feed


def test_cafe_murmur_near_the_floor_does_not_trigger_but_a_conversation_does():
    calls: list = []
    mon, feed = monitor(50, calls)  # "Normal": 6 s in 15 s, 9 dB above the floor
    for _ in range(8):  # murmur the VAD flags half the time, a little above the quiet parts
        feed(tone(2, 0.008))  # -45 dB, not speech: the floor
        feed(tone(2, 0.018))  # -38 dB, "speech" but within the margin
    assert not mon.triggered() and mon.speech_in_window() < 1
    vad_runs = len(calls)
    assert vad_runs < 8  # quiet and murmur steps skip the VAD once the floor is known
    feed(tone(8, 0.3))  # someone talking next to the laptop
    assert mon.triggered()
    mon.close()


def test_mic_opened_mid_conversation_and_long_talk_keep_counting():
    mon, feed = monitor(50)
    feed(tone(8, 0.3))  # opened while people talk: no floor yet, speech still counts
    assert mon.triggered()
    feed(tone(120, 0.3))  # two minutes of continuous talk never becomes "the floor"
    assert mon.triggered() and mon.speech_in_window() > 14
    mon.close()


@pytest.mark.parametrize(("talk", "pause", "fires"), [(1.4, 0.6, True), (1.0, 1.0, False)])
def test_lowest_setting_still_triggers_on_a_real_conversation(talk, pause, fires):
    mon, feed = monitor(0)  # 24 s of speech within 40 s, 15 dB above the floor
    for _ in range(30):
        feed(np.concatenate([tone(talk, 0.3), np.zeros(int(pause * SR), np.float32)]))
    assert mon.triggered() is fires  # ~70% speech triggers, 50% doesn't
    mon.close()
