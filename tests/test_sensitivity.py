"""Sensitivity: 0-100 settings -> effective thresholds, presets, the room noise-floor gate
(pure functions, fake mic source and fake VAD; no models, no real mic).

Stories: US-23.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from recandcc.config import Config, DetectionConfig, load_config
from recandcc.detect.room import WINDOW, NoiseFloor, RoomMonitor, window_levels
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
    assert room_trigger(det) == pytest.approx((12.0, 24.0, 12.0))  # at most half the window
    det.outside_sensitivity = 0
    assert room_trigger(det) == pytest.approx((24.0, 48.0, 15.0))


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

    errors: list[BaseException] = []

    def spam(i):
        try:
            for j in range(20):
                cfg.detection.inside_sensitivity = i * 20 + j
                cfg.save(path)
        except BaseException as e:  # surfaced below: a thread's exception would be lost
            errors.append(e)

    threads = [threading.Thread(target=spam, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    cfg.save(path)  # the last save writes the current values
    assert load_config(path).detection.inside_sensitivity == cfg.detection.inside_sensitivity
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]  # no temp files left


def test_save_falls_back_to_writing_in_place_when_replace_is_refused(tmp_path, monkeypatch):
    import recandcc.config as config_mod

    def refuse(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(config_mod.os, "replace", refuse)
    monkeypatch.setattr(config_mod.time, "sleep", lambda s: None)
    cfg = Config()
    cfg.detection.inside_sensitivity = 70
    cfg.save(tmp_path / "c.toml")
    assert load_config(tmp_path / "c.toml").detection.inside_sensitivity == 70
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]  # temp file cleaned up


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


def test_noise_floor_is_a_low_percentile_of_recent_levels():
    nf = NoiseFloor(history_s=10, min_s=1)
    per_s = int(SAMPLE_RATE / WINDOW)
    assert nf.db is None and not nf.loud(np.array([0.0]), 9).any()  # too little history
    nf.add(np.full(per_s * 8, -40.0))  # babble most of the time ...
    nf.add(np.full(per_s * 2, -60.0))  # ... with some quieter moments
    assert nf.db == pytest.approx(-60.0)  # falls fast: 20% quiet is enough
    nf.add(np.full(per_s * 7, -40.0))
    assert nf.db == pytest.approx(-60.0)  # rises slowly: quiet moments still in the history
    nf.add(np.full(per_s * 2, -40.0))
    assert nf.db == pytest.approx(-40.0)  # ... until the room stayed louder long enough
    assert list(nf.loud(np.array([-35.0, -31.0]), 9)) == [False, True]
    assert window_levels(np.zeros(WINDOW, np.float32))[0] == -120.0
    assert window_levels(np.full(WINDOW, 0.1, np.float32))[0] == pytest.approx(-20.0)


# --- RoomMonitor with an outside sensitivity: synthetic audio, an energy "VAD" ---------------
SR = SAMPLE_RATE


def tone(seconds: float, amp: float, wobble: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    env = 1 + wobble * np.sin(2 * np.pi * 0.7 * t)
    return (amp * env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def talk(cycles: int, on: float, off: float, gap) -> np.ndarray:
    """Turn-taking speech: `on` s of voice, then `off` s of `gap(seconds)`."""
    return np.concatenate([np.concatenate([tone(on, 0.3), gap(off)]) for _ in range(cycles)])


def quiet(seconds: float) -> np.ndarray:
    return tone(seconds, 0.005)  # below the fake VAD's threshold


def babble(seconds: float) -> np.ndarray:
    return tone(seconds, 0.03, wobble=0.3)  # continuous, and the fake VAD calls it speech


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


def test_cafe_babble_never_triggers_nor_retriggers_after_a_session():
    calls: list = []
    mon, feed = monitor(50, calls)  # "Normal": 6 s in 15 s, 9 dB above the floor
    feed(babble(60))  # the mic opens in a café
    assert not mon.triggered() and mon.speech_in_window() == 0
    assert len(calls) < 10  # babble near the floor skips the VAD
    feed(talk(8, 1.4, 0.6, babble))  # a conversation at the table, over the babble
    assert mon.triggered()
    tap = mon.session_source()  # a room session records, then ends: history resets
    tap.start(lambda frames: None)
    feed(talk(5, 1.4, 0.6, babble))
    tap.stop()
    assert not mon.triggered()
    feed(babble(60))  # the floor survived the session: the café alone doesn't re-trigger
    assert not mon.triggered() and mon.speech_in_window() == 0
    mon.close()


def test_mic_opened_mid_conversation_and_long_talk_keep_counting():
    mon, feed = monitor(50)
    feed(talk(8, 1.4, 0.6, quiet))  # opened while people talk: the pauses set the floor
    assert mon.triggered()
    feed(talk(60, 1.4, 0.6, quiet))  # two minutes of conversation never becomes "the floor"
    assert mon.triggered() and mon.speech_in_window() > 8
    mon.close()


@pytest.mark.parametrize(
    ("on", "off", "fires"), [(1.4, 0.6, True), (1.2, 0.8, True), (0.8, 1.2, False)]
)
def test_lowest_setting_still_triggers_on_a_real_conversation(on, off, fires):
    mon, feed = monitor(0)  # 24 s of speech within 48 s, 15 dB above the floor
    feed(talk(30, on, off, quiet))
    assert mon.triggered() is fires  # 60-70% speech triggers, 40% doesn't
    mon.close()
