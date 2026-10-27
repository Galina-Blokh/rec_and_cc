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
    assert room_trigger(det) == pytest.approx((1.5, 15.0, 10.0))
    det.outside_sensitivity = 50
    assert room_trigger(det) == pytest.approx((6.0, 15.0, 13.0))
    det.outside_sensitivity = 25
    assert room_trigger(det) == pytest.approx((12.0, 24.0, 14.5))  # at most half the window
    det.outside_sensitivity = 0
    assert room_trigger(det) == pytest.approx((24.0, 48.0, 16.0))


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


def test_save_survives_a_temp_file_that_cannot_be_removed(tmp_path, monkeypatch):
    import recandcc.config as config_mod

    real_unlink = config_mod.Path.unlink

    def unlink(self, missing_ok=False):
        if self.name.endswith(".tmp"):
            raise PermissionError("still open")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(
        config_mod.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError())
    )
    monkeypatch.setattr(config_mod.time, "sleep", lambda s: None)
    monkeypatch.setattr(config_mod.Path, "unlink", unlink)
    cfg = Config()
    cfg.detection.outside_sensitivity = 40
    cfg.save(tmp_path / "c.toml")  # the in-place fallback worked: no error reported
    assert load_config(tmp_path / "c.toml").detection.outside_sensitivity == 40


def _controller(tmp_path, monkeypatch):
    import recandcc.ui.common as common
    from recandcc.ui.common import Controller

    path = tmp_path / "config.toml"
    monkeypatch.setattr(common, "CONFIG_PATH", path)

    class Monitor:
        def snapshot(self):
            return []

    class Q:
        def submit(self, *_a, **_k):
            pass

    cfg = Config(output_dir=tmp_path)
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    return ctrl, cfg, path


def test_reload_skips_our_own_saves_and_applies_hand_edits(tmp_path, monkeypatch):
    ctrl, cfg, path = _controller(tmp_path, monkeypatch)
    apply_preset(cfg.detection, "Normal")
    cfg.save(path)
    cfg.detection.outside_sensitivity = 75  # a click whose save hasn't run yet
    ctrl._maybe_reload_config()
    assert cfg.detection.outside_sensitivity == 75  # our own file isn't "reloaded" over it
    path.write_text("[detection]\noutside_sensitivity = 25\n")  # a hand edit
    ctrl._maybe_reload_config()
    assert cfg.detection.outside_sensitivity == 25


def test_reload_ignores_an_empty_or_broken_file(tmp_path, monkeypatch):
    ctrl, cfg, path = _controller(tmp_path, monkeypatch)
    cfg.detection.outside_sensitivity = 75
    path.write_text("")  # mid-way through an in-place write
    ctrl._maybe_reload_config()
    assert cfg.detection.outside_sensitivity == 75
    path.write_text("[detection\n")
    ctrl._maybe_reload_config()
    assert cfg.detection.outside_sensitivity == 75
    path.write_text("")
    assert load_config(path).detection.outside_sensitivity is None  # startup: defaults


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
    assert nf.db is None and not nf.loud(np.array([0.0]), 9)  # too little history: gated
    nf.add(np.full(per_s * 8, -40.0))  # babble most of the time ...
    nf.add(np.full(per_s * 2, -60.0))  # ... with some quieter moments
    assert nf.db == pytest.approx(-60.0)  # falls fast: 20% quiet is enough
    nf.add(np.full(per_s * 7, -40.0))
    assert nf.db == pytest.approx(-60.0)  # rises slowly: quiet moments still in the history
    nf.add(np.full(per_s * 2, -40.0))
    assert nf.db == pytest.approx(-40.0)  # ... until the room stayed louder long enough
    # A step counts by its loud windows (90th percentile), so syllable dips don't sink it.
    step = np.array([-60.0] * 5 + [-30.0] * 5)
    assert nf.loud(step, 9) and not nf.loud(step, 11)
    assert not nf.loud(np.full(10, -35.0), 9)
    nf.clear()
    assert nf.db is None
    assert window_levels(np.zeros(WINDOW, np.float32))[0] == -120.0
    assert window_levels(np.full(WINDOW, 0.1, np.float32))[0] == pytest.approx(-20.0)


# --- RoomMonitor with an outside sensitivity ---------------------------------------------
# Deterministic, seeded speech-like signals (from the review simulations): syllabic
# envelopes (60% of 50 ms slots voiced, amplitude 0.3-1.5) over noise carriers, turn-taking
# 1.3 s on / 0.7 s off, and N-talker café babble. The fake VAD is an energy detector with a
# short hangover, like Silero's: it calls babble speech too (the worst case for the gate).
SR = SAMPLE_RATE
SLOT = 800  # 50 ms syllable slots
HANG = 4  # windows of VAD hangover (~130 ms)


def syllabic(n: int, rng: np.random.Generator, lo: float = 0.3, hi: float = 1.5) -> np.ndarray:
    k = n // SLOT + 1
    env = np.repeat((rng.random(k) < 0.6) * rng.uniform(lo, hi, k), SLOT)[:n]
    return np.convolve(env, np.ones(240) / 240, "same") * rng.standard_normal(n)


def quiet_room(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.001).astype(np.float32)  # ~-60 dB


def babble(seconds: float, talkers: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = sum(syllabic(n, rng, 0.5, 1.5) * 0.02 for _ in range(talkers))
    return np.asarray(out, np.float32)


def conversation(seconds: float, bed: np.ndarray, over_db: float, seed: int = 1) -> np.ndarray:
    """Turn-taking voices (1.3 s on, 0.7 s off: 65% speech) `over_db` above the `bed`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    mask = (np.arange(n) % int(2.0 * SR)) < int(1.3 * SR)
    voice = syllabic(n, rng) * mask
    gain = 10 ** (over_db / 20) * np.sqrt(np.mean(bed[:n] ** 2)) / np.sqrt(np.mean(voice**2))
    return (voice * gain + bed[:n]).astype(np.float32)


def fake_vad(audio: np.ndarray) -> np.ndarray:
    act = np.abs(audio.reshape(-1, WINDOW)).max(axis=1) > 0.01
    out = act.copy()
    for k in range(1, HANG + 1):
        out[k:] |= act[:-k]
    return out.astype(np.float32)


class Source:
    def start(self, on_frames):
        self.cb = on_frames

    def stop(self):
        pass


def monitor(outside: float, calls: list | None = None):
    """A listening monitor and feed(signal) -> whether it triggered at any second."""
    det = DetectionConfig(room_listening=True, outside_sensitivity=outside)

    def vad(audio):
        if calls is not None:
            calls.append(audio.size)
        return fake_vad(audio)

    src = Source()
    mon = RoomMonitor(det, lambda: src, vad=vad, clock=lambda: 0.0, thresholds=room_trigger)
    assert mon.start()

    def feed(signal) -> bool:
        fired = False
        for i in range(0, len(signal), SR):
            for j in range(i, min(i + SR, len(signal)), 1600):
                src.cb(signal[j : min(j + 1600, i + SR)])
            assert mon.wait_idle()
            fired |= mon.triggered()
        return fired

    return mon, feed


@pytest.mark.parametrize("outside", [0, 25, 50, 75])
@pytest.mark.parametrize("place", ["quiet", "cafe"])
def test_a_conversation_fires_at_every_setting(outside, place):
    bed = quiet_room(90) if place == "quiet" else babble(90, 4)
    over = 34 if place == "quiet" else 20  # voices vs a -60 dB room / 20 dB over babble
    mon, feed = monitor(outside)
    assert feed(conversation(90, bed, over))
    mon.close()


@pytest.mark.parametrize("talkers", [2, 4, 8])
@pytest.mark.parametrize("outside", [25, 50])
def test_babble_never_triggers_nor_retriggers_at_normal_or_noisy(outside, talkers):
    calls: list = []
    mon, feed = monitor(outside, calls)
    assert not feed(babble(120, talkers, seed=talkers))
    assert len(calls) < 60  # most babble steps sit at the floor and skip the VAD
    assert feed(conversation(30, babble(30, talkers, seed=99), 20))  # a talk at the table
    tap = mon.session_source()  # ... recorded as a room session, which ends
    tap.start(lambda frames: None)
    tap.stop()
    assert not feed(babble(120, talkers, seed=talkers + 1))  # the café alone: no re-trigger
    mon.close()


def test_quiet_room_preset_ignores_four_or_more_talkers():
    for talkers in (4, 8):
        mon, feed = monitor(75)
        assert not feed(babble(120, talkers, seed=talkers))
        mon.close()


@pytest.mark.parametrize("outside", [25, 50])
def test_a_reopened_mic_relearns_the_floor(outside):
    mon, feed = monitor(outside)
    feed(quiet_room(30))  # a quiet office ...
    mon.close_mic()
    assert mon.start()  # ... lid closed, opened in a café
    assert not feed(babble(60, 4))
    mon.close()


def test_mic_opened_mid_conversation_and_long_talk_keep_counting():
    mon, feed = monitor(50)
    talk = conversation(150, quiet_room(150), 34)
    assert feed(talk[: 20 * SR])  # opened while people talk: the pauses set the floor
    feed(talk[20 * SR :])  # two more minutes never become "the floor"
    assert mon.triggered() and mon.speech_in_window() > 7
    mon.close()


def test_lowest_setting_needs_more_than_a_few_remarks():
    mon, feed = monitor(0)  # 24 s of speech within 48 s
    sparse = conversation(90, quiet_room(90), 34)
    sparse[(np.arange(sparse.size) % (4 * SR)) >= 2 * SR] = 0.0005  # half the turns silent
    assert not feed(sparse)
    mon.close()
