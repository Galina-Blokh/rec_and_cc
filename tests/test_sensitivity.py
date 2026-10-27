"""Sensitivity: 0-100 settings -> effective thresholds, presets, the room noise-floor gate
(pure functions, fake mic source and fake VAD; no models, no real mic).

Stories: US-23.
"""

from __future__ import annotations

import functools
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
    assert room_trigger(det) == pytest.approx((1.5, 15.0, 9.0))
    det.outside_sensitivity = 50
    assert room_trigger(det) == pytest.approx((6.0, 15.0, 10.5))
    det.outside_sensitivity = 25
    assert room_trigger(det) == pytest.approx((12.0, 24.0, 11.25))  # at most half the window
    det.outside_sensitivity = 0
    assert room_trigger(det) == pytest.approx((24.0, 48.0, 12.0))


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


def test_reload_notices_a_same_size_edit_within_one_mtime_tick(tmp_path, monkeypatch):
    import os

    ctrl, cfg, path = _controller(tmp_path, monkeypatch)
    cfg.detection.outside_sensitivity = 25
    cfg.save(path)
    stamp = path.stat().st_mtime_ns
    path.write_text(
        path.read_text().replace("outside_sensitivity = 25", "outside_sensitivity = 75")
    )
    os.utime(path, ns=(stamp, stamp))  # coarse mtime: same size, same timestamp
    ctrl._maybe_reload_config()
    assert cfg.detection.outside_sensitivity == 75


def test_a_preset_click_keeps_a_hand_edit_made_just_before(tmp_path, monkeypatch):
    ctrl, cfg, path = _controller(tmp_path, monkeypatch)
    cfg.save(path)
    ctrl._maybe_reload_config()
    path.write_text(path.read_text().replace("min_active_s = 3.0", "min_active_s = 5.0"))
    monkeypatch.setattr(ctrl, "_save_settings", lambda: cfg.save(path))
    ctrl.set_sensitivity("Noisy place")  # before the periodic reload saw the edit
    det = load_config(path).detection
    assert det.min_active_s == 5.0 and (det.outside_sensitivity, det.inside_sensitivity) == (25, 50)


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
    nf.add(np.full(per_s * 8, -40.0))  # chatter ...
    nf.add(np.full(per_s * 2, -60.0))  # ... with a few quiet seconds: not yet a floor change
    assert nf.db == pytest.approx(-40.0)
    nf.add(np.full(per_s * 2, -60.0))
    assert nf.db == pytest.approx(-60.0)  # falls fast: 4 quiet seconds out of 10
    nf.add(np.full(per_s * 4, -40.0))
    assert nf.db == pytest.approx(-60.0)  # rises slowly: quiet moments still in the history
    nf.add(np.full(per_s * 3, -40.0))
    assert nf.db == pytest.approx(-40.0)  # ... until the room stayed louder long enough
    # A step counts by its loud windows (90th percentile), so syllable dips don't sink it.
    step = np.array([-60.0] * 5 + [-30.0] * 5)
    assert nf.loud(step, 9) and not nf.loud(step, 11)
    assert not nf.loud(np.full(10, -35.0), 9)
    nf.clear()
    assert nf.db is None


def test_noise_floor_jumps_to_a_louder_place():
    nf = NoiseFloor()
    per_s = int(SAMPLE_RATE / WINDOW)
    assert not nf.add(np.full(per_s * 30, -60.0), 10)
    assert not nf.add(np.full(per_s, -45.0), 10)  # a second of chatter: not a new place
    shifted = [nf.add(np.full(8, -45.0), 10) for _ in range(per_s * 3 // 8)]
    assert any(shifted) and nf.db == pytest.approx(-45.0)  # quieter moments now 15 dB up
    # A conversation dips back to the floor every few seconds: no shift.
    nf.clear()
    nf.add(np.full(per_s * 10, -60.0), 10)
    talk = np.where(np.arange(per_s * 20) % per_s < per_s * 0.6, -30.0, -60.0)
    assert not any(nf.add(talk[i : i + 16], 10) for i in range(0, talk.size, 16))
    assert window_levels(np.zeros(WINDOW, np.float32))[0] == -120.0
    assert window_levels(np.full(WINDOW, 0.1, np.float32))[0] == pytest.approx(-20.0)


# --- RoomMonitor with an outside sensitivity ---------------------------------------------
# Deterministic, seeded speech-like signals (from the review simulations): syllabic envelopes
# (40/80/150 ms slots, amplitude 0.1-2.0) on a harmonic + noise carrier, random turns of
# 1-3 s with 0.3-1.5 s pauses, a brown-noise fan and N-talker café babble. The fake VAD is a
# level detector with a short hangover, like Silero's; for babble it calls everything audible
# speech (the worst case for the gate).
SR = SAMPLE_RATE
VOICES = [  # syllable slot, share voiced, amplitude range
    dict(slot_ms=40, voiced=0.6),
    dict(slot_ms=80, voiced=0.55, lo=0.1, hi=2.0),
    dict(slot_ms=150, voiced=0.7),
]


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Moving average over k samples (O(n), unlike np.convolve)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([out, np.full(k - 1, out[-1])])


def env_speech(n, rng, slot_ms=50, voiced=0.6, lo=0.3, hi=1.5) -> np.ndarray:
    slot = int(slot_ms * SR / 1000)
    k = n // slot + 1
    env = np.repeat((rng.random(k) < voiced) * rng.uniform(lo, hi, k), slot)[:n]
    env = smooth(env, 240)
    t = np.arange(n) / SR
    carrier = 0.3 * np.sign(np.sin(2 * np.pi * rng.uniform(100, 220) * t))
    return env * (carrier + 0.7 * rng.standard_normal(n, dtype=np.float32))


def quiet_room(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * SR)) * 0.001).astype(np.float32)  # ~-60 dB


@functools.lru_cache(maxsize=64)
def babble(seconds: float, talkers: int, seed: int = 0, level: float = 0.004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    voices = [env_speech(n, rng, 60, 0.55, 0.4, 1.6) * level for _ in range(talkers)]
    out = (sum(voices) + quiet_room(seconds, seed + 1000)).astype(np.float32)
    out.flags.writeable = False  # cached: shared between tests
    return out


def fan(seconds: float, seed: int = 0, level: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    b = np.cumsum(rng.standard_normal(n))
    b -= smooth(b, 400)  # brownish, no DC drift
    wobble = 1 + 0.1 * np.sin(2 * np.pi * 0.3 * np.arange(n) / SR)
    return (b / np.std(b) * level * wobble).astype(np.float32)


def conversation(seconds: float, bed: np.ndarray, over_db: float, seed: int = 1, voice=None):
    """Turn-taking voices (1-3 s turns, 0.3-1.5 s pauses) `over_db` above the `bed`."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    mask = np.zeros(n, bool)
    i = 0
    while i < n:
        on = int(rng.uniform(1.0, 3.0) * SR)
        mask[i : i + on] = True
        i += on + int(rng.uniform(0.3, 1.5) * SR)
    v = env_speech(n, rng, **(voice or VOICES[0])) * mask
    gain = 10 ** (over_db / 20) * np.sqrt(np.mean(bed[:n] ** 2)) / np.sqrt(np.mean(v**2))
    return (v * gain + bed[:n]).astype(np.float32)


def level_vad(thr_db: float = -45.0, hang: int = 4):
    def vad(audio):
        act = window_levels(audio) > thr_db
        out = act.copy()
        for k in range(1, hang + 1):
            out[k:] |= act[:-k]
        return out.astype(np.float32)

    return vad


class Source:
    def start(self, on_frames):
        self.cb = on_frames

    def stop(self):
        pass


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def monitor(outside: float, vad=None, calls: list | None = None, clock=None):
    """A listening monitor and feed(signal) -> whether it triggered after any VAD step."""
    det = DetectionConfig(room_listening=True, outside_sensitivity=outside)
    vad = vad or level_vad()

    def counted(audio):
        if calls is not None:
            calls.append(audio.size)
        return vad(audio)

    src = Source()
    mon = RoomMonitor(
        det, lambda: src, vad=counted, clock=clock or Clock(), thresholds=room_trigger
    )
    fired = [False]
    add_step = mon._add_step

    def watched(gen, n, speech):  # the worker thread: note a trigger after every step
        add_step(gen, n, speech)
        fired[0] |= mon.triggered()

    mon._add_step = watched  # type: ignore[method-assign]
    assert mon.start()

    def feed(signal) -> bool:
        fired[0] = False
        for j in range(0, len(signal), 1600):
            src.cb(signal[j : j + 1600])
        assert mon.wait_idle(30)
        return fired[0]

    return mon, feed


BABBLE_VAD = level_vad(-70.0)  # everything above the -60 dB room counts as speech


@pytest.mark.parametrize("outside", [0, 25, 50, 75])
@pytest.mark.parametrize("voice", range(len(VOICES)))
def test_a_conversation_fires_at_every_setting_quiet_and_in_a_cafe(outside, voice):
    over = 8 if outside >= 50 else 10  # the realistic edge: voices 8-10 dB over the café
    for bed, db in ((quiet_room(90, voice), 30), (babble(90, 4, seed=voice + 100), over)):
        mon, feed = monitor(outside)
        assert feed(conversation(90, bed, db, seed=voice + 11, voice=VOICES[voice]))
        mon.close()


@pytest.mark.parametrize("talkers", [2, 4, 8])
@pytest.mark.parametrize("outside", [25, 50])
def test_babble_never_triggers_across_seeds(outside, talkers):
    for seed in range(20):
        mon, feed = monitor(outside, BABBLE_VAD)
        assert not feed(babble(150, talkers, seed=seed)), f"seed {seed}"
        mon.close()


@pytest.mark.parametrize("outside", [25, 50])
def test_babble_and_a_fan_dont_retrigger_after_a_session(outside):
    calls: list = []
    mon, feed = monitor(outside, BABBLE_VAD, calls)
    assert not feed(babble(60, 4, seed=1))
    assert len(calls) < 30  # most babble steps sit at the floor and skip the VAD
    assert feed(conversation(40, babble(40, 4, seed=2), 12))  # a talk at the table
    tap = mon.session_source()  # ... recorded as a room session, which ends
    tap.start(lambda frames: None)
    tap.stop()
    assert not feed(babble(120, 4, seed=3))  # the café alone: no re-trigger
    assert not feed(fan(60, seed=4))
    mon.close()


@pytest.mark.parametrize("talkers", [2, 4, 8])
@pytest.mark.parametrize("outside", [25, 50, 75])
def test_walking_into_a_cafe_without_reopening_the_mic(outside, talkers):
    mon, feed = monitor(outside, BABBLE_VAD)
    assert not feed(quiet_room(60, seed=3))  # a quiet office, then the laptop is carried
    assert not feed(babble(120, talkers, seed=talkers + 5))  # ... into a café, mic still open
    mon.close()


@pytest.mark.parametrize("outside", [25, 50])
def test_a_sleep_gap_relearns_the_floor(outside):
    clock = Clock()
    mon, feed = monitor(outside, BABBLE_VAD, clock=clock)
    feed(quiet_room(30))
    clock.t += 30  # the lid was closed; the stream resumes somewhere louder
    assert not feed(babble(120, 4, seed=8))
    mon.close()


@pytest.mark.parametrize("outside", [25, 50])
def test_a_reopened_mic_relearns_the_floor(outside):
    mon, feed = monitor(outside, BABBLE_VAD)
    feed(quiet_room(30))
    mon.close_mic()
    assert mon.start()
    assert not feed(babble(60, 4, seed=9))
    mon.close()


def test_mic_opened_mid_conversation_and_long_talk_keep_counting():
    mon, feed = monitor(50)
    talk = conversation(150, quiet_room(150), 30, seed=5)
    assert feed(talk[: 20 * SR])  # opened while people talk: the pauses set the floor
    assert feed(talk[20 * SR :])  # two more minutes never become "the floor"
    assert mon.speech_in_window() > 6
    mon.close()


def test_lowest_setting_needs_more_than_a_few_remarks():
    mon, feed = monitor(0)  # 24 s of speech within 48 s
    sparse = conversation(90, quiet_room(90), 30)
    sparse[(np.arange(sparse.size) % (4 * SR)) >= 2 * SR] *= 0.0  # half of every 4 s silent
    assert not feed(sparse + quiet_room(90, 7))
    mon.close()


def test_live_transcription_replays_the_whole_widened_preroll(tmp_path):
    from recandcc.models import Track, TriggerKind
    from recandcc.recorder import Recorder

    cfg = Config(output_dir=tmp_path)
    cfg.detection.outside_sensitivity = 0  # window 48 s > room_preroll_s
    src = Source()
    mon = RoomMonitor(cfg.detection, lambda: src, vad=level_vad(), thresholds=room_trigger)
    system = Source()
    rec = Recorder(cfg, system_factory=lambda: system, min_session_s=0)
    try:
        assert mon.start()
        for _ in range(0, 40 * SR, 1600):
            src.cb(np.full(1600, 0.01, np.float32))
        rec.start(TriggerKind.ROOM, "Room", mic_source=mon.session_source)
        heard: list[int] = []
        rec.add_listener(lambda track, start, block: heard.append(block.size * (track == Track.ME)))
        rec.stop()
    finally:
        mon.close()
    assert sum(heard) == 40 * SR  # all of it, not just room_preroll_s + 5
