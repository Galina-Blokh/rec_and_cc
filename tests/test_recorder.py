"""Stories: US-01, US-02, US-06, US-07, US-08, US-13."""

import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.capture.base import AudioSource
from recandcc.capture.mic import StreamResampler
from recandcc.config import Config
from recandcc.models import SAMPLE_RATE, AppTarget, SessionInfo, Track, TriggerKind
from recandcc.recorder import Recorder, dbfs, make_private_dir, should_record_mic, slugify


class FakeSource(AudioSource):
    """Pushes pre-generated frames from a background thread, as fast as possible."""

    def __init__(self, signal: np.ndarray, block: int = 1600, fail: bool = False):
        self.signal = signal.astype(np.float32)
        self.block = block
        self.fail = fail
        self.started = threading.Event()
        self.done = threading.Event()
        self.stopped = False
        self._thread = None

    def start(self, on_frames):
        if self.fail:
            raise RuntimeError("no permission")

        def run():
            for i in range(0, len(self.signal), self.block):
                if self.stopped:
                    break
                on_frames(self.signal[i : i + self.block])
            self.done.set()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self.stopped = True
        if self._thread:
            self._thread.join()


def sine(seconds, freq=440.0, amp=0.5):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.fixture
def cfg(tmp_path):
    c = Config(output_dir=tmp_path / "out")
    return c


def make(cfg, sys_signal, mic_signal=None, **kw):
    sources = {}

    def sysf():
        sources["sys"] = FakeSource(sys_signal)
        return sources["sys"]

    def micf():
        sources["mic"] = FakeSource(mic_signal if mic_signal is not None else sine(1))
        return sources["mic"]

    return Recorder(cfg, mic_factory=micf, system_factory=sysf, **kw), sources


def test_records_both_tracks_for_call(cfg):
    sig = np.concatenate([sine(2), np.zeros(2 * SAMPLE_RATE)])
    rec, src = make(cfg, sig, sine(4, 300, 0.3))
    s = rec.start(TriggerKind.CALL, "zoom.us")
    assert rec.is_recording
    src["sys"].done.wait(5)
    src["mic"].done.wait(5)
    out = rec.stop()
    assert out is not None and out.ended_at is not None
    assert out.tracks == {Track.OTHERS: "others.wav", Track.ME: "me.wav"}
    d = out.directory
    assert d.parent.parent == cfg.output_dir
    assert d.name.endswith("_zoom-us")
    data, sr = sf.read(d / "others.wav")
    assert sr == SAMPLE_RATE
    assert sf.info(str(d / "others.wav")).subtype == "PCM_16"
    assert len(data) == len(sig)
    assert np.allclose(data[:1000], sig[:1000], atol=1e-3)
    loaded = SessionInfo.load(d)
    assert loaded.id == s.id and loaded.ended_at is not None
    assert rec.silent_tracks() == []
    assert not rec.is_recording


def test_media_skips_mic_by_policy(cfg):
    rec, src = make(cfg, sine(4))
    rec.start(TriggerKind.MEDIA, "Google Chrome")
    src["sys"].done.wait(5)
    out = rec.stop()
    assert "mic" not in src
    assert out.tracks == {Track.OTHERS: "others.wav"}
    assert not (out.directory / "me.wav").exists()


def test_record_mic_override(cfg):
    rec, src = make(cfg, sine(4), sine(4))
    rec.start(TriggerKind.CALL, "x", record_mic=False)
    src["sys"].done.wait(5)
    assert rec.stop().tracks == {Track.OTHERS: "others.wav"}


def test_discard_short_session(cfg):
    rec, src = make(cfg, sine(1))
    s = rec.start(TriggerKind.MANUAL)
    src["sys"].done.wait(5)
    assert rec.stop() is None
    assert not s.directory.exists()


def test_stop_when_idle_returns_none(cfg):
    rec, _ = make(cfg, sine(1))
    assert rec.stop() is None


def test_silence_tracking(cfg):
    clock = [100.0]
    sig = np.concatenate([sine(1), np.zeros(4 * SAMPLE_RATE)])
    rec, src = make(cfg, sig, clock=lambda: clock[0])
    rec.start(TriggerKind.MEDIA, "x")
    src["sys"].done.wait(5)
    time.sleep(0.05)
    clock[0] = 130.0
    # the last loud frame was observed at t=100 -> 30 s of silence
    assert rec.seconds_since_sound() == pytest.approx(30.0)
    assert rec.levels()[Track.OTHERS] < -100
    rec.stop()


def test_digital_silence_detected(cfg):
    rec, src = make(cfg, np.zeros(4 * SAMPLE_RATE))
    rec.start(TriggerKind.MEDIA, "x")
    src["sys"].done.wait(5)
    rec.stop()
    assert rec.silent_tracks() == [Track.OTHERS]


def test_failing_source_is_skipped(cfg):
    def bad():
        return FakeSource(sine(1), fail=True)

    good = FakeSource(sine(4))
    rec = Recorder(cfg, mic_factory=bad, system_factory=lambda: good)
    s = rec.start(TriggerKind.CALL, "zoom")
    assert s.tracks == {Track.OTHERS: "others.wav"}
    good.done.wait(5)
    assert rec.stop() is not None

    rec2 = Recorder(cfg, mic_factory=bad, system_factory=bad)
    with pytest.raises(RuntimeError):
        rec2.start(TriggerKind.CALL, "zoom")
    assert not rec2.is_recording


def test_unique_dirs_and_double_start(cfg):
    rec, src = make(cfg, sine(4))
    a = rec.start(TriggerKind.MEDIA, "x")
    with pytest.raises(RuntimeError):
        rec.start(TriggerKind.MEDIA, "x")
    src["sys"].done.wait(5)
    rec.stop()
    b = rec.start(TriggerKind.MEDIA, "x")
    src["sys"].done.wait(5)
    rec.stop()
    assert a.directory != b.directory and a.id != b.id


def test_helpers():
    assert slugify("zoom.us") == "zoom-us"
    assert slugify(None, "manual") == "manual"
    assert dbfs(np.zeros(10)) == -120.0
    assert dbfs(np.ones(10)) == pytest.approx(0.0)
    assert should_record_mic("calls", TriggerKind.CALL)
    assert should_record_mic("calls", TriggerKind.MANUAL)
    assert not should_record_mic("calls", TriggerKind.MEDIA)
    assert should_record_mic("always", TriggerKind.MEDIA)
    assert not should_record_mic("never", TriggerKind.CALL)


@pytest.mark.parametrize("src_rate", [48000, 44100, 16000, 8000])
def test_resampler_chunked(src_rate):
    t = np.arange(src_rate * 2) / src_rate
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    r = StreamResampler(src_rate)
    out = np.concatenate([r.process(x[i : i + 4410]) for i in range(0, len(x), 4410)])
    assert abs(len(out) - 2 * SAMPLE_RATE) <= 2
    # dominant frequency preserved
    spec = np.abs(np.fft.rfft(out[1000:17000]))
    freq = np.argmax(spec) * SAMPLE_RATE / 16000
    assert abs(freq - 440) < 5
    # high frequency above new Nyquist is attenuated when downsampling
    if src_rate > SAMPLE_RATE:
        hi = (0.5 * np.sin(2 * np.pi * 12000 * t)).astype(np.float32)
        r2 = StreamResampler(src_rate)
        o2 = np.concatenate([r2.process(hi[i : i + 4410]) for i in range(0, len(hi), 4410)])
        assert np.sqrt(np.mean(o2[2000:] ** 2)) < 0.05


def test_upgrade_media_to_call_adds_aligned_mic(cfg):
    rec, src = make(cfg, sine(3), sine(1, 300, 0.3))
    rec.start(TriggerKind.MEDIA, "Google Chrome")
    src["sys"].done.wait(5)
    assert "mic" not in src
    assert rec.upgrade_to_call("zoom.us") is True
    src["mic"].done.wait(5)
    out = rec.stop()
    assert out.trigger_kind == TriggerKind.CALL and out.trigger_app == "zoom.us"
    assert out.tracks == {Track.OTHERS: "others.wav", Track.ME: "me.wav"}
    me, _ = sf.read(out.directory / "me.wav")
    # Mic joined after 3 s of system audio: padded with 3 s of silence, then 1 s of mic.
    assert len(me) == 4 * SAMPLE_RATE
    assert np.all(me[: 3 * SAMPLE_RATE] == 0)
    assert np.abs(me[3 * SAMPLE_RATE :]).max() > 0.2
    assert SessionInfo.load(out.directory).trigger_kind == TriggerKind.CALL


def test_upgrade_is_noop_for_calls_and_idle(cfg):
    rec, src = make(cfg, sine(4), sine(4))
    assert rec.upgrade_to_call("zoom.us") is False
    rec.start(TriggerKind.CALL, "zoom.us")
    assert rec.upgrade_to_call("zoom.us") is False
    rec.stop()


def test_upgrade_respects_never_policy(cfg):
    cfg.detection.record_mic = "never"
    rec, src = make(cfg, sine(4))
    rec.start(TriggerKind.MEDIA, "Google Chrome")
    assert rec.upgrade_to_call("zoom.us") is True
    src["sys"].done.wait(5)
    assert rec.stop().tracks == {Track.OTHERS: "others.wav"}
    assert "mic" not in src


# --- frame tap (live transcription) ----------------------------------------------------
class Tap:
    def __init__(self):
        self.got: dict[Track, list[tuple[int, np.ndarray]]] = {}
        self.lock = threading.Lock()

    def __call__(self, track, start, frames):
        with self.lock:
            self.got.setdefault(track, []).append((start, frames))

    def audio(self, track) -> tuple[int, np.ndarray]:
        blocks = self.got[track]
        for (s0, b0), (s1, _) in zip(blocks, blocks[1:], strict=False):
            assert s1 == s0 + len(b0)  # contiguous, in order
        return blocks[0][0], np.concatenate([b for _, b in blocks])


def test_listener_gets_all_frames_with_positions(cfg):
    sig = sine(4)
    rec, src = make(cfg, sig)
    rec.start(TriggerKind.MEDIA, "Google Chrome")
    src["sys"].done.wait(5)
    tap = Tap()
    rec.add_listener(tap)  # added late: the recent audio is replayed
    out = rec.stop()
    start, audio = tap.audio(Track.OTHERS)
    assert start == 0 and np.allclose(audio, sig, atol=1e-6)
    wav, _ = sf.read(out.directory / "others.wav")
    assert len(wav) == len(audio)


def test_listener_mic_joining_mid_session_starts_at_padding(cfg):
    rec, src = make(cfg, sine(3), sine(1, 300, 0.3))
    tap = Tap()
    rec.start(TriggerKind.MEDIA, "Google Chrome")
    rec.add_listener(tap)
    src["sys"].done.wait(5)
    rec.upgrade_to_call("zoom.us")
    src["mic"].done.wait(5)
    rec.stop()
    start, audio = tap.audio(Track.ME)
    assert start == 3 * SAMPLE_RATE and len(audio) == SAMPLE_RATE
    assert tap.audio(Track.OTHERS)[0] == 0


def test_listeners_are_per_session_and_errors_contained(cfg):
    rec, src = make(cfg, sine(1))
    rec.start(TriggerKind.MANUAL)

    def bad(*_):
        raise RuntimeError("boom")

    rec.add_listener(bad)
    src["sys"].done.wait(5)
    tap = Tap()
    rec.add_listener(tap)
    rec.stop()  # still records fine
    assert tap.got
    rec.min_session_s = 0
    rec.start(TriggerKind.MANUAL)
    src["sys"].done.wait(5)
    rec.stop()
    assert sum(len(b) for _, b in tap.got[Track.OTHERS]) == SAMPLE_RATE  # nothing new


def test_replay_buffer_is_bounded(cfg):
    from recandcc import recorder as recmod

    rec, src = make(cfg, sine(15))
    rec.start(TriggerKind.MEDIA, "x")
    src["sys"].done.wait(5)
    tap = Tap()
    rec.add_listener(tap)
    rec.stop()
    start, audio = tap.audio(Track.OTHERS)
    assert recmod.REPLAY_SECONDS * SAMPLE_RATE <= len(audio) < 11 * SAMPLE_RATE
    assert start + len(audio) == 15 * SAMPLE_RATE


def _mode(p):
    return stat.S_IMODE(p.stat().st_mode)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_session_folders_are_owner_only(cfg):
    rec, src = make(cfg, sine(4))
    s = rec.start(TriggerKind.MANUAL)
    src["sys"].done.wait(5)
    rec.stop()
    d = s.directory
    assert d.parent.parent == cfg.output_dir
    for p in (cfg.output_dir, d.parent, d):
        assert _mode(p) == 0o700, p


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_make_private_dir_creates_parents_and_keeps_existing(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    make_private_dir(shared)
    assert _mode(shared) == 0o755  # an existing folder keeps its permissions
    deep = shared / "a" / "b"
    make_private_dir(deep)
    assert _mode(shared / "a") == 0o700 and _mode(deep) == 0o700


def test_make_private_dir_rejects_a_file_in_the_way(tmp_path):
    blocker = tmp_path / "out"
    blocker.write_text("not a folder")
    with pytest.raises(FileExistsError):
        make_private_dir(blocker)


def test_make_private_dir_stops_at_a_missing_root(monkeypatch, tmp_path):
    # A missing drive / UNC share: the root is its own parent and never becomes a directory.
    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    with pytest.raises(FileNotFoundError):
        make_private_dir(tmp_path / "out")


# --- per-app system audio (US-08) --------------------------------------------------------
ZOOM_TARGET = AppTarget(("us.zoom",), (10, 11))


class TargetSource(FakeSource):
    def __init__(self, target=None):
        super().__init__(sine(1))
        self.target, self.added = target, []

    def add_target(self, target):
        self.added.append(target)
        return True


def _target_recorder(cfg, fail_per_app=False):
    made = []

    def sysf(*args):
        if args and fail_per_app:
            raise RuntimeError("old helper")
        made.append(TargetSource(*args))
        return made[-1]

    return Recorder(cfg, mic_factory=lambda: FakeSource(sine(1)), system_factory=sysf), made


@pytest.mark.parametrize(
    ("scope", "target", "expected"),
    [
        ("app", ZOOM_TARGET, ZOOM_TARGET),
        ("all", ZOOM_TARGET, None),  # the default: everything the Mac plays
        ("app", None, None),  # manual session: no app to follow
        ("app", AppTarget(), None),  # no processes, no pattern
    ],
)
def test_system_source_gets_the_target_only_in_app_scope(cfg, scope, target, expected):
    cfg.detection.system_audio_scope = scope
    rec, made = _target_recorder(cfg)
    rec.start(TriggerKind.CALL, "zoom.us", target=target)
    rec.stop()
    assert [m.target for m in made] == [expected]


def test_per_app_source_failure_falls_back_to_all_system_audio(cfg):
    cfg.detection.system_audio_scope = "app"
    rec, made = _target_recorder(cfg, fail_per_app=True)
    s = rec.start(TriggerKind.CALL, "zoom.us", target=ZOOM_TARGET)
    assert [m.target for m in made] == [None] and Track.OTHERS in s.tracks
    rec.stop()


def test_upgrade_extends_a_per_app_tap_to_the_call_app(cfg):
    cfg.detection.system_audio_scope = "app"
    rec, made = _target_recorder(cfg)
    rec.start(TriggerKind.MEDIA, "Google Chrome", target=AppTarget(("com.google.chrome",)))
    assert rec.upgrade_to_call("zoom.us", target=ZOOM_TARGET) is True
    rec.stop()
    assert made[0].added == [ZOOM_TARGET]
