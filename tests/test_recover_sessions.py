"""Sessions an earlier run never stopped (crash, force-quit, power loss, a full disk at stop)
still reach the transcription queue; a recording running now is never touched.

Stories: US-10.
"""

import errno
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import soundfile as sf

from recandcc.capture.base import AudioSource
from recandcc.config import Config
from recandcc.models import RECORDING_LOCK, SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.recorder import Recorder
from recandcc.ui.common import (
    RECOVER_QUIET_S,
    Controller,
    recover_interrupted_sessions,
    untranscribed_sessions,
)

# Writes 3 s to a WAV and dies without closing it, like a crash mid-recording.
_CRASH = """
import os, sys
import numpy as np, soundfile as sf
f = sf.SoundFile(sys.argv[1], mode="w", samplerate=16000, channels=1, subtype="PCM_16",
                 format="WAV")
f.write(np.full(int(sys.argv[2]), 0.25, dtype=np.float32))
f.flush()
os._exit(0)
"""


def _abandoned(root, name, seconds=3.0, tracks=(Track.OTHERS,), age_s=3600.0):
    """A session folder left as a crash leaves it: session.json without ended_at, WAVs
    with ``seconds`` of audio (header never finalised), last written ``age_s`` ago."""
    d = root / "2026-10-27" / name
    d.mkdir(parents=True)
    started = datetime.now().astimezone() - timedelta(seconds=age_s + seconds)
    s = SessionInfo(id=name, directory=d, started_at=started, trigger_kind=TriggerKind.CALL)
    for t in tracks:
        wav = d / f"{t}.wav"
        frames = str(int(seconds * SAMPLE_RATE))
        subprocess.run([sys.executable, "-c", _CRASH, str(wav), frames], check=True)
        s.tracks[t] = wav.name
        old = time.time() - age_s
        os.utime(wav, (old, old))
    s.save()
    return d


def test_abandoned_session_is_recovered_and_requeued(tmp_path):
    d = _abandoned(tmp_path, "100000_zoom-us", seconds=3.0, tracks=(Track.OTHERS, Track.ME))
    assert untranscribed_sessions(tmp_path) == []  # the bug: skipped forever

    got = recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone())
    assert [s.id for s in got] == ["100000_zoom-us"]
    s = SessionInfo.load(d)
    assert s.ended_at is not None
    assert abs((s.ended_at - s.started_at).total_seconds() - 3.0) < 0.01
    assert s.metrics["recording"]["recovered"] is True
    assert s.tracks == {Track.OTHERS: "others.wav", Track.ME: "me.wav"}
    assert [x.id for x in untranscribed_sessions(tmp_path)] == ["100000_zoom-us"]
    # idempotent: a second start finds nothing to do
    assert recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone()) == []


def test_empty_wavs_are_left_alone(tmp_path):
    d = _abandoned(tmp_path, "110000_zoom-us", seconds=0.0, tracks=(Track.OTHERS, Track.ME))
    before = (d / SessionInfo.FILENAME).read_bytes()
    assert recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone()) == []
    assert (d / SessionInfo.FILENAME).read_bytes() == before
    assert (d / "me.wav").exists() and (d / "others.wav").exists()  # nothing is deleted
    assert untranscribed_sessions(tmp_path) == []


def test_an_unreadable_track_leaves_the_whole_session_for_the_next_start(tmp_path):
    d = _abandoned(tmp_path, "111000_zoom-us", seconds=2.0, tracks=(Track.OTHERS, Track.ME))
    good = (d / "me.wav").read_bytes()
    (d / "me.wav").write_bytes(b"held open by a sync client")  # sf.info can't read it now
    old = time.time() - 3600
    os.utime(d / "me.wav", (old, old))
    before = (d / SessionInfo.FILENAME).read_bytes()
    now = datetime.now().astimezone()
    assert recover_interrupted_sessions(tmp_path, started_before=now) == []
    assert (d / SessionInfo.FILENAME).read_bytes() == before  # Me is not dropped for good
    (d / "me.wav").write_bytes(good)  # readable again at the next start
    os.utime(d / "me.wav", (old, old))
    got = recover_interrupted_sessions(tmp_path, started_before=now)
    assert [s.id for s in got] == ["111000_zoom-us"]
    assert SessionInfo.load(d).tracks == {Track.OTHERS: "others.wav", Track.ME: "me.wav"}


def test_a_missing_track_also_leaves_the_session_alone(tmp_path):
    d = _abandoned(tmp_path, "112000_zoom-us", seconds=2.0, tracks=(Track.OTHERS, Track.ME))
    (d / "me.wav").unlink()  # e.g. evicted by iCloud
    before = (d / SessionInfo.FILENAME).read_bytes()
    assert recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone()) == []
    assert (d / SessionInfo.FILENAME).read_bytes() == before


def test_tracks_written_within_the_quiet_margin_are_left_alone(tmp_path):
    d = _abandoned(tmp_path, "113000_zoom-us", seconds=2.0)
    cutoff = datetime.now().astimezone()
    recent = cutoff.timestamp() - RECOVER_QUIET_S / 2  # another writer may still be at it
    os.utime(d / "others.wav", (recent, recent))
    assert recover_interrupted_sessions(tmp_path, started_before=cutoff) == []
    quiet = cutoff.timestamp() - RECOVER_QUIET_S - 5
    os.utime(d / "others.wav", (quiet, quiet))
    assert len(recover_interrupted_sessions(tmp_path, started_before=cutoff)) == 1


def test_a_live_cli_recording_lock_is_respected_and_a_stale_one_removed(tmp_path):
    held = _abandoned(tmp_path, "114000_manual")
    (held / RECORDING_LOCK).write_text(str(os.getpid()), encoding="utf-8")  # alive
    stale = _abandoned(tmp_path, "114001_manual")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (stale / RECORDING_LOCK).write_text(str(dead.pid), encoding="utf-8")
    got = recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone())
    assert [s.id for s in got] == ["114001_manual"]
    assert SessionInfo.load(held).ended_at is None and (held / RECORDING_LOCK).exists()
    assert not (stale / RECORDING_LOCK).exists()


def test_a_lock_from_before_the_last_boot_is_stale_even_if_its_pid_is_taken(tmp_path, monkeypatch):
    import recandcc.models as models

    d = _abandoned(tmp_path, "114002_manual")
    lock = d / RECORDING_LOCK
    lock.write_text(str(os.getpid()), encoding="utf-8")  # the pid now belongs to us
    old = time.time() - 7200
    os.utime(lock, (old, old))
    monkeypatch.setattr(models, "boot_time", lambda: time.time() - 3600)  # rebooted since
    got = recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone())
    assert [s.id for s in got] == ["114002_manual"] and not lock.exists()


def test_boot_time_is_in_the_past():
    from recandcc.models import boot_time

    booted = boot_time()
    assert booted is None or 0 < booted <= time.time()
    if sys.platform in ("darwin", "win32"):
        assert booted is not None


def test_cli_record_holds_the_lock_while_recording(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from recandcc import cli

    seen = []
    d = tmp_path / "2026-10-27" / "150000_manual"

    class Rec:
        def __init__(self, cfg):
            pass

        def start(self, kind, app, record_mic=None):
            d.mkdir(parents=True)
            return SessionInfo(id="x", directory=d, started_at=datetime.now(), trigger_kind=kind)

        def levels(self):
            seen.append((d / RECORDING_LOCK).read_text(encoding="utf-8"))
            return {}

        def stop(self):
            return None  # too short: discarded

        def silent_tracks(self):
            return []

    monkeypatch.setattr(cli, "setup_logging", lambda **k: None)
    monkeypatch.setattr(cli, "load_config", lambda: Config(output_dir=tmp_path))
    monkeypatch.setattr("recandcc.recorder.Recorder", Rec)
    monkeypatch.setattr(cli.signal, "signal", lambda *a: None)
    args = SimpleNamespace(verbose=False, no_mic=True, no_transcribe=True, seconds=0.3)
    assert cli.cmd_record(args) == 1
    assert seen and seen[0] == str(os.getpid())
    assert not (d / RECORDING_LOCK).exists()


def test_busy_sessions_are_skipped_even_if_they_become_busy_mid_scan(tmp_path):
    a = _abandoned(tmp_path, "116000_a")
    b = _abandoned(tmp_path, "116001_b")
    calls = []

    def busy():
        calls.append(1)
        # a: busy under another spelling of its path; b: turns busy between scan and write
        spelled = tmp_path / "2026-10-27" / ".." / "2026-10-27" / "116000_a"
        return [spelled] + ([str(b)] if len(calls) > 2 else [])

    now = datetime.now().astimezone()
    assert recover_interrupted_sessions(tmp_path, started_before=now, busy=busy) == []
    assert SessionInfo.load(a).ended_at is None and SessionInfo.load(b).ended_at is None


def test_an_empty_track_is_dropped_but_kept_on_disk(tmp_path):
    d = _abandoned(tmp_path, "120000_zoom-us", seconds=2.0)
    sf.SoundFile(str(d / "me.wav"), mode="w", samplerate=SAMPLE_RATE, channels=1).close()
    s = SessionInfo.load(d)
    s.tracks[Track.ME] = "me.wav"
    s.save()
    old = time.time() - 3600
    os.utime(d / "me.wav", (old, old))
    recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone())
    assert SessionInfo.load(d).tracks == {Track.OTHERS: "others.wav"}
    assert (d / "me.wav").exists()


def test_discarded_and_skipped_and_recent_sessions_are_not_touched(tmp_path):
    discarded = _abandoned(tmp_path, "130000_a")
    (discarded / SessionInfo.DISCARD_MARKER).touch()
    skipped = _abandoned(tmp_path, "130001_b")
    # began before the app started (e.g. a room pre-roll) but its WAV is written right now
    writing = _abandoned(tmp_path, "130002_c", age_s=0.0)
    cutoff = datetime.now().astimezone() - timedelta(seconds=60)
    os.utime(writing / "others.wav")  # written after the cutoff
    s = SessionInfo.load(writing)
    s.started_at = cutoff - timedelta(seconds=10)
    s.save()
    snap = {d: (d / SessionInfo.FILENAME).read_bytes() for d in (discarded, skipped, writing)}
    got = recover_interrupted_sessions(tmp_path, started_before=cutoff, busy=lambda: [skipped])
    assert got == []
    assert {d: (d / SessionInfo.FILENAME).read_bytes() for d in snap} == snap


class _Source(AudioSource):
    """Delivers audio until stopped, like a live capture."""

    def __init__(self):
        self._stop = threading.Event()
        self.flowing = threading.Event()

    def start(self, on_frames):
        def run():
            while not self._stop.is_set():
                on_frames(np.full(1600, 0.2, dtype=np.float32))
                self.flowing.set()
                time.sleep(0.01)

        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join()


def _recorder(cfg, **kw):
    src = _Source()
    return Recorder(cfg, system_factory=lambda: src, mic_factory=_Source, **kw), src


def test_stop_keeps_ended_at_when_the_metrics_write_hits_a_full_disk(tmp_path, monkeypatch):
    rec, src = _recorder(Config(output_dir=tmp_path), min_session_s=0.0)
    rec.start(TriggerKind.MEDIA, "x")
    assert src.flowing.wait(5)

    def full(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(SessionInfo, "update_metrics", full)
    out = rec.stop()
    assert out is not None and out.ended_at is not None  # the engine still submits it
    assert SessionInfo.load(out.directory).ended_at is not None
    assert [s.id for s in untranscribed_sessions(tmp_path)] == [out.id]


def test_stop_returns_the_session_when_both_saves_fail(tmp_path, monkeypatch):
    rec, src = _recorder(Config(output_dir=tmp_path), min_session_s=0.0)
    rec.start(TriggerKind.MEDIA, "x")
    assert src.flowing.wait(5)

    def full(*_a, **_k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(SessionInfo, "save", full)
    out = rec.stop()
    assert out is not None and out.ended_at is not None  # submitted from memory
    assert not rec.is_recording


class _Monitor:
    def snapshot(self):
        return []


class _Q:
    def __init__(self):
        self.submitted = []

    def submit(self, session, **_k):
        self.submitted.append(session.id)

    def pending(self):
        return 0


def test_startup_recovery_requeues_the_crashed_session_but_not_the_live_one(tmp_path):
    crashed = _abandoned(tmp_path, "090000_zoom-us")
    cfg = Config(output_dir=tmp_path)
    cfg.transcription.keep_audio = True
    rec, src = _recorder(cfg)
    q = _Q()
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=rec, tq=q, start_threads=False)
    live = rec.start(TriggerKind.MEDIA, "x")
    try:
        assert src.flowing.wait(5)
        ctrl._resubmit_unfinished()
        assert q.submitted == ["090000_zoom-us"]
        assert SessionInfo.load(crashed).ended_at is not None
        assert SessionInfo.load(live.directory).ended_at is None
        assert "recovered" not in SessionInfo.load(live.directory).metrics.get("recording", {})
    finally:
        rec.stop()


def test_controller_skips_the_engine_session_and_in_flight_folders(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from recandcc.engine import Engine

    live = _abandoned(tmp_path, "080000_live")
    flying = _abandoned(tmp_path, "080001_flying")
    crashed = _abandoned(tmp_path, "080002_crashed")
    cfg = Config(output_dir=tmp_path)
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Q(), start_threads=False)
    state = property(lambda self: SimpleNamespace(session_dir=str(live)))
    monkeypatch.setattr(Engine, "state", state)
    ctrl._in_flight.add(flying)
    ctrl._recover_interrupted()
    assert SessionInfo.load(live).ended_at is None
    assert SessionInfo.load(flying).ended_at is None
    assert SessionInfo.load(crashed).ended_at is not None
