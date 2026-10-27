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
from recandcc.models import SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.recorder import Recorder
from recandcc.ui.common import Controller, recover_interrupted_sessions, untranscribed_sessions

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
    (d / "others.wav").unlink()  # one track gone entirely, the other never got a frame
    before = (d / SessionInfo.FILENAME).read_bytes()
    assert recover_interrupted_sessions(tmp_path, started_before=datetime.now().astimezone()) == []
    assert (d / SessionInfo.FILENAME).read_bytes() == before
    assert (d / "me.wav").exists()  # nothing is deleted
    assert untranscribed_sessions(tmp_path) == []


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
    assert recover_interrupted_sessions(tmp_path, started_before=cutoff, skip=[skipped]) == []
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
