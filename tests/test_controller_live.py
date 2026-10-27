"""Controller wiring of live transcription: real Engine + Recorder + queue, fake model/audio.

Stories: US-11.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import pytest
from test_live import FakeTranscriber, burst, phrase, silence
from test_recorder import FakeSource

from recandcc.config import Config, TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Mode
from recandcc.recorder import Recorder
from recandcc.ui.common import Controller


class Monitor:
    def snapshot(self):
        return []

    def close(self):
        pass


def make_ctrl(tmp_path, live=True, sync=True):
    cfg = Config(output_dir=tmp_path / "out")
    cfg.detection.mode = Mode.OFF
    cfg.transcription = TranscriptionConfig(model="fake", workers=2, live=live)
    sources = {}
    audio = np.concatenate([silence(1), burst(1.5), silence(1.5), burst(1)])  # ends mid-phrase

    def sysf():
        sources["sys"] = FakeSource(audio)
        return sources["sys"]

    def micf():
        sources["mic"] = FakeSource(silence(len(audio) / SAMPLE_RATE))
        return sources["mic"]

    rec = Recorder(cfg, mic_factory=micf, system_factory=sysf)
    tr = FakeTranscriber(cfg.transcription, sync=sync)
    ctrl = Controller(cfg, monitor=Monitor(), recorder=rec, transcriber=tr, start_threads=False)
    ctrl.engine._save_config = lambda: None
    return ctrl, sources, tr


def events_until(ctrl, kind, timeout=10.0):
    out, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = ctrl.events.get(timeout=0.05)
        except queue.Empty:
            continue
        out.append(ev)
        if ev[0] == kind:
            return out
    raise AssertionError(f"no {kind!r} event; got {[e[0] for e in out]}")


def record(ctrl, sources):
    ctrl.engine.start_manual()
    ctrl.engine.tick()
    sources["sys"].done.wait(5)
    sources["mic"].done.wait(5)
    time.sleep(0.6)  # let the live worker scan the audio while "recording"


def test_live_session_events_and_finalize(tmp_path):
    ctrl, sources, tr = make_ctrl(tmp_path)
    tr.transcribe_session = lambda *_a, **_k: pytest.fail("batch transcription must not run")
    record(ctrl, sources)
    evs = events_until(ctrl, "live")
    sid = ctrl.engine.recorder.session.id
    start = next(e for e in evs if e[0] == "live_start")
    assert start == ("live_start", sid, "Manual recording")
    live_segs = [s for e in evs if e[0] == "live" for s in e[2]]
    assert [s.text for s in live_segs] == [phrase(1)]  # before the session stopped
    session_dir = ctrl.engine.recorder.session.directory

    ctrl.engine.stop()
    ctrl.engine.tick()
    evs = events_until(ctrl, "live_end")
    # the tail (no trailing silence) is flushed and shown before live_end
    assert [s.text for e in evs if e[0] == "live" for s in e[2]] == [phrase(4)]
    assert evs[-1] == ("live_end", sid)
    md = (session_dir / "transcript.md").read_text(encoding="utf-8")  # final, already written
    assert phrase(1) in md and phrase(4) in md and "in progress" not in md
    evs = events_until(ctrl, "transcribed")
    _, session, result = evs[-1]
    assert session.id == sid and session.transcribed and len(result) == 2
    assert ctrl.tq.pending() == 0
    ctrl.shutdown()


def test_discard_cancels_live(tmp_path):
    ctrl, sources, tr = make_ctrl(tmp_path)
    record(ctrl, sources)
    session_dir = ctrl.engine.recorder.session.directory
    sid = ctrl.engine.recorder.session.id
    ctrl.engine._do_discard()
    evs = events_until(ctrl, "live_end")
    assert evs[-1] == ("live_end", sid)
    assert not session_dir.exists()
    time.sleep(0.5)
    assert not session_dir.exists()  # nothing written afterwards
    assert ctrl.tq.pending() == 0
    ctrl.shutdown()


def test_live_off_uses_batch(tmp_path):
    ctrl, sources, tr = make_ctrl(tmp_path, live=False)
    done = []
    tr.transcribe_session = lambda s, *a, **k: done.append(s) or []
    record(ctrl, sources)
    ctrl.engine.stop()
    ctrl.engine.tick()
    evs = events_until(ctrl, "transcribed")
    assert not [e for e in evs if e[0].startswith("live")]
    assert len(done) == 1
    ctrl.shutdown()


def test_live_failure_falls_back_to_batch(tmp_path):
    ctrl, sources, tr = make_ctrl(tmp_path)
    tr.fail_at = set(range(100))  # every live region fails
    done = []
    tr.transcribe_session = lambda s, *a, **k: done.append(s) or []
    record(ctrl, sources)
    ctrl.engine.stop()
    ctrl.engine.tick()
    events_until(ctrl, "transcribed")
    assert len(done) == 1
    ctrl.shutdown()


def test_cli_record_prints_live_lines(tmp_path, monkeypatch, capsys):
    import recandcc.cli as cli
    import recandcc.recorder as recmod
    import recandcc.transcribe as tmod

    cfg = Config(output_dir=tmp_path / "out")
    cfg.transcription = TranscriptionConfig(model="fake", workers=2)
    audio = np.concatenate([silence(1), burst(1.5), silence(1.5), burst(1)])
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "setup_logging", lambda **_k: None)

    class Rec(Recorder):
        def __init__(self, cfg):
            super().__init__(
                cfg,
                mic_factory=lambda: FakeSource(audio * 0),
                system_factory=lambda: FakeSource(audio),
            )

    monkeypatch.setattr(recmod, "Recorder", Rec)
    monkeypatch.setattr(tmod, "Transcriber", lambda c: FakeTranscriber(c))
    args = cli.build_parser().parse_args(["record", "--seconds", "1.0"])
    assert cli.cmd_record(args) == 0
    out = capsys.readouterr().out
    assert f"[00:00:00] Others: {phrase(1)}" in out
    assert f"[00:00:03] Others: {phrase(4)}" in out  # the tail, flushed at stop
    assert "Transcript: " in out and "transcript.md" in out
    (md,) = (tmp_path / "out").glob("*/*/transcript.md")
    assert phrase(4) in md.read_text(encoding="utf-8")
