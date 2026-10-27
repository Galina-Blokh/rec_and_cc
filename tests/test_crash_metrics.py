"""Crash reporting (excepthooks -> log + notice event) and per-session pipeline metrics."""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import timedelta

import numpy as np
import pytest
from test_controller_live import make_ctrl
from test_live import FakeTranscriber, Feeder, burst, make_session, silence
from test_recorder import cfg as cfg  # noqa: F401 - fixture
from test_recorder import make, sine
from test_transcribe import _run_fake

from recandcc import logging_setup
from recandcc.config import TranscriptionConfig
from recandcc.metrics import latency_stats, summary_line
from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.transcribe.live import LiveTranscriber


@pytest.fixture
def hooks(monkeypatch):
    """Install the crash handlers for one test (without faulthandler) and undo it after."""
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(logging_setup, "_installed", False)
    monkeypatch.setattr(logging_setup, "_notifier", None)
    monkeypatch.setattr(logging_setup, "_last_notice", {})
    logging_setup.install_crash_handlers(faults_file=None)
    return logging_setup


def _crash(name: str) -> None:
    def boom():
        raise ValueError("kaboom")

    t = threading.Thread(target=boom, name=name)
    t.start()
    t.join()


def _notices(ctrl) -> list[tuple]:
    out = []
    while not ctrl.events.empty():
        ev = ctrl.events.get_nowait()
        if ev[0] == "notice":
            out.append(ev)
    return out


def test_thread_crash_is_logged_and_noticed_once_per_minute(tmp_path, hooks, caplog):
    ctrl, _, _ = make_ctrl(tmp_path)
    hooks.set_crash_notifier(ctrl.post_notice)
    with caplog.at_level(logging.ERROR, logger="recandcc.crash"):
        _crash("worker-a")
        _crash("worker-a")  # same name within a minute: logged, no second toast
        _crash("worker-b")
    recs = [r for r in caplog.records if "uncaught exception in thread" in r.getMessage()]
    assert len(recs) == 3 and recs[0].exc_info[0] is ValueError
    assert "kaboom" in caplog.text and "Traceback" in caplog.text
    notices = _notices(ctrl)
    assert [n[2].split(" ")[0] for n in notices] == ["worker-a", "worker-b"]
    assert notices[0][1] == "Internal error" and "ValueError" in notices[0][2]
    # after the interval the same thread name is reported again
    now = logging_setup._last_notice["worker-a"] + hooks.NOTICE_INTERVAL_S + 1
    assert hooks.notify_thread_crash("worker-a", "ValueError", now=now)
    ctrl.shutdown()


def test_main_thread_exception_logged(hooks, caplog):
    try:
        raise RuntimeError("main boom")
    except RuntimeError:
        exc = sys.exc_info()
    with caplog.at_level(logging.CRITICAL, logger="recandcc.crash"):
        sys.excepthook(*exc)
    assert "uncaught exception" in caplog.text and "main boom" in caplog.text


def test_faulthandler_enabled_into_log_dir(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(logging_setup.faulthandler, "enable", lambda **kw: calls.append(kw))
    monkeypatch.setattr(logging_setup, "_faults_file", None)
    monkeypatch.setattr(logging_setup, "_installed", True)  # hooks already set up
    logging_setup.install_crash_handlers(tmp_path / "logs" / "faults.log")
    assert calls and calls[0]["all_threads"] is True
    assert calls[0]["file"].name == str(tmp_path / "logs" / "faults.log")
    logging_setup._faults_file.close()


# --------------------------------------------------------------------------- metrics
def test_metrics_round_trip_and_old_files_load(tmp_path):
    s = make_session(tmp_path)
    s.update_metrics("live", {"latency_median_s": 1.2})
    s.update_metrics("live", {"phrases": 3})
    s.save()
    assert SessionInfo.load(s.directory).metrics == {
        "live": {"latency_median_s": 1.2, "phrases": 3}
    }
    f = s.directory / SessionInfo.FILENAME
    data = json.loads(f.read_text(encoding="utf-8"))
    del data["metrics"]  # a session.json written before metrics existed
    f.write_text(json.dumps(data), encoding="utf-8")
    old = SessionInfo.load(s.directory)
    assert old.metrics == {} and old.trigger_kind == TriggerKind.CALL


def test_helpers():
    assert latency_stats([]) == {}
    assert latency_stats([1.0, 3.0, 2.0]) == {
        "latency_median_s": 2.0,
        "latency_max_s": 3.0,
        "phrases": 3,
    }
    assert summary_line({"a": {"b": 1, "c": {"d": 2}}}) == "a.b=1 a.c.d=2"


def test_recorder_saves_recording_metrics(cfg):  # noqa: F811
    rec, src = make(cfg, sine(4))
    rec.start(TriggerKind.CALL, "zoom.us")
    src["sys"].done.wait(5)
    src["mic"].done.wait(5)
    out = rec.stop()
    m = SessionInfo.load(out.directory).metrics["recording"]
    assert m["audio_s"] == 4.0
    assert set(m["writer_backlog_max_s"]) == {"others", "me"}
    assert all(v >= 0 for v in m["writer_backlog_max_s"].values())


def test_live_finish_saves_latency_metrics(tmp_path, caplog):
    tr = FakeTranscriber(TranscriptionConfig(model="fake", workers=2))
    session = make_session(tmp_path)
    live = LiveTranscriber(tr, session, step_s=0.02)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(1), silence(1.5), burst(1)]))
    session.ended_at = session.started_at + timedelta(seconds=5)
    with caplog.at_level(logging.INFO, logger="recandcc.metrics"):
        live.finish(session, timeout=30)
    m = SessionInfo.load(session.directory).metrics["live"]
    assert m["phrases"] == 2 and 0 <= m["latency_median_s"] <= m["latency_max_s"]
    assert m["failed_regions"] == 0 and m["final_after_stop_s"] >= 0
    assert "input_lag_max_s" in m
    assert "live.latency_median_s=" in caplog.text


def test_batch_transcription_saves_rtf(tmp_path, monkeypatch):
    s, _segs, _model, _ = _run_fake(tmp_path, monkeypatch, lid=None)
    m = SessionInfo.load(s.directory).metrics["transcription"]
    assert m["chunks"] > 0 and m["rtf"] >= 0 and m["audio_s"] > 0


def test_setup_logging_adds_file_and_stderr_handlers_once(monkeypatch, tmp_path):
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "level", root.level)
    monkeypatch.setattr(logging_setup, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(logging_setup, "LOG_FILE", tmp_path / "logs" / "recandcc.log")
    installed = []
    monkeypatch.setattr(logging_setup, "install_crash_handlers", installed.append)
    try:
        logging_setup.setup_logging(logging.DEBUG)
        logging_setup.setup_logging(logging.DEBUG)  # second call is a no-op
        kinds = sorted(type(h).__name__ for h in root.handlers)
        assert kinds == ["RotatingFileHandler", "StreamHandler"]
        assert root.level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.WARNING
        assert installed == [logging_setup.FAULTS_FILE]
        logging.getLogger("recandcc.test").info("hello")
        for h in root.handlers:
            h.flush()
        assert "recandcc.test: hello" in (tmp_path / "logs" / "recandcc.log").read_text()
    finally:
        for h in root.handlers:
            h.close()
        if hasattr(root, "_recandcc_configured"):
            del root._recandcc_configured


def test_faulthandler_goes_to_the_faults_file(monkeypatch, tmp_path):
    enabled = []
    monkeypatch.setattr(logging_setup, "_faults_file", None)
    monkeypatch.setattr(logging_setup, "_installed", True)  # keep the real excepthooks
    monkeypatch.setattr(
        logging_setup.faulthandler, "enable", lambda file, all_threads: enabled.append(file)
    )
    faults = tmp_path / "logs" / "faults.log"
    logging_setup.install_crash_handlers(faults)
    try:
        assert enabled and enabled[0].name == str(faults)
        assert faults.exists()
    finally:
        logging_setup._faults_file.close()
