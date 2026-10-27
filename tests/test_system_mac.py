"""MacSystemAudioSource builds the helper's argv and stdin commands (fake Popen; any OS).

Stories: US-08.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import pytest

from recandcc.capture import system_mac
from recandcc.capture.system_mac import MacSystemAudioSource
from recandcc.models import AppTarget


class FakeProc:
    stderr_text = b""
    stdout_bytes = b""
    running = False  # poll(): exited (0) unless a test keeps the helper running

    def __init__(self, cmd, stdin, **_kw):
        self.cmd, self.stdin_mode = cmd, stdin
        self.stdin = io.BytesIO() if stdin == subprocess.PIPE else None
        self.stdout, self.stderr = io.BytesIO(self.stdout_bytes), io.BytesIO(self.stderr_text)
        self.sent = b""

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def procs(monkeypatch):
    made: list[FakeProc] = []

    def popen(cmd, **kw):
        made.append(FakeProc(cmd, **kw))
        stdin = made[-1].stdin
        if stdin is not None:  # keep what was written after stop() closes the pipe
            stdin.close = lambda: setattr(made[-1], "sent", stdin.getvalue())
        return made[-1]

    monkeypatch.setattr(system_mac, "find_helper", lambda: Path(__file__))
    monkeypatch.setattr(system_mac, "_supports_targets", lambda helper, mtime: True)
    monkeypatch.setattr(system_mac.subprocess, "Popen", popen)
    return made


def test_global_tap_by_default(procs):
    source = MacSystemAudioSource()
    source.start(lambda frames: None)
    assert "--pid" not in procs[0].cmd and "--match" not in procs[0].cmd
    assert procs[0].stdin_mode == subprocess.DEVNULL
    assert source.add_target(AppTarget(("us.zoom",))) is False  # global: nothing to add
    source.stop()


def test_target_becomes_pid_and_match_arguments(procs):
    source = MacSystemAudioSource(target=AppTarget(("com.google.chrome", " "), (11, 21)))
    source.start(lambda frames: None)
    cmd = procs[0].cmd
    assert cmd[cmd.index("--pid") :] == [
        "--pid",
        "11",
        "--pid",
        "21",
        "--match",
        "com.google.chrome",  # blank patterns are dropped
    ]
    assert procs[0].stdin_mode == subprocess.PIPE
    assert source.add_target(AppTarget(("us.zoom\nx",), (10,))) is True
    source.stop()
    assert procs[0].sent == b"pid 10\nmatch us.zoom x\n"  # one command per line
    assert source.add_target(AppTarget(pids=(1,))) is False  # stopped


def test_factory_passes_the_target_on_macos(procs, monkeypatch):
    from recandcc import capture

    monkeypatch.setattr(sys, "platform", "darwin")
    source = capture.create_system_source(AppTarget(pids=(5,)))
    source.start(lambda frames: None)
    assert "--pid" in procs[0].cmd
    source.stop()


def test_outdated_helper_records_all_system_audio(procs, monkeypatch, caplog):
    """An old bin/rcc-audiotap exits on --pid: probe first, and use the global tap instead."""
    monkeypatch.setattr(system_mac, "_supports_targets", lambda helper, mtime: False)
    source = MacSystemAudioSource(target=AppTarget(("us.zoom",), (10,)))
    source.start(lambda frames: None)
    assert "--pid" not in procs[0].cmd and "--match" not in procs[0].cmd
    assert source.scope == "all" and source.add_target(AppTarget(pids=(1,))) is False
    assert "predates per-app capture" in caplog.text
    source.stop()


def test_helper_fallback_to_the_global_tap_is_noticed(procs, monkeypatch, caplog):
    monkeypatch.setattr(FakeProc, "stderr_text", b"rcc-audiotap: scope: all\n")
    source = MacSystemAudioSource(target=AppTarget(("us.zoom",)))
    assert source.scope == "app"
    source.start(lambda frames: None)
    source.stop()  # joins the stderr reader
    assert source.scope == "all"
    assert any(r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell script as a fake helper")
def test_support_probe_reads_the_helpers_usage(tmp_path):
    for name, usage in (("new", "capture [--pid PID ...] [--match TEXT ...]"), ("old", "capture")):
        helper = tmp_path / name
        helper.write_text(f"#!/bin/sh\necho '{usage}'\n", encoding="utf-8")
        helper.chmod(0o755)
        assert system_mac._supports_targets(helper, 0.0) is (name == "new")


def test_a_failed_support_probe_is_not_remembered(tmp_path, monkeypatch):
    helper = tmp_path / "helper"
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        if len(calls) == 1:
            raise system_mac.subprocess.TimeoutExpired(argv, 5)
        return type("R", (), {"stdout": "capture [--pid PID ...] [--match TEXT ...]"})()

    monkeypatch.setattr(system_mac.subprocess, "run", run)
    assert system_mac._supports_targets(helper, 1.0) is False  # busy machine: no answer
    assert system_mac._supports_targets(helper, 1.0) is True  # asked again, now remembered
    assert system_mac._supports_targets(helper, 1.0) is True
    assert len(calls) == 2


APP, AUDIO = b"rcc-audiotap: scope: app\n", b"\0" * system_mac.BLOCK_BYTES


@pytest.mark.parametrize(
    ("stderr", "audio", "running", "expected"),
    [
        (APP, AUDIO, True, "app"),
        (b"rcc-audiotap: scope: all\n", AUDIO, True, "all"),
        (b"", AUDIO, True, "pending"),  # no report
        (APP, b"", True, "pending"),  # reported, but the device never started
        (APP, AUDIO, False, "pending"),  # setup failed after the report: the helper exited
    ],
)
def test_confirmed_scope_waits_for_the_helpers_report(
    procs, monkeypatch, stderr, audio, running, expected
):
    monkeypatch.setattr(FakeProc, "stderr_text", stderr)
    monkeypatch.setattr(FakeProc, "stdout_bytes", audio)
    monkeypatch.setattr(FakeProc, "running", running)
    source = MacSystemAudioSource(target=AppTarget(("us.zoom",)))
    source.start(lambda frames: None)
    assert source.confirmed_scope(0.5) == expected
    monkeypatch.setattr(FakeProc, "running", False)  # stop(): no SIGTERM to a fake
    source.stop()
    assert MacSystemAudioSource().confirmed_scope(0) == "all"  # a global tap: known at once
