"""MacSystemAudioSource builds the helper's argv and stdin commands (fake Popen; any OS).

Stories: US-08.
"""

from __future__ import annotations

import io
import subprocess
import sys

import pytest

from recandcc.capture import system_mac
from recandcc.capture.system_mac import MacSystemAudioSource
from recandcc.models import AppTarget


class FakeProc:
    def __init__(self, cmd, stdin, **_kw):
        self.cmd, self.stdin_mode = cmd, stdin
        self.stdin = io.BytesIO() if stdin == subprocess.PIPE else None
        self.stdout, self.stderr = io.BytesIO(), io.BytesIO()
        self.sent = b""

    def poll(self):
        return 0

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

    monkeypatch.setattr(system_mac, "find_helper", lambda: "rcc-audiotap")
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
