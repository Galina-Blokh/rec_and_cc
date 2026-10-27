"""permissions: helper JSON parsing and per-platform fallbacks, with the helper faked.

Stories: US-16, US-18.
"""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from recandcc import permissions as perms


@pytest.fixture
def helper(monkeypatch):
    """Fake `rcc-audiotap`: records argv, answers with the queued stdout."""
    calls, replies, kwargs = [], [], []

    def run(argv, **kw):
        calls.append(argv[1:])
        kwargs.append(kw)
        out = replies.pop(0) if replies else ""
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(stdout=out, stderr="note", returncode=0)

    monkeypatch.setattr("recandcc.native.find_helper", lambda: "/fake/rcc-audiotap")
    monkeypatch.setattr(perms.subprocess, "run", run)
    monkeypatch.setattr(sys, "platform", "darwin")
    return SimpleNamespace(calls=calls, replies=replies, kwargs=kwargs)


def test_status_reads_the_last_json_line(helper):
    helper.replies.append('log line\n\n{"microphone": "granted"}\n')
    assert perms.status() == {"microphone": "granted", "system_audio": "unknown"}
    assert helper.calls == [["permissions"]]
    # A wedged helper must not block the permission worker forever.
    assert helper.kwargs[0]["timeout"] == perms.STATUS_TIMEOUT_S
    assert helper.kwargs[0]["stdin"] is subprocess.DEVNULL


@pytest.mark.parametrize("reply", ["", "not json", subprocess.TimeoutExpired("x", 1)])
def test_status_is_unknown_when_the_helper_fails(helper, reply):
    helper.replies.append(reply)
    assert perms.status() == {"microphone": "unknown", "system_audio": "unknown"}


def test_request_passes_the_timeout_and_returns_the_answer(helper):
    helper.replies.append(json.dumps({"system_audio": "denied"}))
    assert perms.request("system_audio", timeout=5) == "denied"
    assert helper.calls == [["request", "system_audio", "--timeout", "5"]]
    assert helper.kwargs[0]["timeout"] == 5 + 15  # the prompt's timeout plus helper slack


def test_request_failure_is_unknown(helper):
    helper.replies.append("")
    assert perms.request("microphone") == "unknown"


def test_unknown_kind_is_rejected(helper):
    with pytest.raises(ValueError):
        perms.request("camera")
    with pytest.raises(ValueError):
        perms.open_settings("camera")


def test_open_settings_falls_back_to_the_settings_url(helper):
    helper.replies += [RuntimeError("helper gone"), ""]
    perms.open_settings("microphone")
    assert helper.calls[0] == ["open-settings", "microphone"]
    assert helper.calls[1][-1].endswith("Privacy_Microphone")


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", {"microphone": "unknown", "system_audio": "granted"}),
        ("linux", {"microphone": "unknown", "system_audio": "unknown"}),
    ],
)
def test_other_platforms_never_call_the_helper(monkeypatch, platform, expected):
    monkeypatch.setattr(perms.subprocess, "run", lambda *a, **k: pytest.fail("no helper"))
    monkeypatch.setattr(sys, "platform", platform)
    assert perms.status() == expected
    assert perms.request("system_audio") == expected["system_audio"]
