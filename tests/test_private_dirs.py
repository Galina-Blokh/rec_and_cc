"""Owner-only recording / transcript folders: an owner-only DACL on Windows, 0700 on POSIX.

Stories: US-13.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from recandcc import recorder as recmod
from recandcc.config import Config
from recandcc.models import TriggerKind
from recandcc.recorder import make_private_dir, owner_only_sddl

SID = "S-1-5-21-111-222-333-1001"


@pytest.fixture
def fake_windows(monkeypatch):
    """Pretend to be Windows; record secure-mkdir calls and create the folder for real."""
    calls: list[Path] = []
    state = {"error": None}

    def secure_mkdir(path):
        calls.append(path)
        if state["error"] is not None:
            raise state["error"]
        path.mkdir()

    monkeypatch.setattr(recmod.sys, "platform", "win32")
    monkeypatch.setattr(recmod, "_win_mkdir_owner_only", secure_mkdir)
    return type("FakeWindows", (), {"calls": calls, "state": state})


def test_sddl_is_protected_owner_and_system_only():
    assert owner_only_sddl(SID) == f"D:P(A;OICI;FA;;;{SID})(A;OICI;FA;;;SY)"


def test_windows_creates_every_missing_level_with_the_owner_only_acl(fake_windows, tmp_path):
    deep = tmp_path / "out" / "2026-10-27"
    make_private_dir(deep)
    assert fake_windows.calls == [tmp_path / "out", deep]
    make_private_dir(deep)  # already there: left untouched
    assert len(fake_windows.calls) == 2


def test_windows_acl_failure_is_logged_and_falls_back_to_mkdir(fake_windows, tmp_path, caplog):
    fake_windows.state["error"] = OSError("ACLs not supported on this share")
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert "owner-only ACL" in caplog.text and "not supported" in caplog.text


def test_windows_file_in_the_way_still_raises(fake_windows, tmp_path):
    (tmp_path / "out").write_text("not a folder")
    fake_windows.state["error"] = FileExistsError("exists")
    with pytest.raises(FileExistsError):
        make_private_dir(tmp_path / "out")


def test_windows_session_folder_gets_the_acl_without_subprocesses(
    fake_windows, monkeypatch, tmp_path
):
    def run(*args, **kwargs):
        raise AssertionError("no subprocess on the recording start path")

    monkeypatch.setattr(subprocess, "run", run)
    cfg = Config()
    cfg.output_dir = tmp_path / "out"
    rec = recmod.Recorder.__new__(recmod.Recorder)  # only _make_dir is exercised
    rec.cfg = cfg
    _, d = rec._make_dir(datetime(2026, 10, 27, 9, 30), "zoom", TriggerKind.MANUAL)
    assert d.is_dir()
    assert fake_windows.calls == [cfg.output_dir, d.parent, d]


def test_posix_never_uses_the_windows_api(monkeypatch, tmp_path):
    def secure_mkdir(path):
        raise AssertionError("Windows API on POSIX")

    monkeypatch.setattr(recmod.sys, "platform", "darwin")
    monkeypatch.setattr(recmod, "_win_mkdir_owner_only", secure_mkdir)
    make_private_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows security API")
def test_real_windows_sid_matches_whoami():
    out = subprocess.run(
        ["whoami", "/user", "/fo", "csv", "/nh"], capture_output=True, text=True, check=True
    ).stdout
    recmod._current_user_sid.cache_clear()
    assert recmod._current_user_sid() == out.strip().rsplit(",", 1)[-1].strip('"')


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS ACLs")
def test_real_windows_folder_is_owner_and_system_only(tmp_path):
    d = tmp_path / "private" / "day"
    make_private_dir(d)
    (d / "probe.txt").write_text("ok")  # the owner can still write inside
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip().lower()
    for p in (d.parent, d, d / "probe.txt"):
        out = subprocess.run(["icacls", str(p)], capture_output=True, text=True, check=True)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        # First line is "<path> <ACE>", then one ACE per line, then a summary line.
        aces = [lines[0][len(str(p)) :].strip(), *lines[1:-1]]
        assert len(aces) == 2, out.stdout
        assert any(a.upper().startswith("NT AUTHORITY\\SYSTEM:") for a in aces), out.stdout
        assert any(a.lower().startswith(user + ":") for a in aces), out.stdout
        if p.is_dir():  # the folders themselves inherit nothing from their parent
            assert not any("(I)" in a for a in aces), out.stdout
