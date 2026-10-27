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

    state["persists"] = True
    monkeypatch.setattr(recmod, "_IS_WINDOWS", True)
    monkeypatch.setattr(recmod, "_win_mkdir_owner_only", secure_mkdir)
    monkeypatch.setattr(
        recmod, "_volume_root_and_acl_support", lambda path: ("E:\\", state["persists"])
    )
    monkeypatch.setattr(recmod, "_warned", set())
    monkeypatch.setattr(recmod, "_privacy_notice", {"pending": False, "sent": False})
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
    make_private_dir(tmp_path / "out2")  # the next session start doesn't warn again
    assert len(caplog.records) == 1


def test_windows_sid_failure_falls_back_to_plain_mkdir_and_warns_once(
    monkeypatch, tmp_path, caplog
):
    def no_sid():
        raise OSError("no process token")

    monkeypatch.setattr(recmod, "_IS_WINDOWS", True)
    monkeypatch.setattr(recmod, "_current_user_sid", no_sid)  # real _win_mkdir_owner_only
    monkeypatch.setattr(recmod, "_warned", set())
    monkeypatch.setattr(recmod, "_privacy_notice", {"pending": False, "sent": False})
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()
    assert len(caplog.records) == 1 and "no process token" in caplog.text


def test_windows_volume_without_persistent_acls_warns_once(fake_windows, tmp_path, caplog):
    fake_windows.state["persists"] = False
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out" / "day")
        make_private_dir(tmp_path / "out" / "day2")
    assert (tmp_path / "out" / "day2").is_dir()  # recording goes on
    assert len(caplog.records) == 1
    assert "E:\\" in caplog.text and "may not keep file permissions" in caplog.text


def test_error_already_exists_maps_to_file_exists(tmp_path):
    e = recmod._exists_error(183, tmp_path / "x")
    assert isinstance(e, FileExistsError) and e.filename == str(tmp_path / "x")
    assert recmod._exists_error(5, tmp_path / "x") is None  # access denied: WinError


def test_concurrent_create_burns_no_warning_or_notice(fake_windows, monkeypatch, tmp_path, caplog):
    def concurrent(path):  # another thread won the race: CreateDirectoryW -> 183
        path.mkdir()
        raise recmod._exists_error(183, path)

    monkeypatch.setattr(recmod, "_win_mkdir_owner_only", concurrent)
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert not caplog.records and not recmod._warned
    assert recmod.take_privacy_notice() is None


def test_unknown_acl_support_is_treated_as_not_private(fake_windows, monkeypatch, tmp_path, caplog):
    def broken(path):
        raise OSError("GetVolumeInformationW failed")

    monkeypatch.setattr(recmod, "_volume_root_and_acl_support", broken)
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert len(caplog.records) == 1 and "file permissions" in caplog.text


def test_not_private_folder_yields_one_notice_per_process(fake_windows, tmp_path):
    assert recmod.take_privacy_notice() is None
    fake_windows.state["persists"] = False
    make_private_dir(tmp_path / "a")
    title, message = recmod.take_privacy_notice()
    assert "other users" in title and "NTFS" in message
    make_private_dir(tmp_path / "b")
    assert recmod.take_privacy_notice() is None  # once per process


def test_windows_ntfs_volume_does_not_warn(fake_windows, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert not caplog.records


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

    monkeypatch.setattr(recmod, "_IS_WINDOWS", False)
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


def _assert_owner_and_system_only(paths):
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip().lower()
    for p in paths:
        out = subprocess.run(["icacls", str(p)], capture_output=True, text=True, check=True)
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        # First line is "<path> <ACE>", then one ACE per line, then a summary line.
        aces = [lines[0][len(str(p)) :].strip(), *lines[1:-1]]
        assert len(aces) == 2, out.stdout
        assert any(a.upper().startswith("NT AUTHORITY\\SYSTEM:") for a in aces), out.stdout
        assert any(a.lower().startswith(user + ":") for a in aces), out.stdout
        if p.is_dir():  # the folders themselves inherit nothing from their parent
            assert not any("(I)" in a for a in aces), out.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS ACLs")
def test_real_windows_folder_is_owner_and_system_only(tmp_path):
    d = tmp_path / "private" / "day"
    make_private_dir(d)
    (d / "probe.txt").write_text("ok")  # the owner can still write inside
    _assert_owner_and_system_only([d.parent, d, d / "probe.txt"])


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS ACLs")
def test_real_windows_session_folder_and_file_are_owner_and_system_only(tmp_path):
    cfg = Config()
    cfg.output_dir = tmp_path / "out"
    rec = recmod.Recorder.__new__(recmod.Recorder)  # the path Recorder.start uses
    rec.cfg = cfg
    _, d = rec._make_dir(datetime(2026, 10, 27, 9, 30), "zoom", TriggerKind.MANUAL)
    (d / "others.wav").write_bytes(b"RIFF")
    _assert_owner_and_system_only([cfg.output_dir, d.parent, d, d / "others.wav"])
