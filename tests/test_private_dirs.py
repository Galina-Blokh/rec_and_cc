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
    state["volume"] = "E:\\ (1234ABCD)"
    monkeypatch.setattr(
        recmod, "_volume_acl_support", lambda path: (state["volume"], state["persists"])
    )
    monkeypatch.setattr(recmod, "_warned", set())
    monkeypatch.setattr(recmod, "_noticed", set())
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
    assert "E:\\ (1234ABCD)" in caplog.text and "may not keep file permissions" in caplog.text


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
        assert make_private_dir(tmp_path / "out") is None
    assert (tmp_path / "out").is_dir()
    assert not caplog.records and not recmod._warned


def test_unknown_acl_support_is_treated_as_not_private(fake_windows, monkeypatch, tmp_path, caplog):
    def broken(path):
        raise OSError("GetVolumeInformationW failed")

    monkeypatch.setattr(recmod, "_volume_acl_support", broken)
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert len(caplog.records) == 1 and "file permissions" in caplog.text


class _Idle:
    def __init__(self, fail=False):
        self.fail = fail

    def start(self, on_frames):
        if self.fail:
            raise RuntimeError("no permission")

    def stop(self):
        pass


def _recorder(tmp_path, fail=False):
    cfg = Config(output_dir=tmp_path / "out")
    return recmod.Recorder(
        cfg, mic_factory=lambda: _Idle(fail), system_factory=lambda *a: _Idle(fail)
    )


def _engine(rec):
    from recandcc.engine import Engine

    notices: list[str] = []
    monitor = type("Mon", (), {"snapshot": lambda self: [], "close": lambda self: None})()
    eng = Engine(
        rec.cfg,
        monitor,
        rec,
        type("Q", (), {"submit": lambda self, s: None})(),
        on_notice=lambda title, message: notices.append(title),
        run_in_background=lambda fn: fn(),
    )
    return eng, notices


def test_not_private_session_notice_once_per_volume_through_the_engine(fake_windows, tmp_path):
    fake_windows.state["persists"] = False
    rec = _recorder(tmp_path)
    eng, notices = _engine(rec)
    for _ in range(2):  # repeat sessions on the same bad volume stay quiet
        eng.start_manual()
        eng.tick()
        assert rec.is_recording  # #37: recording goes on
        eng.stop()
        eng.tick()
    assert notices.count(recmod.PRIVACY_NOTICE[0]) == 1
    fake_windows.state["volume"] = "E:\\ (5678EF01)"  # another USB stick, same letter
    eng.start_manual()
    eng.tick()
    assert notices.count(recmod.PRIVACY_NOTICE[0]) == 2
    eng.stop()
    eng.tick()


def test_failed_start_leaves_no_privacy_notice(fake_windows, tmp_path):
    fake_windows.state["persists"] = False
    rec = _recorder(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        rec.start(TriggerKind.MANUAL)
    assert rec.privacy_notice() is None


def test_ui_folder_creation_does_not_arm_the_session_notice(fake_windows, tmp_path):
    fake_windows.state["persists"] = False
    rec = _recorder(tmp_path)
    assert make_private_dir(rec.cfg.output_dir)  # "Open transcripts folder"
    assert rec.privacy_notice() is None


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
    _, d, issue = rec._make_dir(datetime(2026, 10, 27, 9, 30), "zoom", TriggerKind.MANUAL)
    assert d.is_dir() and issue is None
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
    """Every ACE (``icacls``) is SYSTEM or the current user; folders inherit nothing."""
    user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip().lower()
    user = user.rsplit("\\", 1)[-1]  # tolerate the HOST\ / DOMAIN\ prefix
    for p in paths:
        out = subprocess.run(["icacls", str(p)], capture_output=True, text=True, check=True)
        text = out.stdout.replace(str(p), "", 1)
        aces = [ln.strip() for ln in text.splitlines() if ":(" in ln]  # drops the summary
        aces = [a for a in aces if not a.lower().startswith("mandatory label\\")]
        names = {a.split(":(", 1)[0].lower() for a in aces}
        assert "nt authority\\system" in names, out.stdout
        assert any(n.rsplit("\\", 1)[-1] == user for n in names), out.stdout
        others = {n for n in names if n != "nt authority\\system"}
        assert all(n.rsplit("\\", 1)[-1] == user for n in others), out.stdout
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
    _, d, issue = rec._make_dir(datetime(2026, 10, 27, 9, 30), "zoom", TriggerKind.MANUAL)
    assert issue is None
    (d / "others.wav").write_bytes(b"RIFF")
    _assert_owner_and_system_only([cfg.output_dir, d.parent, d, d / "others.wav"])
