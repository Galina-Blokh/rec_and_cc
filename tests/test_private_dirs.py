"""Owner-only recording / transcript folders on Windows (icacls DACL) and POSIX (0700).

Stories: US-13.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from recandcc import recorder as recmod
from recandcc.recorder import icacls_owner_only_argv, make_private_dir, restrict_to_owner

SID = "S-1-5-21-111-222-333-1001"
ICACLS = str(Path(r"C:\Windows") / "System32" / "icacls.exe")
WHOAMI = str(Path(r"C:\Windows") / "System32" / "whoami.exe")


@pytest.fixture(autouse=True)
def _fresh_sid_cache():
    recmod._current_user_sid.cache_clear()
    yield
    recmod._current_user_sid.cache_clear()


@pytest.fixture
def fake_windows(monkeypatch):
    """Pretend to be Windows and record every subprocess argv instead of running it."""
    calls: list[list[str]] = []
    result = {"icacls_rc": 0}

    def run(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == WHOAMI:
            out = f'"desktop-x\\alice","{SID}"\r\n'
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=result["icacls_rc"], stdout="", stderr="Access denied")

    monkeypatch.setattr(recmod.sys, "platform", "win32")
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    monkeypatch.setattr(recmod.subprocess, "run", run)
    return SimpleNamespace(calls=calls, result=result)


def test_icacls_argv_is_owner_and_system_only_without_inheritance(monkeypatch):
    monkeypatch.setenv("SYSTEMROOT", r"C:\Windows")
    d = Path("C:/Users/alice/Documents/Transcribes")
    assert icacls_owner_only_argv(d, SID) == [
        ICACLS,
        str(d),
        "/inheritance:r",
        "/grant:r",
        f"*{SID}:(OI)(CI)F",
        "*S-1-5-18:(OI)(CI)F",
    ]


def test_restrict_on_windows_looks_up_the_sid_then_runs_icacls(fake_windows, tmp_path):
    restrict_to_owner(tmp_path)
    assert fake_windows.calls == [
        [WHOAMI, "/user", "/fo", "csv", "/nh"],
        icacls_owner_only_argv(tmp_path, SID),
    ]


def test_make_private_dir_restricts_every_folder_it_creates(fake_windows, tmp_path):
    deep = tmp_path / "out" / "2026-10-27"
    make_private_dir(deep)
    icacls = [c for c in fake_windows.calls if c[0] == ICACLS]
    assert [c[1] for c in icacls] == [str(tmp_path / "out"), str(deep)]
    assert sum(c[0] == WHOAMI for c in fake_windows.calls) == 1  # SID looked up once
    make_private_dir(deep)  # already there: left untouched
    assert len([c for c in fake_windows.calls if c[0] == ICACLS]) == 2


def test_icacls_failure_is_logged_not_raised(fake_windows, tmp_path, caplog):
    fake_windows.result["icacls_rc"] = 5
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        make_private_dir(tmp_path / "out")
    assert (tmp_path / "out").is_dir()
    assert "could not restrict" in caplog.text and "Access denied" in caplog.text


def test_missing_tools_are_logged_not_raised(monkeypatch, tmp_path, caplog):
    def run(argv, **kwargs):
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(recmod.sys, "platform", "win32")
    monkeypatch.setattr(recmod.subprocess, "run", run)
    with caplog.at_level(logging.WARNING, logger="recandcc.recorder"):
        restrict_to_owner(tmp_path)
    assert "could not restrict" in caplog.text


def test_unexpected_whoami_output_is_rejected(monkeypatch):
    out = SimpleNamespace(returncode=0, stdout="garbage\n", stderr="")
    monkeypatch.setattr(recmod.subprocess, "run", lambda argv, **kw: out)
    with pytest.raises(ValueError):
        recmod._current_user_sid()


def test_posix_never_runs_icacls(monkeypatch, tmp_path):
    def run(argv, **kwargs):
        raise AssertionError(f"unexpected subprocess: {argv}")

    monkeypatch.setattr(recmod.sys, "platform", "darwin")
    monkeypatch.setattr(recmod.subprocess, "run", run)
    make_private_dir(tmp_path / "a" / "b")
    assert (tmp_path / "a" / "b").is_dir()


@pytest.mark.skipif(sys.platform != "win32", reason="NTFS ACLs")
def test_real_windows_folder_is_owner_and_system_only(tmp_path):
    d = tmp_path / "private" / "day"
    make_private_dir(d)
    for p in (d.parent, d):
        out = subprocess.run(["icacls", str(p)], capture_output=True, text=True, check=True).stdout
        lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
        # First line is "<path> <ACE>", then one ACE per line, then a summary line.
        aces = [lines[0][len(str(p)) :].strip()] + [ln for ln in lines[1:-1]]
        assert len(aces) == 2, out
        assert not any("(I)" in a for a in aces), out  # nothing inherited from the parent
        assert any("SYSTEM" in a.upper() for a in aces), out
        user = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip()
        assert any(a.lower().startswith(user.lower() + ":") for a in aces), out
    (d / "probe.txt").write_text("ok")  # the owner can still write inside
