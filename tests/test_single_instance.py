"""cli._SingleInstance: one tray app at a time, on Windows (fake msvcrt) and POSIX.

Stories: US-17.
"""

import os
import sys
import types

import pytest

from recandcc.cli import _SingleInstance


class _FakeMsvcrt(types.ModuleType):
    """Byte-range locks like msvcrt.locking: from the fd's current position."""

    LK_NBLCK = 2
    LK_UNLCK = 0

    def __init__(self):
        super().__init__("msvcrt")
        self.held: dict[tuple[int, int], int] = {}  # (inode, offset) -> fd
        self.offsets: list[int] = []

    def locking(self, fd, mode, nbytes):
        offset = os.lseek(fd, 0, os.SEEK_CUR)
        self.offsets.append(offset)
        key = (os.fstat(fd).st_ino, offset)
        if mode == self.LK_UNLCK:
            self.held.pop(key, None)
        elif key in self.held and self.held[key] != fd:
            raise PermissionError(13, "Permission denied")
        else:
            self.held[key] = fd


@pytest.fixture
def win(monkeypatch):
    fake = _FakeMsvcrt()
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    return fake


@pytest.fixture
def opened():
    locks: list[_SingleInstance] = []
    yield locks
    for lock in locks:
        if lock._f is not None:
            lock._f.close()


def _new(path, opened):
    lock = _SingleInstance(path)
    opened.append(lock)
    return lock


def test_windows_locks_byte_zero_and_writes_pid(win, tmp_path, opened):
    path = tmp_path / "run.lock"
    assert _new(path, opened).acquire()
    assert win.offsets == [0]
    assert path.read_text() == str(os.getpid())


def test_windows_second_instance_reports_already_running(win, tmp_path, opened):
    path = tmp_path / "run.lock"
    assert _new(path, opened).acquire()
    second = _new(path, opened)
    assert second.acquire() is False  # no PermissionError traceback
    assert win.offsets == [0, 0]
    assert second._f is None
    assert path.read_text() == str(os.getpid())  # first instance's PID untouched


def test_windows_stale_lock_of_other_length_still_locks_byte_zero(win, tmp_path, opened):
    path = tmp_path / "run.lock"
    path.write_text("123456789")  # PID left by a crashed run, different length
    assert _new(path, opened).acquire()
    assert win.offsets == [0]
    assert path.read_text() == str(os.getpid())


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_posix_second_instance_reports_already_running(tmp_path, opened):
    path = tmp_path / "run.lock"
    path.write_text("123456789")
    assert _new(path, opened).acquire()
    assert path.read_text() == str(os.getpid())
    assert _new(path, opened).acquire() is False
