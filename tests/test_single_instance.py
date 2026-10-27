"""cli._SingleInstance: one tray app at a time, on Windows (fake msvcrt) and POSIX.

Stories: US-17.
"""

import os
import sys
import types

import pytest

from recandcc import cli
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
        elif key in self.held and self.held[key] != fd and self._open(*key):
            raise PermissionError(13, "Permission denied")
        else:
            self.held[key] = fd

    def _open(self, ino, offset):
        """Windows drops a lock when its handle closes: a closed holder holds nothing."""
        try:
            return os.fstat(self.held[(ino, offset)]).st_ino == ino
        except OSError:
            return False


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


def test_windows_lock_is_free_again_after_the_holder_closes(win, tmp_path, opened):
    path = tmp_path / "run.lock"
    first = _new(path, opened)
    assert first.acquire()
    assert first._f is not None
    first._f.close()
    first._f = None
    assert _new(path, opened).acquire()


class _FullDisk:
    """A file whose writes fail, as on a full disk; everything else is real."""

    def __init__(self, f):
        self._f = f

    def __getattr__(self, name):
        return getattr(self._f, name)

    def write(self, _data):
        raise OSError(28, "No space left on device")


def test_write_failure_after_lock_is_raised_and_closes_the_handle(
    win, tmp_path, opened, monkeypatch
):
    files: list[_FullDisk] = []

    def fake_open(*args, **kwargs):
        files.append(_FullDisk(open(*args, **kwargs)))  # noqa: SIM115
        return files[-1]

    monkeypatch.setattr(cli, "open", fake_open, raising=False)
    lock = _new(tmp_path / "run.lock", opened)
    with pytest.raises(OSError, match="No space"):
        lock.acquire()  # not misreported as "already running"
    assert lock._f is None
    assert files[0].closed


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl is POSIX-only")
def test_posix_second_instance_reports_already_running(tmp_path, opened):
    path = tmp_path / "run.lock"
    path.write_text("123456789")
    assert _new(path, opened).acquire()
    assert path.read_text() == str(os.getpid())
    assert _new(path, opened).acquire() is False


class _Stop(Exception):
    pass


def _patch_run(monkeypatch, results):
    calls = []

    def acquire(self):
        calls.append(1)
        return results[min(len(calls), len(results)) - 1]

    monkeypatch.setattr(_SingleInstance, "acquire", acquire)
    monkeypatch.setattr(cli, "setup_logging", lambda: None)
    monkeypatch.setattr(cli, "_ACQUIRE_RETRY_S", 0)

    def stop(*_a, **_k):
        raise _Stop

    monkeypatch.setattr(cli, "load_config", stop)
    return calls


def test_cmd_run_retries_past_a_transient_probe_lock(monkeypatch):
    """A held_elsewhere() probe holds the lock for an instant; the app must not exit on it."""
    calls = _patch_run(monkeypatch, [False, True])
    with pytest.raises(_Stop):  # got past the lock to load_config
        cli.cmd_run(None)
    assert len(calls) == 2


def test_cmd_run_still_exits_when_really_running(monkeypatch, capsys):
    calls = _patch_run(monkeypatch, [False])
    assert cli.cmd_run(None) == 1
    assert len(calls) == cli._ACQUIRE_RETRIES
    assert "already running" in capsys.readouterr().err
