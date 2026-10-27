"""Menu toggles save only their own key: hand edits to config.toml survive a menu save.

Stories: US-17.
"""

import errno
import os
import stat
import subprocess
import sys
import threading
import time
import tomllib
import types

import pytest

from recandcc import config
from recandcc.config import Config, load_config, save_keys
from recandcc.ui import common
from recandcc.ui.common import Controller

HAND_EDIT = "[summary]\nenabled = true\n"


class Monitor:
    def snapshot(self):
        return []


class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def _ctrl(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "off"\n')
    cfg = load_config(path)
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    path.write_text(path.read_text() + HAND_EDIT)  # edited by hand, not reloaded
    return ctrl, path


def _join(name: str = "save-config") -> None:
    for t in threading.enumerate():
        if t.name == name:
            t.join(5)


def _kept(path) -> dict:
    data = tomllib.loads(path.read_text())
    assert data["summary"]["enabled"] is True  # the hand edit survived
    assert data["detection"]["mode"] == "off"
    return data


def test_save_keys_writes_only_the_given_keys(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('hotkey = "x"\n[transcription]\nlanguages = ["iw"]\n')
    save_keys(path, {"live_captions": False, "summary.max_tokens": 9})
    assert tomllib.loads(path.read_text()) == {
        "hotkey": "x",
        "live_captions": False,
        "transcription": {"languages": ["iw"]},  # the user's own spelling
        "summary": {"max_tokens": 9},
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_save_keys_none_removes_a_key_and_its_emptied_table(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "off"\noutside_sensitivity = 25\n[x.y]\nz = 1\n')
    save_keys(path, {"detection.outside_sensitivity": None, "live_captions": False})
    data = tomllib.loads(path.read_text())
    assert data["detection"] == {"mode": "off"}  # the other key in the section survives
    assert data["live_captions"] is False
    save_keys(path, {"x.y.z": None})  # the last key: its table and emptied parents go too
    assert "x" not in tomllib.loads(path.read_text())


def test_save_keys_removing_an_absent_key_changes_nothing(tmp_path):
    path = tmp_path / "config.toml"
    text = '# my notes\n[detection]\nmode = "off"\n'
    path.write_text(text)
    mtime = path.stat().st_mtime
    assert save_keys(path, {"detection.outside_sensitivity": None, "nope.key": None}) == mtime
    assert path.read_text() == text  # not even rewritten: comments and all
    assert "nope" not in tomllib.loads(path.read_text())  # no table created


def test_menu_save_of_an_unset_value_removes_the_key(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    path.write_text(
        path.read_text().replace('mode = "off"', 'mode = "off"\ninside_sensitivity = 50')
    )
    ctrl.cfg.detection.inside_sensitivity = None  # in memory: unset
    ctrl.save_settings("detection.inside_sensitivity")
    _join()
    data = _kept(path)
    assert "inside_sensitivity" not in data["detection"]


def test_save_keys_returns_the_mtime_and_keeps_a_symlink(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text("[summary]\nenabled = true\n")
    link = tmp_path / "config.toml"
    try:
        link.symlink_to(real)
    except OSError:  # Windows without symlink rights
        pytest.skip("symlinks unavailable")
    mtime = save_keys(link, {"autostart": True})
    assert link.is_symlink() and mtime == real.stat().st_mtime
    assert tomllib.loads(real.read_text()) == {"summary": {"enabled": True}, "autostart": True}


def test_save_keys_removes_its_temp_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("autostart = false\n")

    def fail(*_a):
        raise OSError("disk full")

    monkeypatch.setattr("recandcc.config.os.replace", fail)
    with pytest.raises(OSError):
        save_keys(path, {"autostart": True})
    assert path.read_text() == "autostart = false\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_save_keys_keep_both_keys(tmp_path, monkeypatch):
    """The app menu and the CLI write config.toml at once: neither loses the other's key,
    and neither trips over the other's temp file."""
    monkeypatch.setattr("recandcc.config.os.fsync", lambda _fd: None)  # test locking, not disk
    path = tmp_path / "config.toml"
    path.write_text("a = 0\nb = 0\n")
    errors: list[BaseException] = []
    start = threading.Barrier(2)

    def writer(key: str) -> None:
        try:
            start.wait(10)
            for i in range(50):
                save_keys(path, {key: i})
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(k,), daemon=True) for k in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
        assert not t.is_alive(), "a writer is stuck (lock never released?)"
    assert errors == []
    assert tomllib.loads(path.read_text()) == {"a": 49, "b": 49}
    assert not list(tmp_path.glob(".*.tmp"))


def test_config_save_waits_for_the_save_keys_lock(tmp_path):
    """A full Config.save (first run, missing-file recovery) shares save_keys' lock file, so
    it can't land in the middle of a CLI read-modify-write."""
    path = tmp_path / "config.toml"
    path.write_text("autostart = true\n")
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with config._file_lock(path.with_name(f".{path.name}.lock")):
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(10)
    saver = threading.Thread(target=Config(autostart=False).save, args=(path,), daemon=True)
    saver.start()
    saver.join(0.3)
    assert saver.is_alive()  # blocked on the lock
    assert path.read_text() == "autostart = true\n"
    release.set()
    for t in (holder, saver):
        t.join(10)
        assert not t.is_alive()
    assert tomllib.loads(path.read_text())["autostart"] is False


def test_config_save_keeps_a_symlink(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text("autostart = false\n")
    link = tmp_path / "config.toml"
    try:
        link.symlink_to(real)
    except OSError:  # Windows without symlink rights
        pytest.skip("symlinks unavailable")
    Config(autostart=True).save(link)
    assert link.is_symlink()
    assert tomllib.loads(real.read_text())["autostart"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_first_save_creates_a_private_dir_and_file(tmp_path):
    path = tmp_path / "RecAndCC" / "config.toml"
    Config().save(path)
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_save_keeps_an_existing_dir_mode(tmp_path):
    folder = tmp_path / "RecAndCC"
    folder.mkdir()
    folder.chmod(0o755)
    Config().save(folder / "config.toml")
    assert stat.S_IMODE(folder.stat().st_mode) == 0o755


def _flaky_replace(monkeypatch, failures: int) -> list:
    """os.replace that fails with PermissionError ``failures`` times (a reader on Windows)."""
    real = os.replace
    calls: list = []

    def replace(src, dst):
        calls.append(src)
        if len(calls) <= failures:
            raise PermissionError("file in use")
        real(src, dst)

    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setattr(config.os, "replace", replace)
    monkeypatch.setattr(config.time, "sleep", lambda _s: None)
    return calls


def test_windows_replace_retries_while_the_file_is_open(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("autostart = false\n")
    calls = _flaky_replace(monkeypatch, failures=2)
    config._write_atomic(path, b"autostart = true\n")
    assert len(calls) == 3
    assert path.read_text() == "autostart = true\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_windows_replace_gives_up_and_cleans_up(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("autostart = false\n")
    calls = _flaky_replace(monkeypatch, failures=99)
    with pytest.raises(PermissionError):
        config._write_atomic(path, b"autostart = true\n")
    assert len(calls) == 5  # four retries, then the final attempt
    assert path.read_text() == "autostart = false\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_does_not_retry_off_windows(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("autostart = false\n")
    calls = _flaky_replace(monkeypatch, failures=1)
    monkeypatch.setattr(config.sys, "platform", "linux")
    with pytest.raises(PermissionError):
        config._write_atomic(path, b"autostart = true\n")
    assert len(calls) == 1


def test_write_error_does_not_overwrite_hand_edits(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    before = path.read_text()

    def lost_temp(*_a):
        raise FileNotFoundError("temp file gone")

    monkeypatch.setattr(common, "save_keys", lost_temp)
    ctrl.cfg.live_captions = False
    ctrl.write_settings("live_captions")
    assert path.read_text() == before  # no full cfg.save over the user's file
    notice = ("notice", "Settings not saved", "config.toml could not be written")
    assert ctrl.events.get_nowait() == notice


def test_cli_autostart_keeps_hand_edits(tmp_path, monkeypatch):
    from recandcc import autostart, cli

    path = tmp_path / "config.toml"
    path.write_text(HAND_EDIT)
    monkeypatch.setattr(cli, "CONFIG_PATH", path)
    monkeypatch.setattr(autostart, "enable", lambda: None)
    monkeypatch.setattr(autostart, "describe", lambda: "on")
    assert cli.cmd_autostart(types.SimpleNamespace(action="on")) == 0
    data = tomllib.loads(path.read_text())
    assert data == {"summary": {"enabled": True}, "autostart": True}


@pytest.mark.parametrize(
    ("key", "value"),
    [("live_captions", False), ("autostart", True), ("transcription.audio_retention_days", 7)],
)
def test_menu_setting_keeps_unrelated_hand_edits(tmp_path, monkeypatch, key, value):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    *parent, leaf = key.split(".")
    setattr(getattr(ctrl.cfg, parent[0]) if parent else ctrl.cfg, leaf, value)
    ctrl.save_settings(key)
    _join()
    data = _kept(path)
    assert (data[parent[0]] if parent else data)[leaf] == value


def test_language_menu_keeps_unrelated_hand_edits(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    ctrl.toggle_language("he")
    _join()
    assert "he" in _kept(path)["transcription"]["languages"]


def test_unparseable_config_is_left_alone_with_a_notice(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    path.write_text("[summary\n")  # mid-edit
    ctrl.cfg.live_captions = False
    ctrl.save_settings("live_captions")
    _join()
    assert path.read_text() == "[summary\n"
    notice = ("notice", "Settings not saved", "config.toml could not be parsed")
    assert ctrl.events.get_nowait() == notice


def test_missing_config_is_created(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    path.unlink()
    ctrl.cfg.live_captions = False
    ctrl.save_settings("live_captions")
    _join()
    assert load_config(path).live_captions is False


def _fake_autostart(monkeypatch, module) -> None:
    state = {"on": False}
    monkeypatch.setattr(module.autostart, "is_enabled", lambda: state["on"])
    monkeypatch.setattr(module.autostart, "enable", lambda: state.update(on=True))


def test_tray_toggles_keep_hand_edits(tmp_path, monkeypatch):
    stubs = {
        "pystray": types.SimpleNamespace(Icon=None, MenuItem=None, Menu=None),
        "PIL": types.SimpleNamespace(Image=None, ImageDraw=None),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "recandcc.ui.tray", raising=False)
    from recandcc.ui import tray

    try:
        ctrl, path = _ctrl(tmp_path, monkeypatch)
        app = tray.TrayApp.__new__(tray.TrayApp)
        app.ctrl, app.cfg, app._live_id = ctrl, ctrl.cfg, None
        app.captions = types.SimpleNamespace(post=lambda *_a: None)
        _fake_autostart(monkeypatch, tray)
        app._toggle_captions()
        _join()
        app._toggle_login()
        _join()
        data = _kept(path)
        assert data["live_captions"] is False and data["autostart"] is True
    finally:
        sys.modules.pop("recandcc.ui.tray", None)  # don't leak the module built on stubs


@pytest.mark.skipif(sys.platform != "darwin", reason="rumps menu")
def test_mac_toggles_keep_hand_edits(tmp_path, monkeypatch):
    from recandcc.ui import mac

    ctrl, path = _ctrl(tmp_path, monkeypatch)
    app = mac.MacApp.__new__(mac.MacApp)
    app.ctrl, app.cfg, app.captions = ctrl, ctrl.cfg, None
    app._render_retention = lambda: None
    monkeypatch.setattr(ctrl, "purge_audio_now", lambda: None)
    _fake_autostart(monkeypatch, mac)
    app._toggle_captions(types.SimpleNamespace(state=1))
    _join()
    app._toggle_login(types.SimpleNamespace(state=0))
    _join()
    app._set_retention(types.SimpleNamespace(retention_days=30))
    _join("retention-set")
    data = _kept(path)
    assert data["live_captions"] is False and data["autostart"] is True
    assert data["transcription"]["audio_retention_days"] == 30


def test_save_keys_from_another_process_keeps_both_keys(tmp_path):
    """The CLI is a separate process: the lock file serializes it with the app's writes."""
    path = tmp_path / "config.toml"
    path.write_text("a = 0\nb = 0\n")
    ready, go = tmp_path / "ready", tmp_path / "go"
    code = (
        "import sys, time\nfrom pathlib import Path\nfrom recandcc.config import save_keys\n"
        "path, ready, go = map(Path, sys.argv[1:4])\n"
        "ready.touch()\n"
        "end = time.monotonic() + 30\n"
        "while not go.exists():\n"
        "    if time.monotonic() > end:\n        sys.exit(3)\n"
        "    time.sleep(0.005)\n"
        "for i in range(50):\n    save_keys(path, {'b': i})\n"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(path), str(ready), str(go)])
    try:
        end = time.monotonic() + 60  # interpreter start-up and the import
        while not ready.exists():
            assert child.poll() is None, "child exited early"
            assert time.monotonic() < end, "child never got ready"
            time.sleep(0.01)
        go.touch()  # both sides start writing together
        for i in range(50):
            save_keys(path, {"a": i})
        assert child.wait(60) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert tomllib.loads(path.read_text()) == {"a": 49, "b": 49}


class _Clock:
    """Fake time: sleep advances monotonic, so lock timeouts cost no real time."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += s


@pytest.fixture
def fake_clock(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(config.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(config.time, "sleep", clock.sleep)
    monkeypatch.setattr(config, "LOCK_TIMEOUT", 1.0)
    return clock


def _fake_fcntl(monkeypatch, held: list[bool]):
    calls = []
    fake = types.SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=8)

    def flock(fd, op):
        calls.append(op)
        if op & fake.LOCK_NB and held[0]:
            raise BlockingIOError(11, "held")

    fake.flock = flock
    monkeypatch.setitem(sys.modules, "fcntl", fake)
    monkeypatch.setattr(sys, "platform", "linux")
    return calls


def test_posix_lock_times_out_when_held(tmp_path, monkeypatch, fake_clock):
    _fake_fcntl(monkeypatch, [True])
    with pytest.raises(config.ConfigLockTimeout), config._file_lock(tmp_path / "x.lock"):
        pytest.fail("must not acquire")
    assert fake_clock.now >= 1.0
    assert issubclass(config.ConfigLockTimeout, OSError)


def test_posix_lock_acquires_when_released(tmp_path, monkeypatch, fake_clock):
    held = [True]
    calls = _fake_fcntl(monkeypatch, held)
    orig_sleep = config.time.sleep

    def sleep(s):
        orig_sleep(s)
        held[0] = False  # the holder lets go after the first wait

    monkeypatch.setattr(config.time, "sleep", sleep)
    with config._file_lock(tmp_path / "x.lock"):
        pass
    assert calls[-1] == 8  # unlocked on exit


def _fake_msvcrt(monkeypatch, fail_times: float, err: int = errno.EACCES):
    state = {"n": 0}
    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, unlocked=False)

    def locking(fd, mode, nbytes):
        if mode == fake.LK_UNLCK:
            fake.unlocked = True
            return
        state["n"] += 1
        if state["n"] <= fail_times:
            raise OSError(err, "held")

    fake.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    return fake


def test_windows_lock_retries_then_succeeds(tmp_path, monkeypatch, fake_clock):
    fake = _fake_msvcrt(monkeypatch, 3, errno.EDEADLK)
    with config._file_lock(tmp_path / "x.lock"):
        pass
    assert fake.unlocked
    assert 0 < fake_clock.now < 1.0


def test_windows_lock_times_out(tmp_path, monkeypatch, fake_clock):
    fake = _fake_msvcrt(monkeypatch, float("inf"))
    with pytest.raises(config.ConfigLockTimeout), config._file_lock(tmp_path / "x.lock"):
        pytest.fail("must not acquire")
    assert not fake.unlocked
    assert fake_clock.now >= 1.0


def test_windows_lock_other_errors_propagate(tmp_path, monkeypatch, fake_clock):
    _fake_msvcrt(monkeypatch, float("inf"), errno.EBADF)
    with pytest.raises(OSError) as ei, config._file_lock(tmp_path / "x.lock"):
        pass
    assert not isinstance(ei.value, config.ConfigLockTimeout)


def test_write_atomic_logs_chmod_failure(tmp_path, monkeypatch, caplog):
    path = tmp_path / "config.toml"
    path.write_text("a = 1\n")

    def boom(*a, **k):
        raise OSError(errno.EPERM, "nope")

    monkeypatch.setattr(config.os, "chmod", boom)
    with caplog.at_level("DEBUG", logger=config.log.name):
        config._write_atomic(path, b"a = 2\n")
    assert path.read_text() == "a = 2\n"
    assert "config.toml" in caplog.text


@pytest.mark.parametrize(
    ("exc", "message"),
    [
        (config.ConfigLockTimeout("locked"), "config.toml is locked by another process"),
        (OSError(errno.EIO, "disk"), "config.toml could not be written"),
    ],
)
def test_failed_menu_save_restores_the_old_value(tmp_path, monkeypatch, caplog, exc, message):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    before = path.read_text()

    def fail(*_a):
        raise exc

    monkeypatch.setattr(common, "save_keys", fail)
    assert ctrl.cfg.live_captions is True
    ctrl.cfg.live_captions = False  # the click
    ctrl.save_settings("live_captions")
    _join()
    assert ctrl.cfg.live_captions is True  # back to what config.toml says
    assert path.read_text() == before
    assert ctrl.events.get_nowait() == ("notice", "Settings not saved", message)
    assert not ctrl._dirty  # dirty mark cleared
    if isinstance(exc, config.ConfigLockTimeout):
        assert all(r.exc_info is None for r in caplog.records)
