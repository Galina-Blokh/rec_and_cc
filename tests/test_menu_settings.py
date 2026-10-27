"""Menu toggles save only their own key: hand edits to config.toml survive a menu save.

Stories: US-17.
"""

import sys
import threading
import tomllib
import types

import pytest

from recandcc.config import load_config, save_keys
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
