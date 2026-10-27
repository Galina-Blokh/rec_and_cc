"""autostart: the macOS LaunchAgent plist and the Windows Run key, both faked on disk."""

import plistlib
import sys

import pytest

from recandcc import autostart


@pytest.fixture
def mac(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(autostart, "PLIST_PATH", tmp_path / "LaunchAgents" / "x.plist")
    monkeypatch.setattr(autostart, "LOG_DIR", tmp_path / "logs")
    monkeypatch.delenv("RCC_APP_BUNDLE", raising=False)
    monkeypatch.setattr(autostart.Path, "home", lambda: tmp_path)
    return tmp_path


def test_mac_enable_writes_a_plist_that_runs_python_without_the_app(mac, monkeypatch):
    monkeypatch.setattr(autostart, "find_app_bundle", lambda: None)
    assert not autostart.is_enabled()
    autostart.enable()
    plist = plistlib.loads(autostart.PLIST_PATH.read_bytes())
    assert plist["ProgramArguments"] == [sys.executable, "-m", "recandcc", "run"]
    assert plist["RunAtLoad"] and plist["Label"] == autostart.LABEL
    assert autostart.is_enabled()
    assert autostart.describe().startswith("enabled")
    autostart.disable()
    autostart.disable()  # idempotent
    assert not autostart.is_enabled()


def test_mac_launches_the_app_bundle_when_present(mac, monkeypatch):
    app = mac / "Bundles" / "RecAndCC.app"
    app.mkdir(parents=True)
    monkeypatch.setenv("RCC_APP_BUNDLE", str(app))
    assert autostart.mac_program_arguments() == ["/usr/bin/open", "-a", str(app)]


def test_find_app_bundle_ignores_non_app_paths(mac, monkeypatch):
    monkeypatch.setenv("RCC_APP_BUNDLE", str(mac))  # a folder, but not a .app
    assert autostart.find_app_bundle() is None


class FakeWinreg:
    HKEY_CURRENT_USER, KEY_SET_VALUE, REG_SZ = "HKCU", 2, 1

    def __init__(self):
        self.values = {}

    def OpenKey(self, root, key, *_):  # noqa: N802 - winreg API
        return _Key(self)

    def SetValueEx(self, _k, name, _r, _t, value):  # noqa: N802
        self.values[name] = value

    def QueryValueEx(self, _k, name):  # noqa: N802
        if name not in self.values:
            raise FileNotFoundError(name)
        return self.values[name], self.REG_SZ

    def DeleteValue(self, _k, name):  # noqa: N802
        if name not in self.values:
            raise FileNotFoundError(name)
        del self.values[name]


class _Key:
    def __init__(self, reg):
        self.reg = reg

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_windows_run_key_round_trip(monkeypatch):
    reg = FakeWinreg()
    monkeypatch.setitem(sys.modules, "winreg", reg)
    monkeypatch.setattr(sys, "platform", "win32")
    assert not autostart.is_enabled()
    autostart.enable()
    assert reg.values[autostart.WIN_VALUE].endswith('" -m recandcc run')
    assert autostart.is_enabled()
    assert "HKCU" in autostart.describe()
    autostart.disable()
    autostart.disable()  # already gone: no error
    assert not autostart.is_enabled()


def test_other_platforms_are_unsupported(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(NotImplementedError):
        autostart.enable()
    with pytest.raises(NotImplementedError):
        autostart.disable()
    assert not autostart.is_enabled()
    assert autostart.describe() == "unsupported platform"
