"""Windows activity detection with fake pycaw / comtypes / winreg / psutil (runs anywhere)."""

from __future__ import annotations

import os
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

from recandcc.detect import win
from recandcc.detect.win import WinActivityMonitor
from recandcc.models import AudioProcess

OWN_EXE = r"C:\Python312\python.exe"


def proc(name: str, pid: int):
    return types.SimpleNamespace(pid=pid, name=lambda: name)


class FakeSession:
    def __init__(self, proc, state: int = 1):
        self._proc, self.State = proc, state

    @property
    def Process(self):
        if isinstance(self._proc, Exception):
            raise self._proc  # process vanished between enumeration and access
        return self._proc


class Key(dict):
    """An open registry key: the node itself, usable as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeWinreg(types.ModuleType):
    """Registry as nested dicts: dict values are subkeys, everything else is a value."""

    HKEY_CURRENT_USER = object()

    def __init__(self, tree: dict):
        super().__init__("winreg")
        self.tree = tree

    def OpenKey(self, parent, sub):
        node = self.tree if parent is self.HKEY_CURRENT_USER else parent
        for part in sub.split("\\"):
            if not isinstance(node.get(part), dict):
                raise FileNotFoundError(sub)
            node = node[part]
        return Key(node)

    def EnumKey(self, key, i):
        subs = [k for k, v in key.items() if isinstance(v, dict)]
        if i >= len(subs):
            raise OSError("no more items")
        return subs[i]

    def QueryValueEx(self, key, name):
        if name not in key or isinstance(key[name], dict):
            raise FileNotFoundError(name)
        return key[name], 11  # REG_QWORD


def mic_tree(packaged: dict, nonpackaged: dict) -> dict:
    node: dict = {"NonPackaged": nonpackaged, **packaged}
    for part in reversed(win._MIC_KEY.split("\\")):
        node = {part: node}
    return node


IN_USE = {"LastUsedTimeStart": 133_000_000, "LastUsedTimeStop": 0}
STOPPED = {"LastUsedTimeStart": 133_000_000, "LastUsedTimeStop": 133_000_500}


@pytest.fixture
def fakes(monkeypatch):
    """Install fake Windows modules; tests fill in sessions, registry and processes."""
    state = types.SimpleNamespace(sessions=[], procs=[], comtypes=MagicMock())
    pycaw = types.ModuleType("pycaw")
    pycaw_pycaw = types.ModuleType("pycaw.pycaw")
    pycaw_pycaw.AudioUtilities = types.SimpleNamespace(GetAllSessions=lambda: state.sessions)
    pycaw.pycaw = pycaw_pycaw
    psutil = types.ModuleType("psutil")
    psutil.process_iter = lambda attrs: iter(state.procs)
    state.winreg = FakeWinreg(mic_tree({}, {}))

    monkeypatch.setitem(sys.modules, "comtypes", state.comtypes)
    monkeypatch.setitem(sys.modules, "pycaw", pycaw)
    monkeypatch.setitem(sys.modules, "pycaw.pycaw", pycaw_pycaw)
    monkeypatch.setitem(sys.modules, "psutil", psutil)
    monkeypatch.setitem(sys.modules, "winreg", state.winreg)
    return state


@pytest.fixture
def mon():
    m = WinActivityMonitor()
    m._own_exe = os.path.normcase(OWN_EXE)  # sys.executable is a POSIX path on macOS CI
    return m


def by_name(procs):
    return {p.bundle_id: p for p in procs}


# ------------------------------------------------------------------ output sessions
def test_active_output_sessions_become_processes(fakes, mon):
    fakes.sessions = [
        FakeSession(proc("Zoom.exe", 101)),
        FakeSession(proc("Spotify.exe", 102), state=0),  # inactive: not playing
        FakeSession(None),  # system sounds session has no process
        FakeSession(proc("python.exe", os.getpid())),  # ourselves
        FakeSession(ProcessLookupError("gone")),  # vanished: skipped, rest still listed
        FakeSession(proc("chrome.exe", 103)),
    ]
    procs = by_name(mon.snapshot())
    assert set(procs) == {"zoom.exe", "chrome.exe"}
    assert procs["zoom.exe"] == AudioProcess(101, "zoom.exe", "Zoom.exe", False, True)


def test_pycaw_failure_still_reports_mic(fakes, mon, caplog):
    def boom():
        raise OSError("COM error")

    sys.modules["pycaw.pycaw"].AudioUtilities = types.SimpleNamespace(GetAllSessions=boom)
    fakes.winreg.tree = mic_tree({"MSTeams_8wekyb3d8bbwe": IN_USE}, {})
    procs = mon.snapshot()
    assert [p.bundle_id for p in procs] == ["msteams"]
    assert "pycaw session enumeration failed" in caplog.text


# ------------------------------------------------------------------ microphone consent store
def test_mic_users_packaged_and_classic(fakes, mon):
    fakes.winreg.tree = mic_tree(
        {
            "MSTeams_8wekyb3d8bbwe": IN_USE,
            "Microsoft.WindowsCamera_8wekyb3d8bbwe": STOPPED,
            "Broken_123": {"LastUsedTimeStop": 0},  # missing value: treated as not in use
            "Never_123": {"LastUsedTimeStart": 0, "LastUsedTimeStop": 0},
        },
        {
            "C:#Program Files#Zoom#bin#Zoom.exe": IN_USE,
            "C:#Tools#old.exe": STOPPED,
        },
    )
    assert sorted(mon._mic_users()) == [
        ("msteams", "MSTeams_8wekyb3d8bbwe"),
        ("zoom.exe", r"C:\Program Files\Zoom\bin\Zoom.exe"),
    ]
    fakes.winreg.tree = {}  # consent store key absent
    assert mon._mic_users() == []


def test_mic_merges_with_output_and_skips_own_exe(fakes, mon):
    fakes.sessions = [FakeSession(proc("Zoom.exe", 101))]
    fakes.procs = [types.SimpleNamespace(pid=555, info={"name": "Discord.exe"})]
    fakes.winreg.tree = mic_tree(
        {},
        {
            "C:#Program Files#Zoom#bin#Zoom.exe": IN_USE,
            "C:#Apps#Discord.exe": IN_USE,
            OWN_EXE.replace("\\", "#"): IN_USE,  # our own recorder: filtered out
        },
    )
    assert by_name(mon.snapshot()) == {
        "zoom.exe": AudioProcess(101, "zoom.exe", "Zoom.exe", True, True),
        "discord.exe": AudioProcess(555, "discord.exe", "discord.exe", True, False),
    }


def test_pid_lookup(fakes, mon, monkeypatch):
    fakes.procs = [
        types.SimpleNamespace(pid=1, info={"name": None}),
        types.SimpleNamespace(pid=2, info={"name": "ZOOM.EXE"}),
    ]
    assert mon._pid_for_exe("zoom.exe") == 2
    assert mon._pid_for_exe("missing.exe") == 0

    def boom(attrs):
        raise RuntimeError("access denied")

    monkeypatch.setattr(sys.modules["psutil"], "process_iter", boom)
    assert mon._pid_for_exe("zoom.exe") == 0


def test_registry_unavailable_still_reports_output(fakes, mon, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "winreg", None)  # import winreg -> ImportError
    fakes.sessions = [FakeSession(proc("Zoom.exe", 101))]
    assert [p.bundle_id for p in mon.snapshot()] == ["zoom.exe"]
    assert "reading microphone consent store failed" in caplog.text


# ------------------------------------------------------------------ COM lifetime
def test_com_initialised_once_per_thread_and_closed(fakes, mon):
    mon.snapshot()
    mon.snapshot()
    assert fakes.comtypes.CoInitialize.call_count == 1
    t = threading.Thread(target=mon.snapshot)
    t.start()
    t.join()
    assert fakes.comtypes.CoInitialize.call_count == 2  # per thread
    mon.close()
    mon.close()  # idempotent
    assert fakes.comtypes.CoUninitialize.call_count == 1


def test_close_uninit_error_is_swallowed(fakes, mon):
    WinActivityMonitor().close()  # never initialised: nothing to undo
    assert fakes.comtypes.CoUninitialize.call_count == 0
    mon.snapshot()
    fakes.comtypes.CoUninitialize.side_effect = OSError("RPC_E_CHANGED_MODE")
    mon.close()
    mon.snapshot()
    assert fakes.comtypes.CoInitialize.call_count == 2  # state was reset despite the error
