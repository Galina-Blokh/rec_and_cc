"""Windows activity detection with fake pycaw / comtypes / winreg / psutil (runs anywhere).

Stories: US-01, US-17, US-18.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import types

import pytest

from recandcc.config import Config, DetectionConfig
from recandcc.detect import win
from recandcc.detect.win import WinActivityMonitor
from recandcc.engine import Engine
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


class FakeComtypes(types.ModuleType):
    """COM apartments are per thread: every CoInitialize needs a CoUninitialize there."""

    def __init__(self):
        super().__init__("comtypes")
        self.depth: collections.Counter = collections.Counter()
        self.inits, self.unbalanced, self.fail_uninit = 0, [], False

    def CoInitialize(self):
        self.inits += 1
        self.depth[threading.get_ident()] += 1

    def CoUninitialize(self):
        if self.fail_uninit:
            raise OSError("RPC_E_CHANGED_MODE")
        tid = threading.get_ident()
        if self.depth[tid] == 0:
            self.unbalanced.append(tid)
        self.depth[tid] -= 1

    def balanced(self) -> bool:
        return not self.unbalanced and not any(self.depth.values())


IN_USE = {"LastUsedTimeStart": 133_000_000, "LastUsedTimeStop": 0}
STOPPED = {"LastUsedTimeStart": 133_000_000, "LastUsedTimeStop": 133_000_500}


@pytest.fixture
def fakes(monkeypatch):
    """Install fake Windows modules; tests fill in sessions, registry and processes."""
    state = types.SimpleNamespace(sessions=[], procs=[], comtypes=FakeComtypes())
    pycaw = types.ModuleType("pycaw")
    pycaw_pycaw = types.ModuleType("pycaw.pycaw")
    pycaw_pycaw.AudioUtilities = types.SimpleNamespace(GetAllSessions=lambda: state.sessions)
    pycaw.pycaw = pycaw_pycaw
    psutil = types.ModuleType("psutil")

    def process_iter(attrs, ad_value="unset"):
        assert ad_value is None  # else protected processes yield AccessDenied in .info
        return iter(list(state.procs))

    psutil.process_iter = process_iter
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


class AccessDenied(Exception):
    """What psutil puts into .info for protected processes when ad_value is not given."""


def test_pid_lookup(fakes, mon, monkeypatch):
    ns = types.SimpleNamespace
    fakes.procs = [
        ns(pid=1, info={"name": None}),  # protected process (ad_value=None)
        ns(pid=3, info={"name": AccessDenied()}),  # not a str: skipped, walk continues
        ns(pid=2, info={"name": "ZOOM.EXE"}),
    ]
    assert mon._pid_for_exe("zoom.exe") == 2
    assert mon._pid_for_exe("absent.exe") == 0

    def boom(attrs, ad_value=None):
        raise RuntimeError("psutil broke")

    monkeypatch.setattr(sys.modules["psutil"], "process_iter", boom)
    assert mon._pid_for_exe("zoom.exe") == 0


def test_registry_unavailable_still_reports_output(fakes, mon, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "winreg", None)  # import winreg -> ImportError
    fakes.sessions = [FakeSession(proc("Zoom.exe", 101))]
    assert [p.bundle_id for p in mon.snapshot()] == ["zoom.exe"]
    assert "reading microphone consent store failed" in caplog.text


# ------------------------------------------------------------------ COM lifetime
def run_in_thread(fn):
    t = threading.Thread(target=fn)
    t.start()
    t.join()


def test_com_is_per_thread_and_balanced(fakes, mon):
    mon.snapshot()
    mon.snapshot()
    assert fakes.comtypes.inits == 1
    run_in_thread(lambda: (mon.snapshot(), mon.close(), mon.close()))  # what the engine does
    assert fakes.comtypes.inits == 2
    mon.close()
    mon.close()  # idempotent
    assert fakes.comtypes.balanced()
    run_in_thread(mon.snapshot)
    mon.close()  # another thread cannot release that apartment
    assert not fakes.comtypes.balanced()


def test_engine_releases_com_on_its_own_thread(fakes, mon, monkeypatch):
    polled = threading.Event()
    snapshot = mon.snapshot
    monkeypatch.setattr(mon, "snapshot", lambda: (snapshot(), polled.set())[0])
    eng = Engine(Config(detection=DetectionConfig(poll_interval_s=0.01)), mon, None)
    eng.start()
    try:
        assert polled.wait(5)
    finally:
        eng.shutdown()
    assert fakes.comtypes.inits == 1 and fakes.comtypes.balanced()


def test_close_uninit_error_is_swallowed(fakes, mon):
    WinActivityMonitor().close()  # never initialised: nothing to undo
    assert fakes.comtypes.balanced()
    mon.snapshot()
    fakes.comtypes.fail_uninit = True
    mon.close()
    mon.snapshot()
    assert fakes.comtypes.inits == 2  # state was reset despite the error
