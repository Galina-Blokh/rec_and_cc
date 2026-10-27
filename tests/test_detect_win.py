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
    state.walks = 0

    def process_iter(attrs, ad_value="unset"):
        assert ad_value is None  # else protected processes yield AccessDenied in .info
        state.walks += 1
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


TEAMS_DIR = r"C:\Program Files\WindowsApps\MSTeams_24243.1309.3132.617_x64__8wekyb3d8bbwe"
ns = types.SimpleNamespace
TEAMS = ns(pid=300, info={"name": "ms-teams.exe", "exe": TEAMS_DIR + r"\ms-teams.exe"})
STARTER = ns(pid=299, info={"name": "MSTeams_autostarter.exe", "exe": TEAMS_DIR + r"\a.exe"})
UPDATER = ns(pid=298, info={"name": "updater.exe", "exe": TEAMS_DIR + r"\updater.exe"})


class AccessDenied(Exception):
    """What psutil puts into .info for protected processes when ad_value is not given."""


def teams_on_mic(fakes):
    fakes.winreg.tree = mic_tree({"MSTeams_8wekyb3d8bbwe": IN_USE}, {})


def test_packaged_mic_user_maps_to_its_running_exe(fakes, mon):
    """The consent store says "MSTeams_8wekyb3d8bbwe"; WASAPI and the app lists say
    "ms-teams.exe". Both must end up as one app, keyed by the exe name."""
    fakes.winreg.tree = mic_tree(
        {"MSTeams_8wekyb3d8bbwe": IN_USE, "Clipchamp.Clipchamp_yxz26nhyzhsrt": IN_USE}, {}
    )
    fakes.sessions = [FakeSession(proc("ms-teams.exe", 300))]
    fakes.procs = [
        ns(pid=1, info={"name": "System", "exe": None}),
        ns(pid=2, info={"name": "x.exe", "exe": r"C:\WindowsApps\x.exe"}),  # no package dir
        ns(pid=3, info={"name": AccessDenied(), "exe": AccessDenied()}),  # not str: skipped
        STARTER,
        TEAMS,
    ]
    assert by_name(mon.snapshot()) == {
        "ms-teams.exe": AudioProcess(300, "ms-teams.exe", "ms-teams.exe", True, True),
        "clipchamp.clipchamp": AudioProcess(0, *["clipchamp.clipchamp"] * 2, True, False),
    }


@pytest.mark.parametrize("order", [[STARTER, UPDATER, TEAMS], [TEAMS, UPDATER, STARTER]])
def test_mic_only_package_prefers_the_app_over_helpers(fakes, mon, order):
    teams_on_mic(fakes)
    fakes.procs = order
    assert [(p.pid, p.bundle_id) for p in mon.snapshot()] == [(300, "ms-teams.exe")]


def test_package_choice_is_stable_and_sticky(fakes, mon, monkeypatch):
    """A key that flips between polls would look like a new app and end the session."""
    now = [0.0]
    monkeypatch.setattr(win, "time", ns(monotonic=lambda: now[0]))
    teams_on_mic(fakes)
    starter = ns(pid=299, info={"name": "starter.exe", "exe": TEAMS_DIR + r"\starter.exe"})
    widget = ns(pid=297, info={"name": "widget.exe", "exe": TEAMS_DIR + r"\widget.exe"})
    fakes.sessions = [FakeSession(proc("widget.exe", 297))]
    fakes.procs = [starter, widget]
    assert [p.bundle_id for p in mon.snapshot()] == ["widget.exe"]  # has the WASAPI session
    fakes.sessions = []  # playback ended: keep the key rather than re-pick "starter.exe"
    for order in ([widget, starter], [starter, widget], [widget, starter]):
        now[0] += 11  # past the TTL: the table is rebuilt in a different order
        fakes.procs = order
        assert [p.bundle_id for p in mon.snapshot()] == ["widget.exe"]
    fakes.procs = [starter]  # the sticky exe exited: fall back to the sorted choice
    now[0] += 11
    assert [p.bundle_id for p in mon.snapshot()] == ["starter.exe"]


def test_helper_playing_audio_does_not_take_over_the_sticky_key(fakes, mon):
    teams_on_mic(fakes)
    fakes.procs = [STARTER, TEAMS]
    assert [p.bundle_id for p in mon.snapshot()] == ["ms-teams.exe"]
    fakes.sessions = [FakeSession(proc("msteams_autostarter.exe", 299))]  # helper plays a sound
    snap = by_name(mon.snapshot())
    assert snap["ms-teams.exe"].input_active  # the call keeps its key
    assert not snap["msteams_autostarter.exe"].input_active
    fakes.winreg.tree = mic_tree({}, {})  # off the mic: the remembered choice is dropped
    mon.snapshot()
    assert mon._chosen == {}


def test_app_replaces_a_helper_that_was_on_the_mic_first(fakes, mon, monkeypatch):
    """Cold start: only the autostarter runs at first. The table is re-walked quickly while
    the package has no exe of its own, so ms-teams.exe takes the key within the engine's
    debounce, even though the helper was the sticky choice and stays alive."""
    now = [0.0]
    monkeypatch.setattr(win, "time", ns(monotonic=lambda: now[0]))
    teams_on_mic(fakes)
    fakes.procs = [STARTER]
    assert [p.bundle_id for p in mon.snapshot()] == ["msteams_autostarter.exe"]
    fakes.procs = [STARTER, TEAMS]
    now[0] += 1
    assert [p.bundle_id for p in mon.snapshot()] == ["msteams_autostarter.exe"]  # cached
    now[0] += win._UNSETTLED_TTL_S
    fakes.sessions = [FakeSession(proc("MSTeams_autostarter.exe", 299))]  # helper plays
    snap = by_name(mon.snapshot())
    assert snap["ms-teams.exe"].input_active and snap["ms-teams.exe"].pid == 300
    assert not snap["msteams_autostarter.exe"].input_active
    assert fakes.walks == 2
    now[0] += win._UNSETTLED_TTL_S
    mon.snapshot()
    assert fakes.walks == 2  # settled on the app: back to the long TTL


def test_package_playing_audio_maps_without_a_walk(fakes, mon):
    """A typical Teams call: ms-teams.exe has a WASAPI session, so no process walk."""
    teams_on_mic(fakes)
    fakes.procs = [STARTER]  # stale table contents must not matter
    fakes.sessions = [FakeSession(proc("ms-teams.exe", 300))]
    assert by_name(mon.snapshot()) == {
        "ms-teams.exe": AudioProcess(300, "ms-teams.exe", "ms-teams.exe", True, True)
    }
    assert fakes.walks == 0


def test_unlisted_package_keeps_its_key(fakes, mon, monkeypatch):
    """A walk that misses the package (failed, exe path unreadable) must not flip the key
    from ms-teams.exe to the family prefix: the engine would see the call end."""
    teams_on_mic(fakes)
    fakes.procs = [TEAMS]
    assert [p.bundle_id for p in mon.snapshot()] == ["ms-teams.exe"]

    def boom(attrs, ad_value=None):
        raise RuntimeError("walk failed")

    mon._table = None
    monkeypatch.setattr(sys.modules["psutil"], "process_iter", boom)
    assert [(p.pid, p.bundle_id) for p in mon.snapshot()] == [(0, "ms-teams.exe")]  # sticky


def test_unlisted_package_without_sticky_is_found_by_name(fakes, mon):
    teams_on_mic(fakes)
    fakes.procs = [ns(pid=300, info={"name": "ms-teams.exe", "exe": None})]  # path unreadable
    assert [(p.pid, p.bundle_id) for p in mon.snapshot()] == [(300, "ms-teams.exe")]


def test_no_walk_when_every_mic_user_already_plays_audio(fakes, mon):
    """A typical call: the app is in a WASAPI session and on the mic, as is our recorder."""
    fakes.sessions = [FakeSession(proc("Zoom.exe", 101))]
    fakes.winreg.tree = mic_tree(
        {}, {"C:#Program Files#Zoom#bin#Zoom.exe": IN_USE, OWN_EXE.replace("\\", "#"): IN_USE}
    )
    assert by_name(mon.snapshot()) == {
        "zoom.exe": AudioProcess(101, "zoom.exe", "Zoom.exe", True, True)
    }
    assert fakes.walks == 0


@pytest.mark.parametrize("flip", [False, True])
def test_choice_without_sticky_or_audio_is_deterministic(fakes, mon, flip):
    teams_on_mic(fakes)
    aaa = ns(pid=290, info={"name": "aaa.exe", "exe": TEAMS_DIR + r"\aaa.exe"})
    zeta = ns(pid=291, info={"name": "zeta.exe", "exe": TEAMS_DIR + r"\zeta.exe"})
    order = [aaa, TEAMS] if not flip else [TEAMS, aaa]
    fakes.procs = order  # the package-name match beats a non-helper that sorts first
    assert [p.bundle_id for p in mon.snapshot()] == ["ms-teams.exe"]
    other = win.WinActivityMonitor()
    other._own_exe = mon._own_exe
    fakes.procs = [zeta, aaa] if not flip else [aaa, zeta]  # no name match: sorted order
    assert [p.bundle_id for p in other.snapshot()] == ["aaa.exe"]


def test_failed_walk_is_not_cached(fakes, mon, monkeypatch):
    teams_on_mic(fakes)
    fakes.procs = [TEAMS]
    calls = []

    def flaky(attrs, ad_value=None):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("walk failed")  # partway: nothing usable this poll
        yield from fakes.procs

    monkeypatch.setattr(sys.modules["psutil"], "process_iter", flaky)
    mon.snapshot()
    assert [p.bundle_id for p in mon.snapshot()] == ["ms-teams.exe"]  # walked again
    assert len(calls) == 2


def test_process_table_is_cached(fakes, mon, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(win, "time", ns(monotonic=lambda: now[0]))
    teams_on_mic(fakes)
    fakes.procs = [TEAMS]
    mon.snapshot()
    now[0] += 1
    mon.snapshot()
    assert fakes.walks == 1  # stable mic set: one walk, not one per poll
    fakes.winreg.tree = mic_tree({"MSTeams_8wekyb3d8bbwe": IN_USE}, {"C:#Zoom.exe": IN_USE})
    mon.snapshot()
    assert fakes.walks == 2  # mic set changed
    now[0] += win._TABLE_TTL_S
    mon.snapshot()
    assert fakes.walks == 3  # TTL expired
    fakes.winreg.tree = mic_tree({}, {})
    mon.snapshot()
    assert fakes.walks == 3  # nobody on the mic: no walk at all


def test_pid_lookup(fakes, mon, monkeypatch):
    fakes.procs = [ns(pid=1, info={"name": None}), ns(pid=2, info={"name": "ZOOM.EXE"})]
    by_family, by_name = mon._process_table(frozenset({"a"}))
    assert by_family == {} and by_name == {"zoom.exe": 2}

    def boom(attrs, ad_value=None):
        raise RuntimeError("psutil broke")

    monkeypatch.setattr(sys.modules["psutil"], "process_iter", boom)
    assert mon._process_table(frozenset({"b"})) == ({}, {})


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
