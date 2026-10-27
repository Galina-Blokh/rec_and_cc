"""Global shortcut: combo parsing, toggle/debounce in the Controller, helper restart logic.

Stories: US-05, US-18.
"""

from __future__ import annotations

import queue
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from recandcc.engine import IDLE, RECORDING, EngineState
from recandcc.hotkey import (
    MOD_ALT,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    MOD_WIN,
    Hotkey,
    HotkeyError,
    HotkeyListener,
    parse_hotkey,
)
from recandcc.models import Mode
from recandcc.native import REPO_ROOT


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize(
    "text, mods, key",
    [
        ("<cmd>+<shift>+r", {"cmd", "shift"}, "r"),
        ("<Command>+<Option>+5", {"cmd", "alt"}, "5"),
        ("ctrl+alt+space", {"ctrl", "alt"}, "space"),
        (" <ctrl> + <shift> + F12 ", {"ctrl", "shift"}, "f12"),
        ("f9", set(), "f9"),
        ("<super>+<opt>+<control>+z", {"cmd", "alt", "ctrl"}, "z"),
    ],
)
def test_parse_ok(text, mods, key):
    hk = parse_hotkey(text)
    assert hk == Hotkey(frozenset(mods), key)


@pytest.mark.parametrize("text", ["", "   ", None])
def test_parse_empty_disables(text):
    assert parse_hotkey(text) is None


@pytest.mark.parametrize(
    "text", ["r", "<cmd>+<shift>", "<cmd>+r+t", "<cmd>+<foo>+r", "<cmd>++r", "<cmd>+f13"]
)
def test_parse_invalid(text):
    with pytest.raises(HotkeyError):
        parse_hotkey(text)


def test_mac_combo_and_label():
    hk = parse_hotkey("<shift>+<cmd>+r")
    assert hk.mac_combo() == "<shift>+<cmd>+r"  # Apple order ⌃⌥⇧⌘
    assert hk.label("darwin") == "⇧⌘R"  # what macOS menus show for ⌘⇧R
    assert parse_hotkey("<cmd>+<ctrl>+<alt>+<shift>+space").label("darwin") == "⌃⌥⇧⌘Space"
    assert parse_hotkey("<win>+<alt>+k").mac_combo() == "<alt>+<cmd>+k"  # win = ⌘ on macOS


def test_windows_mapping():
    hk = parse_hotkey("<cmd>+<shift>+r")
    assert hk.label("win32") == "Ctrl+Shift+R"  # <cmd> = Ctrl on Windows
    assert hk.win32() == (MOD_NOREPEAT | MOD_CONTROL | MOD_SHIFT, ord("R"))
    assert parse_hotkey("<win>+<alt>+7").win32() == (MOD_NOREPEAT | MOD_WIN | MOD_ALT, ord("7"))
    assert parse_hotkey("f1").win32() == (MOD_NOREPEAT, 0x70)
    assert parse_hotkey("<ctrl>+f12").win32()[1] == 0x7B
    assert parse_hotkey("<alt>+space").win32()[1] == 0x20
    assert parse_hotkey("<cmd>+<ctrl>+q").label("win32") == "Ctrl+Q"


# ------------------------------------------------------------------ controller toggle
class FakeEngine:
    def __init__(self):
        self.status = IDLE
        self.calls: list[str] = []

    @property
    def state(self):
        return EngineState(self.status, Mode.ASK)

    def start_manual(self):
        self.calls.append("start")
        self.status = RECORDING

    def stop(self):
        self.calls.append("stop")
        self.status = IDLE


class FakeListener:
    def __init__(self, text, on_press, on_error):
        self.text, self.on_press, self.on_error = text, on_press, on_error
        self.closed = False

    def close(self):
        self.closed = True


def make_ctrl(tmp_path):
    from recandcc.config import Config
    from recandcc.ui.common import Controller

    class Monitor:
        def snapshot(self):
            return []

    class Q:
        def submit(self, *_a, **_k):
            pass

        def pending(self):
            return 0

    ctrl = Controller(
        Config(output_dir=tmp_path),
        monitor=Monitor(),
        recorder=object(),
        tq=Q(),
        start_threads=False,
    )
    ctrl.engine = FakeEngine()
    t = [100.0]
    ctrl.clock = lambda: t[0]
    return ctrl, t


def notices(ctrl):
    out = []
    while True:
        try:
            ev = ctrl.events.get_nowait()
        except queue.Empty:
            return out
        if ev[0] == "notice":
            out.append(ev[1])


def test_toggle_and_debounce(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    ctrl, t = make_ctrl(tmp_path)
    made: list[FakeListener] = []
    ctrl.apply_hotkey(
        "<cmd>+<shift>+r", factory=lambda *a: made.append(FakeListener(*a)) or made[-1]
    )
    assert ctrl.hotkey_label == "⇧⌘R"
    press = made[0].on_press

    press()
    assert ctrl.engine.calls == ["start"]
    assert notices(ctrl) == ["Recording started (⇧⌘R to stop)"]
    t[0] += 0.2
    press()  # bounce: ignored
    assert ctrl.engine.calls == ["start"]
    t[0] += 0.4
    press()
    assert ctrl.engine.calls == ["start", "stop"]
    assert notices(ctrl) == ["Recording stopped"]
    t[0] += 0.6
    press()
    assert ctrl.engine.calls == ["start", "stop", "start"]


def test_stops_any_session(tmp_path):
    ctrl, _ = make_ctrl(tmp_path)
    ctrl.engine.status = RECORDING  # e.g. an auto-detected call
    ctrl.hotkey_pressed()
    assert ctrl.engine.calls == ["stop"]


def test_reload_disable_and_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    ctrl, _ = make_ctrl(tmp_path)
    made: list[FakeListener] = []

    def factory(*a):
        made.append(FakeListener(*a))
        return made[-1]

    ctrl.apply_hotkey("<cmd>+<shift>+r", factory=factory)
    ctrl.apply_hotkey("<ctrl>+<alt>+f5", factory=factory)  # config changed
    assert made[0].closed and not made[1].closed
    assert ctrl.hotkey_label == "⌃⌥F5"

    made[1].on_error("taken", "already registered")
    assert ctrl.hotkey_label == ""
    assert any("used by another app" in n or "Shortcut" in n for n in notices(ctrl))

    ctrl.apply_hotkey("", factory=factory)  # disabled
    assert made[1].closed and len(made) == 2 and ctrl.hotkey_label == ""

    ctrl.apply_hotkey("<cmd>+bogus", factory=factory)  # invalid: toast, no listener
    assert len(made) == 2 and notices(ctrl) == ["Shortcut not set"]


def test_failure_toast_text(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    ctrl, _ = make_ctrl(tmp_path)
    got = []
    ctrl.apply_hotkey("<cmd>+<shift>+r", factory=lambda *a: got.append(FakeListener(*a)) or got[0])
    got[0].on_error("taken", "x")
    ev = ctrl.events.get_nowait()
    assert ev[0] == "notice"
    assert ev[2] == "Shortcut ⇧⌘R is used by another app — change `hotkey` in settings"


def test_shutdown_closes_listener(tmp_path):
    ctrl, _ = make_ctrl(tmp_path)
    made = []
    ctrl.apply_hotkey(
        "<cmd>+<shift>+r", factory=lambda *a: made.append(FakeListener(*a)) or made[0]
    )
    ctrl.engine.shutdown = lambda: None
    ctrl.shutdown()
    assert made[0].closed


# ------------------------------------------------------------------ listener (fake helper)
FAKE_HELPER = textwrap.dedent(
    """
    import json, sys, pathlib
    state = pathlib.Path(sys.argv[1])
    mode = sys.argv[2]
    combo = sys.argv[3]
    runs = int(state.read_text()) if state.exists() else 0
    state.write_text(str(runs + 1))
    def out(o):
        print(json.dumps(o), flush=True)
    if mode == "taken":
        out({"error": "taken", "message": combo + " is already registered by another app"})
        sys.exit(2)
    if mode == "invalid":
        out({"error": "invalid", "message": "bad"})
        sys.exit(2)
    out({"hotkey": "registered", "combo": combo})
    out({"hotkey": "pressed"})
    if mode == "crash" and runs < 2:
        sys.exit(1)  # die twice, then stay up
    sys.stdin.read()  # until the listener closes our stdin
    """
)


def fake_listener(tmp_path, mode, **kw):
    script = tmp_path / "fake_helper.py"
    script.write_text(FAKE_HELPER)
    state = tmp_path / "runs"
    presses, errors = [], []
    lst = HotkeyListener(
        parse_hotkey("<cmd>+<shift>+r"),
        lambda: presses.append(1),
        lambda k, m: errors.append((k, m)),
        command=[sys.executable, str(script), str(state), mode],
        backoff_s=0.05,
        max_backoff_s=0.1,
        **kw,
    )
    return lst, presses, errors, state


def wait_for(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_listener_restarts_after_crash(tmp_path):
    lst, presses, errors, state = fake_listener(tmp_path, "crash")
    try:
        assert wait_for(lambda: lst.starts >= 3 and lst.registered.is_set() and len(presses) >= 3)
        assert errors == []
    finally:
        lst.close()
    assert not lst._thread.is_alive()
    assert int(state.read_text()) == 3  # no further restarts after close


def test_listener_close_stops_helper(tmp_path):
    lst, presses, _, state = fake_listener(tmp_path, "ok")
    assert wait_for(lambda: presses == [1])
    proc = lst._proc
    lst.close()
    assert proc is not None and proc.poll() is not None
    assert lst.starts == 1


def test_listener_taken_reports_once_and_retries(tmp_path):
    lst, presses, errors, state = fake_listener(tmp_path, "taken", taken_retry_s=0.05)
    try:
        assert wait_for(lambda: lst.starts >= 3)
    finally:
        lst.close()
    assert len(errors) == 1 and errors[0][0] == "taken"
    assert presses == []


def test_listener_invalid_gives_up(tmp_path):
    lst, _, errors, state = fake_listener(tmp_path, "invalid")
    assert wait_for(lambda: not lst._thread.is_alive())
    assert errors == [("invalid", "bad")] and lst.starts == 1
    lst.close()


# ------------------------------------------------------------------ real helper (macOS)
HELPER = REPO_ROOT / "bin" / "rcc-audiotap"
needs_helper = pytest.mark.skipif(
    sys.platform != "darwin" or not Path(HELPER).exists(), reason="macOS helper not built"
)


@needs_helper
def test_real_helper_registers_and_exits_on_close(tmp_path):
    # An unusual combo, so neither another app nor a running RecAndCC holds it.
    lst = HotkeyListener(parse_hotkey("<ctrl>+<alt>+<shift>+<cmd>+f11"), lambda: None)
    try:
        assert lst.registered.wait(10), "helper did not register the shortcut"
    finally:
        lst.close()
    assert lst._proc is None or lst._proc.poll() is not None


@needs_helper
def test_real_helper_reports_taken():
    combo = parse_hotkey("<ctrl>+<alt>+<shift>+<cmd>+f10")
    first = HotkeyListener(combo, lambda: None)
    errors = []
    done = threading.Event()
    try:
        assert first.registered.wait(10)
        second = HotkeyListener(combo, lambda: None, lambda k, m: (errors.append(k), done.set()))
        try:
            assert done.wait(10)
        finally:
            second.close()
    finally:
        first.close()
    assert errors == ["taken"]
