"""Windows tray "Name speakers…" / "Forget all voices…" with fake dialogs and a fake
Controller (the Controller's voice workers are tested in test_voices_menu.py). pystray and
Pillow are stubbed; the Tk dialog itself is checked by hand on Windows.

Stories: US-26.
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import time
import types
from pathlib import Path

import pytest


def _menu(*items):
    return items


_menu.SEPARATOR = None  # type: ignore[attr-defined]


@pytest.fixture
def tray(monkeypatch):
    stubs = {
        "pystray": types.SimpleNamespace(
            Icon=lambda *a, **k: None, MenuItem=lambda *a, **k: (*a, k), Menu=_menu
        ),
        "PIL": types.SimpleNamespace(Image=None, ImageDraw=None),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "recandcc.ui.tray", raising=False)
    yield importlib.import_module("recandcc.ui.tray")
    sys.modules.pop("recandcc.ui.tray", None)  # don't leak the module built on stubs


def _app(tray):
    app = tray.TrayApp.__new__(tray.TrayApp)  # only the menu and events: no icon or runtime
    app.ctrl = types.SimpleNamespace(events=queue.Queue())
    app._live_id, app.notices = None, []
    app._notify = lambda title, msg: app.notices.append((title, msg))
    return app


SESSION = Path("/t/2026-10-27/140312_zoom-us")
SPEAKERS = [("Speaker 1", ["hi"]), ("Speaker 2", ["ok"])]


def _wait(pred, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.01)


def _voices_app(tray, monkeypatch, answers, speaker_names=True, recent=True):
    """A tray app whose dialogs answer from ``answers`` (one per dialog, in order; a
    dialog past the end answers None = Cancel). Returns (app, controller calls, dialogs)."""
    app = _app(tray)
    app._naming_open, app._naming_lock = False, threading.Lock()
    app._naming_since, app._forget_open = 0.0, False
    calls: list = []
    asked: list = []
    app.ctrl.speaker_names_enabled = speaker_names
    app.ctrl.recent = [SESSION / "transcript.md"] if recent else []
    app.ctrl.speakers_async = lambda d: calls.append(("speakers", d))
    app.ctrl.name_speaker_async = lambda *a: calls.append(("name", *a))
    app.ctrl.forget_voices_async = lambda: calls.append(("forget",))

    def dialog(title, text, **kw):
        asked.append((text, kw))
        return answers.pop(0) if answers else None

    app._dialog = dialog
    return app, calls, asked


def _top_items(app):
    return {i[0]: i for i in app._menu() if i and isinstance(i[0], str)}


def _name_item(app):
    recent = _top_items(app)["Recent transcripts"][1][0]  # Menu(recent) -> (recent,)
    return next(i for i in recent() if i and i[0] == "Name speakers…")


@pytest.mark.parametrize(("names_on", "has_recent"), [(True, True), (False, True), (True, False)])
def test_name_speakers_item_is_greyed_out_unless_enabled(tray, monkeypatch, names_on, has_recent):
    app, calls, asked = _voices_app(tray, monkeypatch, [0], names_on, has_recent)
    item = _name_item(app)
    assert item[2]["enabled"] is (names_on and has_recent)
    if not item[2]["enabled"]:
        item[1]()  # a stale click on a greyed-out item does nothing either
        time.sleep(0.05)
        assert calls == asked == [] and not app._naming_open
    assert "Forget all voices…" in _top_items(app)  # like macOS: always offered


def test_tray_naming_flow_uses_the_shared_controller_and_keeps_names_out_of_notices(
    tray, monkeypatch
):
    app, calls, asked = _voices_app(tray, monkeypatch, [0, 1, "  Anna  "])
    _name_item(app)[1]()
    _wait(lambda: calls)
    assert calls == [("speakers", SESSION)] and app._naming_open
    assert asked[0] == (
        "Name the speakers of:",
        {"choices": ["2026-10-27 14:03 zoom-us"], "ok": "Next"},
    )
    app._name_speakers()  # a second flow while one is open: refused with a notice
    assert app.notices == [
        ("Naming already in progress", "Finish or cancel the open dialog first.")
    ]
    app.ctrl.events.put(("speakers", SESSION, SPEAKERS))
    app._drain()
    _wait(lambda: len(calls) == 2)
    assert calls[1] == ("name", SESSION, "Speaker 2", "Anna")
    assert asked[1][1]["choices"][1].startswith("2. Speaker 2:")
    assert asked[2] == ("Name for Speaker 2:", {"entry": True, "ok": "Save"})
    assert app._naming_open  # held until the rename is done
    app.ctrl.events.put(("named", "Speaker 2", "Anna", None))
    app._drain()
    assert not app._naming_open
    assert all("Anna" not in t + m and "Speaker 2" not in t + m for t, m in app.notices)


@pytest.mark.parametrize("answers", [[None], [0, None], [0, 0, None], [0, 0, "   "]])
def test_cancelling_any_naming_dialog_names_nobody_and_frees_the_guard(tray, monkeypatch, answers):
    app, calls, asked = _voices_app(tray, monkeypatch, list(answers))
    app._name_speakers()
    _wait(lambda: asked)
    if answers[0] is not None:
        _wait(lambda: calls)
        app.ctrl.events.put(("speakers", SESSION, SPEAKERS))
        app._drain()
    _wait(lambda: not app._naming_open)
    assert len(asked) == len(answers)
    assert [c[0] for c in calls] == ([] if answers[0] is None else ["speakers"])


def test_no_speakers_or_unreadable_transcript_ends_the_flow_with_a_notice(tray, monkeypatch):
    app, _calls, _asked = _voices_app(tray, monkeypatch, [])
    for result in ([], OSError("gone")):
        app._naming_open = True
        app.ctrl.events.put(("speakers", SESSION, result))
        app._drain()
        assert not app._naming_open
    assert [t for t, _m in app.notices] == ["No speakers to name", "Can't read the speakers"]


@pytest.mark.parametrize(("answer", "forgets"), [(True, True), (None, False)])
def test_forget_all_voices_needs_an_explicit_confirmation(tray, monkeypatch, answer, forgets):
    app, calls, asked = _voices_app(tray, monkeypatch, [answer])
    _top_items(app)["Forget all voices…"][1]()
    _wait(lambda: asked)
    if forgets:
        _wait(lambda: calls)
    else:
        time.sleep(0.05)
    assert asked == [(tray.FORGET_VOICES_TEXT, {"ok": "Forget all voices", "danger": True})]
    assert calls == ([("forget",)] if forgets else [])
    app.ctrl.events.put(("forgot", 2))
    app.ctrl.events.put(("forgot", OSError("busy")))
    app._drain()
    assert [t for t, _m in app.notices] == ["Voices deleted", "Can't delete voices"]


def test_tk_dialog_without_tk_counts_as_cancel(tray, monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)  # import fails, as without Tk
    assert tray.tk_dialog("t", "x", ok="Forget all voices", danger=True) is None


def test_a_lost_naming_flow_stops_blocking_after_the_stale_timeout(tray, monkeypatch):
    app, _calls, _asked = _voices_app(tray, monkeypatch, [])
    clock = [1000.0]
    monkeypatch.setattr(tray.time, "monotonic", lambda: clock[0])
    assert app._begin_naming()
    assert not app._begin_naming()  # a flow is open
    clock[0] += tray.NAMING_STALE_S + 1  # its worker never reported back
    assert app._begin_naming()


def test_a_second_forget_click_does_not_stack_a_confirmation(tray, monkeypatch):
    app, calls, asked = _voices_app(tray, monkeypatch, [])
    gate = threading.Event()
    shown: list = []

    def slow_dialog(*a, **k):
        shown.append(a)
        gate.wait(5)
        return None

    app._dialog = slow_dialog
    item = _top_items(app)["Forget all voices…"][1]
    item()
    _wait(lambda: shown)
    item()  # while the first confirmation is still up
    time.sleep(0.05)
    assert len(shown) == 1
    gate.set()
    _wait(lambda: not app._forget_open)
    assert calls == []
