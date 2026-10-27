"""Windows tray wiring that needs no display: the "Search transcripts…" item, ``search``
events reaching the Tk search window, and that window's submit/open hooks; the export and
"Delete audio after" items. pystray and Pillow are stubbed; the Tk widgets themselves are
checked by hand on Windows.

Stories: US-15, US-20, US-25, US-28.
"""

from __future__ import annotations

import importlib
import queue
import sys
import threading
import types
from datetime import datetime

import pytest

from recandcc.search import Hit


class _Item(tuple):
    """A stub pystray.MenuItem: its positional args, plus ``kw`` (checked=, radio=...)."""

    kw: dict


def _item(*a, **k):
    item = _Item(a)
    item.kw = k
    return item


def _menu(*items):
    return items


_menu.SEPARATOR = None  # type: ignore[attr-defined]


@pytest.fixture
def tray(monkeypatch):
    stubs = {
        "pystray": types.SimpleNamespace(Icon=lambda *a, **k: None, MenuItem=_item, Menu=_menu),
        "PIL": types.SimpleNamespace(Image=None, ImageDraw=None),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "recandcc.ui.tray", raising=False)
    yield importlib.import_module("recandcc.ui.tray")
    sys.modules.pop("recandcc.ui.tray", None)  # don't leak the module built on stubs


class FakeWindow:
    def __init__(self):
        self.posted: list[tuple] = []

    def post(self, *cmd) -> None:
        self.posted.append(cmd)


def _app(tray):
    app = tray.TrayApp.__new__(tray.TrayApp)  # only the search wiring: no icon or runtime
    app.ctrl = types.SimpleNamespace(events=queue.Queue())
    app._live_id, app.notices, app.search = None, [], FakeWindow()
    app._notify = lambda title, msg: app.notices.append((title, msg))
    return app


def test_search_menu_item_opens_the_window(tray):
    app = _app(tray)
    actions = {i[0]: i[1] for i in app._menu() if i and isinstance(i[0], str)}
    actions["Search transcripts…"]()
    assert app.search.posted == [("ask",)]


def test_search_events_reach_the_window_including_failures(tray):
    app = _app(tray)
    hit, err = object(), RuntimeError("disk")
    for hits in ([hit], [], err):
        app.ctrl.events.put(("search", "budget", hits))
    app._drain()
    assert app.search.posted == [
        ("results", "budget", [hit]),
        ("results", "budget", []),
        ("results", "budget", err),
    ]
    assert app.notices == []  # the window shows the status; no toast carries the query


def test_search_window_submits_queries_and_opens_matches(tray, tmp_path, monkeypatch):
    queries, opened, done = [], [], threading.Event()
    win = tray.TkSearch(queries.append)
    shown: list[str] = []
    win._status = shown.append
    win.submit("  ")
    win.submit(" budget ")
    assert queries == ["budget"]
    assert shown == [tray.SEARCHING]
    monkeypatch.setattr(tray, "open_path", lambda p: (opened.append(p), done.set()))
    win.hits = [Hit(tmp_path, "s1", "", 1.0, "Me", "en", "x")]
    win.open(5)  # no such row: nothing
    win.open(0)
    assert done.wait(5)
    assert opened == [tmp_path]  # no transcript.md yet: its session folder


def test_search_window_stops_queueing_when_tk_fails(tray, monkeypatch):
    monkeypatch.setitem(sys.modules, "tkinter", None)  # import fails, as without Tk
    win = tray.TkSearch(lambda q: None)
    win.post("ask")
    assert win._thread is not None
    win._thread.join(5)
    assert win.failed
    win.post("ask")
    assert win._q.qsize() == 1  # nothing queued after the failure


def test_notes_menu_item_and_events_reach_the_notes_window(tray):
    app = _app(tray)
    calls: list[str] = []
    app.ctrl.notes_async = lambda: calls.append("scan")
    app.notes = FakeWindow()
    actions = {i[0]: i[1] for i in app._menu() if i and isinstance(i[0], str)}
    actions["Meeting notes…"]()
    assert calls == ["scan"]
    app.ctrl.events.put(("notes", []))
    app._drain()
    assert app.notes.posted == [("notes", [])]


def test_notes_window_actions(tray, tmp_path, monkeypatch):
    from recandcc.ui.notes import Note

    done = threading.Event()
    opened, asked, acted = [], [], []
    ctrl = types.SimpleNamespace(note_action_async=lambda *a: acted.append(a))
    monkeypatch.setattr(tray, "open_path", lambda p: (opened.append(p), done.set()))
    win = tray.TkNotes(ctrl)
    win.notes = [Note(tmp_path, "zoom.us", datetime(2026, 10, 26), 60.0, True)]
    answers = {"rename": "Budget", "delete": False}

    def ask(kind, note):
        asked.append(kind)
        return answers[kind]

    win.act(3, "Rename…", ask)  # no such row: nothing
    win.act(0, "Open summary", ask)  # no summary.md: nothing
    win.act(0, "Rename…", ask)
    win.act(0, "Delete…", ask)  # not confirmed
    answers["delete"] = True
    win.act(0, "Delete…", ask)
    win.act(0, "Reveal folder", ask)
    assert done.wait(5)
    assert asked == ["rename", "delete", "delete"]
    assert acted == [(tmp_path, "rename", "Budget"), (tmp_path, "delete")]
    assert opened == [tmp_path]


def test_export_item_runs_the_controller_export(tray):
    app = _app(tray)
    calls: list[str] = []
    app.ctrl.export_async = lambda: calls.append("export")  # spawns its own worker
    actions = {i[0]: i[1] for i in app._menu() if i and isinstance(i[0], str)}
    actions["Export latest transcript (DOCX + PDF)"]()
    assert calls == ["export"]
    app.ctrl.events.put(("notice", "Export", "Saved transcript.docx"))
    app._drain()
    assert app.notices == [("Export", "Saved transcript.docx")]  # the tray's notify path


def test_retention_submenu_saves_the_choice_and_moves_the_check(tray):
    app = _app(tray)
    app.cfg = types.SimpleNamespace(transcription=types.SimpleNamespace(audio_retention_days=0))
    saved: list[int] = []

    def set_retention(days):
        saved.append(days)
        app.cfg.transcription.audio_retention_days = days

    app.ctrl.set_retention = set_retention
    refreshed: list[int] = []
    app.icon = types.SimpleNamespace(update_menu=lambda: refreshed.append(1))
    submenu = {i[0]: i[1] for i in app._menu() if i and isinstance(i[0], str)}

    def rows():
        (build,) = submenu["Delete audio after"]  # a callable menu, rebuilt on each update
        return {i[0]: i for i in build()}

    def checked():
        return [title for title, i in rows().items() if i.kw["checked"](i)]

    assert checked() == ["Never (keep audio)"]
    assert all(i.kw["radio"] for i in rows().values())
    rows()["30 days"][1]()
    assert saved == [30]
    assert refreshed == [1]  # the cached Win32 menu is rebuilt at once
    assert checked() == ["30 days"]
    app.cfg.transcription.audio_retention_days = 5  # set by hand in config.toml
    extra = rows()["5 days (from settings)"]
    assert extra[1] is None and extra.kw["enabled"] is False
    assert checked() == ["5 days (from settings)"]
