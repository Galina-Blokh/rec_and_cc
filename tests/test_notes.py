"""Meeting notes: listing sessions newest first with their facts, summary excerpts when a
summary.md exists, renaming via session.json, deleting to the Trash or via discard_session,
and the Controller wiring (busy sessions, search index, a fresh list).

Stories: US-28.
"""

from __future__ import annotations

import json
import queue
import shutil
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.ui import notes
from recandcc.ui.common import Controller

T0 = datetime(2026, 10, 26, 14, 3, 12)


def make_session(out: Path, name: str, start: datetime, app: str | None = "zoom.us", **kw):
    d = out / start.strftime("%Y-%m-%d") / name
    d.mkdir(parents=True)
    kind = TriggerKind.CALL if app else TriggerKind.MANUAL
    s = SessionInfo(name, d, start, kind, app, start + timedelta(minutes=5), **kw)
    s.tracks = {Track.ME: "me.wav"}
    s.save()
    return s


def write_transcript(d: Path, speakers=("Me", "Others", "Me"), languages=("en", "ru")):
    segs = [{"start": i, "end": i + 1, "text": "x", "speaker": s, "language": "en"}
            for i, s in enumerate(speakers)]  # fmt: skip
    data = {"duration": 3725.0, "languages": list(languages), "segments": segs}
    (d / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    (d / "transcript.md").write_text("# t\n", encoding="utf-8")


def test_lists_sessions_newest_first_with_their_facts(tmp_path):
    old = make_session(tmp_path, "090000_zoom-us", T0 - timedelta(days=1), transcribed=True)
    new = make_session(tmp_path, "140312_manual", T0, app=None)
    gone = make_session(tmp_path, "150000_x", T0 + timedelta(hours=1))
    (gone.directory / SessionInfo.DISCARD_MARKER).touch()
    write_transcript(old.directory)
    (tmp_path / "2026-10-26" / "junk").mkdir()  # no session.json: skipped
    listed = notes.scan_notes(tmp_path)
    assert [n.directory for n in listed] == [new.directory, old.directory]
    assert notes.note_columns(listed[0]) == (
        "Manual recording", "2026-10-26 14:03", "05:00", "—", "—", "not yet",
    )  # fmt: skip
    assert notes.note_columns(listed[1]) == (
        "zoom.us", "2026-10-25 14:03", "1:02:05", "Me, Others", "en, ru", "transcribed",
    )  # fmt: skip
    assert "zoom.us" in notes.note_label(listed[1])
    assert notes.note_actions(listed[0]) == ["Reveal folder", "Rename…", "Delete…"]
    assert notes.scan_notes(tmp_path, limit=1)[0].directory == new.directory
    assert notes.scan_notes(tmp_path / "missing") == []


def test_summary_md_adds_an_excerpt_and_an_open_action(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    (s.directory / "summary.md").write_text(
        "# Summary\n\n- Budget agreed\n- " + "word " * 50, encoding="utf-8"
    )
    (note,) = notes.scan_notes(tmp_path)
    assert note.has_summary and note.excerpt.startswith("Summary Budget agreed word")
    assert len(note.excerpt) <= notes.EXCERPT_CHARS and note.excerpt.endswith("…")
    assert notes.note_columns(note)[-1] == "transcribed, summary"
    assert "Open summary" in notes.note_actions(note)
    assert notes.action_target(note, "Open summary") == s.directory / "summary.md"
    assert notes.action_target(note, "Reveal folder") == s.directory
    assert "Budget agreed" in notes.actions_script(note)


def test_rename_is_stored_in_session_json_and_survives_a_stale_save(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0)
    raw = json.loads((s.directory / "session.json").read_text("utf-8"))
    assert "title" in raw and raw["title"] is None
    del raw["title"]  # a session.json from before titles existed still loads
    (s.directory / "session.json").write_text(json.dumps(raw), encoding="utf-8")
    assert SessionInfo.load(s.directory).title is None
    notes.rename_session(s.directory, "  Budget   review ")
    s.transcribed = True
    s.save()  # the pipeline's own copy, loaded before the rename
    loaded = SessionInfo.load(s.directory)
    assert loaded.title == "Budget review" and loaded.transcribed
    assert notes.scan_notes(tmp_path)[0].title == "Budget review"
    notes.rename_session(s.directory, "")
    assert notes.scan_notes(tmp_path)[0].title == "zoom.us"


def test_delete_uses_the_trash_or_discard_session(tmp_path, monkeypatch):
    s = make_session(tmp_path, "140312_zoom-us", T0)
    trashed: list[Path] = []
    monkeypatch.setattr(notes.sys, "platform", "darwin")
    monkeypatch.setattr(notes, "move_to_trash", lambda p: trashed.append(p) or False)
    assert notes.delete_session(s.directory) == "failed"  # Trash refused: nothing deleted
    assert trashed == [s.directory] and (s.directory / "session.json").exists()
    assert not (s.directory / SessionInfo.DISCARD_MARKER).exists()
    assert "Trash" in notes.delete_prompt(notes.scan_notes(tmp_path)[0])
    trash = tmp_path / "Trash"
    monkeypatch.setattr(notes, "move_to_trash", lambda p: bool(shutil.move(p, trash)))
    assert notes.delete_session(s.directory) == "trashed"
    assert not s.directory.exists() and not (trash / SessionInfo.DISCARD_MARKER).exists()
    # a partial move (a locked file stayed behind): discard_session finishes the job
    s = make_session(tmp_path, "150000_zoom-us", T0)
    (s.directory / "me.wav").write_bytes(b"x")
    monkeypatch.setattr(notes, "move_to_trash", lambda p: (p / "session.json").unlink())
    assert notes.delete_session(s.directory) == "deleted" and not s.directory.exists()
    s = make_session(tmp_path, "160000_zoom-us", T0)
    monkeypatch.setattr(notes.sys, "platform", "linux")  # no Trash: deleted for good
    assert "permanently" in notes.delete_prompt(notes.scan_notes(tmp_path)[0])
    assert notes.delete_session(s.directory) == "deleted"
    assert not s.directory.exists()


def test_move_to_trash_is_off_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(notes.sys, "platform", "linux")
    assert not notes.move_to_trash(tmp_path)


def test_mac_dialog_scripts_quote_titles_and_parse_choices(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, title='Say "hi"')
    listed = notes.scan_notes(tmp_path)
    assert 'Say \\"hi\\"' in notes.list_script(listed)
    assert 'Say \\"hi\\"' in notes.rename_script(listed[0])
    assert "Delete" in notes.delete_script(listed[0])
    assert notes.chosen_index("1. 2026-10-26 14:03 · x", 1) == 0
    assert notes.chosen_index("false", 1) is None
    assert notes.chosen_index("2. y", 1) is None
    assert s.directory.exists()
    assert notes.summary_excerpt("") == ""


def _ctrl(tmp_path, session_dir=None, pending=0):
    c = Controller.__new__(Controller)  # only the notes wiring: no runtime
    c.cfg = types.SimpleNamespace(output_dir=tmp_path)
    c.events = queue.Queue()
    c.engine = types.SimpleNamespace(state=types.SimpleNamespace(session_dir=session_dir))
    c.tq = types.SimpleNamespace(pending=lambda: pending)
    c.refresh_async = lambda force=False: None
    return c


def _wait_for(c, kind: str):
    while True:
        ev = c.events.get(timeout=5)
        if ev[0] == kind:
            return ev


def test_controller_lists_renames_and_deletes(tmp_path, monkeypatch):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    indexed: list[Path] = []
    monkeypatch.setattr(Controller, "_index_session", staticmethod(indexed.append))
    monkeypatch.setattr(notes, "delete_session", lambda d: notes.discard_session(d) and "x")
    c = _ctrl(tmp_path)
    c.notes_async()
    assert [n.directory for n in _wait_for(c, "notes")[1]] == [s.directory]
    c.note_action_async(s.directory, "rename", "Budget")
    assert _wait_for(c, "notes")[1][0].title == "Budget"
    c.note_action_async(s.directory, "delete")
    assert _wait_for(c, "notes")[1] == []
    assert indexed == [s.directory] and not s.directory.exists()


@pytest.mark.parametrize("recording", [True, False])
def test_controller_refuses_busy_sessions(tmp_path, recording):
    s = make_session(tmp_path, "140312_zoom-us", T0)  # not transcribed yet
    c = _ctrl(tmp_path, str(s.directory) if recording else None, pending=0 if recording else 1)
    assert c.note_busy(s.directory)
    c.note_action_async(s.directory, "delete")
    assert c.events.get(timeout=5)[0] == "notice"
    assert s.directory.exists()
