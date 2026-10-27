"""Meeting notes: listing sessions newest first with their facts, summary excerpts when a
summary.md exists, renaming via session.json, deleting to the Trash or via discard_session,
and the Controller wiring (busy sessions, search index, a fresh list).

Stories: US-28.
"""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from recandcc.models import SessionInfo, Track, TriggerKind, discard_session
from recandcc.ui import notes
from recandcc.ui.common import Controller

T0 = datetime(2026, 10, 26, 14, 3, 12)


@pytest.fixture(autouse=True)
def _no_real_side_effects(monkeypatch):
    """Nothing here reaches the real Trash, search index or voice profiles, even if a guard
    regresses: tests that need them patch these again."""
    monkeypatch.setattr(notes, "move_to_trash", lambda p: False)
    monkeypatch.setattr(Controller, "_index_session", staticmethod(lambda d: None))
    monkeypatch.setattr(Controller, "_prune_voices", staticmethod(lambda: None))


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
    assert note.has_summary and note.excerpt.startswith("Budget agreed word")
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
    assert notes.delete_session(s.directory, tmp_path) == "failed"  # Trash refused: nothing deleted
    assert trashed == [s.directory] and (s.directory / "session.json").exists()
    assert not (s.directory / SessionInfo.DISCARD_MARKER).exists()
    assert "Trash" in notes.delete_prompt(notes.scan_notes(tmp_path)[0])
    trash = tmp_path / "Trash"
    monkeypatch.setattr(notes, "move_to_trash", lambda p: bool(shutil.move(p, trash)))
    assert notes.delete_session(s.directory, tmp_path) == "trashed"
    assert not s.directory.exists() and not (trash / SessionInfo.DISCARD_MARKER).exists()
    # a partial move (a locked file stayed behind): discard_session finishes the job
    s = make_session(tmp_path, "150000_zoom-us", T0)
    (s.directory / "me.wav").write_bytes(b"x")
    monkeypatch.setattr(notes, "move_to_trash", lambda p: (p / "session.json").unlink())
    assert notes.delete_session(s.directory, tmp_path) == "deleted" and not s.directory.exists()
    assert notes.delete_notice("deleted")[0] == "Meeting deleted for good"
    # ... and if a file stays locked: marked, the rest goes at the next start
    s = make_session(tmp_path, "153000_zoom-us", T0)
    monkeypatch.setattr(notes, "discard_session", lambda d: False)
    assert notes.delete_session(s.directory, tmp_path) == "partial"
    monkeypatch.setattr(notes, "discard_session", discard_session)
    assert notes.delete_notice("trashed") is None
    assert notes.delete_notice("failed")[1].endswith("nothing was deleted.")
    shutil.rmtree(s.directory)
    s = make_session(tmp_path, "160000_zoom-us", T0)
    monkeypatch.setattr(notes.sys, "platform", "linux")  # no Trash: deleted for good
    assert "permanently" in notes.delete_prompt(notes.scan_notes(tmp_path)[0])
    assert notes.delete_session(s.directory, tmp_path) == "deleted"
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
    c.cfg = types.SimpleNamespace(
        output_dir=tmp_path,
        transcription=types.SimpleNamespace(live=False),
        summary=types.SimpleNamespace(enabled=False),
    )
    c._lock, c._in_flight, c._notes_scanning = threading.RLock(), set(), threading.Lock()
    c._live, c.transcriber = None, None
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
    monkeypatch.setattr(notes, "delete_session", lambda d, o: notes.discard_session(d) and "x")
    c = _ctrl(tmp_path)
    c.notes_async()
    assert [n.directory for n in _wait_for(c, "notes")[1]] == [s.directory]
    c.note_action_async(s.directory, "rename", "Budget")
    assert _wait_for(c, "notes")[1][0].title == "Budget"
    c.note_action_async(s.directory, "delete")
    assert _wait_for(c, "notes")[1] == []
    assert indexed == [s.directory] and not s.directory.exists()


@pytest.mark.parametrize("case", ["recording", "queued", "in_flight", "unfinished"])
def test_controller_refuses_busy_sessions(tmp_path, case):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=case == "in_flight")
    c = _ctrl(tmp_path, str(s.directory) if case == "recording" else None, int(case == "queued"))
    if case == "in_flight":  # transcribed=True already, but the queue isn't done with it
        c._on_session_start(s)
    if case == "unfinished":  # no ended_at while another session records
        s.ended_at = None
        s.save()
        c.engine.state.session_dir = str(tmp_path / "other")
    assert c.note_busy(s.directory)
    c.note_action_async(s.directory, "delete")
    assert c.events.get(timeout=5)[0] == "notice"
    assert s.directory.exists()


def test_in_flight_ends_when_transcribed_or_dropped(tmp_path):
    a = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    b = make_session(tmp_path, "150000_zoom-us", T0)
    c = _ctrl(tmp_path)
    c._on_session_start(a)
    c._on_session_start(b)
    c._on_transcribed(a, [])
    c._on_session_end(b, kept=False)
    assert c.note_busy(a.directory) is None and not c._in_flight


def test_failed_delete_posts_a_notice_and_keeps_the_meeting(tmp_path, monkeypatch):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    monkeypatch.setattr(Controller, "_index_session", staticmethod(lambda d: None))
    monkeypatch.setattr(notes, "delete_session", lambda d, o: "failed")
    c = _ctrl(tmp_path)
    c.note_action_async(s.directory, "delete")
    assert _wait_for(c, "notice")[1] == "Could not delete the meeting"
    assert [n.directory for n in _wait_for(c, "notes")[1]] == [s.directory]


def test_notes_scan_is_single_flight(tmp_path, monkeypatch):
    gate, scans = threading.Event(), []
    monkeypatch.setattr(notes, "scan_notes", lambda d: (scans.append(d), gate.wait(5), [])[2])
    c = _ctrl(tmp_path)
    c.notes_async()
    c.notes_async()  # dropped: a scan is running
    gate.set()
    assert _wait_for(c, "notes")[1] == [] and len(scans) == 1


def test_delete_refuses_links_and_folders_outside_the_output_folder(tmp_path):
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    s = make_session(elsewhere, "140312_zoom-us", T0)
    (out / "2026-10-26").mkdir(parents=True)
    assert notes.delete_session(s.directory, out) == "refused"
    link = out / "2026-10-26" / "140312_zoom-us"
    try:
        link.symlink_to(s.directory, target_is_directory=True)
    except OSError:
        pytest.skip("no symlink rights (Windows without developer mode)")
    assert notes.delete_session(link, out) == "refused"
    assert (s.directory / "session.json").exists()
    assert notes.delete_notice("refused")[0] == "Could not delete the meeting"


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows shell struct")
def test_shfileop_struct_matches_shellapi():
    import ctypes

    op = notes.shfileop_struct()
    wide = ctypes.sizeof(ctypes.c_void_p) == 8
    assert ctypes.sizeof(op) == (56 if wide else 30)
    assert op.fFlags.offset == (24 if wide else 16)  # pyright: ignore[reportAttributeAccessIssue]
    assert op.fAnyOperationsAborted.offset == (28 if wide else 18)  # pyright: ignore[reportAttributeAccessIssue]


def test_broken_session_json_is_skipped_and_odd_titles_ignored(tmp_path):
    good = make_session(tmp_path, "140312_zoom-us", T0)
    bad = make_session(tmp_path, "150000_zoom-us", T0 + timedelta(hours=1))
    raw = json.loads((bad.directory / "session.json").read_text("utf-8"))
    (bad.directory / "session.json").write_text(json.dumps({**raw, "tracks": []}), "utf-8")
    raw = json.loads((good.directory / "session.json").read_text("utf-8"))
    (good.directory / "session.json").write_text(json.dumps({**raw, "title": 5}), "utf-8")
    (note,) = notes.scan_notes(tmp_path)
    assert note.directory == good.directory and note.title == "zoom.us"
    assert "zoom.us" in notes.rename_script(note)


def test_rename_changes_only_the_title(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0)
    s.update_metrics("diarization", {"speakers": 2})
    s.save()
    notes.rename_session(s.directory, "First")
    assert SessionInfo.load(s.directory).metrics["diarization"] == {"speakers": 2}
    after_first = SessionInfo.load(s.directory)  # e.g. the CLI, holding title "First"
    notes.rename_session(s.directory, "Second")
    after_first.save()
    s.save()  # loaded before any rename
    assert SessionInfo.load(s.directory).title == "Second"


def test_mac_dialog_flow(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    listed = notes.scan_notes(tmp_path)
    acted, opened = [], []

    def run(*answers):
        replies = iter(answers)
        notes.run_dialogs(listed, lambda _script: next(replies), lambda *a: acted.append(a),
                          opened.append)  # fmt: skip

    run("false")  # cancelled
    run("1. x", "Open transcript")
    run("1. x", "Open summary")  # no summary.md: nothing
    run("1. x", "Rename…", "button returned:Rename, text returned:A, text returned: B")
    run("1. x", "Rename…", "")  # cancelled
    run("1. x", "Delete…", "")
    run("1. x", "Delete…", "button returned:Delete")
    assert opened == [s.directory / "transcript.md"]
    assert acted == [(s.directory, "rename", "A, text returned: B"), (s.directory, "delete")]


def test_listing_survives_a_broken_transcript_and_caps_titles(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    (s.directory / "transcript.json").write_text("{not json", encoding="utf-8")
    (note,) = notes.scan_notes(tmp_path)
    assert note.speakers == [] and note.duration_s == 300.0
    notes.rename_session(s.directory, "x" * 500)
    assert len(SessionInfo.load(s.directory).title or "") == notes.TITLE_CHARS
    s.save()  # the stale copy, twice: the title stays on disk
    notes.rename_session(s.directory, "second")
    s.save()
    assert SessionInfo.load(s.directory).title == "second" and s.title is None
