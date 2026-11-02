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
    assert notes.SUMMARIZE in notes.note_actions(note)  # redo it on demand
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
    notes.rename_session(s.directory, "  Budget   review ", tmp_path)
    s.transcribed = True
    s.save()  # the pipeline's own copy, loaded before the rename
    loaded = SessionInfo.load(s.directory)
    assert loaded.title == "Budget review" and loaded.transcribed
    assert notes.scan_notes(tmp_path)[0].title == "Budget review"
    notes.rename_session(s.directory, "", tmp_path)
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
    pruned: list[int] = []
    monkeypatch.setattr(Controller, "_index_session", staticmethod(indexed.append))
    monkeypatch.setattr(Controller, "_prune_voices", staticmethod(lambda: pruned.append(1)))
    monkeypatch.setattr(notes, "delete_session", lambda d, o: notes.discard_session(d) and "x")
    c = _ctrl(tmp_path)
    c.notes_async()
    assert [n.directory for n in _wait_for(c, "notes")[1]] == [s.directory]
    c.note_action_async(s.directory, "rename", "Budget")
    assert _wait_for(c, "notes")[1][0].title == "Budget"
    assert pruned == [] and indexed == []  # a rename leaves index and voices alone
    c.note_action_async(s.directory, "delete")
    assert _wait_for(c, "notes")[1] == []
    assert indexed == [s.directory] and pruned == [1] and not s.directory.exists()
    c.note_action_async(s.directory, "export")  # unknown: ignored, never a delete
    c._resubmitting = True  # startup may be about to re-queue an untranscribed session
    u = make_session(tmp_path, "150000_zoom-us", T0)
    assert c.note_busy(u.directory)


def test_links_out_of_the_output_folder_are_not_listed_or_renamed(tmp_path):
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    s = make_session(elsewhere, "140312_zoom-us", T0)
    (out / "2026-10-26").mkdir(parents=True)
    link = out / "2026-10-26" / "140312_zoom-us"
    try:
        link.symlink_to(s.directory, target_is_directory=True)
    except OSError:
        pytest.skip("no symlink rights (Windows without developer mode)")
    assert notes.scan_notes(out) == []
    with pytest.raises(ValueError):
        notes.rename_session(link, "x", out)
    assert SessionInfo.load(s.directory).title is None


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
    assert op.fFlags.offset == (32 if wide else 16)  # pyright: ignore[reportAttributeAccessIssue]
    assert op.fAnyOperationsAborted.offset == (36 if wide else 18)  # pyright: ignore[reportAttributeAccessIssue]


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
    notes.rename_session(s.directory, "First", tmp_path)
    assert SessionInfo.load(s.directory).metrics["diarization"] == {"speakers": 2}
    after_first = SessionInfo.load(s.directory)  # e.g. the CLI, holding title "First"
    notes.rename_session(s.directory, "Second", tmp_path)
    after_first.save()
    s.save()  # loaded before any rename
    assert SessionInfo.load(s.directory).title == "Second"


def test_mac_dialog_flow(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    listed = notes.scan_notes(tmp_path)
    acted, opened = [], []

    def act(directory, action, title="", reopen=True):
        acted.append((directory, action, title, reopen))

    def run(*answers):
        replies = iter(answers)
        notes.run_dialogs(listed, lambda _script: next(replies), act, opened.append)

    run("false")  # cancelled
    run("1. x", "Open transcript")
    run("1. x", "Open summary")  # no summary.md: nothing
    run("1. x", "Rename…", "button returned:Rename, text returned:A, text returned: B")
    run("1. x", "Rename…", "")  # cancelled
    run("1. x", "Delete…", "")
    run("1. x", "Delete…", "button returned:Delete")
    run("1. x", "Summarize")
    assert opened == [s.directory / "transcript.md"]
    assert acted == [
        (s.directory, "rename", "A, text returned: B", True),
        (s.directory, "delete", "", True),
        (s.directory, "summarize", "", False),  # no Meeting notes list again after it
    ]


def test_listing_survives_a_broken_transcript_and_caps_titles(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    (s.directory / "transcript.json").write_text("{not json", encoding="utf-8")
    (note,) = notes.scan_notes(tmp_path)
    assert note.speakers == [] and note.duration_s == 300.0
    notes.rename_session(s.directory, "x" * 500, tmp_path)
    assert len(SessionInfo.load(s.directory).title or "") == notes.TITLE_CHARS
    s.save()  # the stale copy, twice: the title stays on disk
    notes.rename_session(s.directory, "second", tmp_path)
    s.save()
    assert SessionInfo.load(s.directory).title == "second" and s.title is None


def test_summarize_queues_a_summary_for_a_chosen_meeting(tmp_path, monkeypatch):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    queued: list[Path] = []
    monkeypatch.setattr(Controller, "_summarize", lambda self, d, announce=False: queued.append(d))
    c = _ctrl(tmp_path)
    assert not c.cfg.summary.enabled  # works with automatic summaries off
    c.note_action_async(s.directory, "summarize")
    assert _wait_for(c, "notice")[1] == "Writing the summary"
    assert queued == [s.directory]
    (s.directory / "transcript.json").unlink()
    c.note_action_async(s.directory, "summarize")
    assert _wait_for(c, "notice")[1] == "No transcript to summarize"
    assert queued == [s.directory]


def test_summary_pick_lists_only_meetings_with_a_transcript(tmp_path):
    done = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(done.directory)
    make_session(tmp_path, "150000_manual", T0 + timedelta(hours=1), app=None)  # no transcript
    listed = notes.scan_notes(tmp_path)
    assert [n.directory for n in notes.summary_choices(listed)] == [done.directory]
    shown, acted = [], []

    def act(directory, action, reopen=True):
        acted.append((directory, action, reopen))

    notes.run_summary_pick(listed, lambda s: shown.append(s) or "1. x", act)
    assert "zoom.us" in shown[0] and "Manual recording" not in shown[0]
    assert "Summarize a meeting" in shown[0]
    notes.run_summary_pick(listed, lambda s: "false", act)  # cancelled
    assert acted == [(done.directory, "summarize", False)]


def test_summarize_scan_posts_the_picker_list(tmp_path):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    c = _ctrl(tmp_path)
    c.notes_async(purpose="summarize")
    assert [n.directory for n in _wait_for(c, "summary_notes")[1]] == [s.directory]


def test_only_summaries_asked_for_from_the_menu_are_announced(tmp_path, monkeypatch):
    from recandcc.engine import IDLE
    from recandcc.transcribe import summary

    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    fail, scanned = [False], threading.Event()  # fail: the next model run errors

    def writer(directory, *args, **kwargs):  # stands in for the local model
        if fail[0]:
            raise RuntimeError("model fell over")
        (directory / "summary.md").write_text("# Summary\n", encoding="utf-8")
        return directory / "summary.md"

    def announced():
        return [e for e in list(c.events.queue) if e[0] == "summary_ready"]

    monkeypatch.setattr(summary, "summarize_session", writer)
    # the last step of a menu action (after its notice): the job is queued by then
    monkeypatch.setattr(notes, "scan_notes", lambda d: (scanned.set(), [])[1])
    c = _ctrl(tmp_path)
    c.engine.state.status = IDLE  # the real queue, never busy here
    c._summarize(s.directory)  # automatic, after transcription: opens nothing
    c._summaries.join()
    assert announced() == []
    fail[0] = True
    c.note_action_async(s.directory, "summarize")  # asked for, and it fails
    assert scanned.wait(5)
    c._summaries.join()
    assert ("notice", "Summary not written", "zoom.us") in list(c.events.queue)
    fail[0], scanned = False, threading.Event()
    c._summarize(s.directory)  # the failed request left nothing behind: still nothing opens
    c._summaries.join()
    assert announced() == []
    c.note_action_async(s.directory, "summarize")  # asked for, and written
    assert scanned.wait(5)
    c._summaries.join()
    assert announced() == [("summary_ready", s.directory / "summary.md", "zoom.us")]


def test_a_request_made_while_labels_are_pending_is_one_summary_after_them(tmp_path, monkeypatch):
    from recandcc.engine import IDLE
    from recandcc.transcribe import summary

    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    writes, gate = [], threading.Event()

    def writer(directory, *args, **kwargs):  # stands in for the local model
        writes.append(directory)
        gate.wait(5)  # runs until the test has queued the rest, as a slow model would
        (directory / "summary.md").write_text("# Summary\n", encoding="utf-8")
        return directory / "summary.md"

    class LabelsPending:  # a transcriber whose speaker labels are still to come
        def __init__(self):
            self.waiting = []

        def after_labels(self, directory, fn):
            self.waiting.append(fn)

    monkeypatch.setattr(summary, "summarize_session", writer)
    monkeypatch.setattr(notes, "scan_notes", lambda d: [])
    c = _ctrl(tmp_path)
    c.engine.state.status = IDLE
    c.transcriber = LabelsPending()
    c._summarize_when_labelled(s.directory)  # the automatic one, queued at transcription
    c.note_action_async(s.directory, "summarize")  # the menu's, before the labels are final
    assert _wait_for(c, "notice")[1] == "Writing the summary"
    assert writes == []  # nothing is written from the unlabelled transcript
    for fn in c.transcriber.waiting:  # the labels are final: both waiters run, in order
        fn()
    gate.set()
    c._summaries.join()
    assert writes == [s.directory]  # one summary, and it is the one announced
    announced = [e for e in list(c.events.queue) if e[0] == "summary_ready"]
    assert announced == [("summary_ready", s.directory / "summary.md", "zoom.us")]


def test_a_summary_picked_from_the_picker_does_not_reopen_meeting_notes(tmp_path, monkeypatch):
    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    scanned = threading.Event()
    monkeypatch.setattr(Controller, "_summarize", lambda self, d, announce=False: None)
    monkeypatch.setattr(notes, "scan_notes", lambda d: (scanned.set(), [])[1])
    c = _ctrl(tmp_path)
    c.note_action_async(s.directory, "summarize", reopen=False)
    assert _wait_for(c, "notice")[1] == "Writing the summary"
    assert not scanned.wait(0.2)  # the picker leaves Meeting notes closed
    c.note_action_async(s.directory, "summarize")  # from Meeting notes: a fresh list follows
    assert _wait_for(c, "notes")[1] == []


def test_mac_summary_picker_and_ready_notice(tmp_path, monkeypatch):
    pytest.importorskip("rumps")
    from recandcc.ui import mac

    s = make_session(tmp_path, "140312_zoom-us", T0, transcribed=True)
    write_transcript(s.directory)
    notices, opened, acted, picked = [], [], [], threading.Event()
    monkeypatch.setattr(mac, "osascript", lambda script, timeout=None: "1. x")
    monkeypatch.setattr(mac, "open_path", opened.append)
    app = mac.MacApp.__new__(mac.MacApp)  # only the picker wiring: no menu, no timer
    app._notify = lambda title, message: notices.append((title, message))
    app._notes_lock = threading.Lock()
    app.ctrl = types.SimpleNamespace(
        events=queue.Queue(),
        note_action_async=lambda d, action, reopen=True: (
            acted.append((d, action, reopen)),
            picked.set(),
        ),
    )
    app._show_summary_pick([])
    assert notices == [("No transcripts to summarize yet", "Transcribe a meeting first.")]
    app._show_summary_pick(notes.scan_notes(tmp_path))
    assert picked.wait(5) and acted == [(s.directory, "summarize", False)]
    app.ctrl.events.put(("summary_ready", s.directory / "summary.md", "zoom.us"))
    app._drain_events()
    assert notices[-1] == ("Summary ready", "zoom.us")
    assert opened == [s.directory / "summary.md"]
