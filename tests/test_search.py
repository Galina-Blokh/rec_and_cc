"""Transcript search: incremental SQLite FTS5 index, LIKE fallback, CLI and UI hook.

Stories: US-20.
"""

from __future__ import annotations

import os
import queue
import stat
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from recandcc import cli, search
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.search import TranscriptIndex, has_fts5
from recandcc.transcribe.output import write_outputs

T0 = datetime(2026, 10, 26, 14, 3, 12)
BACKENDS = [pytest.param(True, id="fts5"), pytest.param(False, id="like")]


def make_session(out: Path, n: int, segments: list[Segment]) -> Path:
    started = T0 + timedelta(minutes=n)
    d = out / f"{started:%Y-%m-%d}" / f"{started:%H%M%S}_zoom-us-{n}"
    d.mkdir(parents=True)
    s = SessionInfo(f"s{n}", d, started, TriggerKind.CALL, "zoom.us", started)
    s.save()
    write_outputs(s, segments, 60.0, "fake")
    return d


def seg(start: float, text: str, speaker: str = "Me", lang: str = "en") -> Segment:
    return Segment(start, start + 2, text, speaker, lang)


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def index(tmp_path, fts: bool) -> TranscriptIndex:
    if fts and not has_fts5():
        pytest.skip("this SQLite has no FTS5")
    return TranscriptIndex(tmp_path / "idx" / "search.sqlite3", fts=fts)


@pytest.mark.parametrize("fts", BACKENDS)
def test_english_and_russian_queries(tmp_path, out, fts):
    a = make_session(out, 1, [seg(5, "Let's discuss the quarterly budget"), seg(65, "Bye")])
    b = make_session(out, 2, [seg(1, "Обсудим бюджет на следующий квартал", "Speaker 1", "ru")])
    idx = index(tmp_path, fts)
    assert idx.sync(out) == 2
    [hit] = idx.search("Budget")
    assert (hit.folder, hit.session_id, hit.start, hit.speaker) == (a, "s1", 5.0, "Me")
    assert hit.started_at == T0.replace(minute=4).isoformat()
    assert "[budget]" in hit.snippet.lower()
    assert hit.transcript == a / "transcript.md"
    [hit] = idx.search("БЮДЖЕТ квартал")  # case-insensitive, every word must match
    assert (hit.folder, hit.speaker, hit.language) == (b, "Speaker 1", "ru")
    assert [h.folder for h in idx.search("бюдж")] == [b]  # a word prefix
    assert idx.search("budget nothing") == [] and idx.search("  !! ") == []


def test_fts_prefix_and_diacritics(tmp_path, out):
    make_session(out, 1, [seg(0, "Café meeting with Zoë")])
    idx = index(tmp_path, True)
    idx.sync(out)
    assert len(idx.search("cafe")) == 1 and len(idx.search("zoe")) == 1
    assert len(idx.search("meet")) == 1  # prefix match
    assert len(idx.search('"meeting) (')) == 1  # FTS syntax in a query is just words


@pytest.mark.parametrize("fts", BACKENDS)
def test_incremental_update_and_delete(tmp_path, out, fts):
    a = make_session(out, 1, [seg(0, "alpha topic")])
    b = make_session(out, 2, [seg(0, "beta topic")])
    idx = index(tmp_path, fts)
    assert idx.sync(out) == 2
    assert idx.sync(out) == 0  # nothing changed: nothing re-read
    s = SessionInfo.load(a)
    write_outputs(s, [seg(0, "gamma topic", "Speaker 1")], 60.0, "fake")
    os.utime(a / "transcript.json", ns=(1, 1))  # a changed mtime, however coarse the clock
    assert idx.sync(out) == 1
    assert idx.search("alpha") == [] and idx.search("gamma")[0].speaker == "Speaker 1"
    for f in b.iterdir():
        f.unlink()
    b.rmdir()
    idx.sync(out)
    assert {h.folder for h in idx.search("topic")} == {a}


@pytest.mark.parametrize("fts", BACKENDS)
def test_discarded_sessions_are_skipped_and_dropped(tmp_path, out, fts):
    a = make_session(out, 1, [seg(0, "secret plan")])
    b = make_session(out, 2, [seg(0, "public plan")])
    (b / SessionInfo.DISCARD_MARKER).touch()
    idx = index(tmp_path, fts)
    idx.sync(out)
    assert [h.folder for h in idx.search("plan")] == [a]
    (a / SessionInfo.DISCARD_MARKER).touch()
    assert idx.index_session(a) is False
    assert idx.search("plan") == []


def test_index_session_adds_one_and_skips_broken(tmp_path, out):
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    a = make_session(out, 1, [seg(0, "launch date")])
    assert idx.index_session(a) is True
    assert len(idx.search("launch")) == 1
    (a / "transcript.json").write_text("{not json", encoding="utf-8")
    assert idx.index_session(a) is False  # logged, not raised
    assert idx.index_session(out / "missing") is False


def test_fallback_when_sqlite_has_no_fts5(tmp_path, out, monkeypatch):
    monkeypatch.setattr(search, "has_fts5", lambda: False)
    make_session(out, 1, [seg(0, "100% done_now")])
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    assert idx.fts is False
    idx.sync(out)
    assert len(idx.search("done_now")) == 1
    assert idx.search("don_e") == []  # LIKE wildcards in the query are literal


def test_switching_backend_reindexes(tmp_path, out):
    if not has_fts5():
        pytest.skip("this SQLite has no FTS5")
    make_session(out, 1, [seg(0, "shared words")])
    path = tmp_path / "idx.sqlite3"
    TranscriptIndex(path, fts=True).sync(out)
    plain = TranscriptIndex(path, fts=False)
    assert plain.sync(out) == 1 and len(plain.search("shared")) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_index_is_owner_only_and_outside_output_dir(tmp_path, out):
    make_session(out, 1, [seg(0, "x")])
    search.search_transcripts(out, "x")
    path = search.default_index().path
    assert path == search.INDEX_PATH and out not in path.parents
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_performance_100_sessions(tmp_path, out):
    words = [
        "budget",
        "roadmap",
        "hiring",
        "release",
        "customer",
        "бюджет",
        "релиз",
        "найм",
        "клиент",
    ]
    for n in range(100):
        make_session(
            out,
            n,
            [
                seg(i * 5, f"{words[(n + i) % len(words)]} item {i} of session {n}")
                for i in range(60)
            ],
        )
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    idx.sync(out)
    t0 = time.perf_counter()
    hits = idx.search("budget", limit=50)
    assert time.perf_counter() - t0 < 0.2
    assert len(hits) == 50
    t0 = time.perf_counter()
    assert idx.sync(out) == 0 and idx.search("релиз")
    assert time.perf_counter() - t0 < 0.5  # an unchanged sync only stats the files


def test_cli_search(out, monkeypatch, capsys):
    from recandcc.config import Config

    make_session(out, 1, [seg(83, "the budget is approved")])
    monkeypatch.setattr(cli, "load_config", lambda: Config(output_dir=out))
    assert cli.main(["search", "budget", "--limit", "5"]) == 0
    text = capsys.readouterr().out
    assert "[00:01:23] Me: the [budget] is approved" in text and "transcript.md" in text
    assert cli.main(["search", "nothing"]) == 1


def test_controller_indexes_after_transcription_and_searches_off_thread(tmp_path, out):
    from recandcc.ui.common import Controller, search_hit_label

    a = make_session(out, 1, [seg(0, "kickoff notes")])
    ctrl = Controller.__new__(Controller)  # only the search wiring, no runtime
    ctrl.events, ctrl.cfg = queue.Queue(), type("C", (), {"output_dir": out})()
    ctrl.refresh_async = lambda force=False: None
    ctrl._on_transcribed(SessionInfo.load(a), [])
    assert ctrl.events.get(timeout=1)[0] == "transcribed"
    deadline = time.monotonic() + 5
    while not search.default_index().search("kickoff") and time.monotonic() < deadline:
        time.sleep(0.02)
    assert search.default_index().search("kickoff")
    ctrl.search_async("kickoff")
    kind, query, hits = ctrl.events.get(timeout=5)
    assert (kind, query, len(hits)) == ("search", "kickoff", 1)
    assert search_hit_label(hits[0]).startswith("2026-10-26 14:04 zoom-us-1 · 00:00 Me: ")
