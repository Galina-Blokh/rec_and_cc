"""Transcript search: incremental SQLite FTS5 index, LIKE fallback, CLI and UI hook.

Stories: US-20.
"""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from recandcc import cli, search
from recandcc.config import Config
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


def test_index_session_adds_one_and_skips_broken(tmp_path, out, monkeypatch):
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    a = make_session(out, 1, [seg(0, "launch date")])
    assert idx.index_session(a) is True
    assert len(idx.search("launch")) == 1
    (a / "transcript.json").write_text("{not json", encoding="utf-8")
    assert idx.index_session(a) is False  # logged, not raised
    assert idx.search("launch") == []  # no stale hits from the previous version
    reads = []
    monkeypatch.setattr(search, "_read", lambda d: reads.append(d))
    assert idx.sync(out) == 0 and reads == []  # not re-read until it changes again
    assert idx.index_session(out / "missing") is False


def test_vacuum_fallback_runs_only_when_text_was_removed(tmp_path, out, monkeypatch):
    idx = index(tmp_path, True)
    idx._fts_secure = False  # as on SQLite < 3.42
    purges = []
    monkeypatch.setattr(idx, "_purge", lambda db: purges.append(1))
    a = make_session(out, 1, [seg(0, "new words")])
    idx.index_session(a)
    idx.sync(out)
    assert purges == []  # adding a session removes nothing: no VACUUM
    idx.index_session(a)  # a reindex replaces old rows
    assert purges == [1]


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


@pytest.mark.parametrize("fts", BACKENDS)
def test_performance_100_sessions(out, monkeypatch, fts):
    """US-20: < 200 ms on the path the menu and CLI use (an unchanged sync + the query)."""
    if fts and not has_fts5():
        pytest.skip("this SQLite has no FTS5")
    monkeypatch.setattr(search, "has_fts5", lambda: fts)
    monkeypatch.setattr(search, "_default", None)
    words = ["budget", "roadmap", "hiring", "release", "бюджет", "релиз", "найм", "клиент"]
    for n in range(100):
        segs = [seg(i * 5, f"{words[(n + i) % len(words)]} item {i} of {n}") for i in range(60)]
        make_session(out, n, segs)
    assert len(search.search_transcripts(out, "budget", limit=50)) == 50  # first: indexes
    timings = []
    for query in ("budget", "релиз", "hiring item"):  # best of 3: a loaded runner is noisy
        t0 = time.perf_counter()
        assert search.search_transcripts(out, query, limit=50)
        timings.append(time.perf_counter() - t0)
    assert min(timings) < 0.2, timings


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
    ctrl.events, ctrl.cfg = queue.Queue(), Config(output_dir=out)
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


def test_failed_search_never_logs_the_query(out, monkeypatch, caplog):
    from recandcc.ui.common import Controller

    def fail(output_dir, query, limit, semantic=False):
        raise sqlite3.OperationalError(f'fts5: syntax error near "{query}"')

    monkeypatch.setattr(search, "search_transcripts", fail)
    ctrl = Controller.__new__(Controller)
    ctrl.events, ctrl.cfg = queue.Queue(), Config(output_dir=out)
    caplog.set_level(logging.DEBUG)
    ctrl.search_async("secretword")
    kind, _query, hits = ctrl.events.get(timeout=5)
    assert kind == "search" and isinstance(hits, sqlite3.OperationalError)
    assert "transcript search failed: OperationalError" in caplog.text
    assert "secretword" not in caplog.text


def leftover(path: Path, token: str) -> int:
    """How often ``token`` still occurs in the index file (and its journal/WAL, if any)."""
    files = [path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-journal")]
    return sum(f.read_bytes().count(token.encode()) for f in files if f.exists())


@pytest.mark.parametrize(
    "fts, fts_secure",
    [pytest.param(True, True, id="fts5"), pytest.param(True, False, id="fts5-optimize")]
    + [pytest.param(False, True, id="like")],
)
def test_dropped_and_replaced_text_leaves_the_file(tmp_path, out, fts, fts_secure):
    a = make_session(out, 1, [seg(0, "zebrasecretpassword confidential")])
    b = make_session(out, 2, [seg(0, "ordinary words"), seg(5, "quokkaoldversion")])
    idx = index(tmp_path, fts)
    idx.sync(out)
    if not fts_secure:  # as on SQLite < 3.42: FTS5 keeps deleted terms until optimize
        with sqlite3.connect(idx.path) as db:
            db.execute("INSERT INTO seg_fts(seg_fts, rank) VALUES ('secure-delete', 0)")
        idx._fts_secure = False
    assert leftover(idx.path, "zebrasecretpassword") > 0
    for f in a.iterdir():
        f.unlink()
    a.rmdir()
    write_outputs(SessionInfo.load(b), [seg(0, "ordinary words")], 60.0, "fake")
    idx.sync(out)
    assert leftover(idx.path, "zebrasecretpassword") == 0
    assert leftover(idx.path, "quokkaoldversion") == 0
    assert len(idx.search("ordinary")) == 1


@pytest.mark.parametrize("fts", BACKENDS)
def test_yo_and_ye_match_each_other(tmp_path, out, fts):
    make_session(out, 1, [seg(0, "Ёлка и всё остальное", "Me", "ru")])
    make_session(out, 2, [seg(0, "Елка стоит", "Me", "ru")])
    idx = index(tmp_path, fts)
    idx.sync(out)
    assert len(idx.search("елка")) == 2 and len(idx.search("ЁЛКА")) == 2
    assert len(idx.search("все остальное")) == 1
    [hit] = idx.search("остальное")
    assert hit.snippet == "Ёлка и всё [остальное]"  # shown as written, ё included


def test_same_mtime_but_new_size_is_reindexed(tmp_path, out):
    a = make_session(out, 1, [seg(0, "first")])
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    idx.sync(out)
    f = a / "transcript.json"
    st = f.stat()
    write_outputs(SessionInfo.load(a), [seg(0, "second and longer")], 60.0, "fake")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))  # a coarse clock: mtime unchanged
    assert idx.sync(out) == 1 and idx.search("second")


def test_first_connection_from_several_threads(tmp_path, out):
    make_session(out, 1, [seg(0, "parallel start")])
    TranscriptIndex(tmp_path / "idx.sqlite3").sync(out)
    idx = TranscriptIndex(tmp_path / "idx.sqlite3")
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: len(idx.search("parallel")), range(16)))
    assert results == [1] * 16


def test_controller_drops_deleted_sessions_at_startup(out):
    from recandcc.ui.common import Controller

    a = make_session(out, 1, [seg(0, "forgotten meeting")])
    search.default_index().index_session(a)
    for f in a.iterdir():
        f.unlink()
    a.rmdir()
    ctrl = Controller.__new__(Controller)  # only the startup scan, no runtime
    from recandcc.config import Config

    ctrl.cfg = Config(output_dir=out)
    ctrl._resubmit_unfinished()
    assert search.default_index().search("forgotten") == []


def test_osascript_takes_the_script_on_stdin_not_argv(monkeypatch):
    from recandcc.ui import common

    calls = []

    def run(argv, **kw):
        calls.append((argv, kw.get("input")))
        return type("R", (), {"stdout": "ok\n"})()

    monkeypatch.setattr(common.subprocess, "run", run)
    assert common.osascript('choose from list {"the [budget] secret"}') == "ok"
    [(argv, stdin)] = calls
    assert "secret" not in " ".join(argv) and "secret" in stdin
