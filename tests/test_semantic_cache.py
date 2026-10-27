"""The in-process vector cache of semantic search: reused across queries, dropped whenever the
stored vectors change (in this process or another), released when semantic search is turned
off, and ranking exactly like the streaming scan it replaces.

Stories: US-32.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading

import numpy as np
import pytest
from test_search import _corrupt, make_session, seg
from test_semantic import DIM, FakeEmbedder, sync

from recandcc import search, semantic
from recandcc.search import TranscriptIndex


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


@pytest.fixture(autouse=True)
def fresh_default_index(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "_default", None)
    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "default" / "s.sqlite3")


def make(tmp_path, embedder=None) -> TranscriptIndex:
    return TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=embedder)


def count_vector_loads(idx: TranscriptIndex) -> list[str]:
    """Every SQL statement that reads stored vectors, on any connection of ``idx``."""
    loads: list[str] = []
    real = idx._open

    def traced():
        db = real()
        db.set_trace_callback(lambda sql: loads.append(sql) if "v.vec" in sql else None)
        return db

    idx._open = traced  # type: ignore[method-assign]
    return loads


def rewrite(d, old: str, new: str) -> None:
    f = d / "transcript.json"
    f.write_text(f.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


def test_cache_is_reused_across_queries(tmp_path, out):
    make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    loads = count_vector_loads(idx)
    first = idx.search("payment")
    assert first and len(loads) == 1  # one chunk read builds the cache
    for _ in range(5):
        assert idx.search("payment") == first
        assert idx.search("pizza") and idx.search("budget") == first
    assert len(loads) == 1


def test_results_match_the_streaming_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCAN_CHUNK", 7)  # several chunks, a running top-k
    monkeypatch.setattr(search, "SEMANTIC_MIN_SIM", 0.05)
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3")
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((60, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    with idx._connect() as db:
        for f in range(3):
            db.execute("INSERT INTO sessions VALUES (?, ?, 'T', 's', 0)", (f"f{f}", f"s{f}"))
            db.execute("INSERT INTO sem_sessions VALUES (?, 's', 'fake-384')", (f"f{f}",))
        db.executemany(
            "INSERT INTO seg_vec VALUES (?, ?, 'Me', 'en', ?, ?)",
            [(f"f{i % 3}", float(i), f"seg {i}", vecs[i].tobytes()) for i in range(60)],
        )
    queries = rng.standard_normal((8, DIM)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    current = {"q": queries[0]}

    class Fixed:
        model_id = "fake-384"

        def embed(self, texts):
            return current["q"][None, :]

    idx.embedder = Fixed()
    for q in queries:
        current["q"] = q
        cached = idx._semantic("x", 10)
        assert idx._cache is not None
        monkeypatch.setattr(search, "CACHE_MAX_ROWS", 0)
        idx._drop_cache()
        streamed = idx._semantic("x", 10)
        monkeypatch.setattr(search, "CACHE_MAX_ROWS", 10**6)
        assert cached == streamed and [h.seg for h in cached] == [h.seg for h in streamed]


def test_a_new_session_invalidates_the_cache(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert len(idx.search("payment")) == 1 and idx._cache is not None
    d2 = make_session(out, 2, [seg(4, "оплата до пятницы", "Speaker 1", "ru")])
    idx.index_session(d2)
    assert idx.wait_embedded(30)
    assert idx._cache is None  # dropped by the commit that stored the new vectors
    assert {h.folder for h in idx.search("payment")} >= {d2}


@pytest.mark.parametrize("how", ["sync", "discard", "rmtree"])
def test_a_deleted_session_leaves_no_vectors_in_memory(tmp_path, out, how):
    keep = make_session(out, 1, [seg(1, "the invoice")])
    gone = make_session(out, 2, [seg(2, "payment overdue")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert {h.folder for h in idx.search("budget")} == {keep, gone}
    if how == "discard":
        (gone / search.SessionInfo.DISCARD_MARKER).touch()
        idx.index_session(gone)
    else:
        shutil.rmtree(gone)
        if how == "sync":
            idx.sync(out)
        else:
            idx.index_session(gone)
    assert idx._cache is None
    assert idx.wait_embedded(30)  # a running worker would make the next query stream
    assert {h.folder for h in idx.search("budget")} == {keep}
    assert all(m[0] != str(gone) for m in idx._cache.meta)  # type: ignore[union-attr]


def test_a_rewritten_transcript_replaces_its_cached_rows(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert idx.search("payment")
    rewrite(d, "the invoice", "pizza for lunch")
    sync(idx, out)
    assert idx.search("payment") == []
    assert [h.snippet for h in idx.search("обед")] == ["pizza for lunch"]


def test_switching_off_releases_the_cache(out, monkeypatch):
    monkeypatch.setattr(semantic, "default_embedder", FakeEmbedder)
    make_session(out, 1, [seg(1, "the invoice")])
    idx = search.default_index(True)
    sync(idx, out)
    assert idx.search("payment") and idx._cache is not None
    search.default_index(False)
    assert idx._cache is None
    idx.sync(out)  # and the vectors leave the file
    with idx._connect() as db:
        assert db.execute("SELECT count(*) FROM seg_vec").fetchone()[0] == 0


def test_another_process_writing_vectors_invalidates_the_cache(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    app = make(tmp_path, FakeEmbedder())
    sync(app, out)
    assert len(app.search("payment")) == 1
    cli = make(tmp_path, FakeEmbedder())  # same file, its own locks and generation
    d2 = make_session(out, 2, [seg(3, "budget review")])
    sync(cli, out)
    assert app._cache is not None  # this process didn't see the write ...
    assert {h.folder for h in app.search("payment")} >= {d2}  # ... but the token did
    shutil.rmtree(d2)
    sync(cli, out)
    assert d2 not in {h.folder for h in app.search("payment")}


def test_a_write_during_the_build_is_never_cached(tmp_path, out, monkeypatch):
    monkeypatch.setattr(search, "SCAN_CHUNK", 1)  # one connection per row
    make_session(out, 1, [seg(1, "the invoice"), seg(5, "payment due")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    other = make(tmp_path, FakeEmbedder())
    d2 = make_session(out, 2, [seg(3, "budget review")])
    real, calls = idx._connect, []

    def connect_and_write():
        calls.append(1)
        if len(calls) == 3:  # in the middle of the chunked read: another process commits
            sync(other, out)
        return real()

    monkeypatch.setattr(idx, "_connect", connect_and_write)
    hits = idx.search("payment")
    assert idx._cache is None  # the token changed during the build: not kept
    assert d2 in {h.folder for h in hits}  # and the streaming fallback saw the write
    monkeypatch.setattr(idx, "_connect", real)
    assert idx.search("payment") == hits and idx._cache is not None


def test_an_in_process_write_during_the_build_is_never_cached(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    real = idx._build_cache

    def build_then_write(*a):
        built = real(*a)
        d2 = make_session(out, 2, [seg(3, "budget review")])
        idx.index_session(d2)  # commits after the build's token check, before install
        assert idx.wait_embedded(30)
        return built

    monkeypatch.setattr(idx, "_build_cache", build_then_write)
    assert idx.search("payment")
    assert idx._cache is None


def test_no_cache_while_the_embed_worker_runs_or_above_the_cap(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice"), seg(2, "payment")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    monkeypatch.setattr(search, "CACHE_MAX_ROWS", 1)
    assert len(idx.search("budget")) == 2 and idx._cache is None
    monkeypatch.setattr(search, "CACHE_MAX_ROWS", 10)
    idx._embed_thread = threading.current_thread()  # as if the worker were busy
    try:
        assert len(idx.search("budget")) == 2 and idx._cache is None
    finally:
        idx._embed_thread = None
    assert len(idx.search("budget")) == 2 and idx._cache is not None


def test_an_uncacheable_index_is_not_rebuilt_on_every_query(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice"), seg(2, "payment")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    monkeypatch.setattr(search, "CACHE_MAX_ROWS", 1)
    builds: list[str] = []
    real = idx._build_cache

    def counted(token, model, dim):
        builds.append(token)
        return real(token, model, dim)

    monkeypatch.setattr(idx, "_build_cache", counted)
    loads = count_vector_loads(idx)
    first = idx.search("budget")
    assert len(first) == 2 and idx._cache is None and len(builds) == 1
    per_query = len(loads)  # the streaming scan only: the over-cap build read no vectors
    for _ in range(5):
        assert idx.search("budget") == first
    assert len(builds) == 1 and len(loads) == 6 * per_query
    make_session(out, 2, [seg(3, "budget review")])
    sync(idx, out)  # a write: new token, so one new attempt
    assert len(idx.search("budget")) == 3 and len(builds) == 2
    idx.search("budget")
    assert len(builds) == 2


def test_a_writer_unaware_of_the_token_still_invalidates_the_cache(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert idx.search("payment") and idx._cache is not None
    with sqlite3.connect(idx.path) as db:  # e.g. an older CLI build: no vec_version update
        db.execute("DELETE FROM seg_vec")
    assert idx.search("payment") == []


def test_concurrent_search_and_embedding_never_return_wrong_rows(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice"), seg(7, "lunch")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    money = ("invoice", "payment", "budget")
    errors: list[BaseException] = []
    stop = threading.Event()

    def searcher():
        try:
            while not stop.is_set():
                for h in idx.search("payment"):
                    # a cached row carries its own text: never a food line under a money query
                    assert any(w in h.snippet for w in money), h.snippet
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=searcher) for _ in range(3)]
    for t in threads:
        t.start()
    texts = ["the invoice", "pizza time", "the budget", "lunch break", "payment sent"]
    for i in range(1, 12):
        rewrite(d, texts[(i - 1) % 5], texts[i % 5])
        idx.sync(out)
    stop.set()
    for t in threads:
        t.join(20)
    assert idx.wait_embedded(30) and not errors
    final = idx.search("payment")
    idx._drop_cache()
    search_cap = search.CACHE_MAX_ROWS
    try:
        search.CACHE_MAX_ROWS = 0
        assert final == idx.search("payment")  # identical to the streaming scan
    finally:
        search.CACHE_MAX_ROWS = search_cap


def test_an_idle_cache_is_released_and_rebuilt(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    clock = [1000.0]
    monkeypatch.setattr(search, "_now", lambda: clock[0])
    loads = count_vector_loads(idx)
    first = idx.search("payment")
    assert first and len(loads) == 1
    clock[0] += search.CACHE_IDLE_S - 1  # used recently: kept
    assert idx.release_idle_cache() is False
    assert idx.search("payment") == first and len(loads) == 1  # a hit renews the timestamp
    clock[0] += search.CACHE_IDLE_S - 1
    assert idx.release_idle_cache() is False
    clock[0] += 2
    assert idx.release_idle_cache() is True and idx._cache is None
    assert idx.release_idle_cache() is False  # nothing left to release
    assert idx.search("payment") == first and len(loads) == 2  # rebuilt lazily, same answer


def test_a_speaker_rename_refreshes_the_cached_metadata(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice", speaker="Speaker 1")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert [h.speaker for h in idx.search("payment")] == ["Speaker 1"]
    rewrite(d, "Speaker 1", "Alice")  # what naming a speaker does to transcript.json
    sync(idx, out)
    hits = idx.search("payment")
    assert [h.speaker for h in hits] == ["Alice"]
    assert [h.speaker for h in idx.search("invoice")] == ["Alice"]


def test_other_models_rows_do_not_count_against_the_cap(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice"), seg(2, "payment")])
    d2 = make_session(out, 2, [seg(3, "lunch"), seg(4, "pizza")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    with sqlite3.connect(idx.path) as db:  # session 2 was embedded by another model
        db.execute("UPDATE sem_sessions SET model = 'old' WHERE folder = ?", (str(d2),))
    monkeypatch.setattr(search, "CACHE_MAX_ROWS", 2)  # 4 rows stored, 2 of this model
    assert idx.search("payment")
    assert idx._cache is not None and len(idx._cache.meta) == 2


def test_recovering_a_corrupt_index_drops_the_cache(tmp_path, out):
    make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert idx.search("payment")
    assert idx._cache is not None
    _corrupt(idx.path, "garbage")
    assert idx._recover()  # deletes the file and must drop the in-memory matrix too
    assert idx._cache is None and idx._no_cache is None
    assert idx.search("payment") == []  # no stale text from the deleted index
    sync(idx, out)  # the rebuild re-embeds from the transcripts
    hits = idx.search("payment")
    assert hits and "invoice" in hits[0].snippet
    assert idx._cache is not None


def test_a_missing_vec_version_reported_by_a_trigger_counts_as_missing_schema():
    import sqlite3

    from recandcc.search import _is_missing_schema

    assert _is_missing_schema(sqlite3.OperationalError("no such table: main.vec_version"))
    assert _is_missing_schema(sqlite3.OperationalError("no such table: vec_version"))
    assert not _is_missing_schema(sqlite3.OperationalError("no such table: main.unrelated"))
