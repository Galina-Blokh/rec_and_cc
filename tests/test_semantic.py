"""Opt-in semantic search: pinned one-time model download, embeddings stored next to the FTS
index, reciprocal-rank fusion, and unchanged results when it is off. The fast tests use a
fake embedder; the real model is only in the slow test.

Stories: US-32.
"""

from __future__ import annotations

import hashlib
import io
import logging
import shutil
import sqlite3
import statistics
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_search import make_session, seg
from test_search_latency import _factor

from recandcc import search, semantic
from recandcc.config import Config, load_config
from recandcc.models import SessionInfo
from recandcc.search import TranscriptIndex, rrf

# One dimension per concept; words of different languages share a concept.
CONCEPTS = {
    "money": ["invoice", "payment", "оплата", "счёт", "budget"],
    "travel": ["flight", "trip", "перелёт", "поездка"],
    "food": ["lunch", "pizza", "обед"],
}


class FakeEmbedder:
    def __init__(self, available: bool = True, fail: bool = False) -> None:
        self.available, self.fail, self.calls = available, fail, []

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("boom")
        if not self.available:
            return None
        out = np.zeros((len(texts), len(CONCEPTS)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, words in enumerate(CONCEPTS.values()):
                out[i, j] = sum(w in t.lower() for w in words)
            out[i] /= max(np.linalg.norm(out[i]), 1e-9)
        return out


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


@pytest.fixture(autouse=True)
def fresh_default_index(tmp_path, monkeypatch):
    """The module-level default index is global state: start every test without one."""
    monkeypatch.setattr(search, "_default", None)
    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "default" / "s.sqlite3")


def sync(idx, out) -> int:
    """sync plus the background embedding, which sync itself never waits for."""
    n = idx.sync(out)
    assert idx.wait_embedded(30)
    return n


def make(tmp_path, embedder=None) -> TranscriptIndex:
    return TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=embedder)


def vector_rows(idx) -> int:
    with sqlite3.connect(idx.path) as db:
        return db.execute("SELECT count(*) FROM seg_vec").fetchone()[0]


# --------------------------------------------------------------------------- off by default
def test_semantic_is_off_by_default(tmp_path):
    assert Config().search.semantic is False
    cfg = tmp_path / "config.toml"
    assert load_config(cfg).search.semantic is False  # creates the default file
    assert "semantic = false" in cfg.read_text()
    cfg.write_text("[search]\nsemantic = true\n")
    assert load_config(cfg).search.semantic is True


def test_off_embeds_nothing_and_matches_full_text(tmp_path, out):
    make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    plain = make(tmp_path)
    sync(plain, out)
    assert vector_rows(plain) == 0
    assert plain.search("invoice") == plain._lexical("invoice")
    assert plain.search("payment") == []  # no word match, no semantic fallback
    assert search.default_index(False).embedder is None


# --------------------------------------------------------------------------- on
def test_semantic_finds_by_meaning_and_across_languages(tmp_path, out):
    a = make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    b = make_session(out, 2, [seg(3, "Нужна оплата до пятницы", "Speaker 1", "ru")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert vector_rows(idx) == 3
    hits = idx.search("payment")  # shares no word with either text
    assert {(h.folder, h.start) for h in hits} == {(a, 1.0), (b, 3.0)}
    assert all("[" not in h.snippet for h in hits)  # no word highlight for meaning matches
    assert idx.search("flight") == []  # nothing close enough


def test_fusion_ranks_a_hit_in_both_lists_first_and_keeps_its_highlight(tmp_path, out):
    make_session(
        out, 1, [seg(1, "payment reminder lunch"), seg(5, "send the invoice"), seg(9, "Pizza")]
    )
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    hits = idx.search("payment")
    assert [h.start for h in hits] == [1.0, 5.0]  # word + meaning match, then meaning only
    assert "[payment]" in hits[0].snippet


def test_rrf_orders_by_summed_reciprocal_rank():
    def hit(n):
        return search.Hit(Path(f"/f{n}"), "s", "", float(n), "Me", "en", f"t{n}")

    a, b, c = hit(1), hit(2), hit(3)
    assert rrf([[a, b], [c, b]]) == [b, a, c]
    assert rrf([[a, b], []]) == [a, b] and rrf([]) == []
    assert rrf([[replace(a, snippet="x")], [a]])[0].snippet == "x"  # first list's copy wins


def test_embedding_is_incremental_and_follows_deletes(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    sync(idx, out)
    sync(idx, out)
    assert len(emb.calls) == 1  # unchanged: nothing re-embedded
    make_session(out, 2, [seg(1, "a pizza")])
    sync(idx, out)
    assert len(emb.calls) == 2 and emb.calls[-1] == ["a pizza"]  # only the new session
    shutil.rmtree(d)
    sync(idx, out)
    assert vector_rows(idx) == 1


def test_rewritten_transcript_is_re_embedded(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    sync(idx, out)
    (d / "transcript.json").write_text(
        (d / "transcript.json").read_text(encoding="utf-8").replace("invoice", "pizza trip"),
        encoding="utf-8",
    )
    sync(idx, out)
    assert emb.calls[-1] == ["the pizza trip"]
    assert vector_rows(idx) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_index_stays_owner_only_after_switching_off(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert vector_rows(idx) == 1
    idx.path.chmod(0o644)  # as if the file had been rebuilt with a loose umask
    off = make(tmp_path)
    sync(off, out)
    assert vector_rows(off) == 0
    assert (off.path.stat().st_mode & 0o777) == 0o600
    assert (off.path.parent.stat().st_mode & 0o777) == 0o700


def test_switching_off_forgets_the_vectors(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    path_idx = make(tmp_path, FakeEmbedder())
    sync(path_idx, out)
    assert vector_rows(path_idx) == 1
    off = make(tmp_path)
    sync(off, out)
    assert vector_rows(off) == 0
    assert [h.start for h in off.search("invoice")] == [1.0]


def test_unavailable_model_falls_back_to_full_text_and_retries(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder(available=False)
    idx = make(tmp_path, emb)
    sync(idx, out)
    assert vector_rows(idx) == 0
    assert [h.start for h in idx.search("invoice")] == [1.0]
    emb.available = True
    sync(idx, out)  # the model showed up: the backlog is embedded
    assert vector_rows(idx) == 1


def test_embed_failure_is_contained_and_logs_no_text(tmp_path, out, caplog):
    make_session(out, 1, [seg(1, "the zebrafruit invoice")])
    emb = FakeEmbedder(fail=True)
    idx = make(tmp_path, emb)
    with caplog.at_level(logging.INFO):
        sync(idx, out)  # _embed_pending hits the exception path
    assert emb.calls and vector_rows(idx) == 0
    assert (
        "zebrafruit" not in caplog.text and "semantic indexing failed: RuntimeError" in caplog.text
    )
    assert [h.start for h in idx.search("invoice")] == [1.0]  # query-time failure too


def test_query_text_is_not_logged_when_semantic_search_fails(tmp_path, out, caplog):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    sync(idx, out)
    emb.fail = True
    with caplog.at_level(logging.INFO):
        assert idx.search("invoice quokkaberry") == []
        assert [h.start for h in idx.search("invoice")] == [1.0]
    assert "quokkaberry" not in caplog.text and "semantic search failed" in caplog.text


def test_discard_marker_drops_vectors_via_index_session(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    assert vector_rows(idx) == 1
    (d / SessionInfo.DISCARD_MARKER).write_text("")
    assert idx.index_session(d) is False
    assert idx.wait_embedded(30) and vector_rows(idx) == 0 and idx.search("invoice") == []


def test_index_session_embeds_the_new_transcript(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    assert idx.index_session(d)
    assert idx.wait_embedded(30) and vector_rows(idx) == 1


class SlowEmbedder(FakeEmbedder):
    """Blocks inside embed until released, like a model that is still downloading."""

    def __init__(self) -> None:
        super().__init__()
        self.release, self.entered = threading.Event(), threading.Event()

    def ready(self):
        return self.release.is_set()  # "loaded" only once the download finished

    def embed(self, texts):
        self.entered.set()
        assert self.release.wait(10)
        return super().embed(texts)


def test_sync_and_search_never_wait_for_embedding(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice"), seg(5, "a payment")])
    emb = SlowEmbedder()
    idx = make(tmp_path, emb)
    t0 = time.monotonic()
    assert idx.sync(out) == 1  # returns while the worker is stuck in embed
    assert emb.entered.wait(5)
    assert [h.start for h in idx.search("invoice")] == [1.0]  # full text, no vectors yet
    assert time.monotonic() - t0 < 5 and vector_rows(idx) == 0
    emb.release.set()
    assert idx.wait_embedded(30) and vector_rows(idx) == 2


def test_one_worker_runs_at_a_time(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = SlowEmbedder()
    idx = make(tmp_path, emb)
    threads = [threading.Thread(target=idx.sync, args=(out,)) for _ in range(6)]
    for t in threads:
        t.start()
    assert emb.entered.wait(5)
    for t in threads:
        t.join(5)
    emb.release.set()
    assert idx.wait_embedded(30)
    assert len(emb.calls) == 1 and vector_rows(idx) == 1  # no duplicate work or rows


def test_search_skips_semantics_until_the_model_is_loaded(tmp_path, out):
    class Cold(FakeEmbedder):
        loaded = False

        def ready(self):
            return self.loaded

        def embed(self, texts):
            self.loaded = True  # the worker's warm-up (or first batch) loads it
            return super().embed(texts)

    make_session(out, 1, [seg(1, "send the invoice")])
    emb = Cold()
    idx = make(tmp_path, emb)
    idx.sync(out)
    idx.wait_embedded(30)
    emb.loaded = False
    assert idx.search("payment") == []  # not loaded: no query embedding, no wait
    n = len(emb.calls)
    assert idx.wait_embedded(30) and emb.loaded  # but a worker was started to load it
    assert len(emb.calls) >= n
    assert [h.start for h in idx.search("payment")] == [1.0]


def test_switching_off_while_embedding_stores_nothing(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = SlowEmbedder()
    idx = make(tmp_path, emb)
    idx.sync(out)
    assert emb.entered.wait(5)
    idx.set_embedder(None)
    idx.sync(out)  # the off path clears
    emb.release.set()
    assert idx.wait_embedded(30)
    assert vector_rows(idx) == 0
    with sqlite3.connect(idx.path) as db:
        assert db.execute("SELECT count(*) FROM seg_fts").fetchone()[0] == 1  # text stays
    assert [h.start for h in idx.search("invoice")] == [1.0]


def test_default_index_is_one_index_with_a_swapped_embedder(monkeypatch):
    monkeypatch.setattr(semantic, "default_embedder", lambda: FakeEmbedder())
    on = search.default_index(True)
    off = search.default_index(False)
    assert on is off and off.embedder is None and search.default_index(True).embedder is not None


def test_concurrent_sync_and_search_with_a_changing_transcript(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    errors = []

    def searcher():
        try:
            for _ in range(30):
                idx.search("invoice")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=searcher) for _ in range(3)]
    for t in threads:
        t.start()
    for i in range(3):
        (d / "transcript.json").write_text(
            (d / "transcript.json").read_text(encoding="utf-8").replace("invoice", f"invoice{i}x"),
            encoding="utf-8",
        )
        idx.sync(out)
    for t in threads:
        t.join(20)
    assert idx.wait_embedded(30) and not errors
    with sqlite3.connect(idx.path) as db:  # the stamp re-check left one fresh row set
        assert db.execute("SELECT count(*) FROM seg_vec").fetchone()[0] == 1
        stamp = db.execute("SELECT stamp FROM sessions").fetchone()[0]
        assert db.execute("SELECT stamp FROM sem_sessions").fetchone()[0] == stamp


def test_vectors_of_another_model_are_re_embedded(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    emb.model_id = "v1"
    idx = make(tmp_path, emb)
    sync(idx, out)
    sync(idx, out)
    assert len(emb.calls) == 1
    emb.model_id = "v2"
    sync(idx, out)
    assert len(emb.calls) == 2 and vector_rows(idx) == 1
    with sqlite3.connect(idx.path) as db:
        assert db.execute("SELECT model FROM sem_sessions").fetchone()[0] == "v2"


def test_v2_index_is_rebuilt_and_re_embedded_with_the_model_tag(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    path = tmp_path / "idx" / "s.sqlite3"
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as db:  # a v2 index: sem_sessions without a model column
        db.execute("CREATE TABLE sem_sessions (folder TEXT PRIMARY KEY, stamp TEXT)")
        db.execute("INSERT INTO sem_sessions VALUES ('gone', 'x')")
        db.execute("CREATE TABLE sessions (folder TEXT PRIMARY KEY, session_id TEXT)")
        db.execute("INSERT INTO sessions VALUES ('gone', 'old')")
        db.execute("PRAGMA user_version = 2")
    emb = FakeEmbedder()
    emb.model_id = "m1"
    idx = TranscriptIndex(path, embedder=emb)
    assert sync(idx, out) == 1
    assert [h.start for h in idx.search("invoice")] == [1.0]
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == search.SCHEMA_VERSION
        assert db.execute("SELECT model FROM sem_sessions").fetchall() == [("m1",)]
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 1
    assert vector_rows(idx) == 1


def test_embedder_swapped_during_a_search_falls_back_to_full_text(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    sync(idx, out)
    other = FakeEmbedder()
    other.embed = lambda texts: np.ones((len(texts), 7), dtype=np.float32)  # another dimension
    other.model_id = emb.model_id = "same"  # the stored rows match; only the width differs
    real = emb.embed

    def swapping(texts):
        idx.embedder = other
        return real(texts)

    emb.embed = swapping
    assert [h.start for h in idx.search("invoice")] == [1.0]  # full text, no exception
    assert idx._semantic("invoice", 5) == []


def test_failed_embedder_start_does_not_wedge_the_worker(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice")])
    idx = make(tmp_path, FakeEmbedder())
    real_start = threading.Thread.start

    def broken(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", broken)
    idx.sync(out)
    assert idx._embed_thread is None and idx.wait_embedded(1)
    monkeypatch.setattr(threading.Thread, "start", real_start)
    sync(idx, out)  # a later kick works
    assert vector_rows(idx) == 1


def test_meaning_hits_rank_after_exact_word_hits(tmp_path, out):
    make_session(
        out,
        1,
        [seg(1, "payment due"), seg(2, "the invoice is paid"), seg(3, "an invoice")],
    )
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    hits = idx.search("invoice")
    # words and meaning, words and meaning, then the meaning-only hit
    assert [h.start for h in hits][-1] == 1.0 and len(hits) == 3
    assert {h.start for h in hits[:2]} == {2.0, 3.0}
    assert search.SEMANTIC_MIN_SIM >= 0.45


def test_me_and_others_at_the_same_timestamp_stay_separate():
    a = search.Hit(Path("/f"), "s", "", 1.0, "Me", "en", "x")
    b = replace(a, speaker="Speaker 1", snippet="y")
    assert rrf([[a, b]]) == [a, b]
    assert search.fuse([a], [b]) == [a, b]


def test_semantic_reads_only_the_winners_text(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice"), seg(2, "lunch")])
    idx = make(tmp_path, FakeEmbedder())
    sync(idx, out)
    hits = idx._semantic("payment", 5)
    assert [(h.start, h.snippet) for h in hits] == [(1.0, "the invoice")]


def test_default_index_follows_the_setting(monkeypatch):
    monkeypatch.setattr(semantic, "default_embedder", lambda: FakeEmbedder())
    assert search.default_index(True).embedder is not None
    assert search.default_index(False).embedder is None


# --------------------------------------------------------------------------- model files
def test_pins_are_exact():
    for f in (semantic.MODEL, semantic.TOKENIZER):
        assert f.url.startswith("https://") and semantic.REVISION in f.url
        assert len(f.sha256) == 64 and f.size > 0
    assert len(semantic.REVISION) == 40


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(semantic, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(semantic, "_last_failure", 0.0)
    return tmp_path / "models"


def _left(d: Path) -> list[str]:
    """Files in the model dir, without the (deliberately kept, empty) download lock file."""
    return sorted(p.name for p in d.iterdir() if p.name != ".download.lock")


def _fake_model(data: bytes) -> semantic.ModelFile:
    return semantic.ModelFile(
        "m.onnx", "https://example.invalid/m.onnx", hashlib.sha256(data).hexdigest(), len(data)
    )


def _urlopen(monkeypatch, data: bytes | None):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        if data is None:
            raise OSError("network is unreachable")
        return io.BytesIO(data)

    monkeypatch.setattr(semantic, "_opener", lambda: SimpleNamespace(open=urlopen))
    return calls


def test_model_on_disk_is_used_without_network(model_dir, monkeypatch):
    data = b"onnx" * 100
    model_dir.mkdir()
    (model_dir / "m.onnx").write_bytes(data)
    calls = _urlopen(monkeypatch, None)
    assert semantic.model_path(_fake_model(data)) == model_dir / "m.onnx"
    assert calls == []


def test_model_is_downloaded_once_and_verified(model_dir, monkeypatch):
    data = b"weights" * 1000
    calls = _urlopen(monkeypatch, data)
    path = semantic.model_path(_fake_model(data))
    assert path is not None and path.read_bytes() == data and len(calls) == 1
    assert semantic.model_path(_fake_model(data)) == path and len(calls) == 1
    # no temp file left (the empty lock file stays: deleting it would race another process)
    assert _left(model_dir) == ["m.onnx"]


def test_corrupt_file_of_the_right_size_is_replaced(model_dir, monkeypatch):
    data = b"weights" * 1000
    model_dir.mkdir()
    (model_dir / "m.onnx").write_bytes(b"x" * len(data))  # same size, wrong content
    calls = _urlopen(monkeypatch, data)
    path = semantic.model_path(_fake_model(data))
    assert path is not None and path.read_bytes() == data and len(calls) == 1
    semantic.model_path(_fake_model(data))  # verified once, then trusted
    assert len(calls) == 1


def test_model_that_fails_to_load_is_deleted_for_a_fresh_download(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    import onnxruntime

    files = {}
    for m in (semantic.MODEL, semantic.TOKENIZER):
        files[m.name] = tmp_path / m.name
        files[m.name].write_bytes(b"junk")
    monkeypatch.setattr(semantic, "model_path", lambda m, download=True, **k: files[m.name])

    def boom(*a, **k):
        raise RuntimeError("invalid protobuf")

    monkeypatch.setattr(onnxruntime, "InferenceSession", boom)
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        staticmethod(
            lambda p: type(
                "T", (), {"enable_truncation": lambda *a: None, "enable_padding": lambda *a: None}
            )()
        ),
    )
    emb = semantic.OnnxEmbedder()
    assert emb.embed(["x"]) is None and not emb.ready()
    assert not any(f.exists() for f in files.values())


def test_oversized_download_is_aborted_and_deleted(model_dir, monkeypatch):
    data = b"weights" * 1000
    reads = []

    class Endless(io.RawIOBase):
        def read(self, n=-1):
            reads.append(n)
            return b"x" * n if len(reads) < 1000 else b""

    monkeypatch.setattr(
        semantic, "_opener", lambda: SimpleNamespace(open=lambda url, timeout=None: Endless())
    )
    assert semantic.model_path(_fake_model(data)) is None
    assert len(reads) == 1  # stopped at the first block over the expected size
    assert _left(model_dir) == []


def test_short_download_is_rejected(model_dir, monkeypatch):
    data = b"weights" * 1000
    _urlopen(monkeypatch, data[:-1])
    assert semantic.model_path(_fake_model(data)) is None
    assert _left(model_dir) == []


def test_load_failure_of_verified_files_does_not_redownload(tmp_path, monkeypatch):
    """Stories: US-32."""
    pytest.importorskip("onnxruntime")
    import onnxruntime

    files = {m.name: tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER)}
    for f in files.values():
        f.write_bytes(b"junk")
    calls = []

    def model_path(m, download=True, **k):
        calls.append(m.name)
        return files[m.name]

    monkeypatch.setattr(semantic, "model_path", model_path)
    monkeypatch.setattr(semantic, "_verified", set(files.values()))
    monkeypatch.setattr(semantic, "_last_failure", 0.0)
    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("abi"))
    )
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        staticmethod(
            lambda p: type(
                "T", (), {"enable_truncation": lambda *a: None, "enable_padding": lambda *a: None}
            )()
        ),
    )
    emb = semantic.OnnxEmbedder()
    assert emb.embed(["x"]) is None
    assert emb.embed(["x"]) is None and emb.embed(["x"]) is None
    assert len(calls) == 2  # looked up once (model, tokenizer); never again
    assert all(f.exists() for f in files.values())  # verified files are not corrupt
    assert semantic._last_failure > 0


def test_unverified_files_that_fail_to_load_are_deleted_and_back_off(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    import onnxruntime

    files = {m.name: tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER)}
    for f in files.values():
        f.write_bytes(b"junk")
    monkeypatch.setattr(semantic, "model_path", lambda m, download=True, **k: files[m.name])
    monkeypatch.setattr(semantic, "_verified", set())
    monkeypatch.setattr(semantic, "_last_failure", 0.0)
    monkeypatch.setattr(
        onnxruntime, "InferenceSession", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
    )
    monkeypatch.setattr(
        "tokenizers.Tokenizer.from_file",
        staticmethod(
            lambda p: type(
                "T", (), {"enable_truncation": lambda *a: None, "enable_padding": lambda *a: None}
            )()
        ),
    )
    assert semantic.OnnxEmbedder().embed(["x"]) is None
    assert not any(f.exists() for f in files.values()) and semantic._last_failure > 0


def test_bad_checksum_is_rejected(model_dir, monkeypatch):
    data = b"weights" * 1000
    _urlopen(monkeypatch, b"tampered" + data[8:])
    assert semantic.model_path(_fake_model(data)) is None
    assert _left(model_dir) == []


def test_non_https_url_is_refused(model_dir, monkeypatch):
    calls = _urlopen(monkeypatch, b"x")
    bad = replace(_fake_model(b"x"), url="http://example.invalid/m.onnx")
    assert semantic.model_path(bad) is None and calls == []


def test_offline_download_fails_gracefully_and_backs_off(model_dir, monkeypatch):
    calls = _urlopen(monkeypatch, None)
    assert semantic.model_path(_fake_model(b"weights")) is None
    assert semantic.model_path(_fake_model(b"weights")) is None
    assert len(calls) == 1  # not retried right away
    assert semantic.model_path(_fake_model(b"weights"), download=False) is None


def test_embedder_without_model_returns_none(model_dir, monkeypatch):
    _urlopen(monkeypatch, None)
    assert semantic.OnnxEmbedder().embed(["hello"]) is None


# --------------------------------------------------------------------------- real model
@pytest.mark.slow
def test_real_model_matches_meaning_in_english_and_russian(tmp_path):
    emb = semantic.OnnxEmbedder()
    vecs = emb.embed(["I need to pay the bill", "Нужно оплатить счёт", "The weather is sunny"])
    if vecs is None:
        pytest.skip("search model not downloaded and can't be fetched")
    q = emb.embed(["payment of an invoice"])
    assert q is not None
    sims = vecs @ q[0]
    assert vecs.shape == (3, semantic.DIM) and np.allclose(
        np.linalg.norm(vecs, axis=1), 1, atol=1e-3
    )
    assert sims[0] > sims[2] and sims[1] > sims[2]
    # the floor separates related pairs from unrelated ones, in both languages
    floor = search.SEMANTIC_MIN_SIM
    assert sims[0] >= floor and sims[1] >= floor and sims[2] < floor
    for query in ("invoice", "Galina"):
        unrelated = emb.embed([query])
        assert unrelated is not None and (vecs[2:] @ unrelated[0]).max() < floor
    ru = emb.embed(["оплата счёта"])
    assert ru is not None and (vecs[:2] @ ru[0]).min() >= floor


# --------------------------------------------------------------------------- resources, load
def _fake_onnx(monkeypatch, tmp_path, tracker):
    """Fake onnxruntime + tokenizers (no model files); `tracker` records concurrent entries."""
    files = {}
    for m in (semantic.MODEL, semantic.TOKENIZER):
        files[m.name] = tmp_path / m.name
        files[m.name].write_bytes(b"x")
    monkeypatch.setattr(semantic, "model_path", lambda m, download=True, **k: files[m.name])

    class Enc:
        ids = [1, 2, 3]
        attention_mask = [1, 1, 1]

    class Tok:
        def enable_truncation(self, n):
            pass

        def enable_padding(self):
            pass

        def encode_batch(self, texts):
            with tracker("tok"):
                return [Enc() for _ in texts]

    class Session:
        def run(self, _out, feed):
            with tracker("ort"):
                n = feed["input_ids"].shape[0]
                return [np.ones((n, 3, semantic.DIM), dtype=np.float32)]

    ort = type(
        "ort",
        (),
        {
            "SessionOptions": type("O", (), {}),
            "InferenceSession": staticmethod(lambda path, opts, **kw: Session()),
        },
    )
    tokmod = type(
        "tokmod", (), {"Tokenizer": type("T", (), {"from_file": staticmethod(lambda p: Tok())})}
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", tokmod)


def test_close_unloads_the_model_and_a_later_embed_reloads_it(tmp_path, monkeypatch):
    import contextlib

    loads = []
    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    real = sys.modules["onnxruntime"].InferenceSession
    sys.modules["onnxruntime"].InferenceSession = staticmethod(
        lambda p, o, **k: (loads.append(1), real(p, o, **k))[1]
    )
    emb = semantic.OnnxEmbedder()
    assert emb.embed(["a"]).shape == (1, semantic.DIM) and emb.ready()
    emb.close()
    assert not emb.ready() and emb._loaded is None
    assert emb.embed(["b"]) is None and not emb.ready() and len(loads) == 1  # no reload
    assert emb.prepare() is True and emb.ready() and len(loads) == 2  # only prepare() does


def test_switching_off_unloads_the_default_model(tmp_path, monkeypatch):
    import contextlib

    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    emb = semantic.OnnxEmbedder()
    monkeypatch.setattr(semantic, "_default", emb)
    emb.embed(["a"])
    assert search.default_index(True).embedder is emb and emb.ready()
    search.default_index(False)
    assert not emb.ready() and emb._loaded is None
    assert search.default_index(True).embedder is emb
    assert emb.embed(["a"]) is None and not emb.ready()  # a straggler doesn't reload
    assert emb.prepare() and emb.ready()  # the worker's prepare() after enabling again does


def test_embed_calls_never_overlap_in_the_tokenizer_or_session(tmp_path, monkeypatch):
    import contextlib

    guard, inside, overlaps = threading.Lock(), [0], []

    @contextlib.contextmanager
    def tracker(name):
        with guard:
            inside[0] += 1
            if inside[0] > 1:
                overlaps.append(name)
        time.sleep(0.01)
        try:
            yield
        finally:
            with guard:
                inside[0] -= 1

    _fake_onnx(monkeypatch, tmp_path, tracker)
    emb = semantic.OnnxEmbedder()
    emb.embed(["warm"])
    threads = [
        threading.Thread(target=lambda: [emb.embed(["a", "b"]) for _ in range(5)]) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert overlaps == []


# --------------------------------------------------------------------------- recording pause
def test_embedding_pauses_while_busy_and_resumes(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    state = {"busy": True}
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb, busy=lambda: state["busy"])
    idx.busy_poll_s = 0.01
    idx.sync(out)
    time.sleep(0.3)
    assert emb.calls == [] and vector_rows(idx) == 0  # paused: a recording is running
    state["busy"] = False
    assert idx.wait_embedded(30)
    assert emb.calls and vector_rows(idx) == 1


def test_a_long_session_is_embedded_in_chunks_that_each_wait_for_idle(tmp_path, out, monkeypatch):
    monkeypatch.setattr(search, "EMBED_CHUNK", 2)
    make_session(out, 1, [seg(i, f"invoice {i}") for i in range(5)])
    state = {"busy": False}
    emb = FakeEmbedder()
    first_chunk = threading.Event()
    real = emb.embed

    def embed(texts):
        result = real(texts)
        if len(emb.calls) == 1:
            state["busy"] = True  # a recording starts right after the first chunk
            first_chunk.set()
        return result

    emb.embed = embed
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb, busy=lambda: state["busy"])
    idx.busy_poll_s = 0.01
    idx.sync(out)
    assert first_chunk.wait(10)
    time.sleep(0.3)
    assert [len(c) for c in emb.calls] == [2]  # paused mid-session: no more calls while busy
    assert vector_rows(idx) == 0
    state["busy"] = False
    assert idx.wait_embedded(30)
    assert [len(c) for c in emb.calls] == [2, 2, 1] and vector_rows(idx) == 5


class PrepEmbedder(FakeEmbedder):
    """Has the optional ``prepare`` hook (download + load) that the worker drives."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded, self.prepared = False, 0

    def ready(self):
        return self.loaded

    def prepare(self, wait=None):
        self.prepared += 1
        if wait is not None and not wait():
            return False
        self.loaded = True
        return True


def test_first_download_and_load_never_start_while_busy(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = PrepEmbedder()
    state = {"busy": True}
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb, busy=lambda: state["busy"])
    idx.busy_poll_s = 0.01
    idx.sync(out)
    time.sleep(0.3)
    assert emb.prepared == 0 and emb.calls == []  # a recording is running
    state["busy"] = False
    assert idx.wait_embedded(30)
    assert emb.prepared == 1 and vector_rows(idx) == 1


def test_switching_off_cancels_a_waiting_prepare(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = PrepEmbedder()
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb, busy=lambda: True)
    idx.busy_poll_s = 0.01
    idx.sync(out)
    idx.set_embedder(None)  # does not block on the paused worker
    assert idx.wait_embedded(10) and emb.prepared == 0 and vector_rows(idx) == 0


def test_download_pauses_while_busy_and_resumes(model_dir, monkeypatch):
    data = b"w" * (3 << 20)
    _urlopen(monkeypatch, data)
    busy, polls, paused_at = threading.Event(), [0], []
    busy.set()

    def wait():
        polls[0] += 1
        if polls[0] == 2:  # a recording starts between two blocks
            paused_at.append(time.monotonic())
            time.sleep(0.2)
        return True

    path = semantic.model_path(_fake_model(data), wait=wait)
    assert path is not None and path.read_bytes() == data and polls[0] >= 3
    assert paused_at


class _Resp(io.BytesIO):
    """A response that can raise on read once its connection has sat idle too long."""

    def __init__(self, data, status=200, content_range="", die_after=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range} if content_range else {}
        self.die_after = die_after
        self.sent = 0

    def read(self, n=-1):
        if self.die_after is not None and self.sent >= self.die_after:
            raise ConnectionResetError("idle timeout")
        block = super().read(n)
        self.sent += len(block)
        return block


def _resumable(monkeypatch, data, ranges=True):
    """urlopen whose first connection dies after 1 MiB once it has idled; later ones honour
    Range (or ignore it, with ranges=False)."""
    opened = []

    def urlopen(url, timeout=None):
        rng = url.get_header("Range") if hasattr(url, "get_header") else None
        opened.append(rng)
        if rng is None or not ranges:
            return _Resp(data, die_after=1 << 20 if not opened[:-1] else None)
        start = int(rng.split("=")[1].rstrip("-"))
        return _Resp(data[start:], 206, f"bytes {start}-{len(data) - 1}/{len(data)}")

    monkeypatch.setattr(semantic, "_opener", lambda: SimpleNamespace(open=urlopen))
    return opened


@pytest.mark.parametrize("ranges", [True, False])
def test_download_reopens_after_a_long_pause_without_backoff(model_dir, monkeypatch, ranges):
    """Stories: US-32."""
    data = bytes(range(256)) * (3 << 12)  # 3 MiB
    opened = _resumable(monkeypatch, data, ranges)
    monkeypatch.setattr(semantic, "REOPEN_AFTER_PAUSE_S", 0.05)
    polls = [0]

    def wait():
        polls[0] += 1
        if polls[0] == 1:  # a recording runs after the first block; the server drops the idle link
            time.sleep(0.1)
        return True

    path = semantic.model_path(_fake_model(data), wait=wait)
    assert path is not None and path.read_bytes() == data
    assert semantic._last_failure == 0.0  # not a failure: no backoff
    # Range resume (2 opens), or a rejected Range followed by a clean restart (3 opens)
    assert opened[1] is not None and len(opened) == (2 if ranges else 3)


def test_download_abandoned_midway_leaves_no_part_and_no_backoff(model_dir, monkeypatch):
    data = b"w" * (3 << 20)
    _urlopen(monkeypatch, data)
    polls = [0]

    def wait():
        polls[0] += 1
        return polls[0] < 2  # give up after the first block

    assert semantic.model_path(_fake_model(data), wait=wait) is None
    assert _left(model_dir) == [] and semantic._last_failure == 0.0
    assert semantic.model_path(_fake_model(data)) is not None  # next time it just downloads


def test_embed_never_downloads_but_prepare_does(model_dir, monkeypatch):
    calls = _urlopen(monkeypatch, None)
    emb = semantic.OnnxEmbedder()
    assert emb.embed(["hello"]) is None and calls == []
    assert emb.prepare() is False and len(calls) == 1  # prepare tried (offline: failed)


# --------------------------------------------------------------------------- scale / budget
DIM = semantic.DIM
SEMANTIC_BUDGET_S = 0.2  # measured ~5-10 ms locally: a 20x margin for slow CI runners


class VecEmbedder:
    """384-d unit vectors derived from the text (deterministic), like the real model's shape."""

    model_id = "fake-384"

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = np.random.default_rng(abs(hash(t)) % (2**32)).standard_normal(DIM)
            out[i] /= np.linalg.norm(out[i])
        return out


def _median_of(n, fn):
    """Median of ``n`` runs after one warm-up (first connection, page cache)."""
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _budget() -> float:
    """The US-20 budget with its Windows headroom / RCC_SEARCH_LATENCY_FACTOR (see
    test_search_latency.py)."""
    return SEMANTIC_BUDGET_S * _factor()


def test_semantic_search_over_6000_segments_stays_in_budget(tmp_path):
    """US-32 performance: the meaning ranking over 6,000 stored 384-d vectors (no model)."""
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=VecEmbedder())
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((6000, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    with idx._connect() as db:
        for f in range(100):
            db.execute("INSERT INTO sessions VALUES (?, ?, '', 's', 0)", (f"f{f}", f"s{f}"))
            db.execute("INSERT INTO sem_sessions VALUES (?, 's', 'fake-384')", (f"f{f}",))
        db.executemany(
            "INSERT INTO seg_vec VALUES (?, ?, 'Me', 'en', ?, ?)",
            [(f"f{i % 100}", float(i), f"seg {i}", vecs[i].tobytes()) for i in range(6000)],
        )
    q = vecs[1234] * 1.0  # a stored vector: cosine 1 with itself

    class Fixed:
        model_id = "fake-384"

        def embed(self, texts):
            return q[None, :]

    idx.embedder = Fixed()
    hits = idx._semantic("anything", 80)
    assert hits and hits[0].start == 1234.0 and len(hits) <= 80
    t = _median_of(7, lambda: idx._semantic("anything", 80))
    assert t < _budget()


def test_top_k_scan_matches_a_full_sort(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCAN_CHUNK", 7)
    monkeypatch.setattr(search, "SEMANTIC_MIN_SIM", -1.0)
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3")
    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((50, DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    with idx._connect() as db:
        db.execute("INSERT INTO sessions VALUES ('f', 's', '', 's', 0)")
        db.execute("INSERT INTO sem_sessions VALUES ('f', 's', 'fake-384')")
        db.executemany(
            "INSERT INTO seg_vec VALUES ('f', ?, 'Me', 'en', 't', ?)",
            [(float(i), vecs[i].tobytes()) for i in range(50)],
        )
    q = rng.standard_normal(DIM).astype(np.float32)
    q /= np.linalg.norm(q)

    class Fixed:
        model_id = "fake-384"

        def embed(self, texts):
            return q[None, :]

    idx.embedder = Fixed()
    got = [h.start for h in idx._semantic("x", 10)]
    assert got == [float(i) for i in np.argsort(-(vecs @ q))[:10]]


def test_semantic_search_transcripts_100_sessions_stays_in_budget(out, monkeypatch):
    """US-32 performance: the US-20 path (unchanged sync + query) with semantic on, 6,000
    segments embedded by a fast 384-d fake."""
    monkeypatch.setattr(semantic, "default_embedder", lambda: VecEmbedder())
    words = ["budget", "roadmap", "hiring", "release", "бюджет", "релиз", "найм", "клиент"]
    for n in range(100):
        make_session(
            out,
            n,
            [seg(i * 5, f"{words[(n + i) % len(words)]} item {i} of {n}") for i in range(60)],
        )
    idx = search.default_index(True)
    search.search_transcripts(out, "budget", limit=50, semantic=True)
    assert idx.wait_embedded(120)
    assert vector_rows(idx) == 6000

    def run():
        assert search.search_transcripts(out, "budget", limit=50, semantic=True)

    t = _median_of(7, run)
    assert t < _budget()


# --------------------------------------------------------------------------- locks, failures
def _loaded_embedder(monkeypatch, tmp_path):
    import contextlib

    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    monkeypatch.setattr(semantic, "_verified", set())
    monkeypatch.setattr(semantic, "_last_failure", 0.0)
    return semantic.OnnxEmbedder()


def test_close_during_a_slow_load_deadlocks_nothing_and_later_calls_work(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    real = sys.modules["onnxruntime"].InferenceSession

    def slow(path, opts, **kw):
        entered.set()
        assert release.wait(10)
        return real(path, opts, **kw)

    sys.modules["onnxruntime"].InferenceSession = staticmethod(slow)
    result = []
    loader = threading.Thread(target=lambda: result.append(emb.prepare()), daemon=True)
    loader.start()
    assert entered.wait(10)
    closer = threading.Thread(target=emb.close, daemon=True)
    closer.start()
    closer.join(5)
    assert not closer.is_alive()  # close() didn't wait for the load
    monkeypatch.setattr(semantic, "_default", emb)
    search.default_index(True)
    # close() never waits for the load in flight (it takes only the embedder's state lock), so
    # default_index(False) may call it under search._default_lock and still returns at once
    searcher = threading.Thread(target=lambda: search.default_index(False), daemon=True)
    searcher.start()
    searcher.join(5)
    assert not searcher.is_alive()
    release.set()
    loader.join(10)
    assert not loader.is_alive() and result == [False]  # cancelled: nothing published
    assert not emb.ready()
    sys.modules["onnxruntime"].InferenceSession = staticmethod(real)
    assert emb.prepare() is True and emb.ready()  # later calls work
    assert emb.embed(["a"]).shape == (1, semantic.DIM)


def test_close_while_embedding_and_loading_concurrently_never_deadlocks(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    ort = sys.modules["onnxruntime"]
    real_session = ort.InferenceSession
    real_run = semantic.OnnxEmbedder._run

    def slow_session(path, opts, **kw):
        time.sleep(0.005)  # loads and embeds overlap with close() instead of finishing instantly
        return real_session(path, opts)

    def slow_run(loaded, texts):
        time.sleep(0.005)
        return real_run(loaded, texts)

    ort.InferenceSession = staticmethod(slow_session)
    monkeypatch.setattr(semantic.OnnxEmbedder, "_run", staticmethod(slow_run))
    stop = threading.Event()

    def churn(fn):
        while not stop.is_set():
            fn()

    workers = [
        threading.Thread(target=churn, args=(f,), daemon=True)
        for f in (lambda: emb.embed(["a"]), emb.prepare, emb.close, emb.close)
    ]
    for w in workers:
        w.start()
    time.sleep(0.5)
    stop.set()
    for w in workers:
        w.join(10)
        assert not w.is_alive()
    ort.InferenceSession = real_session


def _failing_session(monkeypatch, exc, times):
    real = sys.modules["onnxruntime"].InferenceSession
    state = {"n": 0}

    def session(path, opts, **kw):
        state["n"] += 1
        if state["n"] <= times:
            raise exc
        return real(path, opts, **kw)

    sys.modules["onnxruntime"].InferenceSession = staticmethod(session)
    return state


def test_a_locked_model_file_is_a_soft_failure_that_retries(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    monkeypatch.setattr(semantic, "LOAD_RETRY_PAUSE_S", 0.0)
    files = [tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER)]
    semantic._verified.update(files)  # intact files, as after a real download
    state = _failing_session(monkeypatch, PermissionError("held by Defender"), 100)
    assert emb.embed(["a"]) is None
    assert state["n"] == semantic.LOAD_ATTEMPTS  # quick retries right away
    assert not emb._broken and all(f.exists() for f in files)  # never "broken for the process"
    assert emb.embed(["a"]) is None and state["n"] == semantic.LOAD_ATTEMPTS  # backoff holds
    emb._retry_at = 0.0  # backoff over; the lock is gone
    state["n"] = 100  # now succeeds
    assert emb.embed(["a"]) is not None and emb.ready()


def test_a_lock_that_clears_during_the_quick_retries_loads(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    monkeypatch.setattr(semantic, "LOAD_RETRY_PAUSE_S", 0.0)
    state = _failing_session(monkeypatch, OSError("sharing violation"), 2)
    assert emb.prepare() is True and state["n"] == 3 and emb.ready()


def test_a_permanently_invalid_model_is_broken_until_close(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    semantic._verified.update(tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER))
    state = _failing_session(monkeypatch, RuntimeError("invalid protobuf"), 1)
    assert emb.embed(["a"]) is None and emb._broken and state["n"] == 1
    assert emb.embed(["a"]) is None and state["n"] == 1  # not retried per search
    emb.close()
    assert not emb._broken
    assert emb.prepare() and emb.ready()  # switching semantic off and on retries


@pytest.mark.parametrize(
    "msg",
    [
        "Файл занят другим процессом. (os error 32)",
        "Не удалось загрузить модель: [WinError 33] часть файла заблокирована",
        "Отказано в доступе. (os error 5)",
        "ошибка 0x80070020 при открытии файла",
        "Error 32: файл используется",
    ],
)
def test_localised_windows_lock_errors_are_soft_by_their_codes(msg):
    """Stories: US-32."""
    assert semantic._is_soft(RuntimeError(msg))


def test_winerror_attribute_is_soft_and_other_failures_are_not():
    e = RuntimeError("локализованный текст")
    e.winerror = 32  # type: ignore[attr-defined]
    assert semantic._is_soft(e)
    assert not semantic._is_soft(RuntimeError("invalid protobuf"))
    assert not semantic._is_soft(RuntimeError("tensor of shape 325 is wrong"))


def test_a_localised_lock_is_retried_not_broken(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    monkeypatch.setattr(semantic, "LOAD_RETRY_PAUSE_S", 0.0)
    semantic._verified.update(tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER))
    _failing_session(monkeypatch, RuntimeError("Файл занят другим процессом (os error 32)"), 100)
    assert emb.embed(["a"]) is None
    assert not emb._broken and emb._retry_at > 0


@pytest.mark.parametrize("soft", [True, False])
def test_a_load_failing_after_close_records_no_failure(tmp_path, monkeypatch, soft):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    monkeypatch.setattr(semantic, "LOAD_RETRY_PAUSE_S", 0.0)
    semantic._verified.update(tmp_path / m.name for m in (semantic.MODEL, semantic.TOKENIZER))
    sys_ort = sys.modules["onnxruntime"]
    exc = PermissionError("locked") if soft else RuntimeError("invalid protobuf")

    def session(path, opts, **kw):
        emb.close()  # switched off while the load is running
        raise exc

    sys_ort.InferenceSession = staticmethod(session)
    assert emb.prepare() is False
    assert not emb._broken and emb._retry_at == 0.0 and semantic._last_failure == 0.0


def test_close_forgets_the_download_backoff(tmp_path, monkeypatch):
    emb = _loaded_embedder(monkeypatch, tmp_path)
    monkeypatch.setattr(semantic, "_last_failure", time.monotonic())
    emb.close()
    assert semantic._last_failure == 0.0


# ------------------------------------------------------------------ round 6
def test_cli_search_returns_meaning_matches(out, monkeypatch, capsys):
    """US-32: a one-shot `recandcc search` loads the model, indexes and waits for it."""
    make_session(out, 1, [seg(1, "please send the invoice")])
    emb = PrepEmbedder()
    monkeypatch.setattr(semantic, "default_embedder", lambda: emb)
    notes = []
    hits = search.search_transcripts(out, "оплата", semantic=True, wait_s=30, notify=notes.append)
    assert emb.prepared == 1 and [h.snippet for h in hits] == ["please send the invoice"]
    assert any("Downloading" in n for n in notes)
    assert not any("invoice" in n for n in notes)  # never transcript text


def test_cli_search_says_when_indexing_is_unfinished(out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = SlowEmbedder()
    monkeypatch.setattr(semantic, "default_embedder", lambda: emb)
    notes = []
    search.search_transcripts(out, "invoice", semantic=True, wait_s=0.01, notify=notes.append)
    assert "still indexing; showing what's ready" in notes
    search.default_index(True).wait_embedded(30)


def test_query_and_segments_are_yo_folded_alike(tmp_path, out):
    make_session(out, 1, [seg(1, "всё про перелёт"), seg(2, "ещё обед")])
    emb = FakeEmbedder()
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb)
    sync(idx, out)
    assert emb.calls[0] == ["все про перелет", "еще обед"]
    idx._semantic("перелёт", 5)
    idx._semantic("ПЕРЕЛЕТ", 5)
    assert emb.calls[-2][0] == "перелет"  # ё folded like the stored text


def test_embed_after_close_does_not_reload(tmp_path, monkeypatch):
    import contextlib

    loads = []
    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    real = sys.modules["onnxruntime"].InferenceSession
    sys.modules["onnxruntime"].InferenceSession = staticmethod(
        lambda p, o, **k: (loads.append(1), real(p, o, **k))[1]
    )
    emb = semantic.OnnxEmbedder()
    emb.close()  # switched off before an in-flight chunk reaches embed()
    assert emb.embed(["a"]) is None and loads == [] and not emb.ready()


def test_download_refuses_a_redirect_to_http():
    import urllib.error
    import urllib.request

    h = semantic._HttpsOnlyRedirect()
    req = urllib.request.Request("https://example.invalid/m.onnx")
    with pytest.raises(urllib.error.URLError, match="non-https"):
        h.redirect_request(req, None, 302, "Found", {}, "http://example.invalid/m.onnx")
    ok = h.redirect_request(req, None, 302, "Found", {}, "https://cdn.example.invalid/m.onnx")
    assert ok is not None and ok.full_url.startswith("https://cdn.")


def test_recording_busy_includes_the_transcription_backlog():
    from recandcc.ui.common import IDLE, Controller

    class Q:
        n = 0

        def pending(self):
            return self.n

    q = Q()
    ctrl = SimpleNamespace(engine=SimpleNamespace(state=SimpleNamespace(status=IDLE)), tq=q)
    assert Controller._recording_busy(ctrl) is False
    q.n = 1
    assert Controller._recording_busy(ctrl) is True  # Whisper backlog: don't compete
    q.n = 0
    ctrl._resubmitting = True
    assert Controller._recording_busy(ctrl) is True
    ctrl._resubmitting = False
    ctrl.engine.state.status = "recording"
    assert Controller._recording_busy(ctrl) is True


def test_a_reused_rowid_between_scan_and_materialise_is_not_shown(tmp_path, out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice is due")])
    idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=FakeEmbedder())
    sync(idx, out)
    real = np.lexsort

    def lexsort_then_swap(keys, *a, **k):
        with idx._connect() as db:  # delete + insert reuses the rowid with another segment
            rid, folder = db.execute("SELECT rowid, folder FROM seg_vec").fetchone()
            db.execute("DELETE FROM seg_vec WHERE rowid = ?", (rid,))
            db.execute(
                "INSERT INTO seg_vec(rowid, folder, start, speaker, language, text, vec) "
                "VALUES (?, ?, 9, 'Me', 'en', 'lunch pizza', ?)",
                (rid, folder, np.eye(len(CONCEPTS), dtype=np.float32)[2].tobytes()),
            )
        return real(keys, *a, **k)

    monkeypatch.setattr(search.np, "lexsort", lexsort_then_swap)
    assert idx._semantic("payment", 5) == []  # not the wrong segment under the old score


def test_search_first_call_keeps_busy_aware_pausing(out, monkeypatch):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = PrepEmbedder()
    monkeypatch.setattr(semantic, "default_embedder", lambda: emb)
    state = {"busy": True}
    busy = lambda: state["busy"]  # noqa: E731
    search.search_transcripts(out, "invoice", semantic=True, busy=busy)  # search first
    idx = search.default_index(True)  # a later caller without busy must not clear it
    assert idx.busy is busy
    idx.busy_poll_s = 0.01
    time.sleep(0.2)
    assert emb.prepared == 0 and emb.calls == []  # paused during the "recording"
    state["busy"] = False
    assert idx.wait_embedded(30) and emb.prepared == 1


def test_a_cancelled_wait_stops_the_load_before_the_session(tmp_path, monkeypatch):
    import contextlib

    loads = []
    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    real = sys.modules["onnxruntime"].InferenceSession
    sys.modules["onnxruntime"].InferenceSession = staticmethod(
        lambda p, o, **k: (loads.append(1), real(p, o, **k))[1]
    )
    emb = semantic.OnnxEmbedder()
    calls = iter([True, True, True, False])  # a recording starts after the files are checked
    assert emb.prepare(lambda: next(calls, False)) is False
    assert loads == [] and not emb.ready()


def test_the_session_is_cpu_only(tmp_path, monkeypatch):
    import contextlib

    seen = []
    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    real = sys.modules["onnxruntime"].InferenceSession
    sys.modules["onnxruntime"].InferenceSession = staticmethod(
        lambda p, o, **k: (seen.append(k), real(p, o, **k))[1]
    )
    assert semantic.OnnxEmbedder().prepare() is True
    assert seen == [{"providers": ["CPUExecutionProvider"]}]


# --------------------------------------------------------------------------- round 7
def test_off_on_interleaving_never_closes_a_reattached_embedder(monkeypatch):
    """default_index(False) drops the embedder, then a concurrent default_index(True)
    re-attaches the same object before the close: it must stay usable."""
    closed = []

    class Emb(FakeEmbedder):
        def close(self):
            closed.append(self)

    emb = Emb()
    monkeypatch.setattr(semantic, "default_embedder", lambda: emb)
    idx = search.default_index(True)
    real_lock, armed = search._default_lock, []

    class Lock:
        """Fires a concurrent default_index(True) right after the first unlock of the off call."""

        def __enter__(self):
            real_lock.acquire()

        def __exit__(self, *exc):
            real_lock.release()
            if armed:
                armed.pop()
                search.default_index(True)  # the interleaved "on"

    monkeypatch.setattr(search, "_default_lock", Lock())
    armed.append(1)
    search.default_index(False)
    assert idx.embedder is emb and closed == []  # re-attached: not closed
    search.default_index(False)  # without interleaving it is closed
    assert idx.embedder is None and closed == [emb]


def test_hand_edited_semantic_setting_is_applied_to_a_running_app(tmp_path, monkeypatch):
    from recandcc.ui.common import Controller

    ctrl = SimpleNamespace(
        cfg=Config(output_dir=tmp_path),
        _save_lock=threading.Lock(),
        _lang_lock=threading.Lock(),
        _config_mtime=-1.0,
        _mtime=lambda: 1.0,
        post_notice=lambda *a: None,
        apply_hotkey=lambda *a: None,
    )
    new = Config(output_dir=tmp_path)
    new.search.semantic = True
    monkeypatch.setattr("recandcc.ui.common.load_config", lambda *a, **k: new)
    assert ctrl.cfg.search.semantic is False
    Controller._maybe_reload_config(ctrl)  # type: ignore[arg-type]
    assert ctrl.cfg.search.semantic is True


def test_download_lock_serialises_two_processes_and_the_second_uses_the_file(
    model_dir, monkeypatch
):
    """The second 'process' (a thread with its own lock object, as a second process has its
    own file descriptor) waits, then uses the verified file instead of downloading again."""
    data = b"weights" * 1000
    model = _fake_model(data)
    monkeypatch.setattr(semantic, "DOWNLOAD_LOCK_POLL_S", 0.01)
    holder = semantic._FileLock(model_dir / ".download.lock")
    assert holder.try_acquire()
    other = semantic._FileLock(model_dir / ".download.lock")
    assert not other.try_acquire()  # exclusive across descriptors
    calls = _urlopen(monkeypatch, data)
    got = []
    t = threading.Thread(target=lambda: got.append(semantic.model_path(model)))
    t.start()
    time.sleep(0.1)
    assert t.is_alive() and calls == []  # waiting for the holder
    (model_dir / "m.onnx").write_bytes(data)  # the other process finishes
    holder.release()
    t.join(5)
    assert got == [model_dir / "m.onnx"] and calls == []
    assert other.try_acquire()  # released
    other.release()


def test_download_lock_wait_is_bounded_and_not_a_failure(model_dir, monkeypatch):
    data = b"weights" * 1000
    monkeypatch.setattr(semantic, "DOWNLOAD_LOCK_WAIT_S", 0.1)
    monkeypatch.setattr(semantic, "DOWNLOAD_LOCK_POLL_S", 0.01)
    holder = semantic._FileLock(model_dir / ".download.lock")
    assert holder.try_acquire()
    calls = _urlopen(monkeypatch, data)
    assert semantic.model_path(_fake_model(data)) is None and calls == []
    assert semantic._last_failure == 0.0  # no 10-minute backoff
    holder.release()
    assert semantic.model_path(_fake_model(data)) == model_dir / "m.onnx"


def test_download_uses_a_per_process_temp_name(model_dir, monkeypatch):
    import os

    data = b"weights" * 1000
    _urlopen(monkeypatch, data)
    seen = []
    real = Path.replace
    monkeypatch.setattr(Path, "replace", lambda self, t: (seen.append(self.name), real(self, t))[1])
    assert semantic.model_path(_fake_model(data)) is not None
    assert seen == [f".m.onnx.part.{os.getpid()}"]


def test_a_search_does_not_queue_behind_the_indexer(tmp_path, monkeypatch):
    """The indexer holds the model's compute lock for a whole chunk: a search then returns
    its full-text hits at once instead of waiting."""
    import contextlib

    _fake_onnx(monkeypatch, tmp_path, lambda name: contextlib.nullcontext())
    emb = semantic.OnnxEmbedder()
    assert emb.prepare() and emb.ready()
    entered, release = threading.Event(), threading.Event()
    real_run = emb._run

    def slow_run(loaded, texts):
        entered.set()
        release.wait(10)
        return real_run(loaded, texts)

    monkeypatch.setattr(emb, "_run", slow_run)
    indexer = threading.Thread(target=lambda: emb.embed(["chunk"] * 64))
    indexer.start()
    assert entered.wait(5)
    try:
        assert emb.embed_query(["q"]) is None  # does not block
        out = tmp_path / "out"
        make_session(out, 0, [seg(0, "the budget review")])
        idx = TranscriptIndex(tmp_path / "idx" / "s.sqlite3", embedder=emb)
        idx.sync(out)
        t0 = time.perf_counter()
        hits = idx.search("budget")
        assert time.perf_counter() - t0 < 2.0 and hits
    finally:
        release.set()
        indexer.join(5)
    assert idx.wait_embedded(10)  # the index's own worker also embedded: the lock is free again
    assert emb.embed_query(["q"]) is not None


def test_a_busy_index_never_embeds_on_the_query_path(tmp_path, out):
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    make_session(out, 0, [seg(0, "invoice for the budget")])
    sync(idx, out)
    emb.calls.clear()
    state = {"busy": True}
    idx.busy = lambda: state["busy"]
    assert idx.search("budget")  # full-text hit
    assert emb.calls == []
    state["busy"] = False
    idx.search("budget")
    assert emb.calls == [["budget"]]
