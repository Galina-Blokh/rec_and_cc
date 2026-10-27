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
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_search import make_session, seg

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

    monkeypatch.setattr(semantic.urllib.request, "urlopen", urlopen)
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
    assert [p.name for p in model_dir.iterdir()] == ["m.onnx"]  # no temp file left


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
    monkeypatch.setattr(semantic, "model_path", lambda m, download=True: files[m.name])

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


def test_bad_checksum_is_rejected(model_dir, monkeypatch):
    data = b"weights" * 1000
    _urlopen(monkeypatch, b"tampered" + data[8:])
    assert semantic.model_path(_fake_model(data)) is None
    assert list(model_dir.iterdir()) == []


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
