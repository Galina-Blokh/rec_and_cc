"""Opt-in semantic search: pinned one-time model download, embeddings stored next to the FTS
index, reciprocal-rank fusion, and unchanged results when it is off. The fast tests use a
fake embedder; the real model is only in the slow test.

Stories: US-32.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sqlite3
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_search import make_session, seg

from recandcc import search, semantic
from recandcc.config import Config, load_config
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
    plain.sync(out)
    assert vector_rows(plain) == 0
    assert plain.search("invoice") == plain._lexical("invoice")
    assert plain.search("payment") == []  # no word match, no semantic fallback
    assert search.default_index(False).embedder is None


# --------------------------------------------------------------------------- on
def test_semantic_finds_by_meaning_and_across_languages(tmp_path, out):
    a = make_session(out, 1, [seg(1, "Please send the invoice"), seg(9, "Lunch at noon")])
    b = make_session(out, 2, [seg(3, "Нужна оплата до пятницы", "Speaker 1", "ru")])
    idx = make(tmp_path, FakeEmbedder())
    idx.sync(out)
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
    idx.sync(out)
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
    idx.sync(out)
    idx.sync(out)
    assert len(emb.calls) == 1  # unchanged: nothing re-embedded
    make_session(out, 2, [seg(1, "a pizza")])
    idx.sync(out)
    assert len(emb.calls) == 2 and emb.calls[-1] == ["a pizza"]  # only the new session
    shutil.rmtree(d)
    idx.sync(out)
    assert vector_rows(idx) == 1


def test_rewritten_transcript_is_re_embedded(tmp_path, out):
    d = make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    idx.sync(out)
    (d / "transcript.json").write_text(
        (d / "transcript.json").read_text(encoding="utf-8").replace("invoice", "pizza trip"),
        encoding="utf-8",
    )
    idx.sync(out)
    assert emb.calls[-1] == ["the pizza trip"]
    assert vector_rows(idx) == 1


def test_switching_off_forgets_the_vectors(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    path_idx = make(tmp_path, FakeEmbedder())
    path_idx.sync(out)
    assert vector_rows(path_idx) == 1
    off = make(tmp_path)
    off.sync(out)
    assert vector_rows(off) == 0
    assert [h.start for h in off.search("invoice")] == [1.0]


def test_unavailable_model_falls_back_to_full_text_and_retries(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder(available=False)
    idx = make(tmp_path, emb)
    idx.sync(out)
    assert vector_rows(idx) == 0
    assert [h.start for h in idx.search("invoice")] == [1.0]
    emb.available = True
    idx.sync(out)  # the model showed up: the backlog is embedded
    assert vector_rows(idx) == 1


def test_failing_embedder_never_breaks_search(tmp_path, out):
    make_session(out, 1, [seg(1, "the invoice")])
    emb = FakeEmbedder()
    idx = make(tmp_path, emb)
    idx.sync(out)
    emb.fail = True
    assert [h.start for h in idx.search("invoice")] == [1.0]
    idx.index_session(out)  # nothing to index here, and no exception either


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
