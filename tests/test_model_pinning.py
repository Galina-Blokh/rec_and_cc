"""Every offered Whisper model is pinned; unpinned repos are not downloaded without opt-in.

Stories: US-14.
"""

import sys
import types
from pathlib import Path

import pytest

from recandcc.config import TranscriptionConfig
from recandcc.transcribe import gpu, revisions
from recandcc.transcribe.engine import Transcriber
from recandcc.transcribe.revisions import UnpinnedModelError

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT  # replaced per test by the _snapshot fixture


def _fake_hub(monkeypatch, cached: bool):
    calls = []

    def snapshot_download(repo, revision=None, local_files_only=False):
        calls.append((revision, local_files_only))
        if local_files_only and not cached:
            raise FileNotFoundError(repo)
        return f"/cache/{repo}"

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    return calls


@pytest.fixture(autouse=True)
def _snapshot(tmp_path, monkeypatch):
    """A faster-whisper snapshot folder that download_model() returns in these tests."""
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(sys.modules[__name__], "SNAPSHOT", tmp_path, raising=False)


def _fake_whisper(monkeypatch, cached: bool):
    """Fake faster_whisper; ``calls`` records (name, revision, local_files_only) lookups."""
    calls = []

    def download_model(name, local_files_only=False, revision=None):
        calls.append((name, revision, local_files_only))
        if local_files_only and not cached:
            raise FileNotFoundError(name)  # what a cache miss (LocalEntryNotFoundError) is
        return str(SNAPSHOT)

    class FakeModel:
        def __init__(self, path, **kwargs):
            pass

    fake = types.SimpleNamespace(
        WhisperModel=FakeModel, utils=types.SimpleNamespace(download_model=download_model)
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", fake.utils)
    return calls


def _transcriber(**kw) -> Transcriber:
    return Transcriber(TranscriptionConfig(model="base", backend="faster-whisper", **kw))


@pytest.mark.parametrize(
    "name", sorted(set(revisions.OFFERED_MODELS) | set(revisions.FASTER_WHISPER_REPOS))
)
def test_every_offered_model_is_pinned_on_both_backends(name):
    assert revisions.faster_whisper_revision(name) is not None
    assert revisions.revision(gpu.mlx_repo(name)) is not None


@pytest.mark.parametrize("doc", ["README.md", "docs/CONFIGURATION.md"])
def test_models_named_in_the_docs_are_offered(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    for name in revisions.OFFERED_MODELS:
        if name != "turbo":  # alias, not listed in the docs
            assert f"`{name}`" in text or f" {name} " in text, name


def test_unpinned_mlx_repo_refused_without_opt_in(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    with pytest.raises(UnpinnedModelError):
        gpu.local_model_path("someone/custom-whisper")
    assert calls == [(None, True)]  # local lookup only, never went online


def test_unpinned_mlx_repo_downloads_with_opt_in(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    gpu.local_model_path("someone/custom-whisper", allow_unpinned=True)
    assert calls == [(None, True), (None, False)]


def test_cached_unpinned_mlx_repo_still_loads(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=True)
    assert gpu.local_model_path("someone/custom-whisper") == "/cache/someone/custom-whisper"
    assert calls == [(None, True)]


def test_pinned_offered_mlx_model_downloads(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    gpu.local_model_path(gpu.mlx_repo("medium"))
    pinned = revisions.HF_REVISIONS["mlx-community/whisper-medium-mlx"]
    assert calls == [(pinned, True), (pinned, False)]


def test_unpinned_faster_whisper_repo_refused_without_opt_in(monkeypatch):
    calls = _fake_whisper(monkeypatch, cached=False)
    with pytest.raises(UnpinnedModelError):
        _transcriber()._load("someone/custom-ct2")
    assert calls == [("someone/custom-ct2", None, True)]


def test_unpinned_faster_whisper_repo_downloads_with_opt_in(monkeypatch):
    calls = _fake_whisper(monkeypatch, cached=False)
    _transcriber(allow_unpinned_models=True)._load("someone/custom-ct2")
    assert calls == [("someone/custom-ct2", None, True), ("someone/custom-ct2", None, False)]


def test_offered_faster_whisper_model_downloads_pinned(monkeypatch):
    calls = _fake_whisper(monkeypatch, cached=False)
    _transcriber()._load("large-v3")
    pinned = revisions.HF_REVISIONS["Systran/faster-whisper-large-v3"]
    assert calls == [("large-v3", pinned, True), ("large-v3", pinned, False)]


def test_mlx_whisper_passes_opt_in_to_the_download(monkeypatch):
    seen = []
    monkeypatch.setattr(gpu, "local_model_path", lambda repo, allow: seen.append(allow) or repo)
    monkeypatch.setattr(gpu.MlxWhisper, "_get_model", lambda self: object())
    model = gpu.MlxWhisper("someone/custom-mlx", allow_unpinned=True)
    model.load()
    model.close()
    assert seen == [True]


def test_cached_unpinned_faster_whisper_repo_still_loads(monkeypatch):
    calls = _fake_whisper(monkeypatch, cached=True)
    _transcriber()._load("someone/custom-ct2")
    assert calls == [("someone/custom-ct2", None, True)]


@pytest.mark.parametrize("allow", [False, True])
def test_transcriber_passes_opt_in_to_mlx(monkeypatch, allow):
    from recandcc.transcribe import engine

    made = []

    class FakeMlx:
        def __init__(self, model, allow_unpinned=False):
            made.append((model, allow_unpinned))

        def load(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(engine, "MlxWhisper", FakeMlx)
    t = Transcriber(
        TranscriptionConfig(model="someone/custom-mlx", allow_unpinned_models=allow),
        language_model=None,
    )
    t.gpu = True
    t.ensure_loaded()
    t._pool.shutdown()
    assert made == [("someone/custom-mlx", allow)]


def test_refused_mlx_model_frees_its_thread(monkeypatch):
    _fake_hub(monkeypatch, cached=False)
    closed = []
    real_close = gpu.MlxWhisper.close

    def close(self):
        closed.append(self.repo)
        real_close(self)

    monkeypatch.setattr(gpu.MlxWhisper, "close", close)
    t = Transcriber(TranscriptionConfig(model="someone/custom-mlx"), language_model=None)
    t.gpu = True
    with pytest.raises(UnpinnedModelError):
        t.ensure_loaded()
    assert closed == ["someone/custom-mlx"]


def test_refused_main_model_keeps_the_language_id_model(monkeypatch):
    _fake_hub(monkeypatch, cached=False)
    t = Transcriber(TranscriptionConfig(model="someone/custom-mlx", languages=["en", "ru"]))
    t.gpu = True
    loaded = []
    monkeypatch.setattr(t, "_load", lambda name: loaded.append(name) or object())
    for _ in range(2):
        with pytest.raises(UnpinnedModelError):
            t.ensure_loaded()
    assert loaded == [t.language_model]  # loaded once, not on every retry
