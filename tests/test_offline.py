"""Models load from disk without contacting Hugging Face once downloaded.

Stories: US-14, US-19.
"""

import hashlib
import io
import sys
import types

import pytest
from huggingface_hub.errors import IncompleteSnapshotError, LocalEntryNotFoundError

from recandcc.transcribe import gpu, revisions, summary_cpu

# What huggingface_hub raises with local_files_only=True for a missing snapshot and for one
# left partial by an interrupted download: both must go online (once) to fetch the model.
CACHE_MISSES = [
    LocalEntryNotFoundError("not cached"),
    IncompleteSnapshotError("partial", snapshot_path="/cache/partial"),
]
BROKEN_CACHE = [RuntimeError("corrupt"), OSError("corrupt"), PermissionError("denied")]


def _fake_hub(
    monkeypatch,
    cached: bool,
    revs: list | None = None,
    local_error: Exception | None = None,
    miss: Exception = CACHE_MISSES[0],
):
    calls = []

    def snapshot_download(repo, revision=None, local_files_only=False):
        calls.append(local_files_only)
        if revs is not None:
            revs.append(revision)
        if local_files_only and local_error is not None:
            raise local_error
        if local_files_only and not cached:
            raise miss
        return f"/cache/{repo}"

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download)
    )
    return calls


def test_cached_model_resolves_without_network(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=True)
    assert gpu.local_model_path("mlx-community/whisper-large-v3-turbo") == (
        "/cache/mlx-community/whisper-large-v3-turbo"
    )
    assert calls == [True]  # local lookup only, no download attempt


@pytest.mark.parametrize("miss", CACHE_MISSES, ids=type)
def test_missing_model_is_downloaded_once(monkeypatch, miss):
    calls = _fake_hub(monkeypatch, cached=False, miss=miss)
    assert gpu.local_model_path("mlx-community/whisper-small-mlx").startswith("/cache/")
    assert calls == [True, False]


@pytest.mark.parametrize("error", BROKEN_CACHE, ids=type)
def test_broken_cache_is_not_downloaded_again(monkeypatch, error, caplog):
    calls = _fake_hub(monkeypatch, cached=True, local_error=error)
    with pytest.raises(type(error)):
        gpu.local_model_path("mlx-community/whisper-large-v3-turbo")
    assert calls == [True]  # no online retry
    assert "models--mlx-community--whisper-large-v3-turbo" in caplog.text
    assert "delete it to download the model again" in caplog.text


def test_local_path_is_used_as_is(tmp_path, monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    assert gpu.local_model_path(str(tmp_path)) == str(tmp_path)
    assert calls == []


def test_pinned_revision_is_used_on_disk_and_for_download(monkeypatch):
    revs = []
    _fake_hub(monkeypatch, cached=False, revs=revs)
    gpu.local_model_path("mlx-community/whisper-large-v3-turbo")
    pinned = revisions.HF_REVISIONS["mlx-community/whisper-large-v3-turbo"]
    assert revs == [pinned, pinned]


def test_unpinned_repo_uses_latest(monkeypatch):
    revs = []
    _fake_hub(monkeypatch, cached=True, revs=revs)
    gpu.local_model_path("someone/custom-whisper")
    assert revs == [None]


def test_revisions_are_full_commit_hashes():
    for repo, rev in revisions.HF_REVISIONS.items():
        assert len(rev) == 40 and int(rev, 16) >= 0, repo
    for repo in revisions.FASTER_WHISPER_REPOS.values():
        assert repo in revisions.HF_REVISIONS


def test_faster_whisper_names_map_to_pinned_revisions():
    turbo = revisions.HF_REVISIONS["mobiuslabsgmbh/faster-whisper-large-v3-turbo"]
    assert revisions.faster_whisper_revision("large-v3-turbo") == turbo
    assert (
        revisions.faster_whisper_revision("mobiuslabsgmbh/faster-whisper-large-v3-turbo") == turbo
    )
    assert revisions.faster_whisper_revision("someone/custom-ct2") is None
    assert revisions.faster_whisper_revision("/some/local/dir") is None


def test_ivrit_hebrew_models_are_pinned_in_both_formats(monkeypatch):
    for ct2, mlx in revisions.IVRIT_MLX_REPOS.items():
        assert revisions.faster_whisper_revision(ct2) == revisions.HF_REVISIONS[ct2]
        assert gpu.mlx_repo(ct2) == mlx and revisions.revision(mlx) is not None
        assert revisions.ct2_repo(mlx) == ct2 and revisions.ct2_repo(ct2) == ct2
    revs = []
    _fake_hub(monkeypatch, cached=False, revs=revs)
    gpu.local_model_path(gpu.mlx_repo("ivrit-ai/whisper-large-v3-turbo-ct2"))
    pinned = revisions.HF_REVISIONS["mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx"]
    assert revs == [pinned, pinned]  # on disk first, then the one pinned download


def _fake_faster_whisper(monkeypatch, tmp_path, local_error=None, tokenizer=True):
    """Fake faster_whisper: download_model returns a snapshot folder under ``tmp_path``."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    if tokenizer:
        (snapshot / "tokenizer.json").write_text("{}")
    downloads, models = [], []

    def download_model(name, local_files_only=False, revision=None):
        downloads.append((name, revision, local_files_only))
        if local_files_only and local_error is not None:
            raise local_error
        return str(snapshot)

    class FakeModel:
        def __init__(self, path, **kwargs):
            models.append((path, kwargs))

    fake = types.SimpleNamespace(
        WhisperModel=FakeModel,
        utils=types.SimpleNamespace(
            download_model=download_model, _MODELS={"base": "Systran/faster-whisper-base"}
        ),
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return snapshot, downloads, models


def _transcriber():
    from recandcc.config import TranscriptionConfig
    from recandcc.transcribe.engine import Transcriber

    return Transcriber(TranscriptionConfig(model="base", backend="faster-whisper"))


BASE_REV = revisions.HF_REVISIONS["Systran/faster-whisper-base"]


def test_faster_whisper_cached_model_loads_from_its_folder(monkeypatch, tmp_path):
    snapshot, downloads, models = _fake_faster_whisper(monkeypatch, tmp_path)
    _transcriber()._load("base")
    assert downloads == [("base", BASE_REV, True)]  # pinned revision, local lookup only
    assert [path for path, _ in models] == [str(snapshot)]  # a folder: no hub access
    assert "local_files_only" not in models[0][1] and "revision" not in models[0][1]


@pytest.mark.parametrize("miss", CACHE_MISSES, ids=type)
def test_faster_whisper_cache_miss_downloads_once(monkeypatch, tmp_path, miss):
    snapshot, downloads, models = _fake_faster_whisper(monkeypatch, tmp_path, local_error=miss)
    _transcriber()._load("base")
    assert downloads == [("base", BASE_REV, True), ("base", BASE_REV, False)]
    assert [path for path, _ in models] == [str(snapshot)]


@pytest.mark.parametrize("error", BROKEN_CACHE, ids=type)
def test_faster_whisper_broken_cache_is_not_downloaded_again(monkeypatch, tmp_path, caplog, error):
    _, downloads, models = _fake_faster_whisper(monkeypatch, tmp_path, local_error=error)
    with pytest.raises(type(error)):
        _transcriber()._load("base")
    assert downloads == [("base", BASE_REV, True)]  # no online retry
    assert models == []
    assert "models--Systran--faster-whisper-base" in caplog.text
    assert "to download the model again" in caplog.text


def test_faster_whisper_missing_tokenizer_is_corrupt_not_fetched(monkeypatch, tmp_path):
    # faster-whisper would fetch a tokenizer from Hugging Face without tokenizer.json.
    _, downloads, models = _fake_faster_whisper(monkeypatch, tmp_path, tokenizer=False)
    with pytest.raises(RuntimeError, match="tokenizer.json missing.*download the model again"):
        _transcriber()._load("base")
    assert downloads == [("base", BASE_REV, True)]
    assert models == []  # WhisperModel never constructed


def test_faster_whisper_user_folder_is_used_as_is(monkeypatch, tmp_path):
    _, downloads, models = _fake_faster_whisper(monkeypatch, tmp_path)
    own = tmp_path / "my-model"
    own.mkdir()
    (own / "tokenizer.json").write_text("{}")
    _transcriber()._load(str(own))
    assert downloads == []
    assert [path for path, _ in models] == [str(own)]


CPU_WEIGHTS = b"cpu weights" * 40


def _pinned_cpu_model(monkeypatch, tmp_path) -> summary_cpu.CpuModel:
    """A one-file CPU summary model, pinned to a made-up commit, stored under tmp_path."""
    monkeypatch.setattr(summary_cpu, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(summary_cpu, "_verified", set())
    file = summary_cpu.ModelFile(
        "onnx/model.onnx", hashlib.sha256(CPU_WEIGHTS).hexdigest(), len(CPU_WEIGHTS)
    )
    return summary_cpu.CpuModel(
        repo="onnx-community/offline-test", revision="c" * 40, files=(file,)
    )


def test_cached_cpu_summary_model_never_touches_the_network(monkeypatch, tmp_path):
    model = _pinned_cpu_model(monkeypatch, tmp_path)
    (model.folder / "onnx").mkdir(parents=True)
    (model.folder / "onnx" / "model.onnx").write_bytes(CPU_WEIGHTS)  # valid, as after a download

    def unexpected(url):
        raise AssertionError(f"a cached model made a request: {url}")

    monkeypatch.setattr(summary_cpu, "_open", unexpected)
    assert summary_cpu.model_dir(model) == model.folder


def test_missing_cpu_summary_model_makes_exactly_one_pinned_request(monkeypatch, tmp_path):
    model = _pinned_cpu_model(monkeypatch, tmp_path)
    requests = []

    def fake_hub(url):
        requests.append(url)
        return io.BytesIO(CPU_WEIGHTS)

    monkeypatch.setattr(summary_cpu, "_open", fake_hub)
    summary_cpu.model_dir(model)
    assert requests == [model.url(model.files[0])]
    assert requests[0].startswith(f"https://huggingface.co/{model.repo}/resolve/{model.revision}/")
    summary_cpu.model_dir(model)  # now cached: no second request
    assert len(requests) == 1
