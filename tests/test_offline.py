"""Models load from disk without contacting Hugging Face once downloaded."""

import sys
import types

from recandcc.transcribe import gpu


def _fake_hub(monkeypatch, cached: bool):
    calls = []

    def snapshot_download(repo, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only and not cached:
            raise FileNotFoundError(repo)
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


def test_missing_model_is_downloaded_once(monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    assert gpu.local_model_path("mlx-community/whisper-small-mlx").startswith("/cache/")
    assert calls == [True, False]


def test_local_path_is_used_as_is(tmp_path, monkeypatch):
    calls = _fake_hub(monkeypatch, cached=False)
    assert gpu.local_model_path(str(tmp_path)) == str(tmp_path)
    assert calls == []
