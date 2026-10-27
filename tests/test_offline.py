"""Models load from disk without contacting Hugging Face once downloaded.

Stories: US-14.
"""

import sys
import types

from recandcc.transcribe import gpu, revisions


def _fake_hub(monkeypatch, cached: bool, revs: list | None = None):
    calls = []

    def snapshot_download(repo, revision=None, local_files_only=False):
        calls.append(local_files_only)
        if revs is not None:
            revs.append(revision)
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
    assert revisions.faster_whisper_revision("medium") is None
    assert revisions.faster_whisper_revision("/some/local/dir") is None


def test_faster_whisper_load_passes_pinned_revision(monkeypatch):
    from recandcc.config import TranscriptionConfig
    from recandcc.transcribe.engine import Transcriber

    seen = []

    class FakeModel:
        def __init__(self, name, **kwargs):
            seen.append((name, kwargs.get("revision"), kwargs.get("local_files_only")))

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel)
    )
    t = Transcriber(TranscriptionConfig(model="base", backend="faster-whisper"))
    t._load("base")
    assert seen == [("base", revisions.HF_REVISIONS["Systran/faster-whisper-base"], True)]
