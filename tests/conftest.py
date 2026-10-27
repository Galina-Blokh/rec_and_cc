import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running tests that load real models")


@pytest.fixture(autouse=True)
def _no_diarization_model(request, monkeypatch):
    """Fast tests never download or load the speaker model: diarization runs but finds no
    model (as when offline), so transcripts keep "Others". Slow tests use the real one."""
    if request.node.get_closest_marker("slow") is None:
        from recandcc.transcribe import diarize

        monkeypatch.setattr(diarize, "model_path", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _private_search_index(tmp_path, monkeypatch):
    """Tests never touch the user's real search index (the Controller updates it after
    every transcription)."""
    from recandcc import search

    monkeypatch.setattr(search, "INDEX_PATH", tmp_path / "search-index" / "search.sqlite3")


@pytest.fixture(autouse=True)
def _private_voices_dir(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's real voice profiles."""
    from recandcc.transcribe import voices

    monkeypatch.setattr(voices, "VOICES_DIR", tmp_path_factory.mktemp("voices"))


@pytest.fixture(autouse=True)
def _private_config(tmp_path, monkeypatch):
    """No test reads or writes the user's real config.toml, even through a bare
    `Config().save()` / `load_config()` or a module that imported CONFIG_PATH by name."""
    import sys

    path = tmp_path / "private-config" / "config.toml"
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] == "recandcc" and hasattr(module, "CONFIG_PATH"):
            monkeypatch.setattr(module, "CONFIG_PATH", path)
