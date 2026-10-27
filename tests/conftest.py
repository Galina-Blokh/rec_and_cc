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
def _private_config(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's real config.toml: CONFIG_PATH (looked up at call
    time by Config.save / load_config) and every module-level copy of it point at a temp file.
    """
    import sys

    from recandcc import config

    path = tmp_path_factory.mktemp("config") / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    for name, mod in list(sys.modules.items()):  # `from .config import CONFIG_PATH` copies
        if name.startswith("recandcc.") and hasattr(mod, "CONFIG_PATH"):
            monkeypatch.setattr(mod, "CONFIG_PATH", path)
