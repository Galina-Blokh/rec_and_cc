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


@pytest.fixture(scope="session", autouse=True)
def _import_path_modules():
    """Import the modules that bind user-level paths by name up front, so the guard below
    patches (and restores) their bindings in every test."""
    import contextlib
    import importlib
    import sys

    names = ["recandcc.config", "recandcc.autostart", "recandcc.logging_setup", "recandcc.cli"]
    names.append("recandcc.ui.common")
    if sys.platform == "darwin":
        names.append("recandcc.ui.mac")
    for name in names:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)


@pytest.fixture(autouse=True)
def _private_config(tmp_path, monkeypatch, _import_path_modules):
    """No test reads or writes the user's real config, logs, LaunchAgent plist, lock file or
    default output folder, even through a bare `Config().save()` or a module that imported
    the constant by name."""
    import sys

    private = {
        "CONFIG_DIR": tmp_path / "private-config",
        "CONFIG_PATH": tmp_path / "private-config" / "config.toml",
        "LOG_DIR": tmp_path / "private-logs",
        "LOG_FILE": tmp_path / "private-logs" / "recandcc.log",
        "FAULTS_FILE": tmp_path / "private-logs" / "faults.log",
        "PLIST_PATH": tmp_path / "private-launchagents" / "recandcc.plist",
        "DEFAULT_OUTPUT_DIR": tmp_path / "private-transcribes",
    }
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "recandcc":
            continue
        for const, value in private.items():
            if hasattr(module, const):
                monkeypatch.setattr(module, const, value)
