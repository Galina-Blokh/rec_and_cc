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
def _no_cpu_summary_model(tmp_path_factory, monkeypatch):
    """Fast tests never download the CPU summary model or use the user's model cache, and
    pick the generator as on a machine without onnxruntime-genai (the MLX fakes apply
    everywhere); test_summary_cpu.py switches the CPU backend on with a fake."""
    from recandcc.transcribe import summary_cpu

    monkeypatch.setattr(summary_cpu, "MODELS_DIR", tmp_path_factory.mktemp("models"))
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: False)


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
    names += ["recandcc.ui.common", "recandcc.transcribe.diarize"]
    if sys.platform == "darwin":
        names.append("recandcc.ui.mac")
    for name in names:
        with contextlib.suppress(ImportError):
            importlib.import_module(name)


def _logging_state():
    import faulthandler
    import logging
    import sys
    import threading

    root = logging.getLogger()
    return (
        list(root.handlers),
        root.level,
        getattr(root, "_recandcc_configured", False),
        sys.excepthook,
        threading.excepthook,
        faulthandler.is_enabled(),
    )


def _undo_setup_logging(before) -> None:
    """Undo what a test's `setup_logging` left behind: root handlers (and the open log file),
    the configured flag, our faults file and the excepthooks. faulthandler is restored to its
    previous state, never switched off under pytest's own plugin."""
    import faulthandler
    import logging
    import sys
    import threading

    from recandcc import logging_setup

    handlers, level, configured, excepthook, thread_hook, fault_enabled = before
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)
    root._recandcc_configured = configured  # type: ignore[attr-defined]
    sys.excepthook = excepthook
    threading.excepthook = thread_hook
    if logging_setup._faults_file is not None:
        # pytest's own faulthandler plugin enables it process-wide: keep it on (pointed at
        # stderr) rather than disabling it, and close only the file our code opened.
        if fault_enabled:
            faulthandler.enable(file=sys.__stderr__, all_threads=True)
        else:
            faulthandler.disable()
        logging_setup._faults_file.close()
        logging_setup._faults_file = None
    logging_setup._installed = False


@pytest.fixture(autouse=True)
def _private_config(request, tmp_path, monkeypatch, _import_path_modules):
    """No test reads or writes the user's real config, logs, LaunchAgent plist, lock file or
    default output folder, even through a bare `Config().save()` or a module that imported
    the constant by name."""
    import sys

    before = _logging_state()
    private = {
        "CONFIG_DIR": tmp_path / "private-config",
        "CONFIG_PATH": tmp_path / "private-config" / "config.toml",
        "LOG_DIR": tmp_path / "private-logs",
        "LOG_FILE": tmp_path / "private-logs" / "recandcc.log",
        "FAULTS_FILE": tmp_path / "private-logs" / "faults.log",
        "PLIST_PATH": tmp_path / "private-launchagents" / "recandcc.plist",
        "DEFAULT_OUTPUT_DIR": tmp_path / "private-transcribes",
    }
    if request.node.get_closest_marker("slow") is None:  # slow tests need the real model cache
        private["MODELS_DIR"] = tmp_path / "private-models"
        private["HF_CACHE_DIR"] = tmp_path / "private-hf-cache"
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] != "recandcc":
            continue
        for const, value in private.items():
            if hasattr(module, const):
                monkeypatch.setattr(module, const, value)
    yield
    _undo_setup_logging(before)
