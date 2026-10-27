"""The conftest guards keep tests away from the user's real data, even if a test undoes
its own monkeypatching (a test's `monkeypatch.undo()` once dropped the private voices folder
and wrote a test profile into the real one).

Stories: none.
"""

from pathlib import Path

from recandcc import autostart, config, logging_setup, search, semantic
from recandcc.transcribe import diarize, voices


def _private(path: Path, tmp_root: Path) -> bool:
    return tmp_root in path.parents


def test_guards_survive_monkeypatch_undo(tmp_path, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(voices, "LOCK_WAIT_S", 0.0)  # any patch, then undo them all
    monkeypatch.undo()
    root = tmp_path_factory.getbasetemp()
    assert _private(Path(voices.VOICES_DIR), root)
    assert _private(Path(search.INDEX_PATH), root)
    assert _private(Path(config.CONFIG_PATH), root)
    assert _private(Path(config.LOG_DIR), root)
    assert _private(Path(logging_setup.LOG_FILE), root)
    assert _private(Path(config.DEFAULT_OUTPUT_DIR), root)
    assert _private(Path(autostart.PLIST_PATH), root)
    assert _private(Path(diarize.MODELS_DIR), root)
    assert _private(Path(semantic.MODELS_DIR), root)
