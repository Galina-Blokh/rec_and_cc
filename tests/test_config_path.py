"""User-level paths are resolved at call time and redirected into tmp_path for every test.

Stories: none.
"""

import importlib

import pytest

from recandcc import config

GUARDED = [
    ("recandcc.config", "CONFIG_DIR"),
    ("recandcc.config", "CONFIG_PATH"),
    ("recandcc.config", "LOG_DIR"),
    ("recandcc.config", "DEFAULT_OUTPUT_DIR"),
    ("recandcc.cli", "CONFIG_PATH"),
    ("recandcc.cli", "CONFIG_DIR"),
    ("recandcc.cli", "LOG_DIR"),
    ("recandcc.cli", "LOG_FILE"),
    ("recandcc.logging_setup", "LOG_DIR"),
    ("recandcc.logging_setup", "LOG_FILE"),
    ("recandcc.logging_setup", "FAULTS_FILE"),
    ("recandcc.autostart", "PLIST_PATH"),
    ("recandcc.autostart", "LOG_DIR"),
    ("recandcc.ui.common", "CONFIG_PATH"),
    ("recandcc.ui.common", "LOG_DIR"),
]


@pytest.mark.parametrize(("module", "name"), GUARDED)
def test_guarded_constant_points_under_tmp(module, name, tmp_path):
    value = getattr(importlib.import_module(module), name)
    assert tmp_path in value.parents or value == tmp_path


def test_bare_output_dir_default_is_private(tmp_path):
    assert tmp_path in config.Config().output_dir.parents


def test_bare_save_and_load_use_the_patched_path(tmp_path, monkeypatch):
    path = tmp_path / "elsewhere" / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    config.Config(hotkey="<ctrl>+<alt>+x").save()
    assert path.exists()
    assert config.load_config().hotkey == "<ctrl>+<alt>+x"
    config.save_languages(["en"])
    assert config.load_config().transcription.languages == ["en"]


def test_writers_go_to_the_private_path(tmp_path):
    """save, save_languages and load_config's first-run default all write the private file."""
    private = config.CONFIG_PATH
    assert tmp_path in private.parents
    assert not private.exists()
    config.load_config()  # first run: creates the default file
    assert private.exists()
    private.unlink()
    config.Config().save()
    config.save_languages(["ru"])
    assert config.load_config().transcription.languages == ["ru"]
