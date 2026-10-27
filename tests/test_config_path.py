"""The config path is resolved at call time, so tests can redirect it.

Stories: none.
"""

from recandcc import config


def test_bare_save_and_load_use_the_patched_path(tmp_path, monkeypatch):
    path = tmp_path / "elsewhere" / "config.toml"
    monkeypatch.setattr(config, "CONFIG_PATH", path)
    config.Config(hotkey="<ctrl>+<alt>+x").save()
    assert path.exists()
    assert config.load_config().hotkey == "<ctrl>+<alt>+x"
    config.save_languages(["en"])
    assert config.load_config().transcription.languages == ["en"]


def test_conftest_redirects_the_real_config_path():
    assert "private-config" in config.CONFIG_PATH.parts


def test_no_writer_touches_the_real_file(tmp_path):
    """Every config.toml writer (save, save_languages, load_config's first-run default)
    goes to the private path and leaves the real file alone."""
    real = config.CONFIG_DIR / "config.toml"
    before = real.stat() if real.exists() else None
    private = config.CONFIG_PATH
    assert private != real

    assert not private.exists()
    config.load_config()  # first run: creates the default file at the private path
    assert private.exists()
    private.unlink()
    config.Config().save()
    config.save_languages(["ru"])
    assert config.load_config().transcription.languages == ["ru"]

    after = real.stat() if real.exists() else None
    assert (before and before.st_mtime_ns, before and before.st_size) == (
        after and after.st_mtime_ns,
        after and after.st_size,
    )
