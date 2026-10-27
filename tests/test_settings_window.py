"""Native settings window: Python side with a fake helper (no SwiftUI, no display).

Stories: US-30.
"""

import json
import queue
import stat
import sys
import threading
import time

import pytest

from recandcc import settings_window
from recandcc.config import Config, load_config, save_settings, validate_settings
from recandcc.models import Mode
from recandcc.ui import common
from recandcc.ui.common import Controller

mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="the helper is macOS-only")


class Monitor:
    def snapshot(self):
        return []


class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def fake_helper(tmp_path, monkeypatch, script: str):
    """An executable standing in for rcc-audiotap; `script` is its Python body."""
    path = tmp_path / "rcc-audiotap"
    path.write_text(f"#!{sys.executable}\n{script}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("RCC_AUDIOTAP", str(path))


def reply(obj, exit_code=0):
    return f"import json,sys\nprint(json.dumps({obj!r}))\nsys.exit({exit_code})\n"


def make_ctrl(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('# keep me\nunknown = 1\n[detection]\nmode = "ask"\n')
    cfg = load_config(path)
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    return ctrl, path


def next_event(ctrl, kinds, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            ev = ctrl.events.get(timeout=0.1)
        except queue.Empty:
            continue
        if ev[0] in kinds:
            return ev
    raise AssertionError(f"no {kinds} event")


GOOD = {
    "mode": "auto",
    "output_dir": "/tmp/rcc-out",
    "languages": ["en", "he"],
    "keep_audio": False,
    "live_captions": False,
    "speaker_names": True,
}


@mac_only
def test_save_writes_toml_keeps_unknown_keys_and_applies(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    # the snapshot handed to the helper is a readable JSON file
    fake_helper(
        tmp_path,
        monkeypatch,
        "import json,sys\n"
        "snap=json.load(open(sys.argv[sys.argv.index('--config')+1]))\n"
        "assert snap['mode']=='ask' and 'hotkey' in snap\n"
        f"print(json.dumps({{'settings':'saved','values':{GOOD!r}}}))\n",
    )
    ctrl.open_settings_async()
    ev = next_event(ctrl, {"settings_saved", "settings_fallback", "notice"})
    assert ev[0] == "settings_saved"
    text = path.read_text()
    assert "unknown = 1" in text and 'mode = "auto"' in text
    reloaded = load_config(path)
    assert reloaded.transcription.languages == ["en", "he"]
    assert reloaded.transcription.keep_audio is False
    assert reloaded.transcription.speaker_names is True
    assert reloaded.live_captions is False
    assert str(reloaded.output_dir) == "/tmp/rcc-out"
    ctrl.apply_settings(ev[1])  # UI thread
    assert ctrl.cfg.transcription.keep_audio is False
    assert ctrl.cfg.detection.mode == Mode.AUTO
    assert ctrl.cfg.transcription.languages == ["en", "he"]


@mac_only
def test_cancel_changes_nothing(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    before = path.read_text()
    fake_helper(tmp_path, monkeypatch, reply({"settings": "cancelled"}))
    ctrl.open_settings_async()
    deadline = time.monotonic() + 5
    while ctrl._settings_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ctrl.events.empty()
    assert path.read_text() == before


@mac_only
def test_invalid_values_are_rejected(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    before = path.read_text()
    bad = dict(GOOD, mode="sometimes")
    fake_helper(tmp_path, monkeypatch, reply({"settings": "saved", "values": bad}))
    ctrl.open_settings_async()
    ev = next_event(ctrl, {"settings_saved", "settings_fallback", "notice"})
    assert ev[0] == "notice" and ev[1] == "Settings not saved"
    assert path.read_text() == before


@pytest.mark.parametrize(
    "bad",
    [
        {"mode": "x"},
        {"output_dir": ""},
        {"languages": ["xx"]},
        {"languages": "en"},
        {"keep_audio": "yes"},
    ],
)
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate_settings(bad)


def test_validate_allows_any_language_and_dedupes():
    assert validate_settings({"languages": []}) == {"languages": []}
    assert validate_settings({"languages": ["iw", "he", "en"]}) == {"languages": ["he", "en"]}


def test_validate_ignores_unknown_and_cleans():
    out = validate_settings({"languages": ["EN", "iw"], "hotkey": "<cmd>+x", "mode": "off"})
    assert out == {"languages": ["en", "he"], "mode": Mode.OFF}


def test_save_settings_creates_missing_sections(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("autostart = false\n")
    save_settings(validate_settings({"keep_audio": False, "mode": "off"}), path)
    cfg = load_config(path)
    assert cfg.transcription.keep_audio is False and cfg.detection.mode == Mode.OFF


def test_missing_helper_falls_back(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    monkeypatch.setenv("RCC_AUDIOTAP", str(tmp_path / "nope"))
    ctrl.open_settings_async()
    assert next_event(ctrl, {"settings_fallback", "settings_saved", "notice"})[0] == (
        "settings_fallback"
    )


@mac_only
def test_old_helper_without_subcommand_falls_back(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    fake_helper(
        tmp_path, monkeypatch, "import sys\nsys.stderr.write('unknown subcommand')\nsys.exit(2)\n"
    )
    ctrl.open_settings_async()
    assert next_event(ctrl, {"settings_fallback", "settings_saved", "notice"})[0] == (
        "settings_fallback"
    )


@mac_only
def test_window_never_blocks_the_caller(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    fake_helper(
        tmp_path,
        monkeypatch,
        'import time\ntime.sleep(1.0)\nprint(\'{"settings": "cancelled"}\')\n',
    )
    start = time.monotonic()
    ctrl.open_settings_async()
    assert time.monotonic() - start < 0.5  # returns at once; the helper runs on a worker
    assert any(t.name == "settings-window" for t in threading.enumerate())
    ctrl.open_settings_async()  # a second click while open is ignored
    deadline = time.monotonic() + 10
    while ctrl._settings_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not ctrl._settings_lock.locked()


def test_other_platforms_are_unavailable(monkeypatch):
    monkeypatch.setattr(settings_window.sys, "platform", "win32")
    with pytest.raises(settings_window.SettingsUnavailable):
        settings_window.run_settings(Config())


def test_snapshot_has_no_content(tmp_path):
    snap = settings_window.snapshot(load_config(tmp_path / "x.toml"))
    assert set(snap) == {
        "mode",
        "output_dir",
        "languages",
        "keep_audio",
        "live_captions",
        "speaker_names",
        "hotkey",
    }
    json.dumps(snap)


@mac_only
def test_hand_edit_made_before_saving_is_still_reloaded(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    ctrl._config_mtime = -1.0  # the file changed on disk since the last reload
    fake_helper(
        tmp_path, monkeypatch, reply({"settings": "saved", "values": {"keep_audio": False}})
    )
    ctrl.open_settings_async()
    assert next_event(ctrl, {"settings_saved", "notice"})[0] == "settings_saved"
    assert ctrl._config_mtime == -1.0


@mac_only
def test_quitting_closes_an_open_window(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    fake_helper(tmp_path, monkeypatch, "import time\ntime.sleep(60)\n")
    ctrl.open_settings_async()
    deadline = time.monotonic() + 5
    while settings_window._proc is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert settings_window._proc is not None
    settings_window.close_settings()
    assert next_event(ctrl, {"settings_fallback", "settings_saved", "notice"})[0] == (
        "settings_fallback"
    )  # no answer from the closed helper; the app is quitting anyway
