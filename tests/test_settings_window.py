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


@pytest.fixture(autouse=True)
def _real_config_untouched():
    """Nothing here may write the owner's real config.toml (all saves use tmp paths)."""
    from recandcc.config import CONFIG_PATH

    def stamp():
        try:
            st = CONFIG_PATH.stat()
            return st.st_mtime_ns, st.st_size
        except OSError:
            return None

    before = stamp()
    settings_window._closing.clear()
    yield
    proc = settings_window._proc
    if proc is not None and proc.poll() is None:
        proc.kill()  # never leak a stub helper into the next test
        proc.wait()
    settings_window._closing.clear()
    assert stamp() == before, "a test wrote the real config.toml"


def join_threads(name, timeout=5.0):
    for t in threading.enumerate():
        if t.name == name:
            t.join(timeout)
            assert not t.is_alive(), f"{name} did not finish"


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
    join_threads("save-config")
    assert ctrl.cfg.transcription.keep_audio is False
    assert ctrl.cfg.detection.mode == Mode.AUTO
    assert ctrl.cfg.transcription.languages == ["en", "he"]
    # the engine's own save (set_mode) merged into the file: unknown keys survive
    assert "unknown = 1" in path.read_text()


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
        {"output_dir": "relative/dir"},
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
        "import signal,time\nsignal.signal(signal.SIGUSR1, signal.SIG_IGN)\ntime.sleep(1.0)\n"
        'print(\'{"settings": "cancelled"}\')\n',
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
    deadline = time.monotonic() + 5
    while ctrl._settings_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not ctrl._settings_lock.locked()
    assert ctrl.events.empty()  # killed on quit = cancelled: no fallback, no config opened


@mac_only
def test_only_changed_values_are_applied(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    # same mode as shown, a changed flag: only the flag comes back from run_settings
    values = dict(settings_window.snapshot(ctrl.cfg), keep_audio=False)
    fake_helper(tmp_path, monkeypatch, reply({"settings": "saved", "values": values}))
    assert settings_window.run_settings(ctrl.cfg) == {"keep_audio": False}


@mac_only
def test_mode_change_keeps_unknown_keys(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    ctrl.engine.set_mode(Mode.OFF)  # the engine's own save
    join_threads("save-config")
    text = path.read_text()
    assert "unknown = 1" in text and 'mode = "off"' in text


@mac_only
def test_baseline_is_the_snapshot_the_window_opened_with(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    fake_helper(
        tmp_path,
        monkeypatch,
        "import json,sys,time\n"
        "snap=json.load(open(sys.argv[sys.argv.index('--config')+1]))\n"
        "time.sleep(0.5)\n"
        "snap['keep_audio']=False\n"
        "print(json.dumps({'settings':'saved','values':snap}))\n",
    )
    threading.Timer(0.2, lambda: setattr(ctrl.cfg.detection, "mode", Mode.AUTO)).start()
    # the mode was changed from the menu while the window was open: not part of the result
    assert settings_window.run_settings(ctrl.cfg) == {"keep_audio": False}


@mac_only
def test_timeout_is_cancelled_and_releases_the_lock(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    fake_helper(tmp_path, monkeypatch, "import time\ntime.sleep(60)\n")
    monkeypatch.setattr(settings_window, "SETTINGS_TIMEOUT_S", 0.3)
    monkeypatch.setattr(
        settings_window,
        "run_settings",
        lambda cfg, _orig=settings_window.run_settings: _orig(cfg, timeout=0.3),
    )
    monkeypatch.setattr(common, "run_settings", settings_window.run_settings)
    ctrl.open_settings_async()
    deadline = time.monotonic() + 10
    while ctrl._settings_lock.locked() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not ctrl._settings_lock.locked()
    assert ctrl.events.empty()  # a timeout is a cancel: nothing saved, no fallback


@mac_only
def test_extra_languages_pass_through(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)
    ctrl.cfg.transcription.languages = ["en", "fr"]
    fake_helper(
        tmp_path,
        monkeypatch,
        "import json,sys\n"
        "snap=json.load(open(sys.argv[sys.argv.index('--config')+1]))\n"
        "assert snap['languages']==['en','fr']\n"
        "snap['languages']=['ru','fr']\n"
        "print(json.dumps({'settings':'saved','values':snap}))\n",
    )
    ctrl.open_settings_async()
    assert next_event(ctrl, {"settings_saved", "notice", "settings_fallback"})[0] == (
        "settings_saved"
    )
    assert load_config(path).transcription.languages == ["ru", "fr"]


@mac_only
def test_closing_before_the_window_starts_kills_it(tmp_path, monkeypatch):
    ctrl, _ = make_ctrl(tmp_path, monkeypatch)
    settings_window.close_settings()
    assert settings_window.run_settings(ctrl.cfg) is None


@mac_only
def test_failed_write_rolls_back_and_keeps_the_mode(tmp_path, monkeypatch):
    ctrl, path = make_ctrl(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise PermissionError("locked")

    monkeypatch.setattr(common, "save_settings", boom)
    fake_helper(tmp_path, monkeypatch, reply({"settings": "saved", "values": GOOD}))
    ctrl.open_settings_async()
    ev = next_event(ctrl, {"settings_saved", "settings_fallback", "notice"})
    assert ev[0] == "notice"
    assert ctrl.cfg.detection.mode == Mode.ASK  # not applied: the file was not written
    assert ctrl.cfg.transcription.keep_audio is True and ctrl.cfg.live_captions is True
