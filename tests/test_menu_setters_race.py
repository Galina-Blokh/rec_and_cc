"""Menu and tray setters mark their key dirty before memory changes: a config reload that
runs while their save is still queued never reverts the click, in memory or on disk.

Stories: US-17.
"""

import contextlib
import sys
import threading
import types

import pytest

from recandcc.config import load_config
from recandcc.ui import common
from recandcc.ui.common import Controller


class Monitor:
    def snapshot(self):
        return []


class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def _ctrl(tmp_path, monkeypatch, body: str = ""):
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "off"\n' + body)
    cfg = load_config(path)
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    return ctrl, path


def _race(ctrl, path, monkeypatch, click, hand_edit: str) -> None:
    """Block the click's save at write_settings, hand-edit the file, reload, then release."""
    in_save, go = threading.Event(), threading.Event()
    real = ctrl.write_settings

    def slow_save(*keys, **kw):
        in_save.set()
        assert go.wait(5)
        return real(*keys, **kw)

    monkeypatch.setattr(ctrl, "write_settings", slow_save)
    click()
    assert in_save.wait(5)  # memory changed, the save is not written yet
    path.write_text(hand_edit)
    ctrl._config_mtime = -1.0
    ctrl._maybe_reload_config()  # reads the hand edit while the click is not on disk
    go.set()
    for t in threading.enumerate():
        if t.name == "save-config":
            t.join(5)


@pytest.mark.skipif(sys.platform != "darwin", reason="rumps menu")
def test_mac_retention_click_survives_a_reload_while_its_save_is_blocked(tmp_path, monkeypatch):
    from recandcc.ui import mac

    ctrl, path = _ctrl(tmp_path, monkeypatch, "[transcription]\naudio_retention_days = 7\n")
    app = mac.MacApp.__new__(mac.MacApp)
    app.ctrl, app.cfg = ctrl, ctrl.cfg
    app._render_retention = lambda: None
    purged = []
    monkeypatch.setattr(ctrl, "purge_audio_now", lambda: purged.append(1))
    hand = "[transcription]\naudio_retention_days = 9\nkeep_audio = false\n"
    _race(
        ctrl,
        path,
        monkeypatch,
        lambda: app._set_retention(types.SimpleNamespace(retention_days=30)),
        hand,
    )
    tr = ctrl.cfg.transcription
    assert tr.audio_retention_days == 30  # the click wins over the hand edit
    assert tr.keep_audio is False  # the rest of the hand edit still reloads
    saved = load_config(path).transcription
    assert saved.audio_retention_days == 30 and saved.keep_audio is False
    assert purged == [1]  # the purge ran after the write
    assert not ctrl._dirty


def test_tray_captions_click_survives_a_reload_while_its_save_is_blocked(tmp_path, monkeypatch):
    stubs = {
        "pystray": types.SimpleNamespace(Icon=None, MenuItem=None, Menu=None),
        "PIL": types.SimpleNamespace(Image=None, ImageDraw=None),
    }
    for name, mod in stubs.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "recandcc.ui.tray", raising=False)
    from recandcc.ui import tray

    try:
        ctrl, path = _ctrl(tmp_path, monkeypatch)
        assert ctrl.cfg.live_captions is True
        app = tray.TrayApp.__new__(tray.TrayApp)
        app.ctrl, app.cfg, app._live_id = ctrl, ctrl.cfg, None
        app.captions = types.SimpleNamespace(post=lambda *_a: None)
        hand = "live_captions = true\n[transcription]\nkeep_audio = false\n"
        _race(ctrl, path, monkeypatch, app._toggle_captions, hand)
        assert ctrl.cfg.live_captions is False
        assert ctrl.cfg.transcription.keep_audio is False
        saved = load_config(path)
        assert saved.live_captions is False and saved.transcription.keep_audio is False
        assert not ctrl._dirty
    finally:
        sys.modules.pop("recandcc.ui.tray", None)  # don't leak the module built on stubs


def test_room_listening_click_survives_a_reload_while_its_save_is_blocked(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    _race(
        ctrl,
        path,
        monkeypatch,
        lambda: ctrl.set_room_listening(True),
        "[detection]\nroom_listening = false\n",
    )
    assert ctrl.cfg.detection.room_listening is True
    assert load_config(path).detection.room_listening is True
    assert not ctrl._dirty


def test_failed_save_thread_unmarks_once_keeping_a_concurrent_click(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl._mark_dirty(("live_captions",), 1)  # another click's save is still pending

    def boom(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", boom)
    with pytest.raises(RuntimeError):
        ctrl.menu_set("live_captions", False)
    assert ctrl._dirty == {"live_captions": 1}  # 2 -> 1, not 2 -> 0


def test_raising_change_unmarks_once_keeping_a_concurrent_click(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl._mark_dirty(("live_captions",), 1)

    def change():
        raise ValueError("boom")

    with contextlib.suppress(ValueError):
        ctrl.menu_apply(("live_captions",), change)
    assert ctrl._dirty == {"live_captions": 1}
