"""Launch in standby: nothing records until the user picks a mode, and a config reload
(a hand edit, or `recandcc autostart` run in a terminal) doesn't end it, since standby is
never saved.

Stories: US-17.
"""

import threading
import tomllib

from recandcc.config import Config
from recandcc.models import Mode


class _Monitor:
    def snapshot(self):
        return []


class _Queue:
    def submit(self, *_a, **_k):
        pass


def _notices(ctrl):
    """The notices queued so far; state events (queued by every engine change) are skipped."""
    out = []
    while not ctrl.events.empty():
        event = ctrl.events.get_nowait()
        if event[0] == "notice":
            out.append(event)
    return out


def test_a_reload_of_the_saved_mode_does_not_end_standby(tmp_path, monkeypatch):
    from recandcc.ui.common import Controller

    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "auto"\n')
    monkeypatch.setattr("recandcc.ui.common.CONFIG_PATH", path)
    cfg = Config(output_dir=tmp_path)
    cfg.detection.mode = Mode.AUTO
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Queue(), start_threads=False)
    ctrl.engine.enter_standby()
    assert cfg.detection.mode == Mode.OFF and ctrl.engine.standby
    # The file still says auto and changed on disk: the reload must keep standby.
    on = Config(output_dir=tmp_path)
    on.detection.mode = Mode.AUTO
    monkeypatch.setattr("recandcc.ui.common.load_config", lambda *a, **k: on)
    ctrl._mtime = lambda: 42.0
    ctrl._maybe_reload_config()
    assert cfg.detection.mode == Mode.OFF and ctrl.engine.standby
    ctrl.engine.set_mode(Mode.ASK, save=False)  # the user picks a mode: standby ends
    assert not ctrl.engine.standby


def test_launch_enters_standby_before_the_engine_starts(tmp_path, monkeypatch):
    from recandcc.engine import Engine
    from recandcc.ui.common import Controller

    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "auto"\nroom_listening = true\n')
    monkeypatch.setattr("recandcc.ui.common.CONFIG_PATH", path)
    seen = []
    # Only the launch state is under test: no engine thread, hotkey or background workers.
    monkeypatch.setattr(Engine, "start", lambda self: seen.append((self.standby, self.det.mode)))
    for name in (
        "_resubmit_unfinished",
        "_retention_loop",
        "_warm_speaker_model",
        "refresh_async",
        "apply_hotkey",
    ):
        monkeypatch.setattr(Controller, name, lambda self, *a, **k: None)
    cfg = Config(output_dir=tmp_path)
    cfg.detection.mode = Mode.AUTO
    cfg.detection.room_listening = True
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Queue())
    assert seen == [(True, Mode.OFF)]  # standby when the engine starts
    assert _notices(ctrl) == []  # no "Room listening on" at launch: the mic stays closed
    assert 'mode = "auto"' in path.read_text()  # standby is not saved


def test_a_mode_chosen_in_settings_during_standby_shows_the_room_notice_once(tmp_path, monkeypatch):
    from recandcc.ui.common import ROOM_LISTENING_NOTICE, Controller

    monkeypatch.setattr("recandcc.ui.common.CONFIG_PATH", tmp_path / "config.toml")
    cfg = Config(output_dir=tmp_path)
    cfg.detection.mode = Mode.AUTO
    cfg.detection.room_listening = True
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Queue(), start_threads=False)
    ctrl.engine.enter_standby()
    ctrl.apply_settings({"mode": Mode.ASK})  # the Settings window: the mic opens now
    assert _notices(ctrl) == [("notice", "Room listening on", ROOM_LISTENING_NOTICE)]
    ctrl.apply_settings({"mode": Mode.AUTO})
    assert _notices(ctrl) == []  # standby already ended: no second notice


def test_a_mode_chosen_from_the_menu_during_standby_shows_the_room_notice(tmp_path, monkeypatch):
    from recandcc.ui.common import ROOM_LISTENING_NOTICE, Controller

    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "auto"\nroom_listening = true\n')
    monkeypatch.setattr("recandcc.ui.common.CONFIG_PATH", path)
    cfg = Config(output_dir=tmp_path)
    cfg.detection.mode = Mode.AUTO
    cfg.detection.room_listening = True
    ctrl = Controller(cfg, monitor=_Monitor(), recorder=object(), tq=_Queue(), start_threads=False)
    ctrl.engine.enter_standby()
    ctrl.set_mode(Mode.ASK)  # the menu
    for t in threading.enumerate():
        if t.name == "save-config":
            t.join(5)
    assert _notices(ctrl) == [("notice", "Room listening on", ROOM_LISTENING_NOTICE)]
    assert tomllib.loads(path.read_text())["detection"]["mode"] == "ask"
