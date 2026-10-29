"""Launch in standby: nothing records until the user picks a mode, and a config reload
(a hand edit, or `recandcc autostart` run in a terminal) doesn't end it, since standby is
never saved.

Stories: US-17.
"""

from recandcc.config import Config
from recandcc.models import Mode


class _Monitor:
    def snapshot(self):
        return []


class _Queue:
    def submit(self, *_a, **_k):
        pass


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
