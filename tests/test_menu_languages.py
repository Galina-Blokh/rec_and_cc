"""Menu "Languages": toggling a language updates the live config and config.toml.

Stories: US-19.
"""

import threading
import tomllib

import numpy as np

from recandcc.config import Config, load_config
from recandcc.models import Mode
from recandcc.transcribe import Transcriber
from recandcc.ui.common import Controller
from recandcc.ui.present import ANY_LANGUAGE, LANGUAGES_KEEP_ONE


class Monitor:
    def snapshot(self):
        return []


class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def _ctrl(tmp_path, monkeypatch, languages: str):
    path = tmp_path / "config.toml"
    path.write_text(f'[detection]\nmode = "off"\n[transcription]\nlanguages = {languages}\n')
    cfg = load_config(path)
    monkeypatch.setattr(cfg, "save", lambda: Config.save(cfg, path))
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    assert cfg.detection.mode == Mode.OFF
    return ctrl, path


def _toggle(ctrl: Controller, code: str) -> None:
    ctrl.toggle_language(code)
    for t in threading.enumerate():
        if t.name == "save-config":
            t.join(5)


def test_menu_language_toggle_saves_and_keeps_hand_written_codes(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["EN", "iw", "de"]')
    tr = ctrl.cfg.transcription
    _toggle(ctrl, "ru")
    assert tr.languages == ["en", "he", "de", "ru"]
    saved = tomllib.loads(path.read_text())["transcription"]["languages"]
    assert saved == ["EN", "iw", "de", "ru"]  # own spelling kept
    _toggle(ctrl, "he")
    assert tr.languages == ["en", "de", "ru"]
    assert load_config(path).transcription.languages == ["en", "de", "ru"]
    _toggle(ctrl, ANY_LANGUAGE)
    assert tr.languages == [] and load_config(path).transcription.languages == []


def test_menu_language_toggle_refuses_to_drop_the_last_language(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["he"]')
    before = path.read_text()
    _toggle(ctrl, "he")
    assert ctrl.cfg.transcription.languages == ["he"]
    assert path.read_text() == before
    assert ctrl.events.get_nowait() == ("notice", "Languages", LANGUAGES_KEEP_ONE)


def test_menu_language_change_reaches_the_transcriber_at_once(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch, '["en", "he"]')
    transcriber = Transcriber(ctrl.cfg.transcription)  # shares the config, as in the app
    _toggle(ctrl, "en")
    # a single language needs no detection: answered without loading a model
    assert transcriber.detect_language(np.zeros(1600, np.float32)) == ("he", 1.0)


def test_hand_edited_languages_are_picked_up_live(tmp_path, monkeypatch):
    from recandcc.ui import common

    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
    path.write_text('[transcription]\nlanguages = ["iw"]\n')
    ctrl._config_mtime = -1.0
    ctrl._maybe_reload_config()
    assert ctrl.cfg.transcription.languages == ["he"]
    _toggle(ctrl, "en")  # the reloaded spelling is kept on the next save
    assert tomllib.loads(path.read_text())["transcription"]["languages"] == ["iw", "en"]
