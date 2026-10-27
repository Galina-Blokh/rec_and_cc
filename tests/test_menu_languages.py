"""Menu "Languages": toggling a language updates the live config and config.toml.

Stories: US-19.
"""

import threading
import tomllib

import numpy as np

from recandcc.config import load_config
from recandcc.models import Mode
from recandcc.transcribe import Transcriber
from recandcc.ui import common
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
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda *_a: load_config(path))
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
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    path.write_text('[transcription]\nlanguages = ["iw"]\n')
    ctrl._config_sig = None
    ctrl._maybe_reload_config()
    assert ctrl.cfg.transcription.languages == ["he"]
    _toggle(ctrl, "en")  # the reloaded spelling is kept on the next save
    assert tomllib.loads(path.read_text())["transcription"]["languages"] == ["iw", "en"]


def test_menu_save_keeps_other_keys_edited_by_hand(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    ctrl._config_sig = common.file_signature(path)
    text = "live_captions = false\n" + path.read_text() + "audio_retention_days = 7\n"
    path.write_text(text)  # edited, not reloaded yet
    _toggle(ctrl, "he")
    saved = load_config(path)
    assert saved.live_captions is False  # the hand edits survive the menu save
    assert saved.transcription.audio_retention_days == 7
    assert saved.transcription.languages == ["en", "ru", "he"]
    ctrl._maybe_reload_config()  # and are still picked up: the edit wasn't swallowed
    assert ctrl.cfg.transcription.languages == ["en", "ru", "he"]
    assert ctrl._config_sig == common.file_signature(path)


def test_menu_save_leaves_an_unparseable_config_alone(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    path.write_text("[transcription\n")  # mid-edit
    _toggle(ctrl, "he")
    assert path.read_text() == "[transcription\n"
    assert ctrl.events.get_nowait()[1] == "Languages not saved"


def test_languages_swapped_mid_transcription_never_index_an_empty_list(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch, '["he"]')
    transcriber = Transcriber(ctrl.cfg.transcription)
    calls = {"n": 0}

    class Flipping(type(ctrl.cfg.transcription)):
        """`languages` turns from ["he"] into [] after the first read."""

        @property
        def languages(self):  # pyright: ignore[reportIncompatibleVariableOverride]
            calls["n"] += 1
            return ["he"] if calls["n"] == 1 else []

        @languages.setter
        def languages(self, _v):
            pass

    transcriber.cfg = Flipping()
    audio = np.zeros(1600, np.float32)
    assert transcriber.detect_language(audio) == ("he", 1.0)
    calls["n"] = 0
    assert transcriber._region_languages(audio, [(0.0, 0.1)]) == [("he", 1.0)]
