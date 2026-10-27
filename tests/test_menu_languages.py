"""Menu "Languages": toggling a language updates the live config and config.toml.

Stories: US-19.
"""

import os
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
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
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
    ctrl._config_mtime = -1.0
    ctrl._maybe_reload_config()
    assert ctrl.cfg.transcription.languages == ["he"]
    _toggle(ctrl, "en")  # the reloaded spelling is kept on the next save
    assert tomllib.loads(path.read_text())["transcription"]["languages"] == ["iw", "en"]


def test_menu_save_keeps_other_keys_edited_by_hand(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    ctrl._config_mtime = ctrl._mtime()
    text = "live_captions = false\n" + path.read_text() + "audio_retention_days = 7\n"
    path.write_text(text)  # edited, not reloaded yet
    os.utime(path, (ctrl._config_mtime + 5, ctrl._config_mtime + 5))  # coarse clocks
    _toggle(ctrl, "he")
    saved = load_config(path)
    assert saved.live_captions is False  # the hand edits survive the menu save
    assert saved.transcription.audio_retention_days == 7
    assert saved.transcription.languages == ["en", "ru", "he"]
    ctrl._maybe_reload_config()  # and are still picked up: the edit's mtime wasn't swallowed
    assert ctrl.cfg.transcription.languages == ["en", "ru", "he"]
    assert ctrl._config_mtime == ctrl._mtime()


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


class _ProbeLock:
    """Stands in for ``_lang_lock``: records whether ``_save_lock`` is free when it is taken."""

    def __init__(self, ctrl: Controller, seen: list[bool]):
        self._inner, self._ctrl, self._seen = threading.Lock(), ctrl, seen

    def __enter__(self):
        free = self._ctrl._save_lock.acquire(blocking=False)
        if free:
            self._ctrl._save_lock.release()
        self._seen.append(free)
        return self._inner.__enter__()

    def __exit__(self, *a):
        return self._inner.__exit__(*a)


def test_reload_applies_values_under_the_save_lock(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en"]')
    seen: list[bool] = []
    ctrl._lang_lock = _ProbeLock(ctrl, seen)  # type: ignore[assignment]
    path.write_text('[transcription]\nlanguages = ["ru"]\n')
    ctrl._config_mtime = -1.0
    ctrl._maybe_reload_config()
    assert seen == [False]  # applying happened while a save could not run
    assert ctrl.cfg.transcription.languages == ["ru"]


def test_menu_click_during_reload_is_not_lost(tmp_path, monkeypatch):
    ctrl, path = _ctrl(tmp_path, monkeypatch, '["en"]')
    in_load, go = threading.Event(), threading.Event()
    real = common.load_config

    def slow_load():
        cfg = real()
        in_load.set()
        assert go.wait(5)
        return cfg

    monkeypatch.setattr(common, "load_config", slow_load)
    path.write_text('[transcription]\nlanguages = ["ru"]\n')
    ctrl._config_mtime = -1.0
    reload_t = threading.Thread(target=ctrl._maybe_reload_config)
    reload_t.start()
    assert in_load.wait(5)  # the reload holds the lock, mid-way
    ctrl.toggle_language("he")  # menu click: memory updated, save queued behind the reload
    go.set()
    reload_t.join(5)
    for t in threading.enumerate():
        if t.name == "save-config":
            t.join(5)
    # the click is not lost: memory and the file agree, and both hold it
    assert ctrl.cfg.transcription.languages == ["en", "he"]
    assert load_config(path).transcription.languages == ["en", "he"]


def test_own_save_does_not_trigger_a_reload(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch, '["en", "ru"]')
    ctrl._config_mtime = ctrl._mtime()
    calls = []
    real = common.load_config
    monkeypatch.setattr(common, "load_config", lambda: calls.append(1) or real())
    _toggle(ctrl, "he")
    assert ctrl._config_mtime == ctrl._mtime()
    ctrl._maybe_reload_config()
    assert calls == []
