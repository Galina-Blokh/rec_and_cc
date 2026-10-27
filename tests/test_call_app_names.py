"""Opt-in speaker names from the call app's participant list, with fakes (no real
Accessibility, no helper): gating, silent trust checks, storage, conservative assignment,
logging and Windows import safety.

Stories: US-31.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_recorder import FakeSource, sine
from test_transcribe import make_session

from recandcc import participants
from recandcc.config import Config
from recandcc.models import AppTarget, SessionInfo, TriggerKind
from recandcc.recorder import Recorder
from recandcc.transcribe import callnames, voices
from recandcc.ui.common import Controller

TARGET = AppTarget(("us.zoom",), (123,))
NAME = "Zelda Quuxington"  # a fake participant name, must never reach a log


def on_cfg(**kw) -> Config:
    cfg = Config()
    cfg.transcription.speaker_names = kw.get("speaker_names", True)
    cfg.transcription.names_from_call_app = kw.get("names", True)
    return cfg


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


def test_sampler_not_started_when_disabled_or_not_macos(mac, monkeypatch):
    assert participants.start_sampler(on_cfg(names=False), TARGET) is None
    assert participants.start_sampler(on_cfg(speaker_names=False), TARGET) is None
    assert participants.start_sampler(on_cfg(), None) is None
    monkeypatch.setattr(sys, "platform", "win32")
    assert participants.start_sampler(on_cfg(), TARGET) is None


def test_nothing_is_read_without_accessibility_trust():
    reads = []
    s = participants.ParticipantSampler(
        TARGET, trusted=lambda: False, read=lambda t: reads.append(t) or [], first_after=0
    )
    s.start()
    s._thread.join(2)
    assert reads == [] and s.stop() == []


def test_sampler_collects_union_in_order():
    seen = iter([["Anna", "Bob (Me)"], ["Anna", "Cleo, Computer audio muted", "anna"], None])
    got = threading.Event()

    def read(_t):
        try:
            return next(seen)
        except StopIteration:
            got.set()
            return None

    s = participants.ParticipantSampler(
        TARGET, trusted=lambda: True, read=read, interval=0.01, first_after=0
    )
    s.start()
    assert got.wait(5)
    assert s.stop() == ["Anna", "Cleo"]


def test_clean_names():
    raw = ["Anna (Host)", "Participants (3)", "7", "Bob (You)", "x" * 99, "", "Дима", "Anna"]
    assert participants.clean_names(raw) == ["Anna", "Дима"]


def test_trust_check_never_prompts(mac, monkeypatch):
    calls = []

    def run(argv, **_kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"trusted": false}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr("recandcc.native.find_helper", lambda *_: "/h/rcc-audiotap")
    assert participants.ax_trusted() is False
    assert participants.read_participants(TARGET) is None
    assert [c[1] for c in calls] == ["ax-status", "participants"]
    assert "--pid" in calls[1] and "us.zoom" in calls[1]


def test_helper_never_uses_the_prompting_api():
    src = Path(participants.__file__).parents[2] / "native/macos/AudioTap/Sources"
    code = "\n".join(
        line
        for p in src.rglob("*.swift")
        for line in p.read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    assert "AXIsProcessTrusted()" in code
    assert "AXIsProcessTrustedWithOptions" not in code
    assert "kAXTrustedCheckOptionPrompt" not in code


def _recorder(tmp_path, cfg):
    cfg.output_dir = tmp_path / "out"
    return Recorder(
        cfg,
        mic_factory=lambda: FakeSource(sine(1)),
        system_factory=lambda: FakeSource(sine(2)),
        min_session_s=0.1,
    )


def test_recorder_stores_names_in_session_json(mac, tmp_path, monkeypatch):
    real = participants.ParticipantSampler
    monkeypatch.setattr(
        participants,
        "ParticipantSampler",
        lambda target, **kw: real(
            target, trusted=lambda: True, read=lambda t: ["Anna"], interval=0.01, first_after=0
        ),
    )
    rec = _recorder(tmp_path, on_cfg())
    s = rec.start(TriggerKind.CALL, "zoom.us", target=TARGET)
    time.sleep(0.3)
    assert rec.stop() is not None
    assert SessionInfo.load(s.directory).participants == ["Anna"]


def test_recorder_without_the_option_has_no_sampler(mac, tmp_path):
    rec = _recorder(tmp_path, on_cfg(names=False))
    rec.start(TriggerKind.CALL, "zoom.us", target=TARGET)
    assert rec._sampler is None
    rec.stop()


# -- assignment ---------------------------------------------------------------------------------
def test_plan_assigns_only_the_one_to_one_case():
    assert callnames.plan(["Me", "Speaker 1"], ["Anna"]) == {"Speaker 1": "Anna"}
    assert callnames.plan(["Me", "Others"], ["Anna"]) == {"Others": "Anna"}
    # several unknowns and several names: nothing is guessed
    assert callnames.plan(["Speaker 1", "Speaker 2"], ["Anna", "Bob"]) == {}
    assert callnames.plan(["Speaker 1"], ["Anna", "Bob"]) == {}
    assert callnames.plan(["Speaker 1", "Speaker 2"], ["Anna"]) == {}
    assert callnames.plan(["Speaker 1"], []) == {}


def test_plan_profile_matches_take_precedence_and_me_is_untouched():
    # Bob was recognized by voice: his name is taken, the one free name goes to Speaker 2
    assert callnames.plan(["Me", "Bob", "Speaker 2"], ["Bob", "Anna"]) == {"Speaker 2": "Anna"}
    # Me is never an unknown speaker, and its label is never a free name
    assert callnames.plan(["Me"], ["Anna"]) == {}
    unnamed, free = callnames.split_candidates(["Me", "Speaker 1"], ["Me", "Anna"])
    assert unnamed == ["Speaker 1"] and free == ["Anna"]


def _session_with_voice(tmp_path, monkeypatch, names):
    monkeypatch.setattr(voices, "VOICES_DIR", tmp_path / "voices")
    s = make_session(tmp_path)
    s.participants = names
    s.save()
    seg = {"language": "en"}
    (tmp_path / "transcript.json").write_text(
        json.dumps(
            {
                "duration": 5,
                "model": "m",
                "segments": [
                    {"start": 0, "end": 1, "text": "hi", "speaker": "Me", **seg},
                    {"start": 1, "end": 2, "text": "hello", "speaker": "Speaker 1", **seg},
                ],
            }
        )
    )
    voices.write_session_voices(s, {"Speaker 1": np.eye(192)[0]})
    return s


def test_apply_names_through_the_naming_function(tmp_path, monkeypatch):
    _session_with_voice(tmp_path, monkeypatch, ["Anna"])
    assert callnames.apply(tmp_path) is True
    assert "Anna" in (tmp_path / "transcript.json").read_text()
    assert [p.name for p in voices.load_profiles()] == ["Anna"]


def test_apply_leaves_ambiguous_sessions_alone(tmp_path, monkeypatch):
    _session_with_voice(tmp_path, monkeypatch, ["Anna", "Bob"])
    assert callnames.apply(tmp_path) is False
    assert "Speaker 1" in (tmp_path / "transcript.json").read_text()
    assert voices.load_profiles() == []
    assert callnames.suggestions(tmp_path) == ["Anna", "Bob"]


def test_logs_never_contain_names(tmp_path, monkeypatch, caplog):
    _session_with_voice(tmp_path, monkeypatch, [NAME])
    caplog.set_level(logging.DEBUG)
    s = participants.ParticipantSampler(
        TARGET, trusted=lambda: True, read=lambda t: [NAME], interval=0.01, first_after=0
    )
    s.start()
    deadline = time.monotonic() + 5
    while not s._names and time.monotonic() < deadline:
        time.sleep(0.01)
    names = s.stop()
    participants.store(SessionInfo.load(tmp_path), names)
    callnames.apply(tmp_path)
    assert names == [NAME]
    assert not [r for r in caplog.records if NAME in r.getMessage()]


# -- menu toggle --------------------------------------------------------------------------------
class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


class Mon:
    def snapshot(self):
        return []


def make_ctrl(**kw):
    cfg = on_cfg(names=False, **kw)
    cfg.save = lambda *_a, **_k: None  # type: ignore[method-assign]
    return Controller(cfg, monitor=Mon(), recorder=object(), tq=Q(), start_threads=False)


def test_toggle_explains_and_opens_settings_only_when_untrusted(monkeypatch):
    opened = []
    monkeypatch.setattr(participants, "ax_trusted", lambda: False)
    monkeypatch.setattr(participants, "open_accessibility_settings", lambda: opened.append(1))
    ctrl = make_ctrl()
    assert ctrl.toggle_call_app_names(lambda: False) is False  # user said "Not now"
    assert opened == [] and not ctrl.cfg.transcription.names_from_call_app
    assert ctrl.toggle_call_app_names(lambda: True) is True
    assert opened == [1] and ctrl.cfg.transcription.names_from_call_app
    assert ctrl.toggle_call_app_names(lambda: pytest.fail("no dialog to turn off")) is False


def test_toggle_trusted_needs_no_dialog_and_speaker_names_is_required(monkeypatch):
    monkeypatch.setattr(participants, "ax_trusted", lambda: True)
    monkeypatch.setattr(participants, "open_accessibility_settings", lambda: pytest.fail("open"))
    assert make_ctrl().toggle_call_app_names(lambda: pytest.fail("dialog")) is True
    off = make_ctrl(speaker_names=False)
    assert off.toggle_call_app_names(lambda: True) is False


def test_module_is_inert_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert participants.ax_trusted() is False
    participants.open_accessibility_settings()  # no-op, no subprocess
    assert participants.enabled(on_cfg()) is False
