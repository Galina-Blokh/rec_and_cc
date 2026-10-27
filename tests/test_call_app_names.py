"""Opt-in speaker names from the call app's participant list, with fakes (no real
Accessibility, no helper): gating, silent trust checks, private storage, conservative
assignment, logging, the menu toggle and Windows import safety.

Stories: US-31.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import numpy as np
import pytest
from test_recorder import FakeSource, sine
from test_transcribe import make_session

from recandcc import config, participants
from recandcc.config import Config
from recandcc.models import AppTarget, SessionInfo, TriggerKind
from recandcc.recorder import Recorder
from recandcc.transcribe import callnames, voices
from recandcc.ui import common
from recandcc.ui.common import Controller

TARGET = AppTarget(("us.zoom",), (123,))
NAME = "Zelda Quuxington"  # a fake participant name, must never reach a log


def wait_for(condition, seconds=5.0):
    """Poll instead of sleeping a fixed time."""
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert condition()


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
        TARGET,
        trusted=lambda: False,
        read=lambda t: reads.append(t) or ["Anna"],
        interval=0.01,
        first_after=0,
    )
    s.start()
    time.sleep(0.2)
    assert reads == [] and s.stop() == []


def test_trust_granted_mid_call_starts_the_sampling():
    state = {"trusted": False}
    s = participants.ParticipantSampler(
        TARGET,
        trusted=lambda: state["trusted"],
        read=lambda t: ["Anna"],
        interval=0.01,
        first_after=0,
    )
    s.start()
    time.sleep(0.15)
    assert s._names == []
    state["trusted"] = True
    deadline = time.monotonic() + 5
    while not s._names and time.monotonic() < deadline:
        time.sleep(0.01)
    assert s.stop() == ["Anna"]


def test_only_names_stable_over_two_samples_count():
    seen = iter(
        [
            ["Anna", "Bob (Me)", "Noise"],
            ["Anna", "Cleo, Computer audio muted"],
            ["Anna", "Cleo"],
            [],
        ]
    )
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
    # "Noise" and the first "Cleo" flickered; an empty read (panel closed) keeps the last result
    assert s.stop() == ["Anna", "Cleo"]


def test_names_that_scroll_out_of_view_are_kept():
    seen = iter([["Ann", "Bob", "Cy"]] * 2 + [["Ann", "Cy"]] * 2)
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
    assert s.stop() == ["Ann", "Bob", "Cy"]


def test_a_helper_starting_after_stop_is_killed():
    killed = []

    class P:
        def kill(self):
            killed.append(1)

    s = participants.ParticipantSampler(TARGET, trusted=lambda: True, read=lambda t: [])
    s.signal()
    s._track(P())  # type: ignore[arg-type]
    assert killed == [1]


def test_clean_names():
    raw = ["Anna (Host)", "Participants (3)", "7", "Bob (You)", "x" * 99, "", "Дима", "Anna"]
    assert participants.clean_names(raw) == ["Anna", "Дима"]
    # a row has several texts: the user's own marker may be any of them, or the whole name
    assert participants.clean_names(["Bob\n(You)", "You", "Cleo\nmuted", "Me"]) == ["Cleo"]


def test_trust_check_never_prompts(mac, monkeypatch):
    calls = []

    class FakeProc:
        returncode = 0

        def communicate(self, timeout=None):
            return '{"trusted": false}\n', None

    def popen(argv, **_kw):
        calls.append(argv)
        return FakeProc()

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr("recandcc.native.find_helper", lambda *_: "/h/rcc-audiotap")
    assert participants.ax_trusted() is False
    assert participants.read_participants(TARGET) is None
    assert [c[1] for c in calls] == ["ax-status", "participants"]
    assert "--pid" in calls[1] and "us.zoom" in calls[1]
    # the helper's participants command only runs for the process that started it
    assert calls[1][-2:] == ["--parent-pid", str(os.getpid())]
    # ax-status takes no options: the exact argv has nothing after the subcommand
    assert calls[0] == ["/h/rcc-audiotap", "ax-status"]
    assert calls[1] == [
        "/h/rcc-audiotap",
        "participants",
        "--pid",
        "123",
        "--match",
        "us.zoom",
        "--parent-pid",
        str(os.getpid()),
    ]


def built_helper():
    from recandcc.native import find_helper

    if sys.platform != "darwin":
        pytest.skip("macOS helper")
    try:
        return str(find_helper())
    except FileNotFoundError:
        pytest.skip("rcc-audiotap is not built")


def test_real_helper_parses_the_arguments_python_sends():
    helper = built_helper()
    out = subprocess.run(  # noqa: S603
        [helper, "ax-status"], capture_output=True, text=True, timeout=20, check=False
    )
    assert out.returncode == 0 and json.loads(out.stdout)["trusted"] in (True, False)
    # started by this test process, which the helper accepts as parent: no usage error (2)
    ok = subprocess.run(  # noqa: S603
        [helper, "participants", "--match", "zzz-none", "--parent-pid", str(os.getpid())],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert ok.returncode in (0, 3)
    # without the parent pid it refuses
    no = subprocess.run(  # noqa: S603
        [helper, "participants", "--match", "zzz-none"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert no.returncode == 3 and json.loads(no.stdout)["refused"] is True


def swift_sources() -> dict[str, str]:
    src = Path(participants.__file__).parents[2] / "native/macos/AudioTap/Sources"
    return {p.name: p.read_text(encoding="utf-8") for p in src.rglob("*.swift")}


def test_helper_never_uses_the_prompting_api():
    code = "\n".join(
        line
        for text in swift_sources().values()
        for line in text.splitlines()
        if not line.lstrip().startswith("//")
    )
    assert "AXIsProcessTrusted()" in code
    assert "AXIsProcessTrustedWithOptions" not in code
    assert "kAXTrustedCheckOptionPrompt" not in code


def test_helper_walk_is_limited_to_call_windows_and_has_budgets():
    code = swift_sources()["Accessibility.swift"]
    assert "browserIDs" in code and "callPageWords" in code  # browsers: call windows only
    assert "AXUIElementCreateSystemWide()" in code and "perAppSeconds" in code
    assert "startedByRecAndCC" in code  # defense in depth for `participants`
    assert "import Darwin" in code
    # every AX read has its own timeout and stops once the app's budget is used up
    assert "if Date() > axDeadline" in code
    assert code.index("AXUIElementSetMessagingTimeout(element") < code.index(
        "AXUIElementCopyAttributeValue"
    )
    # two call windows: fail closed; and no bare "meet - " page word
    assert "callWindows.count == 1" in code
    page_words = code[code.index("let callPageWords") : code.index("let meetTitlePattern")]
    assert "meet - " not in page_words and "meet.google.com" in page_words
    assert "teams.microsoft.com" in page_words and '"zoom"' not in page_words


def _recorder(tmp_path, cfg):
    cfg.output_dir = tmp_path / "out"
    return Recorder(
        cfg,
        mic_factory=lambda: FakeSource(sine(1)),
        system_factory=lambda: FakeSource(sine(2)),
        min_session_s=0.1,
    )


def test_recorder_keeps_names_in_the_private_store_not_the_session(mac, tmp_path, monkeypatch):
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
    wait_for(lambda: rec._sampler is not None and rec._sampler._names)
    assert rec.stop() is not None
    assert voices.read_session_names(s.id) == ["Anna"]
    # the session folder may sync to a cloud: no name in it
    assert not any("Anna" in p.read_text(errors="ignore") for p in s.directory.glob("*.json"))


def test_recorder_without_the_option_has_no_sampler(mac, tmp_path):
    rec = _recorder(tmp_path, on_cfg(names=False))
    rec.start(TriggerKind.CALL, "zoom.us", target=TARGET)
    assert rec._sampler is None
    rec.stop()


def test_stop_signals_first_and_reaps_after_capture_stopped_outside_the_lock(
    mac, tmp_path, monkeypatch
):
    order = []

    class Fake:
        def signal(self):
            order.append("signal")

        def stop(self):
            order.append(("reap", rec._op_lock.locked()))
            return []

    real_stop = FakeSource.stop

    def stop_source(self):
        order.append("capture stopped")
        return real_stop(self)

    monkeypatch.setattr(FakeSource, "stop", stop_source)
    monkeypatch.setattr(participants, "start_sampler", lambda cfg, target: Fake())
    rec = _recorder(tmp_path, on_cfg())
    rec.start(TriggerKind.CALL, "zoom.us", target=TARGET)
    rec.stop()
    assert order[0] == "signal"  # only sets the flag and kills: nothing waits before capture
    assert order[-1] == ("reap", False)  # waiting happens last, outside the lock
    assert order.count("capture stopped") == 2 and "capture stopped" not in order[-1:]


# -- assignment ---------------------------------------------------------------------------------
def test_plan_assigns_only_the_certain_case():
    assert callnames.plan(["Me", "Speaker 1"], ["Anna"]) == {"Speaker 1": "Anna"}
    # several unknowns and several names: nothing is guessed
    assert callnames.plan(["Speaker 1", "Speaker 2"], ["Anna", "Bob"]) == {}
    # the number of names must equal the number of other speakers: an extra name (maybe from an
    # unrelated list), a missing name or none make the panel untrustworthy
    assert callnames.plan(["Speaker 1"], ["Anna", "Bob"]) == {}
    assert callnames.plan(["Speaker 1", "Speaker 2"], ["Anna"]) == {}
    assert callnames.plan(["Speaker 1"], []) == {}
    # "Others" (no diarization) is not a speaker to name
    assert callnames.plan(["Me", "Others"], ["Anna"]) == {}


def test_plan_known_speakers_must_match_names_and_me_is_untouched():
    labels = ["Me", "Bob Smith", "Speaker 2"]
    assert callnames.plan(labels, ["bob", "Anna"]) == {"Speaker 2": "Anna"}  # first-name overlap
    assert callnames.plan(labels, ["BOB SMITH", "Anna"]) == {"Speaker 2": "Anna"}
    # "Robert" is not "Bob Smith": the panel doesn't show him, so it can't be trusted
    assert callnames.plan(["Me", "Robert", "Speaker 2"], ["Bob Smith", "Anna"]) == {}
    assert callnames.plan(["Me", "Bob", "Speaker 2"], ["Anna", "Cleo"]) == {}
    assert callnames.plan(["Me"], ["Anna"]) == {}


def _session_with_voice(tmp_path, names, said="hello"):
    s = make_session(tmp_path)
    voices.write_session_names(s, names)
    seg = {"language": "en"}
    (tmp_path / "transcript.json").write_text(
        json.dumps(
            {
                "duration": 5,
                "model": "m",
                "segments": [
                    {"start": 0, "end": 1, "text": "hi", "speaker": "Me", **seg},
                    {"start": 1, "end": 2, "text": said, "speaker": "Speaker 1", **seg},
                ],
            }
        )
    )
    voices.write_session_voices(s, {"Speaker 1": np.eye(192)[0]})
    return s


def test_apply_renames_the_label_but_creates_no_voice_profile(tmp_path):
    s = _session_with_voice(tmp_path, ["Anna"])
    assert callnames.apply(tmp_path) is True
    assert "Anna" in (tmp_path / "transcript.json").read_text()
    assert voices.load_profiles() == []  # a name from another app is not a voice identity
    assert list(voices.read_session_voices(s.id)) == ["Anna"]  # confirmable with name-speaker
    assert voices.read_session_names(s.id) == []  # used up


def test_apply_leaves_ambiguous_sessions_alone(tmp_path):
    _session_with_voice(tmp_path, ["Anna", "Bob"])
    assert callnames.apply(tmp_path) is False
    assert "Speaker 1" in (tmp_path / "transcript.json").read_text()
    assert voices.load_profiles() == []
    assert callnames.suggestions(tmp_path) == ["Anna", "Bob"]


def test_a_disagreeing_self_introduction_wins(tmp_path):
    _session_with_voice(tmp_path, ["Anna"], said="Hello everyone, my name is Bob and I run this")
    assert callnames.apply(tmp_path) is False
    assert "Speaker 1" in (tmp_path / "transcript.json").read_text()
    # an agreeing introduction does not get in the way
    _session_with_voice(tmp_path, ["Bob Smith"], said="Hello everyone, my name is Bob")
    assert callnames.apply(tmp_path) is True


def test_logs_never_contain_names(tmp_path, caplog):
    _session_with_voice(tmp_path, [])
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


def test_failure_logs_the_type_only(tmp_path, monkeypatch, caplog):
    from recandcc.config import TranscriptionConfig
    from recandcc.transcribe.engine import Transcriber

    def boom(_directory):
        raise ValueError(NAME)

    monkeypatch.setattr(callnames, "apply", boom)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True, names_from_call_app=True))
    caplog.set_level(logging.INFO)
    t._name_from_call_app(make_session(tmp_path))
    assert "ValueError" in caplog.text and NAME not in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="uses a shell script as the helper")
def test_stop_kills_an_in_flight_helper(tmp_path, monkeypatch):
    slow = tmp_path / "rcc-audiotap"
    slow.write_text("#!/bin/sh\nexec sleep 30\n")
    slow.chmod(0o755)
    monkeypatch.setattr("recandcc.native.find_helper", lambda *_: slow)
    s = participants.ParticipantSampler(TARGET, trusted=lambda: True, interval=60, first_after=0)
    s.start()
    wait_for(lambda: s._proc is not None)  # the helper is running
    assert s._proc is not None and s._proc.poll() is None
    started = time.monotonic()
    assert s.stop() == []
    assert time.monotonic() - started < 2.5
    assert s._proc.poll() is not None  # killed, so the next session's sampler can't overlap
    assert not s._thread.is_alive()


def test_discarded_sessions_leave_no_names(tmp_path):
    from test_engine import CHROME, IDLE, DiscardingRecorder, Harness

    from recandcc.models import Mode

    class NamingRecorder(DiscardingRecorder):
        def stop(self):
            s = super().stop()
            voices.write_session_names(s, ["Anna"])  # what Recorder.stop() stores
            return s

    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = NamingRecorder(tmp_path / "days")
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.eng.tick()
    assert h.status == IDLE and h.rec.stops == 1
    assert voices.read_session_names("s") == []


def test_names_are_deleted_not_applied_when_the_option_was_switched_off(tmp_path):
    from recandcc.config import TranscriptionConfig
    from recandcc.transcribe.engine import Transcriber

    s = _session_with_voice(tmp_path, ["Anna"])
    t = Transcriber(
        TranscriptionConfig(model="fake", speaker_names=True, names_from_call_app=False)
    )
    t._name_from_call_app(s)
    assert voices.read_session_names(s.id) == []
    assert "Speaker 1" in (tmp_path / "transcript.json").read_text()  # not applied


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
    path = common.CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[detection]\nmode = "off"\n[transcription]\nlive = false\n', encoding="utf-8")
    cfg = config.load_config(path)
    cfg.transcription.speaker_names = kw.get("speaker_names", True)
    return Controller(cfg, monitor=Mon(), recorder=object(), tq=Q(), start_threads=False)


def settle():
    for t in threading.enumerate():
        if t.name in ("ax-trust", "ax-settings", "save-config"):
            t.join(5)


def saved():
    return tomllib.loads(common.CONFIG_PATH.read_text(encoding="utf-8"))["transcription"]


def test_toggle_checks_trust_off_the_main_thread_and_asks_before_opening_settings(monkeypatch):
    opened, threads = [], []
    monkeypatch.setattr(
        participants, "ax_trusted", lambda: threads.append(threading.current_thread()) or False
    )
    monkeypatch.setattr(participants, "open_accessibility_settings", lambda: opened.append(1))
    ctrl = make_ctrl()
    ctrl.toggle_call_app_names()
    settle()
    assert threads and threads[0] is not threading.main_thread()
    assert ctrl.events.get_nowait() == ("call_names_confirm",)
    assert opened == [] and not ctrl.cfg.transcription.names_from_call_app
    ctrl.confirm_call_app_names()  # the user said "Open Settings"
    settle()
    assert opened == [1] and ctrl.cfg.transcription.names_from_call_app
    # only its own key is written; the rest of the file is as the user had it
    assert saved() == {"live": False, "names_from_call_app": True}
    assert 'mode = "off"' in common.CONFIG_PATH.read_text(encoding="utf-8")
    ctrl.toggle_call_app_names()  # off: immediate
    settle()
    assert saved()["names_from_call_app"] is False


def test_toggle_when_trusted_needs_no_dialog_and_requires_speaker_names(monkeypatch):
    monkeypatch.setattr(participants, "ax_trusted", lambda: True)
    monkeypatch.setattr(participants, "open_accessibility_settings", lambda: pytest.fail("open"))
    ctrl = make_ctrl()
    ctrl.toggle_call_app_names()
    settle()
    assert ctrl.cfg.transcription.names_from_call_app and ctrl.events.empty()
    off = make_ctrl(speaker_names=False)
    off.toggle_call_app_names()
    settle()
    assert not off.cfg.transcription.names_from_call_app
    assert off.events.get_nowait()[0] == "notice"


def test_module_is_inert_off_macos(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert participants.ax_trusted() is False
    participants.open_accessibility_settings()  # no-op, no subprocess
    assert participants.enabled(on_cfg()) is False


def test_names_are_pruned_with_their_session_and_deleted_with_all_voices(tmp_path):
    day = tmp_path / "day"
    folder = day / "s"
    folder.mkdir(parents=True)
    s = make_session(folder)
    voices.write_session_names(s, ["Anna"])
    assert voices.prune_session_voices() == 0 and voices.read_session_names(s.id) == ["Anna"]
    import shutil

    shutil.rmtree(folder)
    assert voices.prune_session_voices() == 1 and voices.read_session_names(s.id) == []
    voices.write_session_names(s, ["Anna"])
    voices.delete_all_profiles()
    assert voices.read_session_names(s.id) == []
