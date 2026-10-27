"""Menu access to speaker names: the Controller's worker functions with fakes (no AppKit).

Stories: US-26.
"""

from __future__ import annotations

import json
import logging
import queue
import time

from test_voices import ANNA, BOB, CARL, _final

from recandcc.config import Config, TranscriptionConfig
from recandcc.transcribe import voices
from recandcc.ui.common import Controller
from recandcc.ui.present import (
    forgot_notice,
    named_notice,
    speaker_choice_label,
    speakers_notice,
)


def controller(speaker_names: bool = True) -> Controller:
    ctrl = Controller.__new__(Controller)
    ctrl.events = queue.Queue()
    ctrl.cfg = Config(transcription=TranscriptionConfig(speaker_names=speaker_names))
    return ctrl


def test_menu_naming_updates_profile_and_transcript_like_the_cli(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    ctrl.speakers_async(tmp_path)
    kind, directory, speakers = ctrl.events.get(timeout=5)
    assert (kind, directory) == ("speakers", tmp_path)
    assert [label for label, _ in speakers] == ["Speaker 1", "Speaker 2"]
    assert speakers_notice(speakers) is None and "hi" in speaker_choice_label(2, *speakers[1])
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert ctrl.events.get(timeout=5) == ("named", "Speaker 2", "Carl", None)
    assert "Carl:** hi" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert "Carl" in {s["speaker"] for s in data["segments"]}
    (profile,) = voices.load_profiles()
    assert profile.name == "Carl" and float(profile.centroid @ CARL) > 0.9
    assert named_notice(None)[0] == "Speaker named"


def test_naming_is_refused_when_speaker_names_is_off(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller(speaker_names=False)
    assert not ctrl.speaker_names_enabled
    ctrl.name_speaker_async(tmp_path, "Speaker 1", "Bob")
    _, _, _, error = ctrl.events.get(timeout=5)
    assert error and voices.load_profiles() == []
    assert "Speaker 1:** hello" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert named_notice(error)[0] == "Can't name the speaker"


def test_session_without_voice_data_has_no_speakers_to_name(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=False)
    ctrl = controller()
    ctrl.speakers_async(tmp_path)
    _, _, speakers = ctrl.events.get(timeout=5)
    assert speakers == [] and speakers_notice(speakers)[0] == "No speakers to name"


def test_forget_all_voices_removes_profiles_and_session_voices(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    voices.save_profile("Anna", ANNA)
    voices.save_profile("Bob", BOB)
    ctrl = controller()
    ctrl.forget_voices_async()
    kind, count = ctrl.events.get(timeout=5)
    assert (kind, count) == ("forgot", 2)
    assert voices.load_profiles() == [] and voices.read_session_voices("s1") == {}
    assert "2 voice profile" in forgot_notice(count)[1]
    assert forgot_notice(OSError())[0] == "Can't delete voices"


def test_names_are_never_logged(tmp_path, monkeypatch, caplog):
    _final(tmp_path, monkeypatch, speaker_names=True)
    caplog.set_level(logging.INFO)
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Zebulon")
    ctrl.events.get(timeout=5)
    ctrl.name_speaker_async(tmp_path, "Speaker 9", "Zebulon")  # fails
    ctrl.events.get(timeout=5)
    ctrl.forget_voices_async()
    ctrl.events.get(timeout=5)
    assert "Zebulon" not in caplog.text and "hello" not in caplog.text


def test_unexpected_error_still_posts_a_name_free_outcome(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    monkeypatch.setattr(voices, "name_speaker", lambda *a: (_ for _ in ()).throw(TypeError("x")))
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Zebulon")
    _, _, _, error = ctrl.events.get(timeout=5)
    assert error and "Zebulon" not in error and "Speaker 2" not in error
    assert "Zebulon" not in " ".join(named_notice(None) + named_notice(error))


def test_forget_waits_for_a_running_naming(tmp_path, monkeypatch):
    import threading

    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    started, release = threading.Event(), threading.Event()
    real = voices.name_speaker

    def slow(*args):
        with voices.LOCK:  # as the real one does
            started.set()
            release.wait(5)
            return real(*args)

    monkeypatch.setattr(voices, "name_speaker", slow)
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert started.wait(5)
    ctrl.forget_voices_async()
    time.sleep(0.2)
    assert ctrl.events.empty()  # forget is waiting for the naming
    release.set()
    assert ctrl.events.get(timeout=5)[0] == "named"
    assert ctrl.events.get(timeout=5) == ("forgot", 1)
    assert voices.load_profiles() == []  # the forget came last: nothing came back


def test_choice_label_flattens_multiline_hints():
    label = speaker_choice_label(1, "Speaker 1", ["hello\nthere", "a\r\n b"])
    assert "\n" not in label and "\r" not in label


def test_two_namings_of_one_session_both_survive(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 1", "Anna")
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert [ctrl.events.get(timeout=5)[3] for _ in range(2)] == [None, None]
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert {"Anna", "Carl"} <= {s["speaker"] for s in data["segments"]}


def test_diarization_started_before_a_forget_cannot_bring_voices_back(tmp_path, monkeypatch):
    from test_transcribe import make_session

    session = make_session(tmp_path)
    started = voices.forget_generation()
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=started)
    assert voices.read_session_voices(session.id)
    voices.delete_all_profiles()
    assert voices.read_session_voices(session.id) == {}
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=started)  # began before
    assert voices.read_session_voices(session.id) == {}
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=voices.forget_generation())
    assert voices.read_session_voices(session.id)


def test_atomic_write_survives_concurrent_writers_and_a_busy_file(tmp_path, monkeypatch):
    import threading
    from pathlib import Path

    from recandcc.transcribe.output import atomic_write

    target = tmp_path / "t.md"
    errors = []

    def write(i):
        try:
            for _ in range(20):
                atomic_write(target, f"text {i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and target.read_text(encoding="utf-8").startswith("text ")
    assert [p.name for p in tmp_path.iterdir()] == ["t.md"]  # no temp left behind

    real, calls = Path.replace, []

    def busy(self, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held")
        return real(self, dst)

    monkeypatch.setattr(Path, "replace", busy)
    atomic_write(target, "after retries")
    assert target.read_text(encoding="utf-8") == "after retries" and len(calls) == 3
