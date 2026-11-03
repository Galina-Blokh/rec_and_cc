"""Final names from explicit introductions; summaries wait for named transcript files.

Stories: US-21, US-27.
"""

import json
from concurrent.futures import Future
from datetime import datetime

import numpy as np
import pytest

from recandcc.config import Config, TranscriptionConfig, load_config
from recandcc.models import Segment, SessionInfo, Track, TriggerKind
from recandcc.transcribe import names, summary, voices
from recandcc.transcribe.diarize import Turn
from recandcc.transcribe.engine import Transcriber
from recandcc.transcribe.intro_hints import confident_names
from recandcc.transcribe.output import write_outputs
from recandcc.ui.common import Controller


def segment(text, speaker="Speaker 1", start=0):
    return Segment(start, start + 1, text, speaker, "en")


def session(tmp_path, segments):
    info = SessionInfo("names", tmp_path, datetime(2026, 11, 3, 10), TriggerKind.CALL, "test")
    info.save()
    write_outputs(info, segments, 10.0, "fake")
    return info


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My name is Sebastian.", "Sebastian"),
        ("Hello, I'm Anna from sales.", "Anna"),
        ("Меня зовут Анна.", "Анна"),
        ("שלום, קוראים לי דנה.", "דנה"),
        ('He said, "My name is Bob."', None),
        ("My name is Anna. My name is Bob.", None),
        ("I'm Sure.", None),
    ],
)
def test_only_direct_unambiguous_introductions_are_assigned(text, expected):
    assert confident_names([segment(text)]) == ({"Speaker 1": expected} if expected else {})


def test_conflicts_collisions_existing_names_and_undifferentiated_tracks_stay_unchanged():
    anna = segment("My name is Anna.")
    for other in (
        segment("My name is Olga.", start=300),
        segment("My name is Anna.", "Speaker 2"),
        segment("My name is Anna.", "Me"),
        segment("Hello.", "Anna"),
    ):
        assert confident_names([anna, other]) == {}
    assert confident_names([segment("My name is Anna.", "Others")], single_labels={"Me"}) == {}
    assert confident_names([segment("My name is Gali.", "Me")]) == {}
    assert confident_names([segment("My name is Gali.", "Me")], single_labels={"Me"}) == {}
    assert confident_names([segment("Hello."), segment("My name is Anna.", start=121)]) == {}
    assert (
        confident_names(
            [anna, segment("Hello.", "Speaker 2"), segment("My name is Anna.", "Speaker 2", 300)]
        )
        == {}
    )


def test_all_formats_are_named_without_reading_or_learning_voice_profiles(tmp_path, monkeypatch):
    session(tmp_path, [segment("My name is Sebastian."), segment("I will send it.", start=2)])
    monkeypatch.setattr(voices, "read_session_voices", lambda *_a: pytest.fail("voices are off"))
    assert names.apply_introductions(tmp_path) == {"Speaker 1": "Sebastian"}
    for suffix in ("json", "md", "srt"):
        text = (tmp_path / f"transcript.{suffix}").read_text("utf-8")
        assert "Sebastian" in text and "Speaker 1" not in text
    assert voices.load_profiles() == []


def test_session_voice_keys_follow_names_but_no_voice_profile_is_learned(tmp_path):
    info = session(tmp_path, [segment("My name is Sebastian.")])
    voices.write_session_voices(info, {"Speaker 1": np.ones(192)})
    names.apply_introductions(tmp_path, sync_voices=True)
    stored = voices.read_session_voices(info.id)
    assert list(stored) == ["Sebastian"] and stored["Sebastian"]["profile"] is None
    assert voices.load_profiles() == []


def test_failed_voice_write_restores_every_transcript_format(tmp_path, monkeypatch):
    info = session(tmp_path, [segment("My name is Sebastian.")])
    voices.write_session_voices(info, {"Speaker 1": np.ones(192)})
    before = {p.name: p.read_bytes() for p in tmp_path.glob("transcript.*")}

    def fail(*args, **kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(voices, "write_session_voices", fail)
    with pytest.raises(OSError):
        names.apply_introductions(tmp_path, sync_voices=True)
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob("transcript.*")}


@pytest.mark.parametrize("slow", [False, True])
def test_automatic_summary_sees_saved_names_after_fast_or_slow_diarization(
    tmp_path, monkeypatch, slow
):
    cfg = Config(output_dir=tmp_path)
    cfg.transcription.model = "fake"
    cfg.summary.enabled = True
    transcriber = Transcriber(cfg.transcription)
    transcriber.diarize_wait_s = 0
    info = session(tmp_path, [segment("My name is Sebastian.", "Others")])
    future = Future()
    if not slow:
        future.set_result([Turn(0, 1, 1)])
    transcriber.write_final(info, [segment("My name is Sebastian.", "Others")], 10.0, future)
    observed = []

    class Generator:
        def __call__(self, messages, max_tokens):
            data = json.loads((tmp_path / "transcript.json").read_text("utf-8"))
            observed.append(data["segments"][0]["speaker"])
            return "- Sebastian: Send the report."

        def close(self):
            pass

    monkeypatch.setattr(summary, "create_generator", lambda *a, **k: Generator())
    ctrl = Controller(
        cfg,
        monitor=object(),
        recorder=object(),
        tq=object(),
        transcriber=transcriber,
        start_threads=False,
    )
    monkeypatch.setattr(ctrl, "refresh_async", lambda **_k: None)
    monkeypatch.setattr(ctrl, "_index_session", lambda _d: None)
    ctrl._on_transcribed(info, [])
    if slow:
        assert not (tmp_path / "summary.md").exists()
        future.set_result([Turn(0, 1, 1)])
    ctrl._summaries.join()
    assert observed and set(observed) == {"Sebastian"}
    assert "Sebastian" in (tmp_path / "summary.md").read_text("utf-8")


def test_naming_can_be_disabled_and_invalid_setting_fails_closed(tmp_path):
    cfg = TranscriptionConfig(model="fake", names_from_transcript=False)
    info = session(tmp_path, [segment("My name is Sebastian.")])
    t = Transcriber(cfg)
    assert t._name_from_transcript(info) == {}
    path = tmp_path / "config.toml"
    path.write_text('[transcription]\nnames_from_transcript = "false"\n')
    assert load_config(path).transcription.names_from_transcript is False


@pytest.mark.parametrize("room", [False, True])
def test_multiple_speakers_get_distinct_names_on_call_or_room_track(tmp_path, room):
    label = "Me" if room else "Others"
    segments = [segment("My name is Sebastian.", label), segment("I'm Gali.", label, 2)]
    info = session(tmp_path, segments)
    info.started_as_room = room
    info.save()
    future = Future()
    turns = [Turn(0, 1, 1), Turn(2, 3, 2)]
    future.set_result({Track.ME: turns} if room else turns)
    result = Transcriber(TranscriptionConfig(model="fake")).write_final(info, segments, 3, future)
    assert [s.speaker for s in result] == ["Sebastian", "Gali"]


@pytest.mark.parametrize("turns", [None, [], [Turn(0, 1, 1, overlap=True)]])
def test_unseparated_remote_track_keeps_its_label(tmp_path, turns):
    segments = [segment("My name is Sebastian.", "Others")]
    info = session(tmp_path, segments)
    future = Future()
    future.set_result(turns)
    result = Transcriber(TranscriptionConfig(model="fake")).write_final(info, segments, 1, future)
    assert result[0].speaker == "Others"


@pytest.mark.parametrize("room", [False, True])
def test_owner_label_stays_me_for_summary_ownership(tmp_path, room):
    segments = [segment("My name is Gali.", "Me")]
    info = session(tmp_path, segments)
    info.trigger_kind = TriggerKind.ROOM if room else TriggerKind.CALL
    info.save()
    future = Future()
    future.set_result([Turn(0, 1, 1)])
    result = Transcriber(TranscriptionConfig(model="fake")).write_final(info, segments, 1, future)
    assert result[0].speaker == "Me"
