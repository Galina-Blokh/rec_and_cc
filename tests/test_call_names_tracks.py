"""Call-app names in a room recording a call joined: each `Speaker N` label remembers the track
it came from, only the call's (system audio) side gets a participant name, a numbered owner on
the mic never does, and older transcripts without track info stay fail-closed. With fakes: no
models, no Accessibility.

Stories: US-35, US-31.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind
from recandcc.transcribe import callnames, voices
from recandcc.transcribe.diarize import Turn, Turns
from recandcc.transcribe.engine import Transcriber


class FakeDiarizer:
    """``Diarizer.submit`` stand-in: the turns for each WAV name, done right away."""

    def __init__(self, turns):
        self.turns = turns

    def submit(self, path, timing=None):
        from concurrent.futures import Future

        done: Future = Future()
        done.set_result(self.turns.get(path.name))
        return done


def _unit(rng, n):
    return [v / np.linalg.norm(v) for v in rng.normal(size=(n, 192))]


def _session(tmp_path, started_as_room=True):
    s = SessionInfo(
        "u1", tmp_path, datetime.now(), TriggerKind.CALL, started_as_room=started_as_room
    )
    for track in (Track.ME, Track.OTHERS):
        sf.write(tmp_path / f"{track}.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        s.tracks[track] = f"{track}.wav"
    s.save()
    return s


def _final(tmp_path, turns, names, started_as_room=True):
    """write_final of a room session a call upgraded: mic segments at 1-4 and 6-9, system
    audio at 2-3 and 11-14; the call app showed `names`."""
    s = _session(tmp_path, started_as_room)
    voices.write_session_names(s, names)
    cfg = TranscriptionConfig(model="fake", speaker_names=True, names_from_call_app=True)
    t = Transcriber(cfg)
    t._diarizer = FakeDiarizer(turns)  # pyright: ignore[reportAttributeAccessIssue]
    segs = [Segment(1, 4, "a", "Me", "en"), Segment(6, 9, "b", "Me", "en")]
    segs += [Segment(2, 3, "c", "Others", "en"), Segment(11, 14, "d", "Others", "en")]
    t.write_final(s, segs, 15.0)
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    return [(x["speaker"], x.get("track", "")) for x in data["segments"]]


def _transcript(tmp_path, rows, names, started_as_room=True):
    """A session folder with a transcript of (speaker, track) rows and kept call-app names."""
    s = _session(tmp_path, started_as_room)
    voices.write_session_names(s, names)
    segs = []
    for i, (speaker, track) in enumerate(rows):
        seg = {"start": i, "end": i + 1, "text": "x", "speaker": speaker, "language": "en"}
        segs.append(seg | ({"track": track} if track else {}))
    data = {"duration": len(rows), "model": "m", "segments": segs}
    (tmp_path / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    return s


def _speakers(tmp_path):
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    return [x["speaker"] for x in data["segments"]]


def test_room_session_upgraded_to_a_call_names_the_remote_speaker_never_the_owner(tmp_path):
    rng = np.random.default_rng(7)
    owner, carl, bob, anna = _unit(rng, 4)
    voices.save_profile("Bob", bob)
    turns = {
        # two people at the laptop: the owner is numbered "Speaker 1"
        "me.wav": Turns([Turn(0, 5, 0), Turn(5, 10, 1)], centroids={0: owner, 1: carl}),
        # the call: Bob (known voice) and one unknown
        "others.wav": Turns([Turn(0, 5, 0), Turn(10, 15, 1)], centroids={0: bob, 1: anna}),
    }
    out = _final(tmp_path, turns, ["Bob", "Anna"])
    assert out == [
        ("Speaker 1", "me"),
        ("Speaker 2", "me"),
        ("Bob", "others"),
        ("Anna", "others"),  # was "Speaker 3": the one unknown on the call's side
    ]
    assert voices.read_session_names("u1") == []  # used up
    assert voices.load_profiles()[0].name == "Bob" and len(voices.load_profiles()) == 1


def test_a_numbered_owner_is_never_counted_or_named(tmp_path):
    # Counting all labels, "Bob" plus the owner would match two names and the owner would be
    # renamed "Zed". On the call's track there is one speaker and two names: nothing is certain.
    _transcript(tmp_path, [("Speaker 1", "me"), ("Bob", "others")], ["Bob", "Zed"])
    assert callnames.plan(["Speaker 1", "Bob"], ["Bob", "Zed"]) == {"Speaker 1": "Zed"}
    assert callnames.apply(tmp_path) is False
    assert _speakers(tmp_path) == ["Speaker 1", "Bob"]
    # The one call-side unknown gets the one name; the room's numbered labels are not counted.
    _transcript(
        tmp_path, [("Speaker 1", "me"), ("Speaker 2", "me"), ("Speaker 3", "others")], ["Anna"]
    )
    assert callnames.apply(tmp_path) is True
    assert _speakers(tmp_path) == ["Speaker 1", "Speaker 2", "Anna"]


@pytest.mark.parametrize(
    "rows",
    [
        [("Speaker 1", ""), ("Speaker 2", "")],  # written before labels carried their track
        [("Speaker 1", "me"), ("Speaker 2", "")],  # partly unknown
        [("Speaker 1", "me"), ("Speaker 2", "others"), ("Speaker 2", "me")],  # mixed
    ],
)
def test_room_sessions_without_clear_track_info_stay_fail_closed(tmp_path, rows):
    s = _transcript(tmp_path, rows, ["Anna"])
    assert callnames.apply(tmp_path) is False
    assert "Anna" not in _speakers(tmp_path)
    assert voices.read_session_names(s.id) == []  # not kept either


def test_the_rename_itself_refuses_a_mic_track_label(tmp_path):
    s = _transcript(tmp_path, [("Speaker 1", "me"), ("Speaker 2", "others")], ["Anna"])
    since = voices.forget_generation()
    with voices.NAMING_LOCK, pytest.raises(ValueError, match="call's track"):
        callnames._rename(tmp_path, s, "Speaker 1", "Anna", since)
    assert _speakers(tmp_path) == ["Speaker 1", "Speaker 2"]


def test_a_forget_since_the_session_began_still_names_nobody(tmp_path):
    s = _transcript(tmp_path, [("Speaker 1", "me"), ("Speaker 2", "others")], ["Anna"])
    since = voices.forget_generation()
    voices.forget_all()
    voices.write_session_names(s, ["Anna"])
    assert callnames.apply(tmp_path, since=since) is False
    assert "Anna" not in _speakers(tmp_path)


def test_tracks_survive_naming_a_speaker_and_suggestions_ignore_the_room(tmp_path):
    rng = np.random.default_rng(8)
    owner, carl, x, y = _unit(rng, 4)
    turns = {
        "me.wav": Turns([Turn(0, 5, 0), Turn(5, 10, 1)], centroids={0: owner, 1: carl}),
        "others.wav": Turns([Turn(0, 5, 0), Turn(10, 15, 1)], centroids={0: x, 1: y}),
    }
    out = _final(tmp_path, turns, ["Anna", "Bob"])  # two unknowns on the call: nothing assigned
    assert [lbl for lbl, _ in out] == ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4"]
    voices.write_session_names(SessionInfo.load(tmp_path), ["Anna", "Carl"])
    voices.name_speaker(tmp_path, "Speaker 2", "Carl")  # the colleague in the room
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert {x["speaker"]: x["track"] for x in data["segments"]}["Carl"] == "me"
    # Carl is in the room, not on the call: his name does not use up a participant name
    assert callnames.suggestions(tmp_path) == ["Anna", "Carl"]


def test_a_plain_call_records_the_track_and_names_as_before(tmp_path):
    rng = np.random.default_rng(9)
    bob, anna = _unit(rng, 2)
    voices.save_profile("Bob", bob)
    turns = {"others.wav": Turns([Turn(0, 5, 0), Turn(10, 15, 1)], centroids={0: bob, 1: anna})}
    out = _final(tmp_path, turns, ["Bob", "Anna"], started_as_room=False)
    assert out == [("Me", ""), ("Me", ""), ("Bob", "others"), ("Anna", "others")]


def test_a_slow_diarization_writes_tracks_even_when_no_label_changes(tmp_path):
    from concurrent.futures import Future

    s = _session(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    segs = [Segment(1, 4, "a", "Me", "en"), Segment(2, 3, "c", "Others", "en")]
    t._write(s, segs, 15.0)  # written before the labels came
    done: Future = Future()
    # one person on each track: both keep their label, only the track is new
    done.set_result({Track.ME: [Turn(0, 5, 0)], Track.OTHERS: [Turn(0, 5, 0)]})
    t._relabel_later(s, segs, 15.0, done, voices.forget_generation())
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [(x["speaker"], x["track"]) for x in data["segments"]] == [
        ("Me", "me"),
        ("Others", "others"),
    ]


def test_suggestions_fail_closed_when_the_tracks_are_unknown(tmp_path):
    _transcript(tmp_path, [("Speaker 1", ""), ("Speaker 2", "")], ["Anna"])
    assert callnames.suggestions(tmp_path) == []  # never the room's (or the owner's) labels
    _transcript(tmp_path, [("Speaker 1", "me"), ("Speaker 2", "others")], ["Anna", "Bob"])
    assert callnames.suggestions(tmp_path) == ["Anna", "Bob"]


def test_a_failed_voice_store_write_puts_the_old_label_back(tmp_path, monkeypatch):
    s = _transcript(tmp_path, [("Speaker 1", "me"), ("Speaker 2", "others")], ["Anna"])
    voices.write_session_voices(s, {"Speaker 2": np.eye(192)[0]})

    def broken(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(voices, "write_session_voices", broken)
    assert callnames.apply(tmp_path) is False
    assert _speakers(tmp_path) == ["Speaker 1", "Speaker 2"]  # transcript and store agree
    assert list(voices.read_session_voices(s.id)) == ["Speaker 2"]
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [x["track"] for x in data["segments"]] == ["me", "others"]  # tracks kept too
