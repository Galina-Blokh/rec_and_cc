"""Voice profiles: matching, naming a speaker once, the profile store and its opt-in, with
synthetic embeddings (no model).

Stories: US-21.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future

import numpy as np
import pytest
from test_transcribe import make_session

from recandcc import cli
from recandcc.config import Config, TranscriptionConfig
from recandcc.models import Segment
from recandcc.transcribe import voices
from recandcc.transcribe.diarize import Turn, Turns, label_speakers
from recandcc.transcribe.engine import Transcriber

RNG = np.random.default_rng(7)
ANNA, BOB, CARL = (v / np.linalg.norm(v) for v in RNG.normal(size=(3, 192)))


def near(v, noise=0.3):
    w = v + noise * RNG.normal(size=v.shape) / np.sqrt(v.size)
    return w / np.linalg.norm(w)


def profiles(**named):
    return [voices.Profile(name, c, 1) for name, c in named.items()]


def seg(start, end, speaker="Others", text="x"):
    return Segment(start, end, text, speaker, "en")


SEGS = [seg(1, 4, text="hello"), seg(2, 3, "Me", "yes"), seg(6, 9, text="hi"), seg(11, 14)]
TURNS = Turns(
    [Turn(0, 5, 3), Turn(5, 10, 8), Turn(10, 15, 3)], centroids={3: near(ANNA), 8: near(CARL)}
)


# --------------------------------------------------------------------------- matching
def test_match_needs_similarity_and_margin():
    known = profiles(Anna=ANNA, Bob=BOB)
    assert voices.match_profiles({0: near(ANNA), 1: near(BOB), 2: CARL}, known) == {
        0: "Anna",
        1: "Bob",
    }
    assert voices.match_profiles({0: near(ANNA)}, known, min_similarity=0.999) == {}
    twins = profiles(Anna=ANNA, Anya=near(ANNA, 0.05))  # two profiles of one voice: ambiguous
    assert voices.match_profiles({0: near(ANNA)}, twins) == {}


def test_each_name_is_used_once_per_session():
    close, closer = near(ANNA, 0.6), near(ANNA, 0.1)
    assert voices.match_profiles({0: close, 1: closer}, profiles(Anna=ANNA)) == {1: "Anna"}


def test_named_speakers_replace_numbers_and_me_is_untouched():
    out, labels = label_speakers(SEGS, TURNS, names={8: "Anna"})
    assert [s.speaker for s in out] == ["Speaker 1", "Me", "Anna", "Speaker 1"]
    assert labels == {3: "Speaker 1", 8: "Anna"}
    one = [Turn(0, 15, 3)]  # a single other speaker: named if known, else stays "Others"
    assert [s.speaker for s in label_speakers(SEGS, one, names={3: "Bob"})[0]][0] == "Bob"
    assert label_speakers(SEGS, one)[1] == {3: "Others"}


# --------------------------------------------------------------------------- store
def test_profiles_are_a_running_mean_in_a_private_folder(tmp_path):
    d = tmp_path / "voices"
    voices.save_profile("Анна", ANNA, d)
    p = voices.save_profile(" анна ", near(ANNA, 1.0), d)
    assert p.count == 2 and p.name == "анна"
    [loaded] = voices.load_profiles(d)
    assert loaded.count == 2 and np.allclose(loaded.centroid, p.centroid, atol=1e-5)
    assert float(loaded.centroid @ ANNA) > 0.8
    if os.name != "nt":
        assert d.stat().st_mode & 0o777 == 0o700
        assert all(f.stat().st_mode & 0o777 == 0o600 for f in d.iterdir())


def test_delete_one_and_all(tmp_path):
    for name, c in (("Anna", ANNA), ("Bob", BOB), ("Carl", CARL)):
        voices.save_profile(name, c, tmp_path)
    assert voices.delete_profile("anna", tmp_path) and not voices.delete_profile("Zed", tmp_path)
    assert [p.name for p in voices.load_profiles(tmp_path)] == ["Bob", "Carl"]
    assert voices.delete_all_profiles(tmp_path) == 2 and voices.load_profiles(tmp_path) == []


# --------------------------------------------------------------------------- sessions
def _final(tmp_path, monkeypatch, speaker_names: bool):
    s = make_session(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=speaker_names))
    done: Future = Future()
    done.set_result(TURNS)
    monkeypatch.setattr(t, "start_diarization", lambda _s: done)
    return t.write_final(s, SEGS, 15.0)


def test_opt_in_off_stores_nothing(tmp_path, monkeypatch):
    voices.save_profile("Anna", ANNA)
    out = _final(tmp_path, monkeypatch, speaker_names=False)
    assert [s.speaker for s in out] == ["Speaker 1", "Me", "Speaker 2", "Speaker 1"]
    assert not (tmp_path / voices.SPEAKERS_NAME).exists()


def test_known_voice_is_named_in_a_new_session(tmp_path, monkeypatch):
    voices.save_profile("Anna", ANNA)
    out = _final(tmp_path, monkeypatch, speaker_names=True)
    assert [s.speaker for s in out] == ["Anna", "Me", "Speaker 1", "Anna"]
    assert set(voices.read_session_speakers(tmp_path)) == {"Anna", "Speaker 1"}


def test_name_speaker_saves_profile_and_rewrites_all_outputs(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    assert voices.session_speakers(tmp_path) == [
        ("Speaker 1", ["hello", "x"]),
        ("Speaker 2", ["hi"]),
    ]
    p = voices.name_speaker(tmp_path, "Speaker 2", "Carl")
    assert p.count == 1 and float(p.centroid @ CARL) > 0.9
    assert "Carl:** hi" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "Carl: hi" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [x["speaker"] for x in data["segments"]] == ["Speaker 1", "Me", "Carl", "Speaker 1"]
    assert set(voices.read_session_speakers(tmp_path)) == {"Speaker 1", "Carl"}
    for label, name in (("Me", "Bob"), ("Speaker 1", "Carl"), ("Speaker 9", "Bob")):
        with pytest.raises(ValueError):
            voices.name_speaker(tmp_path, label, name)
    assert [p.name for p in voices.load_profiles()] == ["Carl"]


def test_cli_naming_needs_the_opt_in_and_voices_can_be_deleted(tmp_path, monkeypatch, capsys):
    _final(tmp_path, monkeypatch, speaker_names=True)
    cfg = Config(transcription=TranscriptionConfig(speaker_names=False))
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    assert cli.main(["name-speaker", str(tmp_path), "Speaker 1", "Bob"]) == 1
    assert voices.load_profiles() == []
    cfg.transcription.speaker_names = True
    assert cli.main(["name-speaker", str(tmp_path), "Speaker 1", "Bob"]) == 0
    assert cli.main(["speakers", str(tmp_path)]) == 0
    assert cli.main(["voices", "list"]) == 0
    assert "Bob  (1 session(s)" in capsys.readouterr().out
    assert cli.main(["voices", "delete", "Zed"]) == 1
    assert cli.main(["voices", "delete", "--all"]) == 0 and voices.load_profiles() == []
