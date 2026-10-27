"""Voice profiles: matching, naming a speaker once, the profile store and its opt-in, with
synthetic embeddings (no model).

Stories: US-21.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from test_transcribe import make_session

from recandcc import cli
from recandcc.config import Config, TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment
from recandcc.transcribe import voices
from recandcc.transcribe.diarize import Diarizer, Turn, Turns, label_speakers
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


def at_similarity(v, sim):
    """A unit vector at exactly cosine `sim` to `v` (and almost orthogonal to the others)."""
    w = np.random.default_rng(int(sim * 1000)).normal(size=v.shape)  # independent of test order
    w -= (w @ v) * v
    return sim * v + np.sqrt(1 - sim**2) * w / np.linalg.norm(w)


def test_a_similar_stranger_is_not_named():
    # The synthetic recall eval (docs/EVALUATION.md): a never-named voice reached 0.68 to a
    # profile with a 0.16 margin and took its name under the old 0.6 threshold.
    known = profiles(Anna=ANNA, Bob=BOB)
    assert voices.MATCH_SIMILARITY >= 0.7
    assert voices.match_profiles({0: at_similarity(ANNA, 0.68)}, known) == {}
    assert voices.match_profiles({0: at_similarity(ANNA, 0.9)}, known) == {0: "Anna"}


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


def test_numbers_skip_labels_taken_by_names():
    turns = [Turn(0, 5, 1), Turn(5, 10, 2), Turn(10, 15, 3)]
    segs = [seg(1, 4), seg(6, 9), seg(11, 14)]
    out, labels = label_speakers(segs, turns, names={1: "Speaker 2"})
    assert [s.speaker for s in out] == ["Speaker 2", "Speaker 1", "Speaker 3"]
    assert len(set(labels.values())) == 3


def test_names_are_checked_and_normalised(tmp_path):
    for bad in ("", " ", "Speaker 2", "speaker  3", "ME", "others", "A\nB", "a\x07", "**x**"):
        with pytest.raises(ValueError):
            voices.check_name(bad)
    nfc, nfd = unicodedata.normalize("NFC", "Йоси"), unicodedata.normalize("NFD", "Йоси")
    assert nfc != nfd
    voices.save_profile(nfc, ANNA, tmp_path)
    assert voices.save_profile(nfd, ANNA, tmp_path).count == 2
    assert [p.name for p in voices.load_profiles(tmp_path)] == [nfc]


# --------------------------------------------------------------------------- store
def test_profiles_are_a_running_mean_in_a_private_folder(tmp_path):
    d = tmp_path / "voices"
    d.mkdir()
    other = {"name": "Old", "model": "another.onnx", "count": 1, "updated": "", "centroid": [1]}
    (d / "old.json").write_text(json.dumps(other))
    voices.save_profile("Анна", ANNA, d)
    p = voices.save_profile(" анна ", near(ANNA, 1.0), d)
    assert p.count == 2 and p.name == "анна"
    [loaded] = voices.load_profiles(d)  # another embedding model's profile is skipped
    assert loaded.count == 2 and np.allclose(loaded.centroid, p.centroid, atol=1e-5)
    assert float(loaded.centroid @ ANNA) > 0.8
    if os.name != "nt":
        (d / "old.json").unlink()
        [f] = d.glob("*.json")
        # left over from a crash, world-readable
        stale = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        stale.write_text("{}")
        stale.chmod(0o644)
        voices.save_profile("Анна", ANNA, d)
        assert d.stat().st_mode & 0o777 == 0o700
        assert f.stat().st_mode & 0o777 == 0o600 and not stale.exists()


def test_delete_one_and_all(tmp_path):
    for name, c in (("Anna", ANNA), ("Bob", BOB), ("Carl", CARL)):
        voices.save_profile(name, c, tmp_path)
    (tmp_path / "rec").mkdir()
    voices.write_session_voices(make_session(tmp_path / "rec"), {"Speaker 1": BOB}, None, tmp_path)
    if os.name != "nt":
        assert (tmp_path / "sessions").stat().st_mode & 0o777 == 0o700
    assert voices.delete_profile("anna", tmp_path) and not voices.delete_profile("Zed", tmp_path)
    assert [p.name for p in voices.load_profiles(tmp_path)] == ["Bob", "Carl"]
    assert voices.delete_all_profiles(tmp_path) == 2 and voices.load_profiles(tmp_path) == []
    assert voices.read_session_voices("s1", tmp_path) == {}


# --------------------------------------------------------------------------- sessions
TRANSCRIPT_FILES = {"session.json", "transcript.md", "transcript.json", "transcript.srt"}


def _final(tmp_path, monkeypatch, speaker_names: bool, turns=TURNS):
    s = make_session(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=speaker_names))
    done: Future = Future()
    done.set_result(turns)
    monkeypatch.setattr(t, "start_diarization", lambda _s: done)
    return t.write_final(s, SEGS, 15.0)


def test_opt_in_off_stores_nothing(tmp_path, monkeypatch):
    voices.save_profile("Anna", ANNA)
    _final(tmp_path, monkeypatch, speaker_names=True)
    assert voices.read_session_voices("s1")
    out = _final(tmp_path, monkeypatch, speaker_names=False)  # turned off: old voices go too
    assert [s.speaker for s in out] == ["Speaker 1", "Me", "Speaker 2", "Speaker 1"]
    assert voices.read_session_voices("s1") == {}
    assert {f.name for f in tmp_path.iterdir()} == TRANSCRIPT_FILES


def test_known_voice_is_named_and_voices_stay_out_of_output_dir(tmp_path, monkeypatch):
    voices.save_profile("Anna", ANNA)
    out = _final(tmp_path, monkeypatch, speaker_names=True)
    assert [s.speaker for s in out] == ["Anna", "Me", "Speaker 1", "Anna"]
    assert set(voices.read_session_voices("s1")) == {"Anna", "Speaker 1"}
    assert {f.name for f in tmp_path.iterdir()} == TRANSCRIPT_FILES
    _final(tmp_path, monkeypatch, speaker_names=True, turns=None)  # not diarized this time
    assert voices.read_session_voices("s1") == {}


def test_name_speaker_saves_profile_and_rewrites_all_outputs(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    assert voices.session_speakers(tmp_path) == [
        ("Speaker 1", ["hello", "x"]),
        ("Speaker 2", ["hi"]),
    ]
    p = voices.name_speaker(tmp_path, "speaker 2", "Carl")
    assert p.count == 1 and float(p.centroid @ CARL) > 0.9
    assert "Carl:** hi" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "Carl: hi" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [x["speaker"] for x in data["segments"]] == ["Speaker 1", "Me", "Carl", "Speaker 1"]
    assert set(voices.read_session_voices("s1")) == {"Speaker 1", "Carl"}
    assert voices.name_speaker(tmp_path, "carl", "Carl").count == 1  # same session: not re-added
    for label, name in (("Me", "Bob"), ("Speaker 1", "carl"), ("Speaker 9", "Bob")):
        with pytest.raises(ValueError):
            voices.name_speaker(tmp_path, label, name)
    stale = make_session(tmp_path)  # voices not in the transcript
    voices.write_session_voices(stale, {"Speaker 7": BOB})
    with pytest.raises(ValueError):
        voices.name_speaker(tmp_path, "Speaker 7", "Bob")
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
    assert voices.read_session_voices("s1") == {}


def test_a_failure_partway_never_counts_a_session_twice(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)

    def boom(*_a, **_k):
        raise OSError("disk full")

    for step in ("write_outputs", "save_profile"):
        with monkeypatch.context() as m:
            m.setattr(voices, step, boom)
            with pytest.raises(OSError):
                voices.name_speaker(tmp_path, "Speaker 2", "Carl")
        assert voices.name_speaker(tmp_path, "Speaker 2", "Carl").count == 1  # the retry
        assert voices.name_speaker(tmp_path, "Carl", "Carl").count == 1
        voices.delete_profile("Carl")
        _final(tmp_path, monkeypatch, speaker_names=True)
    assert "Carl" not in (tmp_path / "transcript.md").read_text(encoding="utf-8")


def test_windows_replace_retries_when_the_file_is_held_open(tmp_path, monkeypatch):
    real, calls = os.replace, []

    def held(src, dst):
        calls.append(dst)
        if len(calls) < 3:
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(voices.os, "replace", held)
    monkeypatch.setattr(voices.time, "sleep", lambda _s: None)
    assert voices.save_profile("Anna", ANNA, tmp_path).count == 1 and len(calls) == 3
    calls.clear()
    monkeypatch.setattr(voices, "REPLACE_TRIES", 2)
    with pytest.raises(PermissionError):
        voices.save_profile("Anna", ANNA, tmp_path)
    assert [f.name for f in tmp_path.iterdir()] == [calls[0].name]  # no temp file left
    assert voices.load_profiles(tmp_path)[0].count == 1


def test_correcting_a_name_moves_the_voice_to_the_new_profile(tmp_path, monkeypatch):
    voices.save_profile("Anna", BOB)  # an earlier session's voice
    _final(tmp_path, monkeypatch, speaker_names=True)
    assert voices.name_speaker(tmp_path, "Speaker 2", "Anna").count == 2
    assert voices.name_speaker(tmp_path, "anna", "Bob").count == 1
    [anna, bob] = voices.load_profiles()
    assert anna.count == 1 and float(anna.centroid @ BOB) > 0.999
    assert bob.name == "Bob" and float(bob.centroid @ CARL) > 0.9
    voices.name_speaker(tmp_path, "Bob", "Carl")  # Bob had only this session: deleted
    assert [p.name for p in voices.load_profiles()] == ["Anna", "Carl"]
    old = {"name": "Old", "model": voices.MODEL, "count": 2, "updated": "", "centroid": list(BOB)}
    (voices.VOICES_DIR / "old.json").write_text(json.dumps(old))  # stored before sums
    p = voices.save_profile("Old", BOB)
    assert p.count == 3 and float(p.centroid @ BOB) > 0.999


def test_deleting_a_name_strips_it_from_session_voices(tmp_path, monkeypatch, capsys):
    _final(tmp_path, monkeypatch, speaker_names=True)
    voices.name_speaker(tmp_path, "Speaker 2", "Carl")
    assert cli.main(["voices", "delete", "carl"]) == 0
    assert set(voices.read_session_voices("s1")) == {"Speaker 1"}
    assert voices.load_profiles() == []
    with pytest.raises(ValueError):  # the voice can't come back from this session
        voices.name_speaker(tmp_path, "Carl", "Carl")
    assert cli.main(["voices", "delete", "carl"]) == 1


def _sessions(*ids):
    return sorted(voices._session_path(i, None).name for i in ids)


def test_voices_of_deleted_sessions_are_pruned(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    gone, kept, other_day = out / "d1" / "gone", out / "d1" / "kept", out / "d2" / "s3"
    for d in (gone, kept, other_day):
        d.mkdir(parents=True)
    s = make_session(gone)
    voices.write_session_voices(replace(s, id="s2"), {"Speaker 1": BOB})
    voices.write_session_voices(replace(s, id="s3", directory=other_day), {"Speaker 1": BOB})
    sessions = voices.VOICES_DIR / "sessions"
    (sessions / "legacy.json").write_text(json.dumps({"session_id": "s0", "speakers": {}}))
    (sessions / "broken.json").write_text("{")
    shutil.rmtree(gone)
    _final(kept, monkeypatch, speaker_names=True)  # a transcript write prunes
    legacy_and_broken = ["broken.json", "legacy.json"]  # no folder known / unreadable: kept
    assert sorted(f.name for f in sessions.iterdir()) == sorted(
        [*_sessions("s1", "s3"), *legacy_and_broken]
    )
    shutil.rmtree(other_day.parent)  # the whole day folder: output_dir shows it is gone
    assert cli.main(["voices", "list"]) == 0  # every voices command prunes too
    assert sorted(f.name for f in sessions.iterdir()) == sorted(
        [*_sessions("s1"), *legacy_and_broken]
    )


def test_an_unreachable_output_dir_prunes_nothing(tmp_path, monkeypatch):
    out = tmp_path / "drive" / "Transcribes"
    session = out / "2026-10-27" / "rec"
    session.mkdir(parents=True)
    _final(session, monkeypatch, speaker_names=True)
    shutil.rmtree(tmp_path / "drive")  # like an unmounted drive or an offline share
    assert voices.prune_session_voices() == 0 and voices.read_session_voices("s1")
    if os.name != "nt":
        session.mkdir(parents=True)
        _final(session, monkeypatch, speaker_names=True)
        shutil.rmtree(session)
        session.parent.chmod(0)  # can't be listed: not taken for a deletion
        try:
            assert voices.prune_session_voices() == 0
        finally:
            session.parent.chmod(0o700)
        assert voices.prune_session_voices() == 1


def test_profile_shapes_never_mix_and_a_crash_leaves_no_voice_data(tmp_path):
    voices.save_profile("Anna", ANNA, tmp_path)
    with pytest.raises(ValueError):
        voices._fold("Anna", ANNA[:10], -1, tmp_path)
    assert voices.load_profiles(tmp_path)[0].count == 1
    assert voices._fold("Zed", ANNA, -1, tmp_path) is None  # nothing to take out of
    (tmp_path / "sessions").mkdir()
    for stale in (tmp_path / ".x.json.1.2.tmp", tmp_path / "sessions" / ".y.json.1.2.tmp"):
        stale.write_text("[0.1]")
    assert voices.delete_all_profiles(tmp_path) == 1
    assert not list(tmp_path.rglob("*.*"))


def test_real_diarization_attaches_speaker_centroids():
    class Stream:
        def accept_waveform(self, _rate, samples):
            self.voice = ANNA if samples.mean() > 0 else CARL

        def input_finished(self):
            pass

    class Extractor:
        def create_stream(self):
            return Stream()

        def compute(self, stream):
            return list(near(stream.voice, 0.1) * 3)  # not unit length, like the model

    d = Diarizer(threads=2, vad=lambda _audio: [(0.0, 8.0), (8.0, 16.0)])
    d._extractor, d._pool = Extractor(), ThreadPoolExecutor(2)  # as if the model loaded
    audio = np.concatenate([np.ones(8 * SAMPLE_RATE), -np.ones(8 * SAMPLE_RATE)])
    turns = d.diarize(audio.astype(np.float32))
    d.close()
    assert isinstance(turns, Turns) and len(turns.centroids) == 2
    first, last = turns.centroids[turns[0].speaker], turns.centroids[turns[-1].speaker]
    assert abs(float(np.linalg.norm(first)) - 1) < 1e-5
    assert float(first @ ANNA) > 0.95 and float(last @ CARL) > 0.95
    assert voices.match_profiles(turns.centroids, profiles(Anna=ANNA)) == {turns[0].speaker: "Anna"}
