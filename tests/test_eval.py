"""scripts/eval.py: metrics on hand-made references, and a run with a fake pipeline.

Stories: none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.models import SAMPLE_RATE, Segment
from recandcc.transcribe.diarize import Turn

_spec = importlib.util.spec_from_file_location(
    "rcc_eval", Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
)
ev = importlib.util.module_from_spec(_spec)
sys.modules["rcc_eval"] = ev  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(ev)


def seg(start, end, text, lang="en", speaker="Others"):
    return Segment(start=start, end=end, text=text, speaker=speaker, language=lang)


def utt(start, end, text, lang="en", speaker="A"):
    return ev.Utterance(start, end, speaker, lang, text)


def test_normalize_case_punctuation_and_yo():
    assert ev.normalize("Hello, World! It's 5 o'clock.") == [
        "hello",
        "world",
        "it's",
        "5",
        "o'clock",
    ]
    assert ev.normalize("Ещё — раз…") == ["еще", "раз"]


@pytest.mark.parametrize(
    ("ref", "hyp", "edits"),
    [
        ("a b c", "a b c", 0),
        ("a b c", "a x c", 1),
        ("a b c", "a c", 1),
        ("a b", "a b c d", 2),
        ("a b", "", 2),
    ],
)
def test_word_edits(ref, hyp, edits):
    assert ev.word_edits(ref.split(), hyp.split()) == edits


def test_text_scores_per_language_and_lid():
    utts = [utt(0, 2, "hello there world"), utt(3, 5, "привет как дела", "ru")]
    segs = [
        seg(0.1, 1.9, "Hello there, word."),  # 1 substitution
        seg(3.0, 4.0, "привет как", "en"),  # tagged wrong, 1 deletion
        seg(4.0, 4.5, "дела", "ru"),  # shorter than the "en" part: LID still wrong
    ]
    s = ev.text_scores(utts, segs)
    assert s["en"] == {"words": 3, "edits": 1, "utts": 1, "lid_ok": 1}
    assert s["ru"] == {"words": 3, "edits": 0, "utts": 1, "lid_ok": 0}


def test_unmatched_segment_goes_to_nearest_utterance_without_language_vote():
    utts = [utt(0, 1, "one"), utt(10, 11, "two")]
    (w0, v0), (w1, v1) = ev.assign_words(utts, [seg(8, 9, "two", "ru")])
    assert w0 == [] and w1 == ["two"] and v1 == {}


def test_unmatched_segment_goes_to_closest_interval_not_midpoint():
    utts = [utt(0, 20, "long one"), utt(30, 31, "short")]
    (w0, _), (w1, _) = ev.assign_words(utts, [seg(21, 22, "extra")])
    assert w0 == ["extra"] and w1 == []  # 1 s after the long one, 8 s before the short one


def test_segment_spanning_utterances_is_split():
    utts = [utt(0, 1, "hello"), utt(1, 2, "world")]
    s = ev.text_scores(utts, [seg(0, 2, "Hello world")])
    assert s["en"] == {"words": 2, "edits": 0, "utts": 2, "lid_ok": 2}


def test_segment_over_overlapping_utterances_splits_words_and_votes():
    utts = [utt(0, 2, "one two", "en", "A"), utt(1, 3, "три четыре", "ru", "B")]
    (w0, v0), (w1, v1) = ev.assign_words(utts, [seg(0, 3, "one two три четыре", "en")])
    assert w0 == ["one", "two"] and w1 == ["три", "четыре"]  # equal overlap: half each
    assert v0 == {"en": 2.0} and v1 == {"en": 2.0}


def test_language_vote_weighs_overlap_not_segment_length():
    utts = [utt(0, 10, "a b c d e f g h i j", "en"), utt(10, 12, "к л", "ru")]
    segs = [seg(0, 10.5, "a b c d e f g h i j", "en"), seg(10.5, 12, "к л", "ru")]
    s = ev.text_scores(utts, segs)  # the long en segment overlaps the ru utterance by 0.5 s only
    assert s["ru"]["lid_ok"] == 1 and s["en"]["lid_ok"] == 1


def test_missing_hypothesis_counts_all_words_and_no_lid():
    s = ev.text_scores([utt(0, 1, "a b c")], [])
    assert s["en"] == {"words": 3, "edits": 3, "utts": 1, "lid_ok": 0}


def test_der_perfect_with_renamed_speakers():
    ref = [(0, 2, "A"), (2, 4, "B")]
    assert ev.der(ref, [(0, 2, "1"), (2, 4, "0")])["der"] == 0.0


def test_der_miss_false_alarm_confusion():
    ref = [(0, 2, "A"), (2, 4, "B")]
    hyp = [(0, 1, "x"), (2, 5, "x")]  # A misses 1 s; B -> x confused (2 s); 1 s false alarm
    d = ev.der(ref, hyp)
    assert d["miss_s"] == pytest.approx(1.0)
    assert d["false_alarm_s"] == pytest.approx(1.0)
    assert d["confusion_s"] == pytest.approx(1.0)  # x maps to B (2 s); A's 1 s is confused
    assert d["der"] == pytest.approx(3.0 / 4.0)


def test_der_many_speakers_is_exact():
    ref = [(i, i + 1, f"R{i}") for i in range(7)]
    hyp = [(i, i + 1, f"H{i}") for i in range(7)] + [(7 + i, 7.5 + i, f"X{i}") for i in range(13)]
    d = ev.der(ref, hyp)  # 7 x 20: permutations would be ~390 million, the DP is instant
    assert d["confusion_s"] == 0.0 and d["false_alarm_s"] == pytest.approx(6.5)


def test_best_mapping_beats_greedy():
    # greedy takes 5 (row 0 -> col 0) and then 0; the optimum pairs 4 + 4
    assert ev._best_mapping([[5, 4], [4, 0]]) == 8
    assert ev._best_mapping([[1, 2, 3]]) == 3 and ev._best_mapping([]) == 0


def test_der_overlapping_reference_speakers():
    ref = [(0, 2, "A"), (1, 3, "B")]  # 1 s of crosstalk: 4 s of reference speech
    perfect = ev.der(ref, [(0, 2, "x"), (1, 3, "y")])
    assert perfect["speech_s"] == pytest.approx(4.0) and perfect["der"] == 0.0
    one_voice = ev.der(ref, [(0, 3, "x")])  # crosstalk heard as one speaker: 1 s missed
    assert one_voice["miss_s"] == pytest.approx(1.0) and one_voice["false_alarm_s"] == 0.0
    assert one_voice["confusion_s"] == pytest.approx(1.0)  # x maps to A or B, not both


def test_best_mapping_greedy_above_twelve_speakers():
    m = [[10 if i == j else 1 for j in range(14)] for i in range(13)]
    assert ev._best_mapping(m) == 130  # diagonal; greedy finds it, the DP would be 2^13 x 14


def test_der_no_reference():
    assert ev.der([], [])["der"] == 0.0


class FakeTranscriber:
    """Like the real one with keep_audio = false: drops the WAV after transcribing."""

    def __init__(self, segs):
        self.segs, self.sessions = segs, []

    def transcribe_session(self, session):
        wav = session.directory / "others.wav"
        assert wav.exists()
        self.sessions.append(session)
        session.update_metrics("transcription", {"audio_s": 5.0, "elapsed_s": 0.5})
        wav.unlink()
        return self.segs


class FakeDiarizer:
    def diarize_file(self, path):
        assert path.exists()  # runs before the transcriber can drop the audio
        return [Turn(0.0, 2.0, 0), Turn(3.0, 5.0, 1)]


def test_manifest_run_and_summary(tmp_path):
    sf.write(tmp_path / "c1.wav", np.zeros(5 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    manifest = {
        "clips": [
            {
                "id": "c1",
                "audio": "c1.wav",
                "utterances": [
                    {"start": 0, "end": 2, "speaker": "A", "language": "en", "text": "hello there"},
                    {"start": 3, "end": 5, "speaker": "B", "language": "ru", "text": "да нет"},
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (clip,) = ev.load_manifest(tmp_path)
    tr = FakeTranscriber([seg(0, 2, "hello there"), seg(3, 5, "да", "ru")])
    result = ev.run_clip(clip, tr, FakeDiarizer())
    assert not tr.sessions[0].directory.exists()  # temporary copy removed
    assert (tmp_path / "c1.wav").exists()  # the eval clip itself is untouched
    assert result["diarization"]["der"] == 0.0 and result["elapsed_s"] == 0.5
    s = ev.summarize([result])
    assert s["languages"]["en"]["wer"] == 0.0
    assert s["languages"]["ru"]["wer"] == 0.5
    assert s["languages"]["all"] == {"wer": 0.25, "lid_accuracy": 1.0, "words": 4, "utterances": 2}
    assert s["der"] == 0.0 and s["rtf"] == 0.1


def test_der_counts_false_alarms_on_clips_without_reference_speech():
    silent = {"text": {}, "diarization": ev.der([], [(0, 5, "x")])}
    clean = {"text": {}, "diarization": ev.der([(0, 10, "A")], [(0, 10, "x")])}
    assert ev.summarize([silent, clean])["der"] == pytest.approx(0.5)  # 5 s errors / 10 s speech


def test_clip_copies_stay_inside_the_given_root(tmp_path):
    sf.write(tmp_path / "c.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    clip = ev.Clip("c", tmp_path / "c.wav", ev.Track.OTHERS, [utt(0, 1, "a")])
    root = tmp_path / "run"
    root.mkdir()
    tr = FakeTranscriber([])
    ev.run_clip(clip, tr, None, root)
    assert tr.sessions[0].directory.parent == root and list(root.iterdir()) == []


def test_rtf_weighs_clips_by_audio_length():
    results = [
        {"text": {}, "audio_s": 1.0, "elapsed_s": 1.0},  # RTF 1.0 on a 1 s clip
        {"text": {}, "audio_s": 99.0, "elapsed_s": 9.9},  # RTF 0.1 on a 99 s clip
    ]
    assert ev.summarize(results)["rtf"] == pytest.approx(0.109)


@pytest.mark.parametrize("audio", ["../outside.wav", "/etc/hosts"])
def test_manifest_rejects_audio_outside_the_set(tmp_path, audio):
    root = tmp_path / "set"
    root.mkdir()
    sf.write(tmp_path / "outside.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    u = {"start": 0, "end": 1, "speaker": "A", "language": "en", "text": "a"}
    m = {"clips": [{"id": "x", "audio": audio, "utterances": [u]}]}
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the eval folder"):
        ev.load_manifest(root)


@pytest.mark.parametrize(
    ("start", "end", "msg"),
    [
        (2, 1, "start < end"),
        (-1, 1, "start < end"),
        (0, float("nan"), "start < end"),
        (0, 9, "after the audio"),
    ],
)
def test_manifest_rejects_bad_times(tmp_path, start, end, msg):
    sf.write(tmp_path / "x.wav", np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    u = {"start": start, "end": end, "speaker": "A", "language": "en", "text": "a"}
    bad = {"clips": [{"id": "x", "audio": "x.wav", "utterances": [u]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        ev.load_manifest(tmp_path)
