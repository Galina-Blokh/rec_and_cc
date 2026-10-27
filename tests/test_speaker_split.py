"""Splitting a segment where the diarized speaker changes (word timings, merge step).

Stories: US-12.
"""

from __future__ import annotations

import json
import types

import numpy as np
from test_transcribe import make_session

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment
from recandcc.transcribe import engine as eng
from recandcc.transcribe.diarize import Turn, assign_speakers
from recandcc.transcribe.merge import split_on_speaker_change
from recandcc.transcribe.output import format_json, format_markdown, format_srt


def words(n, step=0.5, t0=0.0):
    return [(t0 + i * step, t0 + (i + 1) * step, f" w{i}") for i in range(n)]


def seg(ws, speaker="Others"):
    text = "".join(w[2] for w in ws).strip()
    return Segment(ws[0][0], ws[-1][1], text, speaker, "en", words=list(ws))


def labelled(segments, turns):
    return assign_speakers(split_on_speaker_change(segments, turns), turns)


def test_mid_segment_change_splits_at_the_word():
    s = seg(words(8))  # 0-4 s
    out = labelled([s], [Turn(0, 2, 0), Turn(2, 4, 1)])
    assert [(o.speaker, o.text, o.start, o.end) for o in out] == [
        ("Speaker 1", "w0 w1 w2 w3", 0.0, 2.0),
        ("Speaker 2", "w4 w5 w6 w7", 2.0, 4.0),
    ]
    assert all(o.language == "en" for o in out)


def test_short_flicker_is_not_split():
    s = seg(words(8))
    # speaker 1 interjects for one word (0.5 s)
    out = split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 2.5, 1), Turn(2.5, 4, 0)])
    assert len(out) == 1 and out[0] is s


def test_text_is_preserved_word_for_word():
    s = seg(words(12))
    out = split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 4, 1), Turn(4, 6, 0)])
    assert [o.text for o in out] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9 w10 w11"]


def test_a_b_a_with_long_b_splits_into_three():
    s = seg(words(12))
    out = split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 4, 1), Turn(4, 6, 0)])
    assert [(o.start, o.end) for o in out] == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]


def test_flicker_tie_is_absorbed_by_one_neighbour():
    s = seg(words(8))  # equal-length neighbours (1.5 s each) around a 1-word flicker
    out = split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 2.5, 1), Turn(2.5, 4, 2)])
    assert len(out) == 2
    assert "".join(o.text for o in out).replace(" ", "") == "w0w1w2w3w4w5w6w7"


def test_zero_length_word_is_kept_and_assigned():
    ws = [(0.0, 1.0, " a"), (1.0, 2.0, " b"), (2.0, 2.0, " c"), (2.0, 3.0, " d"), (3.0, 4.0, " e")]
    out = split_on_speaker_change([seg(ws)], [Turn(0, 2, 0), Turn(2, 4, 1)])
    assert [o.text for o in out] == ["a b", "c d e"] or [o.text for o in out] == ["a b c", "d e"]
    assert "".join(o.text for o in out).replace(" ", "") == "abcde"


def test_punctuation_stays_attached_to_its_word():
    ws = [
        (0.0, 1.0, " Hello"),
        (1.0, 1.5, ","),
        (1.5, 2.0, " there"),
        (2.0, 3.0, " Hi"),
        (3.0, 4.0, "!"),
    ]
    out = split_on_speaker_change([seg(ws)], [Turn(0, 2, 0), Turn(2, 4, 1)])
    assert [o.text for o in out] == ["Hello, there", "Hi!"]


def test_cjk_and_subword_tokens_are_not_given_spaces():
    cjk = [(0.0, 1.0, "你"), (1.0, 2.0, "好"), (2.0, 3.0, "吗"), (3.0, 4.0, "好"), (4.0, 5.0, "的")]
    out = split_on_speaker_change([seg(cjk)], [Turn(0, 3, 0), Turn(3, 5, 1)])
    assert [o.text for o in out] == ["你好吗", "好的"]
    sub = [
        (0.0, 1.0, " Hel"),
        (1.0, 2.0, "lo"),
        (2.0, 3.0, " wor"),
        (3.0, 4.0, "ld"),
        (4.0, 5.0, "!"),
    ]
    out = split_on_speaker_change([seg(sub)], [Turn(0, 2, 0), Turn(2, 5, 1)])
    assert [o.text for o in out] == ["Hello", "world!"]


def test_no_words_means_unchanged():
    s = Segment(0, 4, "plain text here", "Others", "en")
    assert split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 4, 1)]) == [s]


def test_me_track_is_never_split():
    s = seg(words(8), speaker="Me")
    out = split_on_speaker_change([s], [Turn(0, 2, 0), Turn(2, 4, 1)])
    assert out == [s]


def test_outputs_carry_the_split_segments(tmp_path):
    turns = [Turn(0, 2, 0), Turn(2, 4, 1)]
    out = labelled([seg(words(8))], turns)
    session = make_session(tmp_path)
    md = format_markdown(session, out, 4.0)
    assert "Speaker 1:** w0 w1 w2 w3" in md and "Speaker 2:** w4 w5 w6 w7" in md
    assert "Speaker 2: w4 w5 w6 w7" in format_srt(out)
    data = json.loads(format_json(session, out, 4.0, "m"))
    assert [s["text"] for s in data["segments"]] == ["w0 w1 w2 w3", "w4 w5 w6 w7"]
    assert all("words" not in s for s in data["segments"])


def test_json_without_words_is_byte_identical_to_asdict(tmp_path):
    from dataclasses import asdict

    segs = [Segment(0.0, 1.5, "héllo", "Me", "en"), Segment(1.5, 3.0, "привет", "Others", "ru")]
    session = make_session(tmp_path)
    got = json.loads(format_json(session, segs, 3.0, "m"))["segments"]
    want = [{k: v for k, v in asdict(s).items() if k != "words"} for s in segs]
    assert got == want
    assert list(got[0]) == ["start", "end", "text", "speaker", "language"]


class _W:
    def __init__(self, start, end, word):
        self.start, self.end, self.word = start, end, word


class _Model:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        s = types.SimpleNamespace(
            start=0.0, end=1.0, text=" a b", words=[_W(0.0, 0.5, " a"), _W(0.5, 1.0, " b")]
        )
        return iter([s]), None


def _transcriber(model):
    t = eng.Transcriber(TranscriptionConfig(model="fake"))
    t._model = model
    t._lid_model = None
    return t


def test_faster_whisper_words_are_requested_and_offset():
    model = _Model()
    t = _transcriber(model)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    out = t.transcribe_audio(audio, 10.0, "Others", "en", words=True)
    assert model.calls[0]["word_timestamps"] is True
    assert out[0].words == [(10.0, 10.5, " a"), (10.5, 11.0, " b")]
    t.transcribe_audio(audio, 0.0, "Me", "en")
    assert model.calls[1]["word_timestamps"] is False


def test_mlx_path_never_requests_word_timestamps():
    class FakeMlx(eng.MlxWhisper):
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language, **kw):
            self.calls.append(kw)
            return [(0.0, 1.0, "a b")]

    model = FakeMlx()
    t = _transcriber(model)
    out = t.transcribe_audio(np.zeros(SAMPLE_RATE, np.float32), 0.0, "Others", "en", words=True)
    assert model.calls == [{}] and out[0].words == []


def test_word_times_are_clamped_to_the_chunk_end():
    class Late(_Model):
        def transcribe(self, audio, **kw):
            ws = [_W(0.0, 0.5, " a"), _W(0.5, 1.5, " b"), _W(1.2, 1.4, " c")]
            s = types.SimpleNamespace(start=0.0, end=1.5, text=" a b c", words=ws)
            return iter([s]), None

    t = _transcriber(Late())
    out = t.transcribe_audio(
        np.zeros(2 * SAMPLE_RATE, np.float32), 10.0, "Others", "en", max_end=11.0, words=True
    )
    assert out[0].end == 11.0
    assert out[0].words == [(10.0, 10.5, " a"), (10.5, 11.0, " b")]


def _chunk_words(monkeypatch, present, cfg=None):
    model = _Model()
    t = eng.Transcriber(cfg or TranscriptionConfig(model="fake"))
    t._model = model
    t._lid_model = None
    monkeypatch.setattr(eng, "model_path", lambda *a, **k: "/x" if present else None)
    chunk = eng.Chunk("others", 0.0, 1.0, language="en")
    t._transcribe_chunk(chunk, np.zeros(SAMPLE_RATE, np.float32))
    return model.calls[0]["word_timestamps"]


def test_words_only_requested_when_the_speaker_model_is_on_disk(monkeypatch):
    assert _chunk_words(monkeypatch, True) is True
    assert _chunk_words(monkeypatch, False) is False
    off = TranscriptionConfig(model="fake", split_on_speaker_change=False)
    assert _chunk_words(monkeypatch, True, off) is False


def test_room_sessions_request_words_on_the_mic_track(monkeypatch):
    """In a room session the diarized track is "Me", so that is the one that needs words."""
    model = _Model()
    t = eng.Transcriber(TranscriptionConfig(model="fake"))
    t._model = model
    t._lid_model = None
    monkeypatch.setattr(eng, "model_path", lambda *a, **k: "/x")
    me = eng.Chunk(eng.Track.ME, 0.0, 1.0, language="en")
    t._transcribe_chunk(me, np.zeros(SAMPLE_RATE, np.float32), eng.Track.ME)
    t._transcribe_chunk(me, np.zeros(SAMPLE_RATE, np.float32))  # call session: Me never split
    assert [c["word_timestamps"] for c in model.calls] == [True, False]


def test_split_track_follows_diarized_track_for_room_and_upgraded_sessions(tmp_path):
    """The word split runs on exactly the track that gets diarized: Others in a call, Me in a room
    session, and Me in a room session later upgraded to a call (started_as_room), which today
    diarizes only the mic track. If both tracks get diarized one day, extend this together."""
    from datetime import datetime

    from recandcc.models import SessionInfo, TriggerKind

    def session(kind, started_as_room=False):
        return SessionInfo(
            "s", tmp_path, datetime(2026, 10, 27), kind, started_as_room=started_as_room
        )

    assert eng.diarized_track(session(TriggerKind.CALL)) == eng.Track.OTHERS
    assert eng.diarized_track(session(TriggerKind.ROOM)) == eng.Track.ME
    assert eng.diarized_track(session(TriggerKind.CALL, started_as_room=True)) == eng.Track.ME
