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
    assert len(out) == 3
    assert " ".join(o.text for o in out) == s.text


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
