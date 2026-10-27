"""Splitting a segment where the diarized speaker changes (word timings, merge step).

Stories: US-12.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest
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


def test_adjacent_weak_runs_are_absorbed_by_strong_neighbours_not_each_other():
    # "Yes." (B) / "No." (C) between a strong A (2 s) and a longer strong D (3 s)
    ws = words(4) + [(2.0, 2.4, " Yes."), (2.4, 2.8, " No.")] + words(6, t0=2.8)
    turns = [Turn(0, 2, 0), Turn(2, 2.4, 1), Turn(2.4, 2.8, 2), Turn(2.8, 5.8, 3)]
    out = split_on_speaker_change([seg(ws)], turns)
    # Yes goes to A (its only strong neighbour), then No goes to the longer D; they never fuse
    assert [o.text for o in out] == ["w0 w1 w2 w3 Yes.", "No. w0 w1 w2 w3 w4 w5"]


def test_weak_runs_without_a_strong_neighbour_are_left_alone_and_it_terminates():
    ws = [(i * 0.2, (i + 1) * 0.2, f" w{i}") for i in range(4)]  # four one-word runs
    out = split_on_speaker_change([seg(ws)], [Turn(i * 0.2, (i + 1) * 0.2, i) for i in range(4)])
    assert [o.text for o in out] == ["w0", "w1", "w2", "w3"]


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


def test_json_never_has_words_and_has_overlap_only_when_true(tmp_path):
    segs = [
        Segment(0.0, 1.5, "héllo", "Me", "en", words=[(0.0, 1.5, " héllo")]),
        Segment(1.5, 3.0, "привет", "Others", "ru", overlap=True, words=[(1.5, 3.0, " привет")]),
    ]
    session = make_session(tmp_path)
    got = json.loads(format_json(session, segs, 3.0, "m"))["segments"]
    assert list(got[0]) == ["start", "end", "text", "speaker", "language"]  # as before #96/#97
    assert got[1] == {
        "start": 1.5,
        "end": 3.0,
        "text": "привет",
        "speaker": "Others",
        "language": "ru",
        "overlap": True,
    }


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


class _FakeMlx(eng.MlxWhisper):
    def __init__(self, ws=((0.0, 0.5, " a"), (0.5, 1.5, " b"))):
        self.calls, self.ws = [], list(ws)

    def transcribe(self, audio, language, words=False):
        self.calls.append(words)
        text = "".join(w for _, _, w in self.ws)
        return [(0.0, 1.0, text, self.ws)] if words else [(0.0, 1.0, text)]

    def close(self):
        pass


@pytest.mark.parametrize("aligner", [False, True])
def test_mlx_requests_words_only_when_the_numpy_aligner_is_available(monkeypatch, aligner):
    monkeypatch.setattr(eng, "mlx_word_timestamps", lambda: aligner)
    model = _FakeMlx()
    t = _transcriber(model)
    audio = np.zeros(SAMPLE_RATE, np.float32)
    out = t.transcribe_audio(audio, 10.0, "Others", "en", words=True)
    assert model.calls == [aligner]
    # session time, the word's own leading space, clamped to the chunk end
    assert out[0].words == ([(10.0, 10.5, " a"), (10.5, 11.0, " b")] if aligner else [])
    t.transcribe_audio(audio, 0.0, "Me", "en")
    assert model.calls[-1] is False


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
    # a word was cut off, so the words no longer cover the text: unsplittable, nothing is lost
    assert out[0].text == "a b c" and out[0].words == []
    assert split_on_speaker_change(out, [Turn(0, 10.5, 0), Turn(10.5, 12, 1)]) == out


def test_mlx_words_crossing_max_end_clear_the_segment_words(monkeypatch):
    monkeypatch.setattr(eng, "mlx_word_timestamps", lambda: True)
    t = _transcriber(_FakeMlx([(0.0, 0.5, " a"), (0.5, 1.5, " b"), (1.2, 1.4, " c")]))
    audio = np.zeros(2 * SAMPLE_RATE, np.float32)
    out = t.transcribe_audio(audio, 10.0, "Others", "en", max_end=11.0, words=True)
    assert out[0].text == "a b c" and out[0].words == []  # same rule as faster-whisper
    out = t.transcribe_audio(audio, 10.0, "Others", "en", max_end=12.0, words=True)
    assert out[0].words == [(10.0, 10.5, " a"), (10.5, 11.5, " b"), (11.2, 11.4, " c")]


def test_words_are_kept_when_nothing_is_cut():
    t = _transcriber(_Model())
    out = t.transcribe_audio(
        np.zeros(SAMPLE_RATE, np.float32), 10.0, "Others", "en", max_end=11.0, words=True
    )
    assert out[0].words == [(10.0, 10.5, " a"), (10.5, 11.0, " b")]


def _session(kind=eng.TriggerKind.CALL, started_as_room=False, tmp_path=None):
    from datetime import datetime
    from pathlib import Path

    from recandcc.models import SessionInfo

    return SessionInfo(
        "s", tmp_path or Path("."), datetime(2026, 10, 27), kind, started_as_room=started_as_room
    )


def _word_flags(monkeypatch, present, cfg=None, session=None, tracks=(eng.Track.OTHERS,)):
    """word_timestamps of one chunk per track, with the tracks decided as _run does."""
    model = _Model()
    t = eng.Transcriber(cfg or TranscriptionConfig(model="fake", split_on_speaker_change=True))
    t._model = model
    t._lid_model = None
    monkeypatch.setattr(eng, "model_path", lambda *a, **k: "/x" if present else None)
    word_tracks = t._word_tracks(session or _session())
    for track in tracks:
        chunk = eng.Chunk(track, 0.0, 1.0, language="en")
        t._transcribe_chunk(chunk, np.zeros(SAMPLE_RATE, np.float32), word_tracks)
    return [c["word_timestamps"] for c in model.calls]


def test_words_only_requested_when_enabled_and_the_speaker_model_is_on_disk(monkeypatch):
    assert _word_flags(monkeypatch, True) == [True]
    assert _word_flags(monkeypatch, False) == [False]
    off = TranscriptionConfig(model="fake", split_on_speaker_change=False)
    assert _word_flags(monkeypatch, True, off) == [False]
    assert TranscriptionConfig(model="fake").split_on_speaker_change is False  # opt-in


def test_room_sessions_request_words_on_the_mic_track(monkeypatch):
    """In a room session the diarized track is "Me", so that is the one that needs words."""
    room = _session(eng.TriggerKind.ROOM)
    flags = _word_flags(monkeypatch, True, session=room, tracks=(eng.Track.ME, eng.Track.OTHERS))
    assert flags == [True, False]
    call = _word_flags(monkeypatch, True, tracks=(eng.Track.ME, eng.Track.OTHERS))
    assert call == [False, True]  # call session: Me never split


def test_word_timestamps_are_decided_once_per_session(tmp_path, monkeypatch):
    """The speaker model appears mid-run (download finishing): every chunk of the session
    still gets the same word_timestamps value, decided before the chunks run."""
    model = _Model()
    t = eng.Transcriber(TranscriptionConfig(model="fake", split_on_speaker_change=True))
    t._model = model
    t._lid_model = None
    answers = iter([None, "/x", "/x", "/x"])
    monkeypatch.setattr(eng, "model_path", lambda *a, **k: next(answers))
    word_tracks = t._word_tracks(_session(tmp_path=tmp_path))  # the single check: None
    for i in range(3):
        chunk = eng.Chunk(eng.Track.OTHERS, float(i), i + 1.0, language="en")
        t._transcribe_chunk(chunk, np.zeros(SAMPLE_RATE, np.float32), word_tracks)
    assert [c["word_timestamps"] for c in model.calls] == [False, False, False]
    # and a session decided once with the model present asks for words on all its chunks
    word_tracks = t._word_tracks(_session(tmp_path=tmp_path))
    assert word_tracks == (eng.Track.OTHERS,)


def test_split_tracks_follow_diarized_tracks_for_room_and_upgraded_sessions(tmp_path, monkeypatch):
    """The word split and word timestamps cover exactly the diarized tracks: Others in a call,
    Me in a room session, and both in a room session upgraded to a call (#103)."""
    from datetime import datetime

    from recandcc.models import SessionInfo, TriggerKind

    def session(kind, started_as_room=False):
        return SessionInfo(
            "s", tmp_path, datetime(2026, 10, 27), kind, started_as_room=started_as_room
        )

    upgraded = session(TriggerKind.CALL, started_as_room=True)
    assert eng.diarized_tracks(session(TriggerKind.CALL)) == (eng.Track.OTHERS,)
    assert eng.diarized_tracks(session(TriggerKind.ROOM)) == (eng.Track.ME,)
    assert set(eng.diarized_tracks(upgraded)) == {eng.Track.ME, eng.Track.OTHERS}
    # both tracks of an upgraded session ask for word timings
    model = _Model()
    t = eng.Transcriber(TranscriptionConfig(model="fake", split_on_speaker_change=True))
    t._model = model
    t._lid_model = None
    monkeypatch.setattr(eng, "model_path", lambda *a, **k: "/x")
    for track in (eng.Track.ME, eng.Track.OTHERS):
        chunk = eng.Chunk(track, 0.0, 1.0, language="en")
        t._transcribe_chunk(chunk, np.zeros(SAMPLE_RATE, np.float32), t._word_tracks(upgraded))
    assert [c["word_timestamps"] for c in model.calls] == [True, True]


def test_each_diarized_track_is_split_with_its_own_turns(tmp_path):
    """In an upgraded room session, a Me segment and an Others segment that each change speaker
    mid-way are both split, each by its own track's turns, and numbering doesn't collide."""
    from datetime import datetime

    from recandcc.models import SessionInfo, TriggerKind

    sess = SessionInfo(
        "s", tmp_path, datetime(2026, 10, 27), TriggerKind.CALL, started_as_room=True
    )
    w = [(0.0, 0.5, " one"), (0.5, 1.0, " two"), (1.0, 1.5, " three"), (1.5, 3.0, " four five six")]
    me = Segment(0.0, 3.0, "one two three four five six", "Me", "en", words=list(w))
    shift = [(a + 10, b + 10, x) for a, b, x in w]
    others = Segment(10.0, 13.0, "one two three four five six", "Others", "en", words=shift)
    turns = {
        eng.Track.ME: [Turn(0.0, 1.5, 0), Turn(1.5, 3.0, 1)],
        eng.Track.OTHERS: [Turn(10.0, 11.5, 0), Turn(11.5, 13.0, 1)],
    }
    t = eng.Transcriber(
        TranscriptionConfig(model="fake", speaker_names=False, split_on_speaker_change=True)
    )
    out = t._labelled(sess, [me, others], turns)
    assert len(out) == 4
    labels = [s.speaker for s in out]
    assert len(set(labels)) == 4  # two per track, numbering continues across tracks


def _naive(words_, turns):
    out = []
    for ws, we, _ in words_:
        overlap = {}
        for t in turns:
            o = min(we, t.end) - max(ws, t.start)
            if o > 0:
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + o
        if overlap:
            out.append(max(overlap, key=lambda k: overlap[k]))
        else:
            mid = (ws + we) / 2
            out.append(min(turns, key=lambda t: max(t.start - mid, mid - t.end, 0.0)).speaker)
    return out


def _random_case(rng, n_turns, n_words):
    turns = []
    t = 0.0
    for _ in range(n_turns):
        t += rng.choice([0.0, 0.0, 0.3, 1.5])  # gaps
        d = rng.choice([0.0, 0.4, 2.0, 6.0])
        turns.append(Turn(round(t, 2), round(t + d, 2), rng.randrange(4)))
        t += rng.choice([d, d / 2, 0.0])  # sometimes overlapping
    rng.shuffle(turns)
    ws, x = [], -1.0
    for _ in range(n_words):
        x += rng.choice([0.0, 0.1, 0.5])
        ws.append((round(x, 2), round(x + rng.choice([0.0, 0.2, 0.5]), 2), " w"))
    return ws, turns


def test_word_speakers_match_the_naive_scan_on_random_input():
    import random

    from recandcc.transcribe.merge import _TurnIndex, _word_speakers

    rng = random.Random(1234)  # noqa: S311
    for _ in range(200):
        ws, turns = _random_case(rng, rng.randrange(1, 15), rng.randrange(1, 40))
        assert _word_speakers(ws, _TurnIndex(turns)) == _naive(ws, turns)


def test_split_scales_to_an_hour_long_call():
    """Production shape: ~1,500 segments of 5-20 words against ~1,500 turns.
    Stories: US-12."""
    import random
    import time

    rng = random.Random(7)  # noqa: S311
    turns = [Turn(i * 2.4, i * 2.4 + 2.3, rng.randrange(3)) for i in range(1500)]
    segs, t = [], 0.0
    for _ in range(1500):
        ws = []
        for _ in range(rng.randrange(5, 21)):
            ws.append((t, t + 0.2, " w"))
            t += 0.24
        segs.append(seg(ws))
        t += 0.1
    t0 = time.perf_counter()
    out = split_on_speaker_change(segs, turns)
    assert time.perf_counter() - t0 < 2.0
    assert len(out) >= len(segs)
    assert [(s.start, s.end) for s in out] == sorted((s.start, s.end) for s in out)


def test_split_pieces_keep_the_transcript_in_time_order(tmp_path):
    """Others 0-10 s split at 5 s plus Me 3-4 s: Me sits between the pieces, in every output.
    Stories: US-12."""
    ws = [(i * 1.0, i * 1.0 + 1.0, f" w{i}") for i in range(10)]
    others = seg(ws)
    me = Segment(3.0, 4.0, "hello", "Me", "en")
    turns = [Turn(0, 5, 0), Turn(5, 10, 1)]
    sess = make_session(tmp_path)
    t = eng.Transcriber(
        TranscriptionConfig(model="fake", speaker_names=False, split_on_speaker_change=True)
    )
    out = t._labelled(sess, [others, me], turns)
    assert [s.speaker for s in out] == ["Speaker 1", "Me", "Speaker 2"]
    assert [s.start for s in out] == [0.0, 3.0, 5.0]
    md = format_markdown(sess, out, 10.0)
    assert md.index("w0 w1") < md.index("hello") < md.index("w5 w6")


def test_a_failing_word_pass_keeps_the_session_and_the_chunk_whole(tmp_path, monkeypatch, caplog):
    """MLX word alignment raising on a chunk must not fail the session: that chunk is
    transcribed again without words (unsplit), and the warning carries no transcript text."""
    import logging

    from test_transcribe import FakeModel, _install, _write_tracks

    class Flaky(_FakeMlx):
        def transcribe(self, audio, language, words=False):
            self.calls.append(words)
            if words:
                raise IndexError("jump_times")
            return [(0.0, len(audio) / SAMPLE_RATE, " secret words")]

    regions = _write_tracks(tmp_path)
    monkeypatch.setattr(eng, "find_speech", lambda a: regions)
    monkeypatch.setattr(eng, "mlx_word_timestamps", lambda: True)
    s = make_session(tmp_path, {eng.Track.ME: "me.wav", eng.Track.OTHERS: "others.wav"})
    t = eng.Transcriber(TranscriptionConfig(model="fake", workers=1, split_on_speaker_change=True))
    model = Flaky()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: _install(t, model, FakeModel()))
    monkeypatch.setattr(t, "_word_tracks", lambda _s: (eng.Track.OTHERS,))
    with caplog.at_level(logging.DEBUG, "recandcc.transcribe"):
        segs = t.transcribe_session(s)
    others = [x for x in segs if x.speaker == "Others"]
    assert others and all(x.text == "secret words" and x.words == [] for x in others)
    # one worker, so chunks run in order: the first failure turns words off for the session
    assert len([c for c in model.calls if c]) == 1 < len(others)
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert sum("off for the rest of this session" in m for m in warnings) == 1
    assert not any("secret" in m for m in warnings)
    # a new session starts with words on again
    model.calls.clear()
    t2_segs = t.transcribe_session(s)
    t.close()
    assert t2_segs and len([c for c in model.calls if c]) == 1


def test_an_empty_tail_window_does_not_trip_the_session_breaker(tmp_path, monkeypatch, caplog):
    """A 1-frame tail window aligns to no words (no exception), so every chunk keeps its word
    pass: the breaker must not switch words off for the rest of the session."""
    import logging

    from test_transcribe import FakeModel, _install, _write_tracks

    from recandcc.transcribe import mlx_timing

    class TailWindow(_FakeMlx):
        def transcribe(self, audio, language, words=False):
            self.calls.append(words)
            if words:
                segs = [{"seek": 0, "start": 0.0, "end": 0.02, "tokens": [10, 11, 100]}]
                mlx_timing.add_word_timestamps(
                    segments=segs,
                    model=None,
                    tokenizer=types.SimpleNamespace(eot=100),
                    mel=None,
                    num_frames=1,
                    last_speech_timestamp=0.0,
                )
                assert segs[0].get("words", []) == []
                return [(0.0, len(audio) / SAMPLE_RATE, " hello there", [])]
            return [(0.0, len(audio) / SAMPLE_RATE, " hello there")]

    regions = _write_tracks(tmp_path)
    monkeypatch.setattr(eng, "find_speech", lambda a: regions)
    monkeypatch.setattr(eng, "mlx_word_timestamps", lambda: True)
    s = make_session(tmp_path, {eng.Track.ME: "me.wav", eng.Track.OTHERS: "others.wav"})
    t = eng.Transcriber(TranscriptionConfig(model="fake", workers=1, split_on_speaker_change=True))
    model = TailWindow()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: _install(t, model, FakeModel()))
    monkeypatch.setattr(t, "_word_tracks", lambda _s: (eng.Track.OTHERS,))
    with caplog.at_level(logging.DEBUG, "recandcc.transcribe"):
        segs = t.transcribe_session(s)
    t.close()
    others = [x for x in segs if x.speaker == "Others"]
    assert len(others) > 1
    assert model.calls.count(True) == len(others)  # the word pass ran on every chunk
    assert not any("word timestamps failed" in r.getMessage() for r in caplog.records)


def test_failed_lazy_word_pass_is_closed_before_the_retry():
    """faster-whisper yields segments lazily: a failure mid-iteration must close that
    generator before the same model transcribes the chunk again without words."""
    events = []

    class Lazy:
        def __init__(self, fail):
            self.fail = fail

        def __iter__(self):
            ws = [_W(0.0, 0.5, " a")] if self.fail else []
            yield types.SimpleNamespace(start=0.0, end=0.5, text=" a", words=ws)
            if self.fail:
                raise IndexError("boom")
            yield types.SimpleNamespace(start=0.5, end=1.0, text=" b", words=[])

        def close(self):
            events.append(("close", self.fail))

    class Model(_Model):
        def transcribe(self, audio, **kw):
            events.append(("transcribe", kw["word_timestamps"]))
            return Lazy(kw["word_timestamps"]), None

    t = _transcriber(Model())
    out = t.transcribe_audio(np.zeros(SAMPLE_RATE, np.float32), 0.0, "Others", "en", words=True)
    assert events == [
        ("transcribe", True),
        ("close", True),
        ("transcribe", False),
        ("close", False),
    ]
    assert [x.text for x in out] == ["a", "b"] and all(x.words == [] for x in out)
