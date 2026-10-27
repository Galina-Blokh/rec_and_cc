"""A short audio sample of a speaker in the "Name speakers…" dialog, with fakes (no afplay).

Stories: US-26.
"""

from __future__ import annotations

import json
import logging
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.ui import sample
from recandcc.ui.common import ask_speaker_name, name_dialog_answer, name_dialog_script

RATE = 16000


def _wav(path, seconds: float, tone_from: float = 0.0, tone_to: float = 0.0) -> None:
    x = np.zeros(int(seconds * RATE), np.int16)
    x[int(tone_from * RATE) : int(tone_to * RATE)] = 8000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(x.tobytes())


def _session(tmp_path, segments, kind=TriggerKind.CALL, room=False, wavs=("others.wav",)):
    tracks = {Track.OTHERS: "others.wav", Track.ME: "me.wav"}
    SessionInfo(
        id="s1",
        directory=tmp_path,
        started_at=datetime(2026, 10, 26, 14, 0, 0),
        trigger_kind=kind,
        tracks=tracks,
        started_as_room=room,
    ).save()
    data = {
        "segments": [
            {"start": a, "end": b, "speaker": spk, "text": "secret words"} for a, b, spk in segments
        ]
    }
    (tmp_path / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    for name in wavs:
        _wav(tmp_path / name, 30.0)
    return tmp_path


def test_pick_span_prefers_first_long_segment_and_clamps():
    assert sample.pick_span([(0, 1), (2, 5), (10, 30)]) == (2, 5)  # first >= 2 s
    assert sample.pick_span([(0, 1), (5, 25)]) == (5, 11)  # clamped to MAX_S
    assert sample.pick_span([(0, 0.5), (3, 4.5), (6, 7)]) == (3, 4.5)  # else the longest
    assert sample.pick_span([]) is None
    assert sample.pick_span([(4, 4)]) is None


def test_find_clip_uses_the_speakers_segments_on_the_others_track(tmp_path):
    d = _session(tmp_path, [(0, 1, "Me"), (1, 2, "Speaker 1"), (3, 8, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip == sample.Clip(d / "others.wav", 3.0, 8.0)
    assert sample.find_clip(d, "Speaker 9") is None


def test_room_upgraded_session_picks_the_louder_track(tmp_path):
    d = _session(tmp_path, [(5, 9, "Speaker 1")], room=True, wavs=())
    _wav(d / "others.wav", 30.0)
    _wav(d / "me.wav", 30.0, tone_from=5, tone_to=9)
    assert sample.find_clip(d, "Speaker 1").wav == d / "me.wav"


def test_missing_or_purged_audio_gives_no_clip_and_the_plain_dialog(tmp_path):
    d = _session(tmp_path, [(0, 4, "Speaker 1")], wavs=())
    assert sample.find_clip(d, "Speaker 1") is None  # retention deleted the WAV
    _wav(d / "others.wav", 10.0)
    assert sample.find_clip(d, "Speaker 1") is not None
    _wav(d / "others.wav", 0.0)  # truncated: the clip lies past the end of the audio
    assert sample.find_clip(d, "Speaker 1") is None
    (d / "transcript.json").write_text("{not json", encoding="utf-8")
    assert sample.find_clip(d, "Speaker 1") is None  # never raises

    shown: list[str] = []

    def ask(script: str) -> str:
        shown.append(script)
        return "button returned:Save, text returned:Anna"

    assert ask_speaker_name("Speaker 1", None, ask, lambda _c: None) == "Anna"
    assert "Play sample" not in shown[0]


def test_dialog_plays_the_sample_and_comes_back_with_the_typed_text(tmp_path):
    clip = sample.Clip(tmp_path / "others.wav", 0.0, 3.0)
    answers = iter(
        [
            "button returned:Play sample, text returned:An",
            "button returned:Save, text returned:Anna",
        ]
    )
    shown: list[str] = []
    played: list = []

    def ask(script: str) -> str:
        shown.append(script)
        return next(answers)

    assert ask_speaker_name("Speaker 1", clip, ask, played.append) == "Anna"
    assert played == [clip]
    assert '"Play sample"' in shown[0] and 'default answer "An"' in shown[1]
    assert name_dialog_answer("") == (False, "")  # cancelled / timed out
    assert name_dialog_script("x") == name_dialog_script("x", sample=False, answer="")


class FakeProc:
    def __init__(self, args, **_kw):
        self.args = args
        self.thread = threading.current_thread()
        self.existed = Path(args[-1]).exists()
        with wave.open(args[-1], "rb") as w:
            self.seconds = w.getnframes() / w.getframerate()

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def terminate(self):
        pass


def test_player_plays_off_the_calling_thread_and_deletes_the_clip(tmp_path, caplog):
    d = _session(tmp_path, [(2, 12, "Anna Secret")])
    procs: list[FakeProc] = []

    def popen(args, **kw):
        procs.append(FakeProc(args, **kw))
        return procs[-1]

    player = sample.SamplePlayer(popen=popen)
    clip = sample.find_clip(d, "Anna Secret")
    assert clip is not None
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        player.play(clip).join(5)
    (proc,) = procs
    assert proc.thread is not threading.main_thread()
    assert proc.args[0] == "afplay" and proc.existed
    assert abs(proc.seconds - sample.MAX_S) < 0.01
    folder = Path(proc.args[-1]).parent
    assert list(folder.iterdir()) == []  # deleted after playback
    player.close()
    assert not folder.exists()  # the temp folder goes on exit
    assert all(
        "Anna" not in r.getMessage() and "secret" not in r.getMessage() for r in caplog.records
    )
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_player_failure_still_deletes_and_logs_only_the_error_type(tmp_path, caplog):
    d = _session(tmp_path, [(0, 3, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip is not None
    seen: list[str] = []

    def popen(args, **_kw):
        seen.append(args[-1])
        raise FileNotFoundError(args[-1])  # no afplay

    player = sample.SamplePlayer(popen=popen)
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        player.play(clip).join(5)
    assert seen and not Path(seen[0]).exists()
    assert any("FileNotFoundError" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]
    player.close()


def test_a_new_sample_stops_the_one_playing(tmp_path):
    d = _session(tmp_path, [(0, 3, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip is not None
    release = threading.Event()
    stopped: list[bool] = []

    class Slow(FakeProc):
        def wait(self, timeout=None):
            release.wait(5)
            return 0

        def poll(self):
            return None if not release.is_set() else 0

        def terminate(self):
            stopped.append(True)
            release.set()

    player = sample.SamplePlayer(popen=lambda a, **k: Slow(a))
    first = player.play(clip)
    for _ in range(100):
        if player._proc is not None:
            break
        threading.Event().wait(0.01)
    player.play(clip).join(5)
    first.join(5)
    assert stopped == [True]
    player.close()
