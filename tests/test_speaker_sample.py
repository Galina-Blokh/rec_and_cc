"""A short audio sample of a speaker in the "Name speakers…" dialog, with fakes (no afplay).

Stories: US-26.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import wave
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from recandcc.engine import IDLE, PROMPTING, RECORDING
from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.ui import sample
from recandcc.ui.common import (
    Controller,
    ask_speaker_name,
    name_dialog_answer,
    name_dialog_script,
)

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


def test_plain_room_session_uses_the_mic_track(tmp_path):
    d = _session(tmp_path, [(5, 9, "Speaker 1")], kind=TriggerKind.ROOM, wavs=("me.wav",))
    assert sample.find_clip(d, "Speaker 1") == sample.Clip(d / "me.wav", 5.0, 9.0)


def test_upgraded_room_session_never_plays_the_mic(tmp_path):
    # a room session a call upgraded: the label's track is not recorded, so it is Others
    d = _session(tmp_path, [(5, 9, "Speaker 3")], room=True, wavs=())
    _wav(d / "me.wav", 30.0, tone_from=0, tone_to=30)  # the user talks the whole time
    _wav(d / "others.wav", 30.0, tone_from=5, tone_to=9)
    assert sample.find_clip(d, "Speaker 3") == sample.Clip(d / "others.wav", 5.0, 9.0)
    _wav(d / "others.wav", 30.0, tone_from=20, tone_to=30)  # padding before the upgrade
    assert sample.find_clip(d, "Speaker 3") is None  # silence: no sample, never the mic


def test_missing_or_purged_audio_gives_no_clip_and_the_plain_dialog(tmp_path):
    d = _session(tmp_path, [(0, 4, "Speaker 1")], wavs=())
    assert sample.find_clip(d, "Speaker 1") is None  # retention deleted the WAV
    _wav(d / "others.wav", 10.0)
    assert sample.find_clip(d, "Speaker 1") is not None
    (d / "transcript.json").write_text("{not json", encoding="utf-8")
    assert sample.find_clip(d, "Speaker 1") is None  # never raises

    shown: list[str] = []

    def ask(script: str) -> str:
        shown.append(script)
        return "button returned:Save, text returned:Anna"

    assert ask_speaker_name("Speaker 1", None, ask, lambda _c: None) == "Anna"
    assert "Play sample" not in shown[0]


def test_segment_past_the_end_or_a_corrupt_wav_gives_no_clip(tmp_path):
    d = _session(tmp_path, [(40, 44, "Speaker 1")])  # the WAV is 30 s long
    assert sample.find_clip(d, "Speaker 1") is None
    _session(tmp_path, [(0, 4, "Speaker 1")], wavs=())
    (d / "others.wav").write_bytes(b"RIFF\x00\x00garbage, not a wave file")
    assert sample.find_clip(d, "Speaker 1") is None


def test_no_sample_while_a_session_records(tmp_path):
    d = _session(tmp_path, [(0, 4, "Speaker 1")])
    ctrl = Controller.__new__(Controller)
    for status in (RECORDING, PROMPTING):
        ctrl.engine = SimpleNamespace(state=SimpleNamespace(status=status))  # type: ignore[assignment]
        assert not ctrl.sample_allowed
        assert ctrl.speaker_clip(d, "Speaker 1") is None  # no "Play sample" button
    ctrl.engine = SimpleNamespace(state=SimpleNamespace(status=IDLE))  # type: ignore[assignment]
    assert ctrl.speaker_clip(d, "Speaker 1") == sample.Clip(d / "others.wav", 0.0, 4.0)

    calls: list = []
    player = sample.SamplePlayer(
        popen=lambda *a, **k: calls.append(a), allowed=lambda: False, root=tmp_path
    )
    assert player.play(sample.Clip(d / "others.wav", 0.0, 4.0)) is None
    assert calls == []
    player.close()


def test_overlapping_segments_are_used_only_when_nothing_else_is_left(tmp_path):
    d = _session(tmp_path, [(2, 8, "Speaker 1"), (10, 11, "Speaker 1")])
    data = json.loads((d / "transcript.json").read_text(encoding="utf-8"))
    data["segments"][0]["overlap"] = True
    (d / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    assert sample.find_clip(d, "Speaker 1") == sample.Clip(d / "others.wav", 10.0, 11.0)
    data["segments"] = data["segments"][:1]  # only the overlap is left
    (d / "transcript.json").write_text(json.dumps(data), encoding="utf-8")
    assert sample.find_clip(d, "Speaker 1") == sample.Clip(d / "others.wav", 2.0, 8.0)


def test_many_replays_keep_the_typed_name(tmp_path):
    clip = sample.Clip(tmp_path / "others.wav", 0.0, 3.0)
    replays = 50  # no cap: the name typed before the last replay is kept
    answers = iter(
        ["button returned:Play sample, text returned:Anna"] * replays
        + ["button returned:Save, text returned:Anna"]
    )
    shown: list[str] = []
    played: list = []

    def ask(script: str) -> str:
        shown.append(script)
        return next(answers)

    assert ask_speaker_name("Speaker 1", clip, ask, played.append) == "Anna"
    assert len(played) == replays
    assert all('default answer "Anna"' in s for s in shown[1:])


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
    """Plays until killed (``block``) or at once."""

    pid = 1234

    def __init__(self, args, block: bool = False, **_kw):
        self.args = args
        self.thread = threading.current_thread()
        self.existed = Path(args[-1]).exists()
        self.dir_mode = Path(args[-1]).parent.stat().st_mode & 0o777
        with wave.open(args[-1], "rb") as w:
            self.seconds = w.getnframes() / w.getframerate()
        self.done = threading.Event()
        if not block:
            self.done.set()
        self.killed = False

    def wait(self, timeout=None):
        if not self.done.wait(5 if timeout is None else timeout):
            raise subprocess.TimeoutExpired(self.args, timeout or 5)
        return 0

    def poll(self):
        return 0 if self.done.is_set() else None

    def kill(self):
        self.killed = True
        self.done.set()


def _start(player: sample.SamplePlayer, clip: sample.Clip) -> threading.Thread:
    t = player.play(clip)
    assert t is not None
    return t


def test_player_plays_off_the_calling_thread_and_deletes_the_clip(tmp_path, caplog):
    d = _session(tmp_path, [(2, 12, "Anna Secret")])
    procs: list[FakeProc] = []

    def popen(args, **kw):
        procs.append(FakeProc(args, **kw))
        return procs[-1]

    player = sample.SamplePlayer(popen=popen, root=tmp_path)
    clip = sample.find_clip(d, "Anna Secret")
    assert clip is not None
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        _start(player, clip).join(5)
    (proc,) = procs
    assert proc.thread is not threading.main_thread()
    assert proc.args[0] == "/usr/bin/afplay" and proc.existed  # never a PATH lookup
    if os.name == "posix":
        assert proc.dir_mode == 0o700
    assert abs(proc.seconds - sample.MAX_S) < 0.01
    folder = Path(proc.args[-1]).parent
    assert folder.name.startswith("rcc-sample-")
    assert list(folder.iterdir()) == []  # deleted after playback
    player.close()
    assert not folder.exists()  # the temp folder goes on quit / exit
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

    player = sample.SamplePlayer(popen=popen, root=tmp_path)
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        _start(player, clip).join(5)
    assert seen and not Path(seen[0]).exists()
    assert any("FileNotFoundError" in r.getMessage() for r in caplog.records)
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]
    player.close()


def test_quick_plays_leave_one_player_and_close_kills_it(tmp_path):
    d = _session(tmp_path, [(0, 3, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip is not None
    procs: list[FakeProc] = []

    def popen(args, **kw):
        procs.append(FakeProc(args, block=True))
        return procs[-1]

    player = sample.SamplePlayer(popen=popen, root=tmp_path)
    threads = [_start(player, clip), _start(player, clip)]
    for _ in range(500):
        if len(procs) == 2:
            break
        threading.Event().wait(0.01)
    assert len(procs) == 2
    assert sum(not p.killed for p in procs) == 1  # the newer one stopped the older one
    player.close()
    assert all(p.killed for p in procs)  # quitting stops the one still playing
    for t in threads:
        t.join(5)
        assert not t.is_alive()
    assert not list(tmp_path.glob("rcc-sample-*"))
    assert player.play(clip) is None  # closed


def _wait_for(cond) -> None:
    for _ in range(500):
        if cond():
            return
        threading.Event().wait(0.01)
    raise AssertionError("timed out")


def test_a_session_starting_during_playback_kills_the_sample(tmp_path):
    d = _session(tmp_path, [(0, 3, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip is not None
    idle = [True]
    procs: list[FakeProc] = []

    def popen(args, **kw):
        procs.append(FakeProc(args, block=True))
        return procs[-1]

    player = sample.SamplePlayer(popen=popen, allowed=lambda: idle[0], root=tmp_path)
    t = _start(player, clip)
    _wait_for(lambda: len(procs) == 1)
    idle[0] = False  # the engine starts recording: the player's own poll kills afplay
    t.join(5)
    assert not t.is_alive() and procs[0].killed
    assert player.play(clip) is None  # no new sample while recording

    idle[0] = True  # the menu's stop() (main thread) kills without closing the player
    t = _start(player, clip)
    _wait_for(lambda: len(procs) == 2)
    player.stop()
    t.join(5)
    assert not t.is_alive() and procs[1].killed
    _start(player, clip)
    _wait_for(lambda: len(procs) == 3)  # later plays still work
    player.close()
    assert procs[2].killed


def test_stale_sample_folders_are_swept_at_start(tmp_path):
    old = tmp_path / "rcc-sample-old"
    old.mkdir()
    (old / "sample-1.wav").write_bytes(b"x")
    other = tmp_path / "keep-me"
    other.mkdir()
    sample.SamplePlayer(root=tmp_path).close()
    assert not old.exists() and other.exists()


def test_the_playing_sample_is_our_own_process_for_detection(tmp_path):
    from test_engine import Harness

    from recandcc.models import AudioProcess, Mode

    h = Harness(Mode.AUTO, ignore_apps=[])  # the user removed afplay from ignore_apps
    afplay = AudioProcess(
        pid=4321, bundle_id="", name="afplay", input_active=False, output_active=True
    )
    h.mon.procs = [afplay]

    d = _session(tmp_path, [(0, 6, "Speaker 1")])
    clip = sample.find_clip(d, "Speaker 1")
    assert clip is not None
    release = threading.Event()

    class Proc(FakeProc):
        pid = 4321

        def wait(self, timeout=None):
            if not release.wait(5 if timeout is None else timeout):
                raise subprocess.TimeoutExpired(self.args, timeout or 5)
            return 0

    player = sample.SamplePlayer(
        popen=lambda a, **k: Proc(a), own_pid=h.eng.set_own_pid, root=tmp_path
    )
    t = _start(player, clip)
    _wait_for(lambda: 4321 in h.eng.own_pids)
    h.run(10)
    assert h.rec.starts == []  # never recorded while the sample plays
    release.set()
    t.join(5)
    assert 4321 not in h.eng.own_pids and 999 in h.eng.own_pids  # unregistered after
    h.run(10)
    assert h.rec.starts  # the same pid is an ordinary app again
    player.close()
