"""Live transcription with real models and macOS `say` audio, fed like a recording.

Run: uv run pytest -m slow tests/test_live_integration.py -v -s
Model: RECANDCC_TEST_MODEL (default "large-v3-turbo", the app default).

Stories: US-11.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta

import pytest
import soundfile as sf
from test_transcribe_integration import EN_ME, EN_OTHERS, RU_OTHERS, place, tts

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.transcribe import Transcriber
from recandcc.transcribe.live import LiveTranscriber
from recandcc.transcribe.merge import similarity

pytestmark = pytest.mark.slow

SPEEDUP = 10.0  # feed at 10x real time
BLOCK = SAMPLE_RATE // 10  # 100 ms


def test_live_end_to_end(tmp_path):
    en_o, ru_o, en_me = (
        tts(EN_OTHERS, "Samantha", tmp_path),
        tts(RU_OTHERS, "Milena", tmp_path),
        tts(EN_ME, "Samantha", tmp_path),
    )
    total = 40.0
    others = place(total, [(1.0, en_o), (12.0, ru_o)])
    me = place(total, [(1.15, 0.3 * en_o), (24.0, en_me)])  # echo + own sentence
    session = SessionInfo(
        id="live-itest",
        directory=tmp_path / "sess",
        started_at=datetime.now().astimezone(),
        trigger_kind=TriggerKind.CALL,
        tracks={Track.ME: "me.wav", Track.OTHERS: "others.wav"},
    )
    session.directory.mkdir()
    sf.write(session.directory / "others.wav", others, SAMPLE_RATE, subtype="FLOAT")
    sf.write(session.directory / "me.wav", me, SAMPLE_RATE, subtype="FLOAT")
    session.save()

    model = os.environ.get("RECANDCC_TEST_MODEL", "large-v3-turbo")
    # Two voices on the others track; diarization has its own test (test_diarize_integration).
    t = Transcriber(TranscriptionConfig(model=model, diarize=False, live_speaker_labels=False))
    t.ensure_loaded()  # don't count model loading
    got, lock = [], threading.Lock()

    def on_segments(segs):
        with lock:
            got.extend(segs)

    live = LiveTranscriber(t, session, on_segments)
    try:
        for i in range(0, int(total * SAMPLE_RATE), BLOCK):
            live.feed(Track.OTHERS, i, others[i : i + BLOCK])
            live.feed(Track.ME, i, me[i : i + BLOCK])
            time.sleep(BLOCK / SAMPLE_RATE / SPEEDUP)
        time.sleep(3.0)  # what a few seconds of real time after the last phrase gives
        with lock:
            before = list(got)
        session.ended_at = session.started_at + timedelta(seconds=total)
        t0 = time.monotonic()
        segs = live.finish(session, timeout=120)
        finish_s = time.monotonic() - t0
    finally:
        t.close()
    for s in segs:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.speaker:6} {s.language}: {s.text}")
    print(f"live before finish: {len(before)} segments; finish() took {finish_s:.2f}s")

    # live captions label system audio "Speaker N" (or "Others"): pick it as "not Me"
    others_live = [s for s in before if s.speaker != "Me"]
    assert all(s.speaker == "Others" or s.speaker.startswith("Speaker ") for s in others_live)
    assert similarity(EN_OTHERS, " ".join(s.text for s in others_live if s.language == "en")) > 0.7
    assert similarity(RU_OTHERS, " ".join(s.text for s in others_live if s.language == "ru")) > 0.6
    # echo of the Others sentence on the mic is not shown live
    assert all(s.start > 20 for s in before if s.speaker == "Me")

    o = [s for s in segs if s.speaker == "Others"]
    m = [s for s in segs if s.speaker == "Me"]
    assert {s.language for s in o} == {"en", "ru"}
    assert similarity(EN_OTHERS, " ".join(s.text for s in o if s.language == "en")) >= 0.7
    assert similarity(RU_OTHERS, " ".join(s.text for s in o if s.language == "ru")) >= 0.6
    assert m and all(s.start > 20 for s in m)
    assert similarity(EN_ME, " ".join(s.text for s in m)) >= 0.7
    assert 0.5 < o[0].start < 1.5 and 11.5 < [s for s in o if s.language == "ru"][0].start < 13
    assert SessionInfo.load(session.directory).transcribed
    md = (session.directory / "transcript.md").read_text(encoding="utf-8")
    assert "Others:**" in md and "Me:**" in md and "in progress" not in md
    assert finish_s < 5.0  # only the tail was left
