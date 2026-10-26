"""End-to-end transcription with a real Whisper model and macOS `say` audio.

Run: uv run pytest -m slow tests/test_transcribe_integration.py -v
Model: RECANDCC_TEST_MODEL (default "small").
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.transcribe import Transcriber
from recandcc.transcribe.merge import similarity

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (shutil.which("say") and shutil.which("afconvert")), reason="needs macOS say/afconvert"
    ),
]

EN_OTHERS = "The quarterly budget review is scheduled for next Thursday afternoon."
RU_OTHERS = "Давайте обсудим новый план разработки на следующую неделю."
EN_ME = "I will prepare the slides and send them to everyone tomorrow morning."


def tts(text: str, voice: str, tmp: Path) -> np.ndarray:
    aiff, wav = tmp / f"{voice}{abs(hash(text))}.aiff", tmp / f"{voice}{abs(hash(text))}.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEF32@16000", "-c", "1", str(aiff), str(wav)], check=True
    )
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SAMPLE_RATE
    return audio


def place(total_s: float, parts: list[tuple[float, np.ndarray]]) -> np.ndarray:
    out = np.zeros(int(total_s * SAMPLE_RATE), dtype=np.float32)
    for at, a in parts:
        i = int(at * SAMPLE_RATE)
        out[i : i + len(a)] += a
    return out


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> SessionInfo:
    tmp = tmp_path_factory.mktemp("session")
    en_o, ru_o, en_me = (
        tts(EN_OTHERS, "Samantha", tmp),
        tts(RU_OTHERS, "Milena", tmp),
        tts(EN_ME, "Samantha", tmp),
    )
    others = place(40, [(1.0, en_o), (14.0, ru_o)])
    # Me: own sentence later, plus a quiet delayed "echo" of the Others English sentence.
    me = place(40, [(1.15, 0.3 * en_o), (28.0, en_me)])
    sf.write(tmp / "others.wav", others, SAMPLE_RATE, subtype="FLOAT")
    sf.write(tmp / "me.wav", me, SAMPLE_RATE, subtype="FLOAT")
    s = SessionInfo(
        id="itest",
        directory=tmp,
        started_at=datetime.now(),
        trigger_kind=TriggerKind.MANUAL,
        tracks={Track.ME: "me.wav", Track.OTHERS: "others.wav"},
    )
    s.save()
    return s


def test_end_to_end(session):
    model = os.environ.get("RECANDCC_TEST_MODEL", "small")
    t = Transcriber(TranscriptionConfig(model=model))
    try:
        segs = t.transcribe_session(session)
    finally:
        t.close()
    for s in segs:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.speaker:6} {s.language}: {s.text}")

    others = [s for s in segs if s.speaker == "Others"]
    me = [s for s in segs if s.speaker == "Me"]
    en = " ".join(s.text for s in others if s.language == "en")
    ru = " ".join(s.text for s in others if s.language == "ru")
    assert {s.language for s in others} == {"en", "ru"}
    assert similarity(EN_OTHERS, en) >= 0.7, en
    assert similarity(RU_OTHERS, ru) >= 0.6, ru
    # echo removed: Me only has its own sentence, starting around 28 s
    assert me and all(s.start > 20 for s in me), me
    assert similarity(EN_ME, " ".join(s.text for s in me)) >= 0.7
    assert SessionInfo.load(session.directory).transcribed
    md = (session.directory / "transcript.md").read_text()
    assert "Others:**" in md and "Me:**" in md
