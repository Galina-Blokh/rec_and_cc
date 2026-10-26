"""Speaker diarization end to end: real speaker model, real Whisper, macOS `say` voices.

Run: uv run pytest -m slow tests/test_diarize_integration.py -v -s
The speaker model is downloaded once into the cache dir; after that the test runs with all
network access blocked (proxy to a closed port + urlopen patched to fail).
Whisper model: RECANDCC_TEST_MODEL (default "large-v3-turbo", the app default, so it is cached).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, SessionInfo, Track, TriggerKind
from recandcc.transcribe import Transcriber, diarize
from recandcc.transcribe.diarize import Diarizer
from recandcc.transcribe.live import LiveTranscriber

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not shutil.which("say"), reason="needs macOS say"),
]

# (voice, text): three clearly different voices taking turns, English and Russian.
PHRASES = [
    ("Samantha", "The quarterly numbers look better than we expected this time."),
    ("Daniel", "I think we should move the launch to next Tuesday afternoon."),
    ("Milena", "Давайте проверим результаты тестирования ещё раз."),
    ("Samantha", "Can you send me the latest version of the design document please?"),
    ("Daniel", "We still need to hire two more engineers for the platform team."),
    ("Milena", "Мне кажется, этот вариант гораздо лучше предыдущего."),
    ("Samantha", "Let's schedule a follow up meeting with the marketing department."),
    ("Daniel", "The customer feedback from the beta program was mostly positive."),
    ("Milena", "Отправьте, пожалуйста, отчёт до конца рабочего дня."),
]
ME_PHRASE = ("Fred", "Sounds good to me, I will take care of the budget.")


def tts(voice: str, text: str, tmp: Path) -> np.ndarray:
    wav = tmp / f"{voice}-{abs(hash(text))}.wav"
    subprocess.run(
        ["say", "-v", voice, "-o", str(wav), "--data-format=LEF32@16000", text], check=True
    )
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SAMPLE_RATE
    return audio


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    """others.wav with PHRASES separated by 0.7-1.2 s pauses + their true time spans."""
    tmp = tmp_path_factory.mktemp("voices")
    rng = np.random.default_rng(1)
    parts, truth, t = [np.zeros(SAMPLE_RATE // 2, np.float32)], [], 0.5
    for voice, text in PHRASES:
        a = tts(voice, text, tmp)
        parts.append(a)
        truth.append((t, t + len(a) / SAMPLE_RATE, voice))
        gap = int(rng.uniform(0.7, 1.2) * SAMPLE_RATE)
        parts.append(np.zeros(gap, np.float32))
        t += (len(a) + gap) / SAMPLE_RATE
    me = tts(*ME_PHRASE, tmp)
    return np.concatenate(parts), truth, me


@pytest.fixture(scope="module")
def offline():
    """Download the speaker model once (if needed), then block the network."""
    assert diarize.model_path() is not None, "speaker model download failed"
    mp = pytest.MonkeyPatch()
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
        mp.setenv(var, "http://127.0.0.1:9")
    # No HF_HUB_OFFLINE: huggingface_hub reads it once at import, so it would leak into later
    # tests in the same process. The dead proxy + urlopen patch already block all network.

    def no_network(*a, **k):
        raise AssertionError("network access after the model was downloaded")

    mp.setattr(diarize.urllib.request, "urlopen", no_network)
    yield
    mp.undo()


def labels_per_phrase(truth, spans):
    """Majority label (by overlap) of `spans` [(start, end, label)] for each true phrase."""
    out = []
    for s, e, _ in truth:
        ov = Counter()
        for ts, te, label in spans:
            o = min(e, te) - max(s, ts)
            if o > 0:
                ov[label] += o
        out.append(ov.most_common(1)[0][0] if ov else None)
    return out


def check(truth, labels, min_ok: int):
    voices = [v for *_, v in truth]
    # each voice -> its most common label; distinct voices get distinct labels
    mapping = {
        v: Counter(lb for lb, vv in zip(labels, voices, strict=True) if vv == v).most_common(1)[0][
            0
        ]
        for v in set(voices)
    }
    assert len(set(mapping.values())) == len(mapping), mapping
    ok = sum(lb == mapping[v] for lb, v in zip(labels, voices, strict=True))
    changes = sum(a != b for a, b in zip(labels, labels[1:], strict=False))
    print(f"labels {labels}: {ok}/{len(labels)} correct, {changes}/{len(labels) - 1} changes")
    assert ok >= min_ok
    assert changes >= len(labels) - 1 - (len(labels) - min_ok) * 2  # alternation preserved


def test_diarizer_separates_voices(track, offline):
    audio, truth, _ = track
    d = Diarizer()
    try:
        t0 = time.monotonic()
        turns = d.diarize(audio)
        print(f"diarized {len(audio) / SAMPLE_RATE:.0f}s in {time.monotonic() - t0:.2f}s")
    finally:
        d.close()
    assert turns and len({t.speaker for t in turns}) >= 2
    labels = labels_per_phrase(truth, [(t.start, t.end, t.speaker) for t in turns])
    check(truth, labels, min_ok=len(truth) - 1)


def _session(tmp: Path, audio: np.ndarray, me: np.ndarray) -> SessionInfo:
    sf.write(tmp / "others.wav", audio, SAMPLE_RATE, subtype="PCM_16")
    mic = np.zeros_like(audio)
    at = int(len(audio) * 0.45)  # Me speaks once, mid-call, over the silence-free others
    mic[at : at + len(me)] = me[: len(mic) - at]
    sf.write(tmp / "me.wav", mic, SAMPLE_RATE, subtype="PCM_16")
    started = datetime.now()
    s = SessionInfo(
        id="dtest",
        directory=tmp,
        started_at=started,
        ended_at=started + timedelta(seconds=len(audio) / SAMPLE_RATE),
        trigger_kind=TriggerKind.CALL,
        trigger_app="zoom.us",
        tracks={Track.ME: "me.wav", Track.OTHERS: "others.wav"},
    )
    s.save()
    return s


def test_transcript_gets_speaker_labels(track, offline, tmp_path):
    audio, truth, me = track
    session = _session(tmp_path, audio, me)
    model = os.environ.get("RECANDCC_TEST_MODEL", "large-v3-turbo")
    t = Transcriber(TranscriptionConfig(model=model, keep_audio=False))
    try:
        t0 = time.monotonic()
        segs = t.transcribe_session(session)
        print(f"transcribed + diarized in {time.monotonic() - t0:.1f}s")
    finally:
        t.close()
    for s in segs:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.speaker:9} {s.language}: {s.text}")
    assert "Others" not in {s.speaker for s in segs}
    assert {s.speaker for s in segs} >= {"Me", "Speaker 1", "Speaker 2"}
    others = [s for s in segs if s.speaker != "Me"]
    assert others[0].speaker == "Speaker 1"  # numbered by first appearance
    labels = labels_per_phrase(truth, [(s.start, s.end, s.speaker) for s in others])
    check(truth, labels, min_ok=len(truth) - 1)
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:**" in md and "Speaker 2:**" in md and "Me:**" in md
    assert not (tmp_path / "others.wav").exists()  # keep_audio = False, after diarization


def test_live_final_transcript_gets_speaker_labels(track, offline, tmp_path):
    """Live path: captions keep "Others"; finish() returns / writes the labelled transcript."""
    audio, truth, me = track
    session = _session(tmp_path, audio, me)
    session.ended_at = None
    mic, _ = sf.read(tmp_path / "me.wav", dtype="float32")
    model = os.environ.get("RECANDCC_TEST_MODEL", "large-v3-turbo")
    t = Transcriber(TranscriptionConfig(model=model))
    t.ensure_loaded()
    shown, lock = [], threading.Lock()

    def on_segments(segs):
        with lock:
            shown.extend(segs)

    live = LiveTranscriber(t, session, on_segments)
    block, speedup = SAMPLE_RATE // 10, 4.0
    try:
        for i in range(0, len(audio), block):
            live.feed(Track.OTHERS, i, audio[i : i + block])
            live.feed(Track.ME, i, mic[i : i + block])
            time.sleep(block / SAMPLE_RATE / speedup)
        session.ended_at = session.started_at + timedelta(seconds=len(audio) / SAMPLE_RATE)
        t0 = time.monotonic()
        segs = live.finish(session, timeout=120)
        print(f"finish() incl. diarization: {time.monotonic() - t0:.2f}s")
    finally:
        t.close()
    assert {s.speaker for s in shown} <= {"Me", "Others"}  # captions unchanged
    others = [s for s in segs if s.speaker != "Me"]
    labels = labels_per_phrase(truth, [(s.start, s.end, s.speaker) for s in others])
    check(truth, labels, min_ok=len(truth) - 1)
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 3:**" in md and "in progress" not in md
