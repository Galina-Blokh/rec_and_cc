"""Hebrew end to end: real Whisper + diarization on offline macOS `say` audio (he/en/ru).

Run: uv run pytest -m slow tests/test_hebrew_integration.py -v -s
Uses only models already on disk (skips otherwise) with the network blocked; prints the
language-ID accuracy and timings. Model: RECANDCC_TEST_MODEL (default "large-v3-turbo").
Only one Hebrew `say` voice exists (Carmit), so the dialogue's second speaker is Carmit
pitched down by resampling.

Stories: US-19.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind
from recandcc.transcribe import Transcriber, diarize
from recandcc.transcribe.engine import LANGUAGE_ID_MODEL
from recandcc.transcribe.gpu import hf_cache_folder, mlx_repo, use_gpu
from recandcc.transcribe.live import LiveTranscriber
from recandcc.transcribe.merge import similarity
from recandcc.transcribe.output import PDI, RLI, RLM
from recandcc.transcribe.revisions import FASTER_WHISPER_REPOS, ct2_repo, revision

VOICES = {"he": "Carmit", "en": "Samantha", "ru": "Milena"}
IVRIT = "ivrit-ai/whisper-large-v3-turbo-ct2"
MODEL = os.environ.get("RECANDCC_TEST_MODEL", "large-v3-turbo")
LANGS = ["en", "ru", "he"]


def _voices() -> set[str]:
    if sys.platform != "darwin":
        return set()
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    return {line.split()[0] for line in out.splitlines() if line.strip()}


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS say"),
    pytest.mark.skipif(
        not set(VOICES.values()) <= _voices(), reason=f"needs the say voices {VOICES}"
    ),
]

# he -> en -> ru -> he, two sentences per block (8 regions)
SWITCHING = [
    ("he", "שלום לכולם, היום נדבר על התוכנית לרבעון הבא."),
    ("he", "אני חושבת שצריך להעביר את הפגישה ליום שלישי."),
    ("en", "The quarterly budget review is scheduled for next Thursday afternoon."),
    ("en", "Can you send me the latest version of the design document please?"),
    ("ru", "Давайте обсудим новый план разработки на следующую неделю."),
    ("ru", "Отправьте, пожалуйста, отчёт до конца рабочего дня."),
    ("he", "הלקוחות היו מרוצים מאוד מהגרסה החדשה של המוצר."),
    ("he", "בואו נסכם את המשימות ונשלח אותן לכל הצוות."),
]
# (speaker, text): a Hebrew dialogue, "l" = Carmit pitched down
DIALOGUE = [
    ("h", "בוקר טוב, סיימת כבר לבדוק את הדוח החודשי?"),
    ("l", "כמעט, נשארו לי רק שני פרקים לקרוא."),
    ("h", "מצוין, אז נוכל לשלוח אותו ללקוח מחר בבוקר."),
    ("l", "כן, אבל קודם אני רוצה לתקן את הטבלה בעמוד השלישי."),
    ("h", "בסדר גמור, תעדכן אותי כשתסיים."),
    ("l", "בטח, אני אשלח לך הודעה אחרי הצהריים."),
]
HEBREW = re.compile("[֐-׿]")


def tts(voice: str, text: str, tmp: Path) -> np.ndarray:
    wav = tmp / f"{voice}-{abs(hash(text))}.wav"
    subprocess.run(
        ["say", "-v", voice, "-o", str(wav), "--data-format=LEF32@16000", text], check=True
    )
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == SAMPLE_RATE
    return audio


def lower(audio: np.ndarray, factor: float = 1.3) -> np.ndarray:
    """``audio`` played ``factor`` times slower: pitch down by the same factor."""
    n = int(len(audio) * factor)
    return np.interp(np.arange(n) / factor, np.arange(len(audio)), audio).astype(np.float32)


def build(clips: list[tuple[str, np.ndarray]], gap_s: float = 1.2):
    """Clips separated by ``gap_s`` pauses + their true spans [(start, end, tag)]."""
    parts, truth, t = [np.zeros(SAMPLE_RATE // 2, np.float32)], [], 0.5
    gap = np.zeros(int(gap_s * SAMPLE_RATE), np.float32)
    for tag, a in clips:
        parts += [a, gap]
        truth.append((t, t + len(a) / SAMPLE_RATE, tag))
        t += (len(a) + len(gap)) / SAMPLE_RATE
    return np.concatenate(parts), truth


def make_session(tmp: Path, others: np.ndarray, sid: str) -> SessionInfo:
    tmp.mkdir(parents=True, exist_ok=True)
    sf.write(tmp / "others.wav", others, SAMPLE_RATE, subtype="FLOAT")
    sf.write(tmp / "me.wav", np.zeros_like(others), SAMPLE_RATE, subtype="FLOAT")
    started = datetime.now().astimezone()
    s = SessionInfo(
        id=sid,
        directory=tmp,
        started_at=started,
        ended_at=started + timedelta(seconds=len(others) / SAMPLE_RATE),
        trigger_kind=TriggerKind.CALL,
        tracks={Track.ME: "me.wav", Track.OTHERS: "others.wav"},
    )
    s.save()
    return s


def majority(truth, segs: list[Segment], key) -> list[str | None]:
    """Per true span, the ``key`` of the segments overlapping it most."""
    out = []
    for s, e, _ in truth:
        ov = Counter()
        for g in segs:
            if (o := min(e, g.end) - max(s, g.start)) > 0:
                ov[key(g)] += o
        out.append(ov.most_common(1)[0][0] if ov else None)
    return out


def cached(repo: str) -> bool:
    """Weights of ``repo`` (at its pinned commit) are in the Hugging Face cache."""
    snaps = hf_cache_folder(repo) / "snapshots"
    rev = revision(repo)
    dirs = [snaps / rev] if rev else list(snaps.glob("*"))
    return any(any(d.glob(p)) for d in dirs for p in ("model.bin", "*.safetensors", "*.npz"))


@pytest.fixture(scope="module")
def offline():
    """Never download: skip without the cached models, then block every network path."""
    from huggingface_hub import constants

    ct2 = FASTER_WHISPER_REPOS.get
    gpu = use_gpu(TranscriptionConfig(model=MODEL))
    need = [mlx_repo(MODEL) if gpu else ct2(MODEL, MODEL), ct2(LANGUAGE_ID_MODEL, "")]
    if missing := [r for r in need if not cached(r)]:
        pytest.skip(f"models not cached: {missing}")
    if diarize.model_path(download=False) is None:
        pytest.skip("speaker model not cached")
    mp = pytest.MonkeyPatch()
    mp.setenv("HF_HUB_OFFLINE", "1")
    mp.setattr(constants, "HF_HUB_OFFLINE", True)  # read once at import; undone below
    for var in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY"):
        mp.setenv(var, "http://127.0.0.1:9")

    def no_network(*a, **k):
        raise AssertionError("network access in an offline test")

    mp.setattr(diarize.urllib.request, "urlopen", no_network)
    yield
    mp.undo()


@pytest.fixture(scope="module")
def switching(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("say")
    return build([(lang, tts(VOICES[lang], text, tmp)) for lang, text in SWITCHING])


def check_switching(truth, segs: list[Segment], directory: Path, label: str, elapsed: float):
    for s in segs:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.speaker:9} {s.language}: {len(s.text)} chars")
    got = majority(truth, segs, lambda g: g.language)
    ok = sum(g == lang for g, (*_, lang) in zip(got, truth, strict=True))
    print(f"{label}: language ID {ok}/{len(truth)} {got}; {elapsed:.1f}s")
    assert {s.language for s in segs} <= set(LANGS)  # never Arabic, Yiddish, ...
    assert ok >= len(truth) - 1
    he = " ".join(s.text for s in segs if s.language == "he")
    want = " ".join(text for lang, text in SWITCHING if lang == "he")
    print(f"{label}: Hebrew similarity {similarity(want, he):.2f}")
    assert similarity(want, he) >= 0.5

    md = (directory / "transcript.md").read_text(encoding="utf-8")
    isolated = re.findall(f"{RLI}([^{PDI}]*){PDI}", md)
    assert isolated and all(HEBREW.search(t) for t in isolated)
    assert not HEBREW.search(re.sub(f"{RLI}[^{PDI}]*{PDI}", "", md))  # all Hebrew isolated
    srt = (directory / "transcript.srt").read_text(encoding="utf-8").splitlines()
    texts = [line for line in srt if ":" in line and "-->" not in line]
    assert texts and all(line.startswith(RLM) == bool(HEBREW.search(line)) for line in texts)
    data = json.loads((directory / "transcript.json").read_text(encoding="utf-8"))
    raw = [s["text"] for s in data["segments"]]
    assert any(HEBREW.search(t) for t in raw)
    assert not any(c in t for t in raw for c in (RLI, PDI, RLM))  # raw text kept


def run_final(session: SessionInfo, cfg: TranscriptionConfig, truth=()):
    """(segments, seconds) of a post-session transcription; prints the language-ID
    confidence of each span in ``truth`` (small ID model / main model)."""
    t = Transcriber(cfg)
    t.diarize_wait_s = 120.0
    try:
        t.ensure_loaded()  # don't time model loading
        t0 = time.monotonic()
        segs = t.transcribe_session(session)
        elapsed = time.monotonic() - t0
        audio = sf.read(session.directory / "others.wav", dtype="float32")[0]
        for s, e, want in truth:
            clip = audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
            (a, ca), (b, cb) = t.detect_language(clip), t.detect_language(clip, main=True)
            print(f"  {want} {e - s:4.1f}s: id model {a} {ca:.3f}, main {b} {cb:.3f}")
        return segs, elapsed
    finally:
        t.close()


def test_language_switching_he_en_ru_he(offline, switching, tmp_path):
    audio, truth = switching
    session = make_session(tmp_path, audio, "he-switch")
    segs, elapsed = run_final(
        session, TranscriptionConfig(model=MODEL, languages=LANGS, diarize=False), truth
    )
    check_switching(truth, segs, tmp_path, "final", elapsed)


def test_live_language_switching(offline, switching, tmp_path):
    audio, truth = switching
    session = make_session(tmp_path, audio, "he-live")
    session.ended_at = None
    t = Transcriber(TranscriptionConfig(model=MODEL, languages=LANGS, diarize=False))
    t.ensure_loaded()
    shown, lock = [], threading.Lock()

    def on_segments(segs):
        with lock:
            shown.extend(segs)

    live, block = LiveTranscriber(t, session, on_segments), SAMPLE_RATE // 10
    silence = np.zeros(block, np.float32)
    try:
        t0 = time.monotonic()
        for i in range(0, len(audio), block):
            live.feed(Track.OTHERS, i, audio[i : i + block])
            live.feed(Track.ME, i, silence[: len(audio[i : i + block])])
            time.sleep(block / SAMPLE_RATE / 10)  # 10x real time
        session.ended_at = session.started_at + timedelta(seconds=len(audio) / SAMPLE_RATE)
        segs = live.finish(session, timeout=120)
        elapsed = time.monotonic() - t0
    finally:
        t.close()
    with lock:
        assert any(s.language == "he" and HEBREW.search(s.text) for s in shown)  # live captions
    check_switching(truth, segs, tmp_path, "live", elapsed)


def test_hebrew_dialogue_gets_two_speakers(offline, tmp_path):
    clips = []
    for who, text in DIALOGUE:
        a = tts("Carmit", text, tmp_path)
        clips.append((who, lower(a) if who == "l" else a))
    audio, truth = build(clips, gap_s=0.9)
    session = make_session(tmp_path / "sess", audio, "he-dialogue")
    segs, elapsed = run_final(session, TranscriptionConfig(model=MODEL, languages=LANGS))
    for s in segs:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.speaker:9} {s.language}: {len(s.text)} chars")
    labels = majority(truth, segs, lambda g: g.speaker)
    print(f"dialogue: speakers {labels}; {elapsed:.1f}s")
    assert {s.speaker for s in segs} == {"Speaker 1", "Speaker 2"}
    pairs = list(zip(labels, [w for *_, w in truth], strict=True))
    label_of = {w: Counter(lb for lb, ww in pairs if ww == w).most_common(1)[0][0] for w in "hl"}
    assert label_of["h"] != label_of["l"]
    assert sum(lb == label_of[w] for lb, w in pairs) >= len(truth) - 1
    assert sum(s.language == "he" for s in segs) >= len(segs) - 1


def test_hebrew_routed_to_ivrit_model(offline, switching, tmp_path):
    cfg = TranscriptionConfig(
        model=MODEL, languages=LANGS, diarize=False, language_models={"he": IVRIT}
    )
    repo = mlx_repo(IVRIT) if use_gpu(cfg) else ct2_repo(IVRIT)
    if not cached(repo):
        pytest.skip(f"{repo} not cached")
    audio, truth = switching
    session = make_session(tmp_path, audio, "he-ivrit")
    t = Transcriber(cfg)
    used: list[tuple[str, bool]] = []
    model_for = t._model_for

    def spy(language: str, wait: bool = True):
        m = model_for(language, wait)
        used.append((language, m is not t._model))
        return m

    t._model_for = spy  # type: ignore[method-assign]
    t.diarize_wait_s = 120.0
    try:
        t0 = time.monotonic()
        segs = t.transcribe_session(session)
        elapsed = time.monotonic() - t0
    finally:
        t.close()
    print(f"routed: {sorted(Counter(used).items())}; loads {session.metrics.get('models')}")
    assert used and all(routed == (lang == "he") for lang, routed in used)
    assert any(k.startswith(IVRIT) for k in session.metrics["models"]["load_s"])
    check_switching(truth, segs, tmp_path, "ivrit", elapsed)
