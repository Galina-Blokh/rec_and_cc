"""Per-language model routing (transcription.language_models): Hebrew regions go to a
Hebrew-tuned Whisper, loaded once on first use; everything else to the main model.

Stories: US-19.
"""

from __future__ import annotations

import logging
import sys
import threading
import types

import numpy as np
import pytest
from test_live import FakeTranscriber, Feeder, burst, manual, silence
from test_live import make_session as make_live_session
from test_transcribe import HebrewFakeModel, _install, _write_tracks, make_session

from recandcc.config import TranscriptionConfig, load_config, normalize_language_models
from recandcc.models import SAMPLE_RATE, Track
from recandcc.transcribe import gpu
from recandcc.transcribe.engine import Transcriber

IVRIT = "ivrit-ai/whisper-large-v3-turbo-ct2"


def _cfg(**kw) -> TranscriptionConfig:
    kw.setdefault("languages", ["en", "ru", "he"])
    kw.setdefault("language_models", {"he": IVRIT})
    return TranscriptionConfig(model="fake", workers=2, **kw)


def _batch(tmp_path, monkeypatch, cfg, load):
    """Transcribe the two-track fake session; ``load`` replaces Transcriber._load (routed)."""
    import recandcc.transcribe.engine as eng

    regions = _write_tracks(tmp_path)
    monkeypatch.setattr(eng, "find_speech", lambda a: regions)
    t = Transcriber(cfg)
    main = HebrewFakeModel()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: _install(t, main, HebrewFakeModel()))
    monkeypatch.setattr(t, "_load", load)
    session = make_session(tmp_path, {Track.ME: "me.wav", Track.OTHERS: "others.wav"})
    segs = t.transcribe_session(session)
    t.close()
    return segs, main


def test_hebrew_regions_go_to_the_routed_model(tmp_path, monkeypatch):
    routed, loads = HebrewFakeModel(), []

    def load(name):
        loads.append(name)
        return routed

    segs, main = _batch(tmp_path, monkeypatch, _cfg(), load)
    assert loads == [IVRIT]  # loaded lazily, once, although both tracks had Hebrew chunks
    assert [c["language"] for c in routed.calls] == ["he", "he"]
    assert {c["language"] for c in main.calls} == {"en"}
    assert [(s.start, s.language, s.text) for s in segs][1] == (5.0, "he", "שלום")


def test_routed_model_not_loaded_without_its_language(tmp_path, monkeypatch):
    loads = []
    cfg = _cfg(languages=["en"])  # no Hebrew detected -> the Hebrew model is never needed
    _, main = _batch(tmp_path, monkeypatch, cfg, lambda name: loads.append(name))
    assert loads == [] and main.calls and {c["language"] for c in main.calls} == {"en"}


def test_routed_model_that_fails_to_load_falls_back(tmp_path, monkeypatch, caplog):
    loads = []

    def load(name):
        loads.append(name)
        raise RuntimeError("no such model")

    with caplog.at_level(logging.INFO, logger="recandcc"):
        segs, main = _batch(tmp_path, monkeypatch, _cfg(), load)
    assert loads == [IVRIT]  # the failure is remembered: not retried for every region
    assert {c["language"] for c in main.calls} == {"en", "he"}
    assert any(s.text == "שלום" for s in segs)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(IVRIT in r.getMessage() and "using fake" in r.getMessage() for r in warnings)
    assert "שלום" not in caplog.text and "word" not in caplog.text  # no transcript text


def test_unpinned_routed_repo_is_not_downloaded(monkeypatch, tmp_path):
    downloads = []

    def download_model(name, local_files_only=False, revision=None):
        downloads.append((name, revision, local_files_only))
        raise FileNotFoundError(name)  # not cached

    fake_utils = types.SimpleNamespace(download_model=download_model)
    fake = types.SimpleNamespace(WhisperModel=object, utils=fake_utils)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", fake_utils)
    t = Transcriber(_cfg(backend="faster-whisper", language_models={"he": "someone/custom"}))
    main = HebrewFakeModel()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: main)
    assert t._model_for("he") is main  # refused (allow_unpinned_models off) -> default model
    assert downloads == [("someone/custom", None, True)]  # local lookup only, no download


def test_live_hebrew_region_goes_to_the_routed_model(tmp_path):
    class RoutingTranscriber(FakeTranscriber):
        """Live fakes for VAD and language ID, but the real per-language model choice."""

        transcribe_audio = Transcriber.transcribe_audio

        def __init__(self):
            super().__init__(_cfg())
            self.main, self.routed, self.loads = HebrewFakeModel(), HebrewFakeModel(), []

        def _ensure_loaded(self):
            return self.main

        def _load(self, name):
            self.loads.append(name)
            return self.routed

    tr = RoutingTranscriber()
    tr.language = lambda a: ("he", 0.95) if len(a) > 2 * SAMPLE_RATE else ("en", 0.95)
    emitted = []
    live = manual(tr, make_live_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([burst(3), silence(1.5), burst(1), silence(1.5), burst(3), silence(1.5)]))
    live._step(False)
    assert tr.loads == [IVRIT]
    assert [c["language"] for c in tr.routed.calls] == ["he", "he"]
    assert [c["language"] for c in tr.main.calls] == ["en"]
    assert [s.text for s in emitted] == ["שלום", "word", "שלום"]


def test_mlx_models_share_one_gpu_thread_and_keep_their_weights(monkeypatch, tmp_path):
    loads = []

    class Holder:  # mlx_whisper.transcribe.ModelHolder: caches one model only
        model = model_path = None

        @classmethod
        def get_model(cls, path, dtype):
            if cls.model is None or path != cls.model_path:
                loads.append(path)
                cls.model, cls.model_path = f"weights:{path}", path
            return cls.model

    def transcribe(audio, path_or_hf_repo, **kw):
        model = Holder.get_model(path_or_hf_repo, "f16")
        text = f"{model} {threading.get_ident()}"
        return {"segments": [{"start": 0.0, "end": 1.0, "text": text}]}

    mlx = types.ModuleType("mlx")
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", types.SimpleNamespace(float16="f16"))
    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=transcribe))
    monkeypatch.setitem(
        sys.modules, "mlx_whisper.transcribe", types.SimpleNamespace(ModelHolder=Holder)
    )
    (tmp_path / "main").mkdir()
    (tmp_path / "he").mkdir()
    main = gpu.MlxWhisper(str(tmp_path / "main"))
    hebrew = gpu.MlxWhisper(str(tmp_path / "he"), share=main)
    main.load()
    hebrew.load()
    audio = np.zeros(SAMPLE_RATE, np.float32)
    texts = [m.transcribe(audio, "en")[0][2] for m in (main, hebrew, main, hebrew)]
    assert len(loads) == 2  # alternating models never reloads weights
    assert [t.split()[0] for t in texts] == [f"weights:{tmp_path / n}" for n in ("main", "he") * 2]
    assert len({t.split()[1] for t in texts}) == 1  # one GPU thread drives both models
    hebrew.close()
    assert main.transcribe(audio, "en")  # closing the routed model keeps the shared thread
    main.close()


def test_normalize_language_models(caplog):
    with caplog.at_level(logging.WARNING, logger="recandcc.config"):
        out = normalize_language_models({"IW": f" {IVRIT} ", "xx": "m", "en": "", "ru": 5})
    assert out == {"he": IVRIT}
    assert caplog.text.count("ignoring") == 3
    assert normalize_language_models(["he"]) == {}


@pytest.mark.parametrize(
    ("toml", "expected", "unused"),
    [
        (f'languages = ["en", "he"]\nlanguage_models = {{ iw = "{IVRIT}" }}', {"he": IVRIT}, 0),
        (f'languages = ["en"]\nlanguage_models = {{ he = "{IVRIT}" }}', {"he": IVRIT}, 1),
        ('language_models = { klingon = "x" }', {}, 0),
        ("", {}, 0),
    ],
)
def test_load_config_validates_language_models(tmp_path, caplog, toml, expected, unused):
    path = tmp_path / "config.toml"
    path.write_text(f"[transcription]\n{toml}\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc.config"):
        cfg = load_config(path)
    assert cfg.transcription.language_models == expected
    assert caplog.text.count("is unused") == unused


def _cached(repo: str) -> bool:
    from huggingface_hub import snapshot_download

    from recandcc.transcribe.revisions import revision

    try:
        snapshot_download(repo, revision=revision(repo), local_files_only=True)
    except Exception:  # noqa: BLE001 - not on disk (or unreadable): skip, never download
        return False
    return True


@pytest.mark.slow
def test_real_ivrit_model_transcribes_hebrew(tmp_path):
    import shutil
    import subprocess

    import soundfile as sf

    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("needs macOS say/afconvert")
    cfg = _cfg(languages=["he"])
    t = Transcriber(cfg)
    repo = gpu.mlx_repo(IVRIT) if t.gpu else IVRIT
    if not _cached(repo):
        pytest.skip(f"{repo} not cached (download it once to run this test)")
    aiff, wav = tmp_path / "he.aiff", tmp_path / "he.wav"
    say = subprocess.run(["say", "-v", "Carmit", "-o", str(aiff), "שלום, מה שלומך היום?"])
    if say.returncode != 0:
        pytest.skip("Hebrew voice Carmit not installed")
    afconvert = ["afconvert", "-f", "WAVE", "-d", "LEF32@16000", "-c", "1"]
    subprocess.run([*afconvert, str(aiff), str(wav)], check=True)
    audio, _ = sf.read(wav, dtype="float32")
    try:
        segs = t.transcribe_audio(audio, 0.0, "Others", "he")
        assert t._routed[IVRIT] is not None  # the ivrit.ai model, not the fallback
    finally:
        t.close()
    assert any("֐" <= ch <= "׿" for s in segs for ch in s.text)
