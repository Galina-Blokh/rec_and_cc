"""Per-region language ID with Hebrew: smoothing rules on fake confidences (fast), and a slow
accuracy check of the real small language-ID model on the synthetic he/en/ru `say` set.

Run slow: uv run pytest -m slow tests/test_language_id_he.py -v -s
Uses only models already on disk (skips otherwise); never downloads.

Stories: US-19.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE
from recandcc.transcribe.engine import LANGUAGE_ID_MODEL, MIN_LANGUAGE_CONFIDENCE
from recandcc.transcribe.merge import language_confidence, smooth_languages
from recandcc.transcribe.revisions import FASTER_WHISPER_REPOS, faster_whisper_revision

ROOT = Path(__file__).resolve().parents[1]


def test_short_uncertain_hebrew_region_inherits_the_neighbouring_language():
    regions = [(0, 4), (4.5, 5.2), (5.7, 9)]
    det = [("ru", 0.99), ("he", 0.40), ("ru", 0.98)]
    assert smooth_languages(regions, det, MIN_LANGUAGE_CONFIDENCE) == ["ru", "ru", "ru"]


def test_short_uncertain_region_becomes_hebrew_from_a_hebrew_neighbour():
    regions = [(0, 3), (3.4, 4.0)]
    det = [("he", 0.97), ("en", 0.50)]
    assert smooth_languages(regions, det, MIN_LANGUAGE_CONFIDENCE) == ["he", "he"]


def test_confident_or_long_hebrew_regions_are_kept():
    regions = [(0, 4), (4.5, 5.2), (5.7, 9), (9.5, 13)]
    det = [("en", 0.99), ("he", 0.92), ("en", 0.95), ("he", 0.60)]  # short+sure; long+unsure
    assert smooth_languages(regions, det, MIN_LANGUAGE_CONFIDENCE) == ["en", "he", "en", "he"]


def test_en_ru_smoothing_is_the_same_with_hebrew_in_the_language_list():
    """For en/ru speech the tiny Hebrew mass is renormalised away, so the languages and the
    smoothed result match whether or not 'he' is configured."""
    regions = [(0, 5), (5.5, 6.0), (10, 15)]
    raw = [
        [("ru", 0.95), ("en", 0.04), ("he", 0.001)],
        [("en", 0.55), ("ru", 0.44), ("he", 0.01)],
        [("en", 0.90), ("ru", 0.09), ("he", 0.01)],
    ]
    with_he = [language_confidence(p, ["en", "ru", "he"]) for p in raw]
    without_he = [language_confidence(p, ["en", "ru"]) for p in raw]
    assert [lang for lang, _ in with_he] == [lang for lang, _ in without_he] == ["ru", "en", "en"]
    expected = ["ru", "ru", "en"]  # the 0.5 s uncertain 'en' takes its ru neighbour
    assert smooth_languages(regions, with_he, MIN_LANGUAGE_CONFIDENCE) == expected
    assert smooth_languages(regions, without_he, MIN_LANGUAGE_CONFIDENCE) == expected


def test_language_id_is_restricted_to_the_configured_languages():
    probs = [("en", 0.2), ("ru", 0.1), ("he", 0.3), ("ar", 0.35), ("yi", 0.05)]
    assert language_confidence(probs, ["en", "ru", "he"])[0] == "he"
    assert language_confidence(probs, ["en", "ru"])[0] == "en"


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS say")
def test_language_id_accuracy_on_the_synthetic_mixed_set(tmp_path):
    from recandcc.transcribe import Transcriber
    from recandcc.transcribe.chunking import find_speech, load_track
    from recandcc.transcribe.gpu import hf_cache_folder

    # Only the pinned language-ID snapshot counts: _load() resolves it with local_files_only and
    # would download it otherwise. The main model is never loaded here.
    repo = FASTER_WHISPER_REPOS.get(LANGUAGE_ID_MODEL, "")
    rev = faster_whisper_revision(LANGUAGE_ID_MODEL)
    if not repo or rev is None:
        pytest.skip("language-ID model has no pinned revision")
    snap = hf_cache_folder(repo) / "snapshots" / rev
    if not ((snap / "model.bin").is_file() and (snap / "tokenizer.json").is_file()):
        pytest.skip("language-ID model not cached")

    spec = importlib.util.spec_from_file_location(
        "rcc_synth_lid", ROOT / "scripts" / "make_synthetic_evalset.py"
    )
    mk = importlib.util.module_from_spec(spec)
    sys.modules["rcc_synth_lid"] = mk
    spec.loader.exec_module(mk)
    out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    clips, skipped = mk.plan(mk.parse_voices(out))
    if skipped:
        pytest.skip(f"say voices missing for clips {skipped}")
    manifest = mk.build_set(tmp_path, clips, mk.say_synth(tmp_path))

    langs = ["en", "ru", "he"]
    tr = Transcriber(TranscriptionConfig(languages=langs))
    tr._lid_model = tr._load(LANGUAGE_ID_MODEL)  # only the language-ID model, offline
    ok = total = 0
    try:
        for clip in manifest["clips"]:
            audio = load_track(tmp_path / clip["audio"])
            for s, e in find_speech(audio):
                votes: dict[str, float] = {}
                for u in clip["utterances"]:
                    ov = min(e, u["end"]) - max(s, u["start"])
                    if ov > 0:
                        votes[u["language"]] = votes.get(u["language"], 0) + ov
                if not votes:
                    continue
                piece = audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)]
                lang, _ = tr._detect(tr._lid_model, piece, langs)
                ok += lang == max(votes, key=votes.__getitem__)
                total += 1
    finally:
        tr.close()
    print(f"language-ID accuracy {ok}/{total}")
    assert total >= 30 and ok / total >= 0.95
