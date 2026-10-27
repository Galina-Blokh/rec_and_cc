"""scripts/make_synthetic_evalset.py: timing and manifest from fake TTS (no `say` needed).

Stories: none.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


mk = _load("rcc_synth", "make_synthetic_evalset.py")
ev = _load("rcc_eval_for_synth", "eval.py")


def fake_synth(voice: str, text: str) -> np.ndarray:
    """0.1 s of silence, then 0.05 s of tone per character, then 0.2 s of silence."""
    tone = 0.5 * np.ones(int(0.05 * mk.SR) * len(text), dtype=np.float32)
    return np.concatenate([np.zeros(1600, np.float32), tone, np.zeros(3200, np.float32)])


def test_parse_voices_keeps_names_with_spaces():
    listing = (
        "Carmit              he_IL    # שלום, שמי כרמית.\n"
        "Reed (English (US)) en_US    # Hello! My name is Reed.\n"
        "Samantha            en_US    # Hello! My name is Samantha.\n"
        "garbage line\n"
    )
    assert mk.parse_voices(listing) == {"Carmit", "Reed (English (US))", "Samantha"}


def test_plan_skips_clips_whose_voice_is_missing():
    clips, skipped = mk.plan(set(mk.VOICES.values()) - {"Carmit"})
    assert skipped and all(any(s == "he" for s, _ in mk.CLIPS[c]) for c in skipped)
    assert all(all(s != "he" for s, _ in parts) for parts in clips.values())
    assert set(clips) | set(skipped) == set(mk.CLIPS)
    assert mk.plan(set()) == ({}, list(mk.CLIPS))


def test_trim_and_layout():
    a = np.array([0, 0, 0.5, 0, -0.5, 0, 0], dtype=np.float32)
    assert mk.trim(a).tolist() == [0.5, 0, -0.5]
    assert mk.trim(np.zeros(5, np.float32)).size == 0
    starts, total = mk.layout([100, 200], sr=1000)
    lead, gap, tail = (round(x * 1000) for x in (mk.LEAD_S, mk.GAP_S, mk.TAIL_S))
    assert starts == [lead, lead + 100 + gap]
    assert total == lead + 100 + gap + 200 + tail
    assert mk.layout([], sr=1000) == ([], lead + tail)


def test_build_clip_times_come_from_the_rendered_lengths():
    parts = [("en1", "hello"), ("ru", "да")]
    audio, utts = mk.build_clip(parts, fake_synth)
    assert [u["language"] for u in utts] == ["en", "ru"]
    assert [u["speaker"] for u in utts] == ["en1", "ru"]
    assert utts[0]["start"] == mk.LEAD_S
    assert utts[0]["end"] == pytest.approx(mk.LEAD_S + 5 * 0.05)  # leading silence trimmed
    assert utts[1]["start"] == pytest.approx(utts[0]["end"] + mk.GAP_S)
    assert len(audio) / mk.SR == pytest.approx(utts[1]["end"] + mk.TAIL_S)
    for u in utts:  # the tone sits exactly inside each utterance
        s, e = round(u["start"] * mk.SR), round(u["end"] * mk.SR)
        assert np.all(audio[s:e] == 0.5)
        assert audio[s - 1] == 0 and audio[e] == 0


def test_build_set_is_a_valid_eval_set(tmp_path):
    clips, _ = mk.plan(set(mk.VOICES.values()))
    manifest = mk.build_set(tmp_path / "set", clips, fake_synth)
    loaded = ev.load_manifest(tmp_path / "set")  # eval.py's own validation
    assert [c.id for c in loaded] == list(mk.CLIPS)
    tracks = {c["id"]: c["track"] for c in manifest["clips"]}
    assert tracks["en-01"] == "me" and tracks["en-3spk-01"] == "others"
    assert {u.language for c in loaded for u in c.utterances} == {"en", "ru", "he"}
    assert len({u.speaker for u in loaded[-1].utterances}) == 3


def test_refuses_the_repo_and_documents(tmp_path):
    assert mk.forbidden(ROOT / "eval") is not None
    assert mk.forbidden(Path.home() / "Documents" / "x") is not None
    assert mk.forbidden(tmp_path / "set") is None


def test_non_macos_exits_with_a_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mk.sys, "platform", "win32")
    assert mk.main([str(tmp_path / "set")]) == 2
    assert "needs macOS" in capsys.readouterr().err
    assert not (tmp_path / "set").exists()
