"""Voice-profile recall across sessions (issue #1), with macOS `say` voices and the real
diarizer: scripts/voice_recall_eval.py, asserted with a floor below the measured value.

Run: uv run pytest -m slow tests/test_voice_recall.py -v -s
Skipped unless macOS, the `say` voices and the cached speaker model are all there: nothing is
downloaded, and the network is blocked while it runs. Profiles go to the conftest temp dir.

Stories: US-21.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("say"), reason="macOS say"),
]

ROOT = Path(__file__).resolve().parents[1]
# Measured 2026-10-27: 21/21 later segments of the four named voices (100%); the floor is the
# issue #1 target and allows two misses. False names: 5/9 (one unnamed voice takes "Anna",
# docs/EVALUATION.md). That ceiling only guards against getting worse; it is not a target.
RECALL_FLOOR = 0.90
FALSE_NAME_CEILING = 0.60


def _load():
    spec = importlib.util.spec_from_file_location(
        "rcc_voice_recall", ROOT / "scripts" / "voice_recall_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rcc_voice_recall"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_named_voices_are_recognized_in_later_sessions(tmp_path, monkeypatch):
    from recandcc.transcribe import diarize

    ev = _load()
    if missing := ev.missing_voices():
        pytest.skip(f"say voices not installed: {missing}")
    if diarize.model_path(download=False) is None:
        pytest.skip("speaker model not cached")

    def no_network(*a, **k):
        raise AssertionError("network access during the recall eval")

    monkeypatch.setattr(diarize.urllib.request, "urlopen", no_network)
    (tmp_path / "say").mkdir()
    enrolled, later = ev.run(tmp_path / "sessions", ev.mk.say_synth(tmp_path / "say"))
    rows = ev.score(later)
    for row in rows:
        print(row)
    seg_recall, dur_recall, false_rate = ev.totals(rows)
    print(f"recall {seg_recall:.1%} segments, {dur_recall:.1%} seconds; false {false_rate:.1%}")
    assert all(enrolled.values()), enrolled  # every voice could be named once
    assert seg_recall >= RECALL_FLOOR
    assert false_rate <= FALSE_NAME_CEILING
