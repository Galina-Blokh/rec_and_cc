"""scripts/make_synthetic_meeting_transcript.py: deterministic, loadable, safe locations.

Stories: none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from recandcc.models import Segment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "rcc_synth_meeting", ROOT / "scripts" / "make_synthetic_meeting_transcript.py"
)
assert spec and spec.loader
mk = importlib.util.module_from_spec(spec)
sys.modules["rcc_synth_meeting"] = mk
spec.loader.exec_module(mk)


def test_deterministic_and_seeded():
    assert mk.build(3) == mk.build(3)
    assert mk.build(3) != mk.build(4)


def test_shape_is_an_hour_of_speech_the_app_can_load():
    segs = [Segment(**s) for s in mk.build(0)["segments"]]
    words = sum(len(s.text.split()) for s in segs)
    assert 8000 <= words <= 9500
    assert {s.speaker for s in segs} == set(mk.SPEAKERS)
    assert segs[-1].end <= mk.DURATION_S
    assert all(a.end <= b.start for a, b in zip(segs, segs[1:], strict=False))


def test_writes_json_into_a_fresh_folder(tmp_path):
    out = tmp_path / "m"
    assert mk.main([str(out)]) == 0
    assert json.loads((out / "transcript.json").read_text())["segments"]
    assert mk.main([str(out)]) == 0  # an earlier meeting folder may be rewritten


def test_refusals_use_tmp_paths_only(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)  # a fake checkout: works in a source tarball too
    afile = tmp_path / "f"
    afile.write_text("x")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "other.txt").write_text("x")
    for target in (home / "Documents" / "x", checkout / "sub", afile, busy):
        assert mk.main([str(target)]) == 2, target
        assert not (target / "transcript.json").exists() if target.is_dir() else True
    err = capsys.readouterr().err
    assert err.count("refusing") == 4
