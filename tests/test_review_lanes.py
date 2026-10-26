"""scripts/review_lanes.sh with a stub Cursor agent: a failed reviewer must fail the run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script")

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "review_lanes.sh"

# Fails for the security lane, answers for every other lane.
STUB = """#!/usr/bin/env bash
prompt="$2"
if [[ "$prompt" == *"security and privacy"* ]]; then
  echo "Error: stub agent failed" >&2
  exit 1
fi
echo "No issues found."
"""


def run(tmp_path, *lanes):
    agent = tmp_path / "agent"
    agent.write_text(STUB)
    agent.chmod(0o755)
    diff = tmp_path / "pr.diff"
    diff.write_text("")
    env = {**os.environ, "CURSOR_AGENT": str(agent), "NO_FOOTER": "1"}
    return subprocess.run(
        ["bash", str(SCRIPT), str(diff), *lanes],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_all_lanes_succeed(tmp_path):
    r = run(tmp_path, "correctness", "tests")
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("No issues found.") == 2


def test_failed_lane_fails_run_but_reports_every_lane(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 1
    for title in (
        "Correctness & threading",
        "Security & privacy",
        "Platforms",
        "Tests & performance",
    ):
        assert f"### {title}" in r.stdout
    assert r.stdout.count("No issues found.") == 3
    assert "_Reviewer failed:_" in r.stdout
    assert "stub agent failed" in r.stdout
