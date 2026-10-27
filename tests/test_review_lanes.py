"""scripts/review_lanes.sh: lanes run in parallel, every lane is reported, and any failed
reviewer makes the script exit 1 (so CI never shows a green check for a review that didn't
happen). Uses a stub agent instead of the real Cursor CLI.

Stories: none.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash"
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "review_lanes.sh"

# Fails for the lanes listed in FAIL_LANES (matched against the prompt), succeeds otherwise.
STUB = """#!/usr/bin/env bash
prompt="$2"
for lane in $FAIL_LANES; do
  case "$prompt" in *"$lane"*) echo "boom: $lane reviewer crashed ${LONG_ERR:-}" >&2; exit 3 ;; esac
done
echo "No issues found."
"""


def run(
    tmp_path: Path, lanes: list[str], fail: str = "", long_err: str = ""
) -> subprocess.CompletedProcess:
    agent = tmp_path / "agent"
    agent.write_text(STUB, encoding="utf-8")
    agent.chmod(0o755)
    diff = tmp_path / "pr.diff"
    diff.write_text("diff --git a/x b/x\n", encoding="utf-8")
    env = {**os.environ, "CURSOR_AGENT": str(agent), "FAIL_LANES": fail, "NO_FOOTER": "1"}
    env["LONG_ERR"] = long_err
    return subprocess.run(  # noqa: S603 - fixed argv
        ["bash", str(SCRIPT), str(diff), *lanes],  # noqa: S607
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def test_all_lanes_pass(tmp_path):
    r = run(tmp_path, ["correctness", "security", "platform", "tests"])
    assert r.returncode == 0, r.stderr
    for title in ("Correctness", "Security", "Platforms", "Tests"):
        assert f"### {title}" in r.stdout
    assert r.stdout.count("No issues found.") == 4


def test_mixed_failure_reports_every_lane_and_exits_1(tmp_path):
    # "security and privacy" is in the security lane's focus text.
    r = run(tmp_path, ["correctness", "security"], fail="privacy")
    assert r.returncode == 1
    assert "### Correctness & threading" in r.stdout and "No issues found." in r.stdout
    assert "### Security & privacy" in r.stdout and "_Reviewer failed:_" in r.stdout
    assert "boom" in r.stdout


def test_error_text_is_capped(tmp_path):
    r = run(tmp_path, ["tests"], fail="coverage", long_err="x" * 2000)
    assert r.returncode == 1
    assert "boom" in r.stdout  # the error is shown...
    assert "x" * 500 not in r.stdout  # ...but capped
    assert all(len(line) <= 400 for line in r.stdout.splitlines())


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "cursor-review.yml"


def _prepare_step() -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("- name: Prepare diff and trusted tree")
    return text[start : text.index("- name:", start + 10)]


def test_workflow_only_runs_the_base_branch_script():
    """Security: the script that sees CURSOR_API_KEY must come from the trusted base branch,
    never from the PR (no fallback), and a missing base script must fail the job."""
    step = _prepare_step()
    checkout_base = step.index('git checkout --quiet --detach "$BASE"')
    copies = [i for i in range(len(step)) if step.startswith("cp scripts/review_lanes.sh", i)]
    assert copies, "the review script is never copied"
    assert all(i > checkout_base for i in copies), "script copied from the PR tree"
    assert "review_lanes.pr.sh" not in WORKFLOW.read_text(encoding="utf-8")
    missing = step.index("if [ ! -f scripts/review_lanes.sh ]")
    assert checkout_base < missing < copies[0]
    assert "exit 1" in step[missing : copies[0]]


def test_workflow_runs_the_copied_script_and_fails_red():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'bash /tmp/review_lanes.sh /tmp/pr.diff "${{ matrix.lane }}"' in text
    assert 'exit "$status"' in text  # a failed reviewer fails the lane check
    assert "--sandbox enabled" in SCRIPT.read_text(encoding="utf-8")
