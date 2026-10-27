"""Build-report collector: metrics from a tiny fixture and price maths. No network."""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "report"
sys.path.insert(0, str(ROOT))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect = _load("collect")
diff = _load("diff")


def _pr(n, opened, merged=None, author="dev", branch=None, body="", commits=1, review=None):
    p = {
        "number": n,
        "title": f"PR {n}",
        "user": {"login": author},
        "created_at": opened,
        "merged_at": merged,
        "closed_at": merged,
        "updated_at": merged or opened,
        "head": {"ref": branch or f"b{n}"},
        "merged_by": {"login": author} if merged else None,
        "additions": 10 * n,
        "deletions": n,
        "changed_files": n,
        "commits": commits,
        "comments": 0,
        "review_comments": 0,
        "body": body,
        "html_url": f"https://example.invalid/pull/{n}",
    }
    comments = [{"body": review, "user": {"login": "github-actions[bot]"}}] if review else []
    msg = "feat: x\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
    cs = [{"commit": {"message": msg, "committer": {"date": opened}}}]
    return {"pr": p, "commits": cs, "reviews": [], "comments": comments}


def _data():
    review = "## 🤖 Cursor review\n\n### Lane\n\n- **a.py:1** — bug\n- **b.py:2** — bug\n"
    detail = {
        "1": _pr(
            1, "2026-10-26T10:00:00Z", "2026-10-26T10:30:00Z", body="Closes #5", review=review
        ),
        "2": _pr(2, "2026-10-26T11:00:00Z", "2026-10-26T13:00:00Z", author="dependabot[bot]"),
        "3": _pr(3, "2026-10-26T12:00:00Z"),
    }
    runs = [
        {
            "id": 1,
            "name": "CI",
            "event": "pull_request",
            "head_branch": "b1",
            "conclusion": "failure",
            "created_at": "2026-10-26T10:05:00Z",
            "updated_at": "2026-10-26T10:08:00Z",
            "html_url": "u",
        },
        {
            "id": 2,
            "name": "CI",
            "event": "pull_request",
            "head_branch": "b1",
            "conclusion": "success",
            "created_at": "2026-10-26T10:20:00Z",
            "updated_at": "2026-10-26T10:25:00Z",
            "html_url": "u",
        },
    ]
    issues = [
        {
            "number": 5,
            "title": "do x",
            "state": "closed",
            "user": {"login": "dev"},
            "labels": [],
            "assignees": [],
            "created_at": "2026-10-26T09:00:00Z",
            "closed_at": "2026-10-26T10:30:00Z",
            "comments": 1,
            "html_url": "i5",
        },
        {
            "number": 6,
            "title": "do y",
            "state": "open",
            "user": {"login": "dev"},
            "labels": [{"name": "question"}],
            "assignees": [],
            "created_at": "2026-10-26T09:00:00Z",
            "closed_at": None,
            "comments": 0,
            "html_url": "i6",
        },
    ]
    return {"detail": detail, "runs": runs, "issues": issues}


def test_pr_metrics_cycle_ci_and_review():
    raw = _data()
    now = collect.ts("2026-10-26T14:00:00Z")
    prs = {p["number"]: p for p in collect.build_prs(raw, raw["runs"], now)}
    assert prs[1]["cycle_min"] == 30
    assert prs[1]["ci_failures"] == 1 and not prs[1]["merged_red"]
    assert prs[1]["review_rounds"] == 1 and prs[1]["review_findings"] == 2
    assert prs[1]["closes"] == [5] and prs[1]["ai_commits"] == 1
    assert prs[2]["bot"] and prs[3]["state"] == "open" and prs[3]["age_min"] == 120


def test_summary_and_issue_links():
    raw = _data()
    now = collect.ts("2026-10-26T14:00:00Z")
    prs = collect.build_prs(raw, raw["runs"], now)
    issues = collect.build_issues(raw["issues"], prs)
    s = collect.summarize(prs, issues, now)
    assert (s["merged"], s["merged_human"], s["open"]) == (2, 1, 1)
    assert s["issues_closed_no_pr"] == 0 and s["issues_unassigned"] == 1
    assert s["self_merged"] == 1 and s["ever_red"] == 1
    assert sum(b["n"] for b in s["hist_cycle"]) == 2


def test_price_counts_cache_writes_by_ttl():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
        "cache_creation_input_tokens": 2_000_000,
        "cache_creation": {
            "ephemeral_1h_input_tokens": 1_000_000,
            "ephemeral_5m_input_tokens": 1_000_000,
        },
    }
    # Opus 5.5: 4 in + 20 out + 0.20 read + 5 (5 min write) + 8 (1 h write)
    assert round(collect.price("claude-opus-5-5", usage), 2) == 37.20


def test_sessions_count_prompts_and_dedupe_usage(tmp_path):
    d = tmp_path / "proj"
    (d / "s1" / "subagents").mkdir(parents=True)
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 0,
    }
    lines = [
        {
            "type": "user",
            "timestamp": "2026-10-26T10:00:00Z",
            "gitBranch": "b1",
            "message": {"content": "secret prompt"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-26T10:00:01Z",
            "gitBranch": "b1",
            "message": {"id": "m1", "model": "claude-opus-5-5", "usage": usage},
        },
        {
            "type": "assistant",
            "timestamp": "2026-10-26T10:00:02Z",
            "gitBranch": "b1",
            "message": {"id": "m1", "model": "claude-opus-5-5", "usage": usage},
        },
    ]
    (d / "s1.jsonl").write_text("\n".join(json.dumps(x) for x in lines))
    (d / "s1" / "subagents" / "agent-x.jsonl").write_text(
        json.dumps(lines[1] | {"message": lines[1]["message"] | {"id": "m2"}})
    )
    out = collect.summarize_sessions(collect.read_sessions([d]), {"b1": 1})
    assert (out["sessions"], out["subagents"], out["prompts"]) == (1, 1, 1)
    assert out["tokens"] == 12 and out["unlinked_cost"] == 0
    assert "secret" not in json.dumps(out)


def test_diff_lists_moved_headline_figures():
    old = {"summary": {"merged": 1, "open": 2}, "ci": {}, "claude": {"cost": 1.0}}
    new = {"summary": {"merged": 3, "open": 2}, "ci": {}, "claude": {"cost": 2.0}}
    assert diff.changes(old, new) == [
        "merged pull requests: 1 -> 3",
        "estimated Claude cost ($): 1.0 -> 2.0",
    ]
