"""Build-report scripts: metrics from a tiny fixture, rendering and price maths. No network.

Stories: none.
"""

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
render = _load("render")
diff = _load("diff")


def _pr(
    n, opened, merged=None, author="dev", branch=None, body="", commits=1, review=None, fix_at=None
):
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
    if fix_at:
        cs.append({"commit": {"message": "fix: y", "committer": {"date": fix_at}}})
    return {"pr": p, "commits": cs, "reviews": [], "comments": comments}


def _data():
    review = "## 🤖 Cursor review\n\n### Lane\n\n- **a.py:1** — bug\n- **b.py:2** — bug\n"
    detail = {
        "1": _pr(
            1,
            "2026-10-26T10:00:00Z",
            "2026-10-26T10:30:00Z",
            body="Closes #5",
            review=review,
            fix_at="2026-10-26T10:10:00Z",
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
    assert prs[1]["fix_pushes"] == 1 and prs[1]["wait_after_green_min"] == 5
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


def _report_data(claude=None, docs=None):
    raw = _data()
    now = collect.ts("2026-10-26T14:00:00Z")
    prs = collect.build_prs(raw, raw["runs"], now)
    issues = collect.build_issues(raw["issues"], prs)
    ledger = [
        {
            "sha": "a",
            "merge": False,
            "author": "dev",
            "email": "e",
            "at": "2026-10-26T10:00:00+00:00",
            "subject": "feat: x",
            "ai": True,
        }
    ]
    return {
        "repo": "o/r",
        "generated_at": "2026-10-26T14:00:00+00:00",
        "window": {"start": "2026-10-26T10:00:00+00:00", "end": "2026-10-26T14:00:00+00:00"},
        "refresh": {"on_merge": True, "minutes": 360},
        "summary": collect.summarize(prs, issues, now),
        "series": collect.series(prs, ledger),
        "commits": collect.build_commits(ledger),
        "ci": collect.build_ci(raw["runs"]),
        "claude": claude or collect.summarize_sessions({}, {}),
        "docs": docs or {"adrs": [], "roadmap": []},
        "prs": prs,
        "issues": issues,
        "prices": {"claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.2}},
        "sources": {"github_calls": 0, "ref": "origin/main"},
    }


SECTIONS = (
    "tldr",
    "how",
    "delivery",
    "prd",
    "size",
    "cycle",
    "quality",
    "team",
    "work",
    "ai",
    "lessons",
    "deferrals",
    "dq",
    "recos",
    "allprs",
    "allissues",
    "allci",
    "sources",
    "glossary",
)


def test_render_produces_every_section():
    page = render.render(_report_data())
    for sid in SECTIONS:
        assert f'id="{sid}"' in page
    assert "None" not in page.replace("NoneType", "")
    assert "<script src" not in page and "https://cdn" not in page


def test_render_with_claude_usage_and_docs_and_fragment():
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 3}
    sessions = {
        "s": {
            "subagent": False,
            "prompts": 2,
            "first": None,
            "last": None,
            "branches": {},
            "usage": [("claude-opus-5-5", usage, None, "b1")],
        }
    }
    claude = collect.summarize_sessions(sessions, {"b1": 1})
    docs = {
        "adrs": [
            {"id": "0001", "title": "Python core", "status": "Accepted", "path": "docs/adr/x.md"}
        ],
        "roadmap": [{"section": "Future work", "done": False, "text": "Hebrew support"}],
    }
    page = render.render(_report_data(claude, docs))
    assert "Python core" in page and "Hebrew support" in page and "claude-opus-5-5" in page
    frag = render.fragment(page)
    assert frag.startswith("<title>") and "<html" not in frag and "<body" not in frag
    assert 'name="claude:viewport-fit-sides"' in frag and 'charset="utf-8"' not in frag


def _titles(data):
    return {f["title"]: f["severity"] for f in render.findings(data)}


def test_findings_fire_from_the_evidence():
    data = _report_data()
    t = _titles(data)
    assert t["No review by a person"] == "medium"
    assert t["Open issues have no owner"] == "low"
    assert "Changes merged while CI was red" not in t and "main went red after a merge" not in t

    data["summary"]["merged_red"] = 2
    findings = render.findings(data)
    assert (findings[0]["title"], findings[0]["severity"]) == (
        "Changes merged while CI was red",
        "high",
    )
    assert "2 merged pull requests" in findings[0]["evidence"]

    data["ci"]["main_failed"], data["ci"]["main_runs"] = 1, 4
    data["claude"].update(cost=10.0, unlinked_share=0.4)
    data["summary"].update(open=5, closed_unmerged=1, reviewed_human=1)
    t = _titles(data)
    assert t["main went red after a merge"] == "medium"
    assert t["Most Claude spend cannot be tied to a pull request"] == "medium"
    assert t["Open pull requests pile up"] == "medium"
    assert t["Work thrown away"] == "low"
    assert "No review by a person" not in t


def test_render_survives_an_empty_record():
    empty_summary = collect.summarize([], [], collect.ts("2026-10-26T14:00:00Z"))
    data = {
        **_report_data(),
        "window": {"start": None, "end": "2026-10-26T14:00:00+00:00"},
        "summary": empty_summary,
        "series": collect.series([], []),
        "commits": collect.build_commits([]),
        "ci": collect.build_ci([]),
        "prs": [],
        "issues": [],
    }
    page = render.render(data)
    assert 'id="tldr"' in page and "0 merged pull requests" in page


def test_day_names_do_not_depend_on_the_locale():
    assert render.day_name("2026-10-27") == "Tue 27/10"


def test_fill_hours_fills_gaps_and_switches_to_days_on_long_spans():
    short = render.fill_hours(
        [{"hour": "2026-10-26 10:00", "n": 1}, {"hour": "2026-10-26 13:00", "n": 2}], ["n"]
    )
    assert [r["n"] for r in short] == [1, 0, 0, 2]
    long = render.fill_hours(
        [
            {"hour": "2026-10-01 10:00", "n": 1},
            {"hour": "2026-10-01 11:00", "n": 1},
            {"hour": "2026-10-14 09:00", "n": 3},
        ],
        ["n"],
    )
    assert len(long) == 14 and long[0] == {"hour": "2026-10-01", "n": 2} and long[-1]["n"] == 3
    assert render.hour_label(long[0]["hour"]) == "01/10"


def test_diff_lists_moved_headline_figures():
    old = {"summary": {"merged": 1, "open": 2}, "ci": {}, "claude": {"cost": 1.0}}
    new = {"summary": {"merged": 3, "open": 2}, "ci": {}, "claude": {"cost": 2.0}}
    assert diff.changes(old, new) == [
        "merged pull requests: 1 -> 3",
        "estimated Claude cost ($): 1.0 -> 2.0",
    ]


def test_ci_commits_and_series():
    raw = _data()
    ci = collect.build_ci(
        raw["runs"]
        + [{**raw["runs"][1], "id": 3, "event": "dynamic", "name": "uv in /. - Update #1"}]
    )
    assert (ci["ci_runs"], ci["ci_passed"], ci["ci_failed"], ci["ci_rate"]) == (2, 1, 1, 0.5)
    assert any(w["name"] == "Dependabot updates" for w in ci["workflows"])
    ledger = [
        {
            "sha": "a",
            "merge": False,
            "author": "dev",
            "email": "e",
            "at": "2026-10-26T10:00:00+00:00",
            "subject": "feat: x",
            "ai": True,
        },
        {
            "sha": "b",
            "merge": False,
            "author": "dev",
            "email": "e",
            "at": "2026-10-26T10:00:00+00:00",
            "subject": "feat: x",
            "ai": True,
        },
        {
            "sha": "c",
            "merge": True,
            "author": "dev",
            "email": "e",
            "at": "2026-10-26T11:00:00+00:00",
            "subject": "Merge",
            "ai": False,
        },
        {
            "sha": "d",
            "merge": False,
            "author": "dev",
            "email": "e",
            "at": "2026-10-26T12:00:00+00:00",
            "subject": "fix(x): y",
            "ai": False,
        },
    ]
    cm = collect.build_commits(ledger)
    assert (cm["authored"], cm["ai"], cm["merges"], cm["ai_floor"]) == (2, 1, 1, 0.5)
    assert {t["type"] for t in cm["types"]} == {"feat", "fix"}
    now = collect.ts("2026-10-26T14:00:00Z")
    ser = collect.series(collect.build_prs(raw, raw["runs"], now), ledger)
    assert ser["peak"]["n"] == 2 and [c["n"] for c in ser["cumulative"]] == [1, 2]


def test_offline_makes_no_network_call(tmp_path, monkeypatch):
    calls = []
    real = collect.subprocess.run

    def spy(args, *a, **k):
        calls.append(args)
        return real(args, *a, **k)

    monkeypatch.setattr(collect.subprocess, "run", spy)
    monkeypatch.setattr(
        collect, "fetch_github", lambda gh: {"detail": {}, "issues": [], "runs": []}
    )
    monkeypatch.setattr(collect, "git_ledger", lambda root, ref: [])
    collect.collect(tmp_path, "o/r", tmp_path / "cache", offline=True, logs=tmp_path / "logs")
    assert not [c for c in calls if "fetch" in c or c[:2] == ["gh", "api"]]
