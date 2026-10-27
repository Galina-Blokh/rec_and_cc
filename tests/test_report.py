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
    out = collect.claude_usage([d], {"b1": 1})
    assert (out["sessions"], out["subagents"], out["prompts"]) == (1, 1, 1)
    assert out["tokens"] == 12 and out["unlinked_cost"] == 0
    assert "secret" not in json.dumps(out)


def test_streaming_session_totals_keep_the_summary_shape(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    big = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}  # Opus 5.5: $24
    small = {"output_tokens": 1_000_000, "cache_creation_input_tokens": 10}  # Haiku: ~$0.50

    def rec(mid, model, usage, when, branch):
        msg = {"id": mid, "model": model, "usage": usage}
        return json.dumps(
            {"type": "assistant", "timestamp": when, "gitBranch": branch, "message": msg}
        )

    lines = [
        rec("a", "claude-opus-5-5", big, "2026-10-25T10:00:00Z", "b1"),
        rec("a", "claude-opus-5-5", big, "2026-10-25T10:00:01Z", "b1"),  # repeated id: once
        rec("b", "claude-haiku-5-5", small, "2026-10-26T10:00:00Z", "main"),
        rec("c", "<synthetic>", big, "2026-10-26T10:00:00Z", "b1"),  # skipped
        json.dumps({"type": "user", "message": {"content": [{"type": "tool_result"}]}}),
        "not json",
    ]
    (d / "s.jsonl").write_text("\n".join(lines))
    assert collect.claude_usage([d], {"b1": 7}) == {
        "sessions": 1,
        "subagents": 0,
        "prompts": 0,
        "tokens": 3_000_010,
        "cost": 24.5,
        "kinds": {"input": 1_000_000, "output": 2_000_000, "cache_read": 0, "cache_write": 10},
        "by_model": [
            {"model": "claude-opus-5-5", "tokens": 2_000_000, "cost": 24.0},
            {"model": "claude-haiku-5-5", "tokens": 1_000_010, "cost": 0.5},
        ],
        "by_day": [
            {"day": "2026-10-25", "tokens": 2_000_000, "cost": 24.0},
            {"day": "2026-10-26", "tokens": 1_000_010, "cost": 0.5},
        ],
        "by_pr": {"7": {"tokens": 2_000_000, "cost": 24.0}},
        "linked_cost": 24.0,
        "unlinked_cost": 0.5,
        "unlinked_share": 0.02,
    }


def _fake_api(gh, responses, log):
    """Serve `gh._api` from {endpoint: JSON} and record each call: no subprocess, no network."""

    def api(endpoint, paginate=False):
        log.append(endpoint)
        with gh._lock:
            gh.calls += 1
        return responses[endpoint]

    gh._api = api


def test_incremental_list_stops_at_cached_page_and_merges(tmp_path):
    ep = "pulls?state=all&sort=updated&direction=desc&per_page=2"

    def item(i, updated):
        return {"id": i, "updated_at": updated}

    gh, log = collect.GitHub("o/r", tmp_path), []
    _fake_api(
        gh,
        {
            f"{ep}&page=1": [item(4, "t4"), item(3, "t3")],
            f"{ep}&page=2": [item(2, "t2"), item(1, "t1")],
            f"{ep}&page=3": [],
        },
        log,
    )
    assert [i["id"] for i in gh.get_list(ep)] == [4, 3, 2, 1]  # cold: every page
    assert len(log) == 3

    # #5 is new and #3 changed; page 2 holds #4 cached unchanged, so page 3 is never asked for.
    gh, log = collect.GitHub("o/r", tmp_path), []
    _fake_api(
        gh,
        {
            f"{ep}&page=1": [item(5, "t5"), item(3, "t3b")],
            f"{ep}&page=2": [item(4, "t4"), item(2, "t2")],
        },
        log,
    )
    got = gh.get_list(ep)
    assert [(i["id"], i["updated_at"]) for i in got] == [
        (5, "t5"),
        (3, "t3b"),
        (4, "t4"),
        (2, "t2"),
        (1, "t1"),
    ]
    assert log == [f"{ep}&page=1", f"{ep}&page=2"] and gh.calls == 2

    off, log = collect.GitHub("o/r", tmp_path, offline=True), []
    _fake_api(off, {}, log)
    assert off.get_list(ep) == got and not log


def test_runs_list_reads_the_workflow_runs_key(tmp_path):
    ep = "actions/runs?per_page=100"
    gh, log = collect.GitHub("o/r", tmp_path), []
    _fake_api(gh, {f"{ep}&page=1": {"workflow_runs": [{"id": 9, "updated_at": "t"}]}}, log)
    assert gh.get_list(ep, key="workflow_runs") == [{"id": 9, "updated_at": "t"}]
    assert len(log) == 1  # a short page is the last one


def test_unfinished_runs_below_the_stop_point_are_refetched(tmp_path):
    ep = "actions/runs?per_page=2"
    gh = collect.GitHub("o/r", tmp_path)
    page1 = [
        {"id": 9, "updated_at": "a", "status": "completed"},
        {"id": 8, "updated_at": "a", "status": "completed"},
    ]
    old = [
        {"id": 7, "updated_at": "a", "status": "completed"},
        {"id": 6, "updated_at": "a", "status": "in_progress"},
    ]
    _fake_api(
        gh,
        {
            f"{ep}&page=1": {"workflow_runs": page1},
            f"{ep}&page=2": {"workflow_runs": old},
            f"{ep}&page=3": {"workflow_runs": []},
        },
        [],
    )
    gh.get_list(ep, key="workflow_runs")  # cold: all pages, run 6 cached as in progress
    log = []
    done6 = {"id": 6, "updated_at": "b", "status": "completed", "conclusion": "success"}
    _fake_api(gh, {f"{ep}&page=1": {"workflow_runs": page1}, "actions/runs/6": done6}, log)
    runs = gh.get_list(ep, key="workflow_runs")
    assert log == [f"{ep}&page=1", "actions/runs/6"]  # stops early, refetches only run 6
    assert [r["id"] for r in runs] == [9, 8, 7, 6] and runs[-1]["conclusion"] == "success"


def test_a_corrupt_cache_file_counts_as_empty(tmp_path):
    ep = "issues?per_page=100"
    gh = collect.GitHub("o/r", tmp_path)
    gh._path(ep).write_text('{"stamp": null, "data": [')  # a run killed mid-write
    _fake_api(gh, {f"{ep}&page=1": [{"id": 1, "updated_at": "t"}]}, [])
    assert gh.get_list(ep) == [{"id": 1, "updated_at": "t"}]
    gh._path("pulls/1").write_text("garbage")
    _fake_api(gh, {"pulls/1": {"number": 1}}, [])
    assert gh.get("pulls/1", "s", paginate=False) == {"number": 1}


def test_rate_limited_calls_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "BACKOFF", 0)
    replies = iter([(1, "", "gh: HTTP 429: secondary rate limit"), (0, "[1]", "")])

    def fake_run(args, **kwargs):
        code, out, err = next(replies)
        return collect.subprocess.CompletedProcess(args, code, out, err)

    monkeypatch.setattr(collect.subprocess, "run", fake_run)
    gh = collect.GitHub("o/r", tmp_path)
    assert gh._api("issues") == [1] and gh.calls == 2


def test_pooled_pr_details_match_sequential(tmp_path, monkeypatch):
    pulls = [{"id": n, "number": n, "updated_at": f"u{n}"} for n in range(1, 8)]

    def responses():
        r = {"pulls?state=all&sort=updated&direction=desc&per_page=100&page=1": pulls}
        r["issues?state=all&sort=updated&direction=desc&per_page=100&page=1"] = []
        r["actions/runs?per_page=100&page=1"] = {"workflow_runs": []}
        for n in range(1, 8):
            r[f"pulls/{n}"] = {"number": n}
            r[f"pulls/{n}/commits?per_page=100"] = [[{"sha": f"c{n}"}]]
            r[f"pulls/{n}/reviews?per_page=100"] = [[{"id": n}]]
            r[f"issues/{n}/comments?per_page=100"] = [[{"body": f"x{n}"}]]
        return r

    gh, log = collect.GitHub("o/r", tmp_path / "a"), []
    _fake_api(gh, responses(), log)
    pooled = collect.fetch_github(gh)
    seq, seq_log = collect.GitHub("o/r", tmp_path / "b"), []
    _fake_api(seq, responses(), seq_log)
    sequential = collect.fetch_github(seq, workers=1)
    assert pooled == sequential and list(pooled["detail"]) == list(range(1, 8))
    assert pooled["detail"][3]["commits"] == [{"sha": "c3"}]
    assert gh.calls == len(log) == 3 + 4 * 7

    # Unchanged PRs are served from the per-PR stamp cache: only the three list calls remain.
    again, again_log = collect.GitHub("o/r", tmp_path / "a"), []
    _fake_api(again, responses(), again_log)
    assert collect.fetch_github(again) == pooled and len(again_log) == 3


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
        "claude": claude or collect.claude_usage([], {}),
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


def test_render_with_claude_usage_and_docs_and_fragment(tmp_path):
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 3}
    d = tmp_path / "proj"
    d.mkdir()
    rec = {
        "type": "assistant",
        "gitBranch": "b1",
        "timestamp": "2026-10-26T10:00:00Z",
        "message": {"id": "m1", "model": "claude-opus-5-5", "usage": usage},
    }
    (d / "s.jsonl").write_text(json.dumps(rec))
    claude = collect.claude_usage([d], {"b1": 1})
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


def test_cursor_review_rule_flips_severity_at_one_finding_per_round():
    data = _report_data()
    data["summary"].update(review_rounds=4, review_findings=2)
    assert _titles(data)["Cursor reviews mostly came back clean"] == "low"
    data["summary"].update(review_findings=6)
    t = _titles(data)
    assert t["Every Cursor review round found something"] == "medium"
    assert "Cursor reviews mostly came back clean" not in t
    data["summary"].update(review_rounds=0, review_findings=0)
    assert not {k for k in _titles(data) if "Cursor review" in k}


def test_stale_banner_follows_the_refresh_cadence():
    data = _report_data()  # generated 26/10 14:00 UTC, refreshed every 6 h
    fresh = render.render(data, now=collect.ts("2026-10-26T20:00:00Z"))
    late = render.render(data, now=collect.ts("2026-10-27T03:00:00Z"))
    assert "Stale: the scheduled refresh has not run." not in fresh
    assert "Stale: the scheduled refresh has not run." in late


def test_ci_hour_table_matches_the_gap_filled_chart():
    data = _report_data()
    data["ci"]["per_hour"] = [
        {"hour": "2026-10-26 10:00", "passed": 1, "failed": 0},
        {"hour": "2026-10-26 13:00", "passed": 0, "failed": 1},
    ]
    page = render.render(data)
    figure = page.split('id="c-ciday"', 1)[1].split("</figure>", 1)[0]
    table = figure.split("<tbody>", 1)[1]
    assert table.count("<tr>") == 4 and "2026-10-26 11:00" in table


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


def _item(i, updated=None):
    return {"id": i, "updated_at": updated or f"t{i}"}


def test_list_drops_deleted_items_and_keeps_cached_tail(tmp_path):
    ep = "pulls?state=all&sort=updated&direction=desc&per_page=2"
    gh = collect.GitHub("o/r", tmp_path)
    _fake_api(
        gh,
        {
            f"{ep}&page=1": [_item(5), _item(4)],
            f"{ep}&page=2": [_item(3), _item(2)],
            f"{ep}&page=3": [_item(1)],
        },
        [],
    )
    assert [i["id"] for i in gh.get_list(ep)] == [5, 4, 3, 2, 1]

    # #5 and #4 were deleted; page 1 holds #3 unchanged, so the walk stops there and keeps
    # the cached tail after it (#2, #1). Items before the overlap that vanished are dropped.
    gh = collect.GitHub("o/r", tmp_path)
    _fake_api(gh, {f"{ep}&page=1": [_item(6), _item(3)]}, [])
    assert [i["id"] for i in gh.get_list(ep)] == [6, 3, 2, 1]
    assert [i["id"] for i in gh.dropped[ep]] == [5, 4]

    # A complete walk replaces the list: #2 and #1 are gone.
    gh = collect.GitHub("o/r", tmp_path)
    _fake_api(gh, {f"{ep}&page=1": [_item(7, "new")]}, [])
    assert [i["id"] for i in gh.get_list(ep)] == [7]
    assert {i["id"] for i in gh.dropped[ep]} == {6, 3, 2, 1}


def test_list_dedupes_ids_across_pages(tmp_path):
    ep = "pulls?state=all&sort=updated&direction=desc&per_page=2"
    gh = collect.GitHub("o/r", tmp_path)
    _fake_api(
        gh,
        {
            f"{ep}&page=1": [_item(3), _item(2)],
            f"{ep}&page=2": [_item(2, "other"), _item(1)],
            f"{ep}&page=3": [],
        },
        [],
    )
    got = gh.get_list(ep)
    assert [(i["id"], i["updated_at"]) for i in got] == [(3, "t3"), (2, "t2"), (1, "t1")]


def test_missing_pr_is_skipped_and_its_cache_removed(tmp_path):
    ep = "pulls?state=all&sort=updated&direction=desc&per_page=100"
    issues = "issues?state=all&sort=updated&direction=desc&per_page=100&page=1"
    runs = "actions/runs?per_page=100&page=1"

    def pr(n):
        return {"id": n, "number": n, "updated_at": f"u{n}"}

    def detail(n):
        return {
            f"pulls/{n}": {"number": n},
            f"pulls/{n}/commits?per_page=100": [[{"sha": "c"}]],
            f"pulls/{n}/reviews?per_page=100": [[]],
            f"issues/{n}/comments?per_page=100": [[]],
        }

    gh = collect.GitHub("o/r", tmp_path)
    r = {f"{ep}&page=1": [pr(2), pr(1)], issues: [], runs: {"workflow_runs": []}}
    r.update(detail(1) | detail(2))
    _fake_api(gh, r, [])
    assert set(collect.fetch_github(gh)["detail"]) == {1, 2}
    assert gh._path("pulls/1").exists()

    # #1 leaves the list: its detail files go. #2 is listed but 404s: skipped, run survives.
    gh = collect.GitHub("o/r", tmp_path)
    r = {
        f"{ep}&page=1": [
            pr(3),
            pr(
                2,
            ),
            pr(2),
        ],
        issues: [],
        runs: {"workflow_runs": []},
    }
    r[f"{ep}&page=1"][1]["updated_at"] = "u2b"
    r[f"{ep}&page=1"][2] = r[f"{ep}&page=1"][1]
    r.update(detail(3))

    def api(endpoint, paginate=False):
        if endpoint.startswith("pulls/2"):
            raise RuntimeError(f"gh api {endpoint} failed: HTTP 404: Not Found")
        return r[endpoint]

    gh._api = api
    got = collect.fetch_github(gh)
    assert list(got["detail"]) == [3]
    assert not gh._path("pulls/1").exists()
    assert not gh._path("pulls/1/commits?per_page=100").exists()


def test_cache_writes_are_atomic(tmp_path, monkeypatch):
    replaced = []
    real = collect.os.replace

    def spy(src, dst):
        replaced.append(Path(dst).name)
        real(src, dst)

    monkeypatch.setattr(collect.os, "replace", spy)
    ep = "pulls?state=all&sort=updated&direction=desc&per_page=2"
    gh = collect.GitHub("o/r", tmp_path)
    _fake_api(gh, {f"{ep}&page=1": [_item(1)], "pulls/1": {"n": 1}}, [])
    gh.get_list(ep)
    gh.get("pulls/1", "s", paginate=False)
    assert len(replaced) == 2
    assert not list(gh.cache.glob("*.tmp"))
