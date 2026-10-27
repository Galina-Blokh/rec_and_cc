"""Build-report collector: metrics from a tiny fixture and price maths. No network.

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
