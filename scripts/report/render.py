"""Render `report-data.json` into one self-contained HTML page (inline CSS, JS and SVG).

Every number and sentence on the page is computed here from the data file, so a refresh
only needs `collect.py` then `render.py`. No CDN, no network.

Usage: python scripts/report/render.py [--data .report/report-data.json] [--out .report/report.html]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime, timedelta, tzinfo
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402

UK: tzinfo
try:
    UK = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:  # Windows without the tzdata package: report in UTC
    UK = UTC
TITLE = "RecAndCC Workshop Report"


# ---------- formatting ----------


def ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def uk(value: str | None, with_day: bool = True) -> str:
    d = ts(value)
    if not d:
        return "–"
    d = d.astimezone(UK)
    return d.strftime("%d/%m, %H:%M") if with_day else d.strftime("%H:%M")


def dur(mins: float | None) -> str:
    if mins is None:
        return "–"
    if mins < 1:
        return "<1 min"
    if mins < 60:
        return f"{mins:.0f} min"
    h, m = divmod(int(round(mins)), 60)
    if h < 24:
        return f"{h} h {m} min" if m else f"{h} h"
    d, h = divmod(h, 24)
    return f"{d} day{'s' * (d > 1)} {h} h" if h else f"{d} day{'s' * (d > 1)}"


def n(v: float | None) -> str:
    return "–" if v is None else f"{v:,.0f}"


def pct(a: float, b: float) -> str:
    return f"{a / b:.0%}" if b else "–"


def money(v: float | None) -> str:
    return "–" if v is None else f"${v:,.2f}" if v < 100 else f"${v:,.0f}"


def big(v: float) -> str:
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if v >= size:
            return f"{v / size:.1f}{suffix}"
    return f"{v:.0f}"


def plural(k: int, one: str, many: str | None = None) -> str:
    return f"{k:,} {one if k == 1 else (many or one + 's')}"


def e(text: object) -> str:
    return escape(str(text))


# ---------- page pieces ----------


def table(
    headers: list[str],
    rows: list[list],
    num: set[int] | None = None,
    sortable: bool = False,
    cls: str = "",
) -> str:
    num = num or set()
    head = "".join(
        f'<th class="{"num" if i in num else ""}">{e(h)}</th>' for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    sort = " sortable" if sortable else ""
    return f'<div class="twrap {cls}"><table class="{sort.strip()}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def figure(
    fid: str,
    title: str,
    why: str,
    svg: str,
    headers: list[str],
    rows: list[list],
    legend: list[tuple[str, str]] | None = None,
) -> str:
    leg = ""
    if legend:
        leg = (
            '<div class="legend">'
            + "".join(f'<span><i style="background:var({c})"></i>{e(t)}</span>' for t, c in legend)
            + "</div>"
        )
    num = set(range(1, len(headers)))
    return (
        f'<figure class="chart" id="{fid}"><figcaption><b>{e(title)}</b><span class="why">{e(why)}</span></figcaption>'
        f'{leg}<div class="plot">{svg}</div><div class="tools"><button class="tbtn" data-tv type="button">Table</button></div>'
        f'<div class="tview">{table(headers, rows, num)}</div></figure>'
    )


def kpi(value: str, label: str, detail: str = "", hero: bool = False) -> str:
    return f'<div class="kpi{" hero" if hero else ""}"><div class="v">{value}</div><div class="l">{label}</div><div class="d">{detail}</div></div>'


def callout(title: str, items: list[str], kind: str = "") -> str:
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'<div class="callout {kind}"><b class="t">{title}</b><ul class="tight">{lis}</ul></div>'


def section(sid: str, title: str, lede: str, body: str) -> str:
    return f'<section id="{sid}"><h2>{e(title)}</h2><p class="lede">{lede}</p>{body}</section>'


def pr_link(p: dict) -> str:
    return f'<a href="{e(p["url"])}">#{p["number"]}</a>'


def issue_link(i: dict) -> str:
    return f'<a href="{e(i["url"])}">#{i["number"]}</a>'


def epoch(value: str) -> float:
    return ts(value).timestamp()


def day_labels(start: float, end: float, k: int = 6) -> list[tuple[float, str]]:
    """About `k` evenly spaced axis labels between start and end, in UK time."""
    if end <= start:
        return []
    fmt = "%d/%m %H:%M" if end - start < 4 * 86400 else "%d/%m"
    return [
        (t, datetime.fromtimestamp(t, UK).strftime(fmt))
        for t in (start + i * (end - start) / (k - 1) for i in range(k))
    ]


MAX_SLOTS = 96  # beyond four days of hours, chart by day instead


def fill_hours(rows: list[dict], keys: list[str]) -> list[dict]:
    """Every UK hour between the first and last row, zeros where nothing happened.

    When that would be more than MAX_SLOTS hours, rows are summed per day instead (keyed
    "YYYY-MM-DD"), so the chart and its table grow with the days covered, not the hours.
    """
    if not rows:
        return []
    first = datetime.strptime(rows[0]["hour"], "%Y-%m-%d %H:00")
    last = datetime.strptime(rows[-1]["hour"], "%Y-%m-%d %H:00")
    daily = (last - first) / timedelta(hours=1) >= MAX_SLOTS
    fmt, step = ("%Y-%m-%d", timedelta(days=1)) if daily else ("%Y-%m-%d %H:00", timedelta(hours=1))
    have: dict[str, dict] = {}
    for r in rows:
        key = datetime.strptime(r["hour"], "%Y-%m-%d %H:00").strftime(fmt)
        slot = have.setdefault(key, {"hour": key, **dict.fromkeys(keys, 0)})
        for name in keys:
            slot[name] += r.get(name, 0)
    cur = first.replace(hour=0) if daily else first
    out = []
    while cur <= last:
        key = cur.strftime(fmt)
        out.append(have.get(key) or {"hour": key, **dict.fromkeys(keys, 0)})
        cur += step
    return out


def hour_label(key: str) -> str:
    if len(key) == 10:
        return datetime.strptime(key, "%Y-%m-%d").strftime("%d/%m")
    return datetime.strptime(key, "%Y-%m-%d %H:00").strftime("%d/%m %Hh")


# ---------- findings (the "lessons the record shows") ----------


def findings(d: dict) -> list[dict]:
    """Rule-based observations from the record, each with evidence and a lever."""
    s, ci, cl = d["summary"], d["ci"], d["claude"]
    out = []

    def add(theme: str, sev: str, title: str, evidence: str, lever: str, proof: str) -> None:
        out.append(
            {
                "theme": theme,
                "severity": sev,
                "title": title,
                "evidence": evidence,
                "lever": lever,
                "proof": proof,
            }
        )

    if s["merged_red"]:
        add(
            "CI",
            "high",
            "Changes merged while CI was red",
            f"{plural(s['merged_red'], 'merged pull request')} went in with the last CI run failing.",
            "Make the CI workflow a required status check on <span class='mono'>main</span>.",
            "Merges while red stay at zero.",
        )
    if ci["main_failed"]:
        add(
            "CI",
            "medium",
            "main went red after a merge",
            f"{ci['main_failed']} of {ci['main_runs']} CI runs on <span class='mono'>main</span> failed, although every pull request had passed CI.",
            "Require branches to be up to date before merging, so CI runs on what will land.",
            "No failed CI runs on main.",
        )
    human = s["merged_human"]
    if human and s["reviewed_human"] == 0:
        add(
            "Review",
            "medium",
            "No review by a person",
            f"0 of {human} merged pull requests by a person had a GitHub review; the Cursor review commented on {s['reviewed_cursor']} of them and {s['self_merged']} were merged by their author.",
            "Keep the Cursor review, and ask a person to review changes to the hard-rule areas (threads, TCC, offline).",
            "Changes to <span class='mono'>engine.py</span>, <span class='mono'>recorder.py</span> and <span class='mono'>capture/</span> carry a human review.",
        )
    if s["review_rounds"]:
        per = s["review_findings"] / s["review_rounds"]
        add(
            "Review",
            "low" if per < 1 else "medium",
            "Every Cursor review round found something"
            if per >= 1
            else "Cursor reviews mostly came back clean",
            f"{plural(s['review_rounds'], 'review round')} posted {plural(s['review_findings'], 'finding')} ({per:.1f} per round).",
            "Run the four review lanes locally before opening a pull request, so the GitHub round confirms rather than discovers.",
            "Findings per round fall.",
        )
    if s["open"] and s["open"] >= max(3, s["merged_human"] // 2):
        stale = [p for p in d["prs"] if p["state"] == "open" and (p["age_min"] or 0) > 24 * 60]
        add(
            "Flow",
            "medium",
            "Open pull requests pile up",
            f"{s['open']} pull requests are open against {s['merged']} merged; {len(stale)} have been open over a day. The peak was {d['series']['peak']['n']} open at once.",
            "Limit work in progress: merge or close before opening the next stack.",
            "Open pull requests stay under five.",
        )
    if s["closed_unmerged"]:
        add(
            "Flow",
            "low",
            "Work thrown away",
            f"{plural(s['closed_unmerged'], 'pull request')} closed without merging.",
            "Settle the approach in the issue before writing the change.",
            "Closed-unmerged stays rare.",
        )
    if cl["cost"] and (cl["unlinked_share"] or 0) > 0.3:
        add(
            "Cost",
            "medium",
            "Most Claude spend cannot be tied to a pull request",
            f"{cl['unlinked_share']:.0%} of the recorded {money(cl['cost'])} ran on branches with no pull request (often <span class='mono'>main</span> or detached worktrees).",
            "Start each piece of work on its own branch before the first prompt.",
            "The unlinked share falls under 30%.",
        )
    if s["issues_open"] and s["issues_unassigned"] == s["issues_open"]:
        add(
            "Backlog",
            "low",
            "Open issues have no owner",
            f"All {s['issues_open']} open issues are unassigned.",
            "Assign an issue when work starts (the claim-before-starting rule).",
            "Every in-progress issue has an assignee.",
        )
    return out


# ---------- main page ----------


CSS = Path(__file__).with_name("report.css")
JS = Path(__file__).with_name("report.js")


def render(d: dict) -> str:
    s, ser, cm, ci, cl = d["summary"], d["series"], d["commits"], d["ci"], d["claude"]
    prs, issues = d["prs"], d["issues"]
    merged = [p for p in prs if p["state"] == "merged"]
    people = [p for p in s["by_person"] if not p["bot"]]
    names = ", ".join(p["name"] for p in people) or "nobody"
    start, end = d["window"]["start"], d["window"]["end"]
    span_h = (ts(end) - ts(start)).total_seconds() / 3600
    gen = ts(d["generated_at"])
    refresh = d.get("refresh") or {}
    stale_after = refresh.get("minutes")
    stale = bool(stale_after) and datetime.now(UTC) - gen > timedelta(minutes=2 * stale_after)
    fs = findings(d)

    # ---- headline ----
    h1 = f"{span_h / 24:.1f} days on GitHub, {plural(len(people), 'person', 'people')}, {plural(s['merged'], 'merged pull request')}: what the RecAndCC build's trail says"
    top5 = [
        f"<b>{plural(s['merged'], 'pull request')} merged ({s['merged_human']} by a person, {s['merged_bot']} by Dependabot) and {s['issues_closed']} of {s['issues']} issues closed</b> in {plural(ser['active_hours'], 'active hour')}; {s['open']} pull requests are still open.",
        f"<b>The typical change is small and fast:</b> a median of {n(s['lines_median'])} lines across {n(s['files_median'])} files, open to merge in {dur(s['cycle_median'])} (p90 {dur(s['cycle_p90'])}).",
        f"<b>Review is automated:</b> the Cursor review (four parallel reviewer lanes) posted {plural(s['review_rounds'], 'round')} with {plural(s['review_findings'], 'finding')}; no person reviewed a pull request on GitHub, and {pct(s['self_merged'], s['merged_human'])} of human merges were pressed by the author.",
        f"<b>CI held:</b> {pct(ci['ci_passed'], ci['ci_passed'] + ci['ci_failed'])} of {ci['ci_runs']} CI runs passed and {plural(s['merged_red'], 'merge')} went in while CI was red; {ci['main_failed']} runs on main failed.",
        f"<b>Claude did most of the writing:</b> {pct(cm['ai'], cm['authored'])} of authored commits on main carry a Claude trailer (a floor, not a rate), across {plural(cl['sessions'], 'session')}, {plural(cl['subagents'], 'helper agent')} and an estimated {money(cl['cost'])} at API list prices.",
    ]
    headline = section(
        "tldr",
        "Headline",
        f"RecAndCC is an always-on menu-bar recorder that turns calls into local transcripts. This page reads the repository's own record from the first commit ({uk(start)}) to {uk(end)}, UK time: built by {e(names)} through Claude Code agents, with a Cursor agent reviewing every pull request.",
        '<div class="kpis">'
        + kpi(
            n(s["merged"]),
            "pull requests merged",
            f"{s['merged_human']} by a person · {s['merged_bot']} by Dependabot · {s['open']} open · {s['closed_unmerged']} closed unmerged",
            True,
        )
        + kpi(
            f"{s['issues_closed']} of {s['issues']}",
            "issues closed",
            f"{s['issues_open']} open · {s['issues_unassigned']} unassigned",
            True,
        )
        + kpi(
            dur(s["cycle_median"]),
            "median pull request, open to merge",
            f"p90 {dur(s['cycle_p90'])} · slowest {dur(s['cycle_max'])}",
            True,
        )
        + kpi(
            n(s["additions"]),
            "lines added in merged pull requests",
            f"{n(s['deletions'])} deleted · {n(s['file_changes'])} file changes",
            True,
        )
        + "</div><div class='kpis'>"
        + kpi(
            pct(cm["ai"], cm["authored"]),
            "authored commits with a Claude trailer",
            "a floor, not a rate",
        )
        + kpi(
            pct(s["reviewed_human"], s["merged"]),
            "merges reviewed by a person",
            f"Cursor reviewed {s['reviewed_cursor']} of {s['merged_human']} human merges",
        )
        + kpi(
            pct(ci["ci_passed"], ci["ci_passed"] + ci["ci_failed"]),
            "CI runs that passed",
            f"{ci['ci_failed']} failed of {ci['ci_runs']} · {s['merged_red']} merges went in red",
        )
        + kpi(
            money(cl["cost"]),
            "estimated Claude cost",
            f"{big(cl['tokens'])} tokens · API list prices",
        )
        + kpi(n(ser["peak"]["n"]), "pull requests open at the peak", uk(ser["peak"]["at"]))
        + kpi(
            f"{s['merged'] / max(ser['active_hours'], 1):.1f}",
            "merges per active hour",
            f"{ser['active_hours']} hours with activity in a {span_h:.0f} h window",
        )
        + "</div>"
        + callout("Five things to take from this page", top5, "good"),
    )
    if stale:
        headline = headline.replace(
            "<h2>Headline</h2>",
            '<h2>Headline</h2><p class="stale">Stale: the scheduled refresh has not run.</p>',
            1,
        )

    # ---- what was measured ----
    measured = section(
        "how",
        "What was measured, and how to read it",
        "This page reads what the project left behind rather than what anyone remembers. Each source says what it can prove and what it cannot.",
        '<div class="grid2">'
        + callout(
            "Pull requests, issues and CI, from GitHub",
            [
                f"{s['prs']} pull requests with their commits, reviews and comments; {s['issues']} issues; {ci['runs_total']} workflow runs, {ci['ci_runs']} of them the CI workflow (lint and tests on macOS and Windows).",
                "Proves what was proposed, merged, reviewed and checked, and when.",
                "Cannot prove how much thought went into a change, or work done outside GitHub.",
            ],
            "good",
        )
        + callout(
            "Commit trailers",
            [
                f"{cm['authored']} authored commits on main ({cm['merges']} merge commits left out).",
                "A commit counts as AI-assisted when its message carries a <span class='mono'>Co-Authored-By: Claude</span> line.",
                "A floor, not a rate: an unmarked commit is not proof it was written by hand.",
            ],
            "good",
        )
        + callout(
            "Cursor review comments",
            [
                f"{s['review_rounds']} review comments posted by the <span class='mono'>Cursor review</span> workflow, one per pushed head, with four reviewer lanes each.",
                'A finding is a bullet in a lane that did not say "No issues found".',
                "Cannot show whether a finding was right, or how it was resolved.",
            ],
            "warn",
        )
        + callout(
            "Claude Code session logs (local)",
            [
                f"{cl['sessions']} sessions and {cl['subagents']} helper agents on this machine, read for counts and token usage only; no prompt or transcript text is copied.",
                "Cost is an estimate at Anthropic list prices; a subscription bills differently.",
                "Misses work on other machines and in claude.ai chat.",
            ],
            "warn",
        )
        + "</div>",
    )

    # ---- delivery ----
    hours = fill_hours(ser["per_hour"], ["opened", "merged"])
    t0 = epoch(start) if start else 0
    t1 = epoch(end)
    xl = day_labels(t0, t1)
    lanes = [p["name"] for p in s["by_person"]]
    swim = []
    for p in prs:
        end_at = p["merged"] or p["closed"] or end
        color = (
            "--s1" if p["state"] == "merged" else ("--gray" if p["state"] == "closed" else "--warn")
        )
        swim.append(
            {
                "lane": p["author"],
                "start": epoch(p["opened"]),
                "end": epoch(end_at),
                "color": color,
                "tip": f"#{p['number']} {p['title'][:70]} · {p['state']} · {dur(p['cycle_min'] or p['age_min'])}",
            }
        )
    by_day: dict[str, dict] = {}
    for p in prs:
        k = ts(p["opened"]).astimezone(UK).strftime("%a %d/%m")
        by_day.setdefault(k, {"opened": 0, "merged": 0, "lines": 0})["opened"] += 1
    for p in merged:
        k = ts(p["merged"]).astimezone(UK).strftime("%a %d/%m")
        row = by_day.setdefault(k, {"opened": 0, "merged": 0, "lines": 0})
        row["merged"] += 1
        row["lines"] += p["additions"]
    commits_by_day = {
        datetime.fromisoformat(x["day"]).strftime("%a %d/%m"): x for x in cm["by_day"]
    }
    delivery = section(
        "delivery",
        "Delivery and throughput",
        f"The first commit landed at {uk(start)} and the first pull request opened at {uk(min((p['opened'] for p in prs), default=None))}. {plural(ser['active_hours'], 'clock hour')} saw a commit or a pull request event. {s['open']} pull requests are open now.",
        figure(
            "c-timeline",
            "Pull requests opened and merged per hour",
            "grouped bars · UK time",
            charts.bars(
                [hour_label(h["hour"]) for h in hours],
                [
                    {"name": "opened", "color": "--s1", "values": [h["opened"] for h in hours]},
                    {"name": "merged", "color": "--s3", "values": [h["merged"] for h in hours]},
                ],
            ),
            ["Hour", "opened", "merged"],
            [[h["hour"], h["opened"], h["merged"]] for h in hours],
            [("opened", "--s1"), ("merged", "--s3")],
        )
        + '<div class="grid2">'
        + figure(
            "c-inflight",
            "Pull requests open at once",
            "step line · open to merge or close",
            charts.steps(
                [
                    {
                        "name": "open",
                        "color": "--s1",
                        "points": [(epoch(x["at"]), x["n"]) for x in ser["inflight"]],
                    }
                ],
                xl,
            ),
            ["At", "open"],
            [[uk(x["at"]), x["n"]] for x in ser["inflight"]],
        )
        + figure(
            "c-cummerged",
            "Cumulative merges",
            "area · one series",
            charts.steps(
                [
                    {
                        "name": "merged",
                        "color": "--s3",
                        "points": [(epoch(x["at"]), x["n"]) for x in ser["cumulative"]],
                    }
                ],
                xl,
                area=True,
            ),
            ["At", "merged"],
            [[uk(x["at"]), x["n"]] for x in ser["cumulative"]],
        )
        + "</div>"
        + figure(
            "c-swim",
            "Every pull request as a bar, one lane per author",
            "swimlane · open to merge · open bars run to now · hover for the pull request",
            charts.swimlanes(lanes, swim, t0, t1, xl),
            ["PR", "author", "state", "opened", "open to merge"],
            [
                [pr_link(p), e(p["author"]), p["state"], uk(p["opened"]), dur(p["cycle_min"])]
                for p in prs
            ],
            [("merged", "--s1"), ("closed unmerged", "--gray"), ("open now", "--warn")],
        )
        + "<h3>Day by day</h3>"
        + table(
            ["Day", "opened", "merged", "lines added", "commits on main", "Claude-signed"],
            [
                [
                    k,
                    v["opened"],
                    v["merged"],
                    n(v["lines"]),
                    commits_by_day.get(k, {}).get("commits", 0),
                    pct(
                        commits_by_day.get(k, {}).get("ai", 0),
                        commits_by_day.get(k, {}).get("commits", 0),
                    ),
                ]
                for k, v in sorted(
                    by_day.items(), key=lambda kv: datetime.strptime(kv[0][4:], "%d/%m")
                )
            ],
            {1, 2, 3, 4, 5},
        ),
    )

    # ---- backlog ----
    label_counts: dict[str, list[int]] = {}
    for i in issues:
        for lab in i["labels"] or ["(no label)"]:
            row = label_counts.setdefault(lab, [0, 0])
            row[0 if i["state"] == "open" else 1] += 1
    labs = sorted(label_counts, key=lambda k: -sum(label_counts[k]))
    days: dict[str, list[int]] = {}
    for i in issues:
        days.setdefault(i["opened"][:10], [0, 0])[0] += 1
        if i["closed"]:
            days.setdefault(i["closed"][:10], [0, 0])[1] += 1
    open_issues = [i for i in issues if i["state"] == "open"]
    rm = d["docs"]["roadmap"]
    rm_sections: dict[str, list[int]] = {}
    for r in rm:
        rm_sections.setdefault(r["section"], [0, 0])[0 if r["done"] else 1] += 1
    backlog = section(
        "prd",
        "Backlog burndown",
        f"The backlog lives in GitHub issues and <span class='mono'>docs/ROADMAP.md</span>. {s['issues_closed']} of {s['issues']} issues are closed; {s['issues_open']} are open, {s['issues_unassigned']} of them with nobody assigned. {plural(s['issues_closed_no_pr'], 'issue was', 'issues were')} closed without a merged pull request naming it.",
        '<div class="grid2">'
        + figure(
            "c-labels",
            "Issues by label and state",
            "stacked bars",
            charts.bars(
                labs,
                [
                    {"name": "open", "color": "--s2", "values": [label_counts[k][0] for k in labs]},
                    {
                        "name": "closed",
                        "color": "--s3",
                        "values": [label_counts[k][1] for k in labs],
                    },
                ],
                stacked=True,
            ),
            ["Label", "open", "closed"],
            [[e(k), *label_counts[k]] for k in labs],
            [("open", "--s2"), ("closed", "--s3")],
        )
        + figure(
            "c-issuesday",
            "Issues opened and closed per day",
            "grouped bars · UTC day",
            charts.bars(
                sorted(days),
                [
                    {
                        "name": "opened",
                        "color": "--s2",
                        "values": [days[k][0] for k in sorted(days)],
                    },
                    {
                        "name": "closed",
                        "color": "--s3",
                        "values": [days[k][1] for k in sorted(days)],
                    },
                ],
            ),
            ["Day", "opened", "closed"],
            [[k, *days[k]] for k in sorted(days)],
            [("opened", "--s2"), ("closed", "--s3")],
        )
        + "</div>"
        + (
            figure(
                "c-roadmap",
                "Roadmap items by section",
                "stacked bars · docs/ROADMAP.md checkboxes",
                charts.hbars(list(rm_sections), [sum(v) for v in rm_sections.values()]),
                ["Section", "done", "not done"],
                [[e(k), *v] for k, v in rm_sections.items()],
            )
            if rm_sections
            else ""
        )
        + "<h3>Open issues that need an owner</h3>"
        + table(
            ["Issue", "title", "labels", "opened", "assigned"],
            [
                [
                    issue_link(i),
                    e(i["title"]),
                    e(", ".join(i["labels"]) or "–"),
                    uk(i["opened"]),
                    e(", ".join(i["assignees"]))
                    if i["assignees"]
                    else '<span class="pill warn">nobody</span>',
                ]
                for i in open_issues
            ],
        ),
    )

    # ---- size ----
    big10 = sorted(merged, key=lambda p: -p["lines"])[:10]
    groups = s["cycle_groups"]
    size = section(
        "size",
        "Pull request size and shape",
        f"The median merged pull request changed {n(s['lines_median'])} lines across {n(s['files_median'])} files in {n(s['commits_median'])} commit(s). {plural(s['big_prs'], 'merged pull request')} changed 1,000 lines or more.",
        '<div class="grid3">'
        + "".join(
            figure(
                fid,
                title,
                "histogram",
                charts.bars(
                    [b["label"] for b in hist],
                    [{"name": "pull requests", "color": "--s1", "values": [b["n"] for b in hist]}],
                    height=220,
                ),
                ["Bucket", "pull requests"],
                [[b["label"], b["n"]] for b in hist],
            )
            for fid, title, hist in (
                ("c-lineshist", "Lines changed per merged pull request", s["hist_lines"]),
                ("c-fileshist", "Files changed", s["hist_files"]),
                ("c-commitshist", "Commits per pull request", s["hist_commits"]),
            )
        )
        + "</div><h3>What moves cycle time</h3><div class='grid2'>"
        + figure(
            "c-groups",
            "Median open to merge, by group",
            "horizontal bars · minutes",
            charts.hbars(
                [f"{g['group']}: {g['label']}" for g in groups if g["n"]],
                [round(g["median"] or 0, 1) for g in groups if g["n"]],
                unit=" min",
            ),
            ["Group", "PRs", "median", "p90"],
            [
                [e(f"{g['group']}: {g['label']}"), g["n"], dur(g["median"]), dur(g["p90"])]
                for g in groups
            ],
        )
        + "<div><p class='small muted'>Correlation with cycle time across merged pull requests (Pearson r; 0 is none, 1 is perfect; – means too little variation to tell):</p>"
        + table(
            ["Factor", "r"],
            [
                [e(c["factor"]), "–" if c["r"] is None else f"{c['r']:+.2f}"]
                for c in s["correlations"]
            ],
            {1},
        )
        + "</div></div><h3>The ten biggest merged pull requests</h3>"
        + table(
            ["PR", "title", "author", "lines", "files", "cycle"],
            [
                [
                    pr_link(p),
                    e(p["title"]),
                    e(p["author"]),
                    n(p["lines"]),
                    p["files"],
                    dur(p["cycle_min"]),
                ]
                for p in big10
            ],
            {3, 4, 5},
        ),
    )

    # ---- cycle ----
    slow = sorted(merged, key=lambda p: -p["cycle_min"])[:10]
    fast = sorted(merged, key=lambda p: p["cycle_min"])[:5]
    still = sorted((p for p in prs if p["state"] == "open"), key=lambda p: -(p["age_min"] or 0))
    pc = [p for p in s["by_person"] if p["cycle_median"] is not None]
    cycle = section(
        "cycle",
        "Cycle time",
        f"Half of all merged pull requests went from open to merged in {dur(s['cycle_median'])}; 90% within {dur(s['cycle_p90'])}. Once CI was green, the median wait to merge was {dur(s['wait_after_green'])}: the time goes into review rounds and fixes, not into waiting for someone to press merge.",
        '<div class="grid2">'
        + figure(
            "c-cyclehist",
            "Open to merge",
            "histogram · merged pull requests",
            charts.bars(
                [b["label"] for b in s["hist_cycle"]],
                [
                    {
                        "name": "pull requests",
                        "color": "--s1",
                        "values": [b["n"] for b in s["hist_cycle"]],
                    }
                ],
            ),
            ["Bucket", "pull requests"],
            [[b["label"], b["n"]] for b in s["hist_cycle"]],
        )
        + figure(
            "c-cycledev",
            "Median open to merge, per author",
            "horizontal bars · minutes",
            charts.hbars(
                [p["name"] for p in pc],
                [round(p["cycle_median"], 1) for p in pc],
                ["--s2" if p["bot"] else "--s1" for p in pc],
                " min",
            ),
            ["Author", "median"],
            [[e(p["name"]), dur(p["cycle_median"])] for p in pc],
            [("person", "--s1"), ("bot", "--s2")],
        )
        + "</div><h3>The ten slowest merges</h3>"
        + table(
            ["PR", "title", "cycle", "lines", "review rounds", "CI failures"],
            [
                [
                    pr_link(p),
                    e(p["title"]),
                    dur(p["cycle_min"]),
                    n(p["lines"]),
                    p["review_rounds"],
                    p["ci_failures"],
                ]
                for p in slow
            ],
            {2, 3, 4, 5},
        )
        + "<h3>The fastest merges</h3>"
        + table(
            ["PR", "title", "author", "cycle"],
            [[pr_link(p), e(p["title"]), e(p["author"]), dur(p["cycle_min"])] for p in fast],
            {3},
        )
        + "<h3>Still open</h3>"
        + table(
            ["PR", "title", "open for", "lines", "review rounds", "findings"],
            [
                [
                    pr_link(p),
                    e(p["title"]),
                    dur(p["age_min"]),
                    n(p["lines"]),
                    p["review_rounds"],
                    p["review_findings"],
                ]
                for p in still
            ],
            {2, 3, 4, 5},
        ),
    )

    # ---- quality ----
    wf = ci["workflows"]
    ci_hours = fill_hours(ci["per_hour"], ["passed", "failed"])
    reviewed = sorted((p for p in prs if p["review_rounds"]), key=lambda p: -p["review_findings"])[
        :10
    ]
    closed = [p for p in prs if p["state"] == "closed"]
    quality = section(
        "quality",
        "Quality gates",
        f"No person reviewed a pull request on GitHub: {s['reviewed_human']} of {s['merged']} merges had a human review and {s['approved']} a formal approval. The review step is the <span class='mono'>Cursor review</span> workflow, which runs four reviewer lanes (correctness and threading, security and privacy, platforms, tests and docs) and posts one comment per pushed head.",
        '<div class="kpis">'
        + kpi(
            pct(s["reviewed_human"], s["merged"]),
            "merges reviewed by a person",
            f"{s['changes_requested']} changes requested · {s['approved']} approvals",
        )
        + kpi(
            f"{s['reviewed_cursor']} of {s['merged_human']}",
            "human merges with a Cursor review",
            f"{s['review_rounds']} rounds · {s['review_findings']} findings",
        )
        + kpi(
            pct(s["self_merged"], s["merged_human"]),
            "human merges pressed by the author",
            "a single-developer repository",
        )
        + kpi(
            pct(s["ever_red"], s["merged"]),
            "merged pull requests with a red CI run",
            f"{s['fix_pushes']} fix pushes after a red run · {s['merged_red']} merged red",
        )
        + "</div><div class='grid2'>"
        + figure(
            "c-reviews",
            "Cursor review rounds per human pull request",
            "bars · one series",
            charts.bars(
                [b["label"] for b in s["hist_reviews"]],
                [
                    {
                        "name": "pull requests",
                        "color": "--s1",
                        "values": [b["n"] for b in s["hist_reviews"]],
                    }
                ],
                height=220,
            ),
            ["Rounds", "pull requests"],
            [[b["label"], b["n"]] for b in s["hist_reviews"]],
        )
        + figure(
            "c-cifix",
            "Fix pushes after a red CI run, per merged pull request",
            "bars · one series",
            charts.bars(
                [b["label"] for b in s["hist_fix"]],
                [
                    {
                        "name": "pull requests",
                        "color": "--s1",
                        "values": [b["n"] for b in s["hist_fix"]],
                    }
                ],
                height=220,
            ),
            ["Pushes", "pull requests"],
            [[b["label"], b["n"]] for b in s["hist_fix"]],
        )
        + "</div>"
        + figure(
            "c-ciday",
            "CI runs per hour, red and green",
            "stacked bars · status colours",
            charts.bars(
                [hour_label(h["hour"]) for h in ci_hours],
                [
                    {
                        "name": "passed",
                        "color": "--good",
                        "values": [h["passed"] for h in ci_hours],
                    },
                    {
                        "name": "failed",
                        "color": "--crit",
                        "values": [h["failed"] for h in ci_hours],
                    },
                ],
                stacked=True,
            ),
            ["Hour", "passed", "failed"],
            [[h["hour"], h["passed"], h["failed"]] for h in ci["per_hour"]],
            [("passed", "--good"), ("failed", "--crit")],
        )
        + "<h3>Workflows</h3>"
        + table(
            ["Workflow", "runs", "success", "failure", "other"],
            [
                [
                    e(w["name"]),
                    w["total"],
                    w.get("success", 0),
                    w.get("failure", 0),
                    w["total"] - w.get("success", 0) - w.get("failure", 0),
                ]
                for w in wf
            ],
            {1, 2, 3, 4},
        )
        + "<h3>Pull requests the Cursor review flagged most</h3>"
        + table(
            ["PR", "title", "rounds", "findings", "state"],
            [
                [pr_link(p), e(p["title"]), p["review_rounds"], p["review_findings"], p["state"]]
                for p in reviewed
            ],
            {2, 3},
        )
        + "<h3>Closed without merging</h3>"
        + (
            table(
                ["PR", "title", "opened", "closed"],
                [[pr_link(p), e(p["title"]), uk(p["opened"]), uk(p["closed"])] for p in closed],
            )
            if closed
            else "<p>Nothing was closed without merging.</p>"
        ),
    )

    # ---- team ----
    bp = s["by_person"]
    cbp = cm["by_person"]
    team = section(
        "team",
        "Team working patterns",
        f"This is not a leaderboard. The repository has {plural(len(people), 'human author')} ({e(names)}); Dependabot opens dependency updates. Most code is written by Claude Code agents running in parallel git worktrees, one branch per change.",
        '<div class="grid2">'
        + figure(
            "c-devprs",
            "Pull requests per author",
            "stacked bars · merged, open, closed unmerged",
            charts.bars(
                [p["name"] for p in bp],
                [
                    {"name": "merged", "color": "--s1", "values": [p["merged"] for p in bp]},
                    {"name": "open", "color": "--warn", "values": [p["open"] for p in bp]},
                    {
                        "name": "closed unmerged",
                        "color": "--gray",
                        "values": [p["closed"] for p in bp],
                    },
                ],
                stacked=True,
                height=220,
            ),
            ["Author", "merged", "open", "closed"],
            [[e(p["name"]), p["merged"], p["open"], p["closed"]] for p in bp],
            [("merged", "--s1"), ("open", "--warn"), ("closed unmerged", "--gray")],
        )
        + figure(
            "c-devlines",
            "Lines changed per author, merged pull requests",
            "bars · one series",
            charts.hbars([p["name"] for p in bp], [p["lines"] for p in bp]),
            ["Author", "lines"],
            [[e(p["name"]), n(p["lines"])] for p in bp],
        )
        + figure(
            "c-devcommits",
            "Commits on main per author, with the Claude-signed share",
            "stacked bars",
            charts.bars(
                [p["name"] for p in cbp],
                [
                    {"name": "Claude trailer", "color": "--s1", "values": [p["ai"] for p in cbp]},
                    {
                        "name": "no trailer",
                        "color": "--gray",
                        "values": [p["commits"] - p["ai"] for p in cbp],
                    },
                ],
                stacked=True,
                height=220,
            ),
            ["Author", "commits", "Claude-signed"],
            [[e(p["name"]), p["commits"], p["ai"]] for p in cbp],
            [("Claude trailer", "--s1"), ("no trailer", "--gray")],
        )
        + figure(
            "c-devhours",
            "Hours with at least one commit, per author",
            "bars · one series",
            charts.hbars([p["name"] for p in cbp], [p["active_hours"] for p in cbp]),
            ["Author", "hours"],
            [[e(p["name"]), p["active_hours"]] for p in cbp],
        )
        + "</div>"
        + figure(
            "c-devhod",
            "When each author commits (UK hour of day)",
            "heat strips · one row per author",
            charts.heatstrips([(p["name"], p["hod"]) for p in cbp]),
            ["Author", *[f"{h:02d}" for h in range(24)]],
            [[e(p["name"]), *p["hod"]] for p in cbp],
        ),
    )

    # ---- work by type ----
    types = cm["types"]
    work = section(
        "work",
        "Work by type",
        "What the commits on main were, read from their conventional-commit prefix (<span class='mono'>feat:</span>, <span class='mono'>fix:</span>, ...). Commits made by merging a GitHub pull request with its title carry no prefix and count as \"other\".",
        figure(
            "c-types",
            "Commits on main by conventional prefix",
            "horizontal bars",
            charts.hbars([t["type"] for t in types], [t["n"] for t in types]),
            ["Prefix", "commits"],
            [[e(t["type"]), t["n"]] for t in types],
        )
        + "<h3>Most discussed issues</h3>"
        + table(
            ["Issue", "title", "comments", "state"],
            [
                [issue_link(i), e(i["title"]), i["comments"], i["state"]]
                for i in sorted(issues, key=lambda i: -i["comments"])[:10]
                if i["comments"]
            ],
            {2},
        ),
    )

    # ---- AI ----
    ai = section(
        "ai",
        "AI signature and Claude use",
        f"About {pct(cm['ai'], cm['authored'])} of the {cm['authored']} authored commits on main carry a Claude trailer. That is a floor, not a rate: an unmarked commit is not proof of hand work, and Dependabot's commits never carry one. The session logs on this machine count {plural(cl['sessions'], 'session')}, {plural(cl['prompts'], 'prompt')} and {plural(cl['subagents'], 'helper agent')}.",
        '<div class="kpis">'
        + kpi(
            pct(cm["ai"], cm["authored"]),
            "commits with a Claude trailer",
            f"{cm['ai']} of {cm['authored']} authored commits",
        )
        + kpi(
            big(cl["tokens"]),
            "tokens recorded",
            f"{pct(cl['kinds'].get('cache_read', 0), cl['tokens'])} re-read from cache",
        )
        + kpi(money(cl["cost"]), "estimated cost", "Anthropic API list prices")
        + kpi(
            pct(cl["unlinked_cost"], cl["cost"]),
            "of that cost not tied to a pull request",
            "session branch has no pull request",
        )
        + "</div><div class='grid2'>"
        + figure(
            "c-aiday",
            "Claude-signed share of commits, by day",
            "bars · percent",
            charts.bars(
                [x["day"] for x in cm["by_day"]],
                [
                    {
                        "name": "signed",
                        "color": "--s1",
                        "values": [
                            round(100 * x["ai"] / x["commits"]) if x["commits"] else 0
                            for x in cm["by_day"]
                        ],
                    }
                ],
                height=220,
                unit="%",
            ),
            ["Day", "commits", "signed"],
            [[x["day"], x["commits"], x["ai"]] for x in cm["by_day"]],
        )
        + figure(
            "c-costday",
            "Estimated Claude cost per day",
            "bars · USD",
            charts.bars(
                [x["day"] for x in cl["by_day"]],
                [{"name": "cost", "color": "--s7", "values": [x["cost"] for x in cl["by_day"]]}],
                height=220,
            ),
            ["Day", "tokens", "cost"],
            [[x["day"], big(x["tokens"]), money(x["cost"])] for x in cl["by_day"]],
        )
        + figure(
            "c-models",
            "Cost by model",
            "horizontal bars · USD",
            charts.hbars([m["model"] for m in cl["by_model"]], [m["cost"] for m in cl["by_model"]]),
            ["Model", "tokens", "cost"],
            [[e(m["model"]), big(m["tokens"]), money(m["cost"])] for m in cl["by_model"]],
        )
        + figure(
            "c-kinds",
            "Where the tokens went",
            "horizontal bars",
            charts.hbars(
                ["cache read", "cache write", "output", "input"],
                [cl["kinds"].get(k, 0) for k in ("cache_read", "cache_write", "output", "input")],
            ),
            ["Kind", "tokens"],
            [
                [k, n(cl["kinds"].get(k, 0))]
                for k in ("cache_read", "cache_write", "output", "input")
            ],
        )
        + "</div>",
    )

    # ---- lessons ----
    sev_order = ["high", "medium", "low"]
    themes = sorted({f["theme"] for f in fs})
    lessons = section(
        "lessons",
        "Lessons: what the record shows cost time",
        f"The repository keeps no lessons file, so this section is built from the record: {plural(len(fs), 'observation')}, each with the evidence behind it. They are recomputed on every refresh and disappear when the evidence does.",
        (
            figure(
                "c-lessons",
                "Observations by theme and severity",
                "stacked bars · status colours",
                charts.bars(
                    themes,
                    [
                        {
                            "name": sv,
                            "color": c,
                            "values": [
                                sum(1 for f in fs if f["theme"] == t and f["severity"] == sv)
                                for t in themes
                            ],
                        }
                        for sv, c in zip(sev_order, ["--crit", "--warn", "--info"], strict=True)
                    ],
                    stacked=True,
                    height=200,
                ),
                ["Theme", *sev_order],
                [
                    [
                        t,
                        *[
                            sum(1 for f in fs if f["theme"] == t and f["severity"] == sv)
                            for sv in sev_order
                        ],
                    ]
                    for t in themes
                ],
                [("high", "--crit"), ("medium", "--warn"), ("low", "--info")],
            )
            if fs
            else ""
        )
        + "".join(
            f'<div class="callout {"crit" if f["severity"] == "high" else "warn" if f["severity"] == "medium" else ""}"><b class="t"><span class="pill {"crit" if f["severity"] == "high" else "warn" if f["severity"] == "medium" else "info"}">{f["severity"]}</span> {f["title"]}</b>{f["evidence"]}</div>'
            for f in sorted(fs, key=lambda f: sev_order.index(f["severity"]))
        ),
    )

    # ---- open questions ----
    oq = [
        i
        for i in open_issues
        if set(i["labels"]) & {"question", "future work", "enhancement", "process"}
    ] or open_issues
    adrs = d["docs"]["adrs"]
    questions = section(
        "deferrals",
        "Open questions and deferrals",
        f"Open questions come from open issues ({len(oq)} shown, labelled question, future work, enhancement or process where labelled) and from roadmap items not yet ticked. Decisions already made are the {plural(len(adrs), 'architecture decision record')} in <span class='mono'>docs/adr/</span>.",
        table(
            ["Issue", "title", "labels", "assigned"],
            [
                [
                    issue_link(i),
                    e(i["title"]),
                    e(", ".join(i["labels"]) or "–"),
                    e(", ".join(i["assignees"]) or "nobody"),
                ]
                for i in oq
            ],
        )
        + "<h3>Roadmap items not yet done</h3>"
        + table(["Section", "item"], [[e(r["section"]), e(r["text"])] for r in rm if not r["done"]])
        + "<h3>Decisions on record</h3>"
        + table(
            ["ADR", "title", "status"],
            [[e(a["id"]), e(a["title"]), e(a["status"] or "–")] for a in adrs],
        ),
    )

    # ---- data quality ----
    dq_rows = [
        [
            "warn",
            "Review does not show up as GitHub reviews.",
            f"The Cursor review posts comments, not GitHub reviews, so GitHub counts {s['merged']} of {s['merged']} merges as unreviewed. This page counts the comments as review rounds.",
        ],
        [
            "warn",
            "Claude usage covers this machine only.",
            f"{cl['sessions']} sessions were read from the local Claude Code logs. Cost is an estimate at list prices; a Claude subscription bills differently.",
        ],
        [
            "info",
            "The AI figure is a floor.",
            "Only commits with a Claude trailer count. Squash merges keep the trailers of the squashed commits only when the merge message includes them.",
        ],
        [
            "info",
            "Phase, priority and role charts are left out.",
            "The reference report charts issues by phase, priority and role label. This repository has no such labels, so issues are charted by their own labels instead.",
        ],
        [
            "info",
            "Lessons are derived, not reported.",
            "There is no lessons file or debrief; the Lessons section lists rule-based observations from the record.",
        ],
    ]
    dq = section(
        "dq",
        "Data quality and caveats",
        "Inside this repository the record is complete: every pull request, issue and workflow run GitHub keeps is counted. The caveats are about what the record cannot see.",
        table(
            ["Level", "Finding", "Why it matters"],
            [[f'<span class="pill {lv}">{lv}</span>', e(a), e(b)] for lv, a, b in dq_rows],
        ),
    )

    # ---- recommendations ----
    recos = section(
        "recos",
        "Recommendations",
        "Ordered by severity of the evidence. Each one names the evidence and how to tell it worked; the list is rebuilt from the data on every refresh.",
        '<ol class="reco">'
        + "".join(
            f"<li><b>{f['lever']}</b><span class='ev'>Evidence: {f['evidence']}</span><span class='ev'>Proof it worked: {f['proof']}</span></li>"
            for f in sorted(fs, key=lambda f: sev_order.index(f["severity"]))
        )
        + "</ol>",
    )

    # ---- appendix ----
    pr_rows = [
        [
            pr_link(p),
            e(p["title"]),
            e(p["author"]),
            p["state"],
            uk(p["opened"]),
            dur(p["cycle_min"]),
            n(p["lines"]),
            p["files"],
            p["commits"],
            p["review_rounds"],
            p["review_findings"],
            p["ci_failures"],
        ]
        for p in prs
    ]
    is_rows = [
        [
            issue_link(i),
            e(i["title"]),
            i["state"],
            e(", ".join(i["labels"]) or "–"),
            e(", ".join(i["assignees"]) or "–"),
            uk(i["opened"]),
            uk(i["closed"]),
            " ".join(f"#{x}" for x in i["closed_by_pr"]) or "–",
        ]
        for i in issues
    ]
    ci_rows = [
        [
            f'<a href="{e(r["url"])}">{e(r["workflow"])}</a>',
            e(r["branch"] or ""),
            e(r["event"]),
            e(r["conclusion"]),
            uk(r["at"]),
            r["minutes"],
        ]
        for r in ci["list"]
    ]
    appendix = (
        section(
            "allprs",
            f"Appendix A: all {len(prs)} pull requests",
            "Sortable: click a column heading. Filter by text below.",
            '<div class="filters"><label>Filter <input type="search" data-filter="t-prs" placeholder="title, author, state"></label></div>'
            + table(
                [
                    "PR",
                    "title",
                    "author",
                    "state",
                    "opened",
                    "cycle",
                    "lines",
                    "files",
                    "commits",
                    "review rounds",
                    "findings",
                    "CI failures",
                ],
                pr_rows,
                {5, 6, 7, 8, 9, 10, 11},
                True,
                "t-prs",
            ),
        )
        + section(
            "allissues",
            f"Appendix B: all {len(issues)} issues",
            "Sortable and filterable.",
            '<div class="filters"><label>Filter <input type="search" data-filter="t-issues" placeholder="title, label, state"></label></div>'
            + table(
                ["Issue", "title", "state", "labels", "assigned", "opened", "closed", "closed by"],
                is_rows,
                set(),
                True,
                "t-issues",
            ),
        )
        + section(
            "allci",
            f"Appendix C: CI and review runs ({len(ci_rows)})",
            "The CI and Cursor review workflows, newest first.",
            table(
                ["Workflow", "branch", "event", "result", "started", "minutes"], ci_rows, {5}, True
            ),
        )
        + section(
            "sources",
            "Appendix D: sources and method",
            "Every figure on this page is computed once from one data file, so the words and the charts cannot disagree.",
            callout(
                "How the data was read",
                [
                    f"GitHub, via the <span class='mono'>gh</span> command line at {uk(d['generated_at'])}: {s['prs']} pull requests (each with its commits, reviews and comments), {s['issues']} issues and {ci['runs_total']} workflow runs of <span class='mono'>{e(d['repo'])}</span>.",
                    f"The commit ledger is <span class='mono'>git log {e(d['sources']['ref'])}</span>: {cm['ledger']} commits, {cm['merges']} merge commits left out.",
                    "Cycle time is pull request opened to merged. CI figures count the CI workflow only; a fix push is a commit added after the pull request's first red CI run.",
                    "Claude usage is read from Claude Code's local session logs for this repository's worktrees; each assistant message is counted once. Prices: "
                    + "; ".join(
                        f"{e(k)} ${v['input']}/${v['output']} per million tokens in/out"
                        for k, v in d["prices"].items()
                        if any(m["model"] == k for m in cl["by_model"])
                    )
                    + ", cache writes at 1.25x (5 min) or 2x (1 h) input, cache reads as listed.",
                    "Refresh: <span class='mono'>scripts/report/refresh.sh</span> runs the collector and the renderer; it runs after each merge and at least every six hours while work is active.",
                ],
            ),
        )
        + section(
            "glossary",
            "Appendix E: glossary",
            "",
            table(["Term", "Meaning"], [[a, b] for a, b in GLOSSARY]),
        )
    )

    toc = [
        (
            "Executive",
            [
                ("tldr", "Headline"),
                ("how", "What was measured"),
                ("delivery", "Delivery and throughput"),
                ("prd", "Backlog burndown"),
            ],
        ),
        (
            "Engineering",
            [
                ("size", "Pull request size and shape"),
                ("cycle", "Cycle time"),
                ("quality", "Quality gates"),
                ("team", "Team working patterns"),
                ("work", "Work by type"),
                ("ai", "AI signature and Claude use"),
            ],
        ),
        (
            "Learning",
            [
                ("lessons", "Lessons"),
                ("deferrals", "Open questions"),
                ("dq", "Data quality and caveats"),
                ("recos", "Recommendations"),
            ],
        ),
        (
            "Appendix",
            [
                ("allprs", f"All {len(prs)} pull requests"),
                ("allissues", f"All {len(issues)} issues"),
                ("allci", "CI runs"),
                ("sources", "Sources and method"),
                ("glossary", "Glossary"),
            ],
        ),
    ]
    nav = "".join(
        f"<h2>{g}</h2>" + "".join(f'<a href="#{a}">{e(t)}</a>' for a, t in items)
        for g, items in toc
    )
    cadence = (
        f"Refreshed after every merge and at least every {stale_after // 60} h"
        if stale_after
        else "One-off snapshot, not refreshed"
    )
    meta = (
        f'<span>Repository: <a class="mono" href="https://github.com/{e(d["repo"])}">{e(d["repo"])}</a></span>'
        f"<span>Data window: {uk(start)} to {uk(end)} (UK time)</span><span>Generated: {uk(d['generated_at'])}</span><span>{cadence}</span>"
        + ('<span class="stale">Stale</span>' if stale else "")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{TITLE}</title><meta name="claude:viewport-fit-sides" content="cover"><style>{CSS.read_text(encoding="utf-8")}</style></head>
<body><div class="tip" id="tip"></div><div class="layout">
<nav class="toc"><div class="brand">{TITLE}</div><div class="muted small">{plural(len(people), "person", "people")}, {plural(s["prs"], "pull request")}, Claude Code agents and a Cursor reviewer · {uk(start)} to {uk(end)}</div>{nav}
<div class="topbar"><button class="tbtn" id="themeBtn" type="button">Toggle dark mode</button></div></nav>
<main><header class="hero"><h1>{e(h1)}</h1><div class="sub">This page reads the repository's own record: pull requests, issues, automated checks (CI), the Claude trailers in commit messages and the local Claude Code session logs. It is written for anyone deciding whether to keep building this way, and what to change if so.</div>
<div class="meta">{meta}</div></header>
{headline}{measured}{delivery}{backlog}{size}{cycle}{quality}{team}{work}{ai}{lessons}{questions}{dq}{recos}{appendix}
</main></div><script>{JS.read_text(encoding="utf-8")}</script></body></html>"""


GLOSSARY = [
    (
        "Pull request",
        "A proposed change to the code, opened for checking before it joins the shared version.",
    ),
    ("Merge", "Accepting a pull request into <span class='mono'>main</span>."),
    ("Self-merge", "The author merges their own pull request."),
    (
        "Review round",
        "One Cursor review comment on a pull request: four reviewer lanes read the diff of one pushed head.",
    ),
    ("Finding", "One issue a reviewer lane raised in a review round."),
    (
        "CI",
        "Continuous integration: the lint and test jobs (macOS and Windows) that run on every pull request and push.",
    ),
    ("Red run", "A CI run that failed."),
    ("Fix push", "A commit added to a pull request after its first red CI run."),
    ("Cycle time", "Time from opening a pull request to merging it."),
    ("Median", "The middle value: half were faster, half slower."),
    ("p90", "The value nine in ten stayed under; it shows the slow tail."),
    (
        "Claude trailer",
        "A <span class='mono'>Co-Authored-By: Claude</span> line at the end of a commit message.",
    ),
    ("Floor", "The AI share counts only signed commits, so the true share can only be higher."),
    (
        "Helper agent",
        "A Claude Code subagent a session started for a side task, often in its own git worktree.",
    ),
    (
        "Cache read",
        "Tokens the model re-read from the prompt cache; billed at a fraction of normal input.",
    ),
    (
        "ADR",
        "Architecture decision record: a short file in <span class='mono'>docs/adr/</span> recording one decision and why.",
    ),
]


def fragment(page: str) -> str:
    """The page without its document wrapper, for hosts that add their own (Claude artifacts)."""
    head = page.split("<head>", 1)[1].split("</head>", 1)[0]
    head = re.sub(r'<meta (charset|name="viewport")[^>]*>', "", head)
    body = page.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    return head.strip() + "\n" + body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path(".report") / "report-data.json")
    ap.add_argument("--out", type=Path, default=Path(".report") / "report.html")
    ap.add_argument(
        "--fragment", action="store_true", help="page body without the html/head/body wrapper"
    )
    args = ap.parse_args(argv)
    ap_page = render(json.loads(args.data.read_text(encoding="utf-8")))
    if args.fragment:
        ap_page = fragment(ap_page)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(ap_page, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
