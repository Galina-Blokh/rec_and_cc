"""Collect the build record of this repository into one data file for the report.

Reads GitHub (pull requests, issues, workflow runs and PR comments) through the `gh` CLI,
the commit ledger from local git, and the local Claude Code session logs, then writes
`report-data.json`. Every figure on the report page comes from that file.

Credentials: `gh` uses `GH_TOKEN` from the environment (falling back to `git_api_key_0307`).
The token is never printed, logged or written. Session logs contribute counts only: no
prompt or transcript text leaves this script.

Usage: python scripts/report/collect.py [--repo OWNER/NAME] [--out .report/report-data.json]
       [--offline]  (cache only: no gh call and no git fetch)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger("report.collect")

REPO = "Galina-Blokh/rec_and_cc"
try:
    UK = ZoneInfo("Europe/London")
except ZoneInfoNotFoundError:  # Windows without the tzdata package: report in UTC
    UK = UTC
TIMEOUT = 120  # seconds for any one gh or git call
RETRIES, BACKOFF = 3, 2.0  # retry rate-limited gh calls after 2, 4, 8 s
RATE_LIMITED = re.compile(r"HTTP 429|rate limit", re.I)  # not a bare 403 (permissions)
NOT_FOUND = re.compile(r"HTTP 404|Not Found")
FULL_WALK = timedelta(hours=24)  # walk every list end to end at least this often
MAX_RUN_REFRESH = 20  # unfinished runs refetched by id per collect
CACHE = Path.home() / ".cache" / "recandcc-report"
CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
REVIEW_MARK = "## 🤖 Cursor review"
CI_WORKFLOW = "CI"
REVIEW_WORKFLOW = "Cursor review"
AI_TRAILER = re.compile(r"^Co-Authored-By:\s*Claude\b", re.I | re.M)
CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.I)
PREFIX = re.compile(r"^(\w+)(?:\([^)]*\))?!?:")
# Anthropic list prices, USD per million tokens: input, output, cache read. Cache writes are
# 1.25x input (5 min) and 2x input (1 h). Unknown models are priced as Opus 5.5.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


# ---------- small helpers ----------


def ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat(timespec="seconds") if value else None


def minutes(a: datetime | None, b: datetime | None) -> float | None:
    return (b - a).total_seconds() / 60 if a and b else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def hour_key(value: datetime) -> str:
    return value.astimezone(UK).strftime("%Y-%m-%d %H:00")


def day_key(value: datetime) -> str:
    return value.astimezone(UK).strftime("%Y-%m-%d")


def is_bot(login: str | None) -> bool:
    return not login or login.endswith("[bot]") or login in {"Copilot", "copilot"}


def bucket(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def histogram(values: list[float], edges: list[float], labels: list[str]) -> list[dict]:
    counts = Counter(bucket(v, edges) for v in values)
    return [{"label": label, "n": counts.get(i, 0)} for i, label in enumerate(labels)]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return round(statistics.correlation(xs, ys), 2)


# ---------- GitHub ----------


def _dedupe(items: list, field: str = "id") -> list:
    """Keep the first occurrence of each `field` value, in order."""
    seen: set = set()
    out = []
    for i in items:
        if i[field] not in seen:
            seen.add(i[field])
            out.append(i)
    return out


class GitHub:
    """`gh api` with an on-disk cache keyed by endpoint (and an `updated_at` stamp).

    Safe to share between threads: each endpoint has its own cache file and `calls` is
    guarded by a lock.
    """

    def __init__(self, repo: str, cache: Path, offline: bool = False):
        self.repo = repo
        self.cache = cache / repo.replace("/", "__")
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.calls = 0
        self._lock = threading.Lock()
        self.dropped: dict[str, list] = {}  # endpoint -> cached items that left the list

    def _path(self, key: str) -> Path:
        return self.cache / (re.sub(r"[^A-Za-z0-9_.-]+", "_", key) + ".json")

    def _load(self, path: Path) -> dict | None:
        """The cached record at `path`, or None when missing or unreadable (a killed run)."""
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def get(self, endpoint: str, stamp: str | None = None, paginate: bool = True):
        """Fetch `repos/<repo>/<endpoint>`. With a `stamp`, reuse the cache while it matches."""
        path = self._path(endpoint)
        cached = self._load(path)
        if cached is not None and (
            self.offline or (stamp is not None and cached.get("stamp") == stamp)
        ):
            return cached["data"]
        if self.offline:
            return []
        data = self._api(endpoint, paginate)
        if paginate:
            pages = data or []
            data = []
            for page in pages:
                data.extend(page.get("workflow_runs", []) if isinstance(page, dict) else page)
        self._write(path, {"stamp": stamp, "data": data})
        return data

    @staticmethod
    def _write(path: Path, payload) -> None:
        """Atomic cache write: temp file in the same directory, then `os.replace`."""
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def get_list(self, endpoint: str, key: str | None = None) -> list:
        """Incremental fetch of a newest-first list endpoint.

        Reads page by page and stops at the first page holding an item already cached
        unchanged (same `id` and `updated_at`). A complete walk (short or empty last page)
        yields the fetched items only. An early stop yields the fetched items plus the cached
        tail after the last cached item on that page. Duplicates (head inserts shift pages)
        collapse to the first occurrence by `id`. Cached items that left the list are dropped
        and recorded in `self.dropped[endpoint]`. Early stops are allowed only within
        FULL_WALK of the last complete walk, so deletions anywhere are seen at least daily.
        `key` names the list inside a dict page.
        """
        path = self._path(endpoint)
        record = self._load(path) or {}
        cached = record.get("data") or []
        if self.offline:
            return cached
        # Deletions below the stop point are only seen on a complete walk, so force one when
        # the last complete walk is older than FULL_WALK.
        last_full = ts(record.get("full_at"))
        allow_stop = bool(last_full) and datetime.now(UTC) - last_full < FULL_WALK
        known = {(i["id"], i.get("updated_at")) for i in cached}
        pos = {i["id"]: n for n, i in enumerate(cached)}
        m = re.search(r"per_page=(\d+)", endpoint)
        per_page = int(m.group(1)) if m else 30
        sep = "&" if "?" in endpoint else "?"
        fresh: list = []
        tail_from = None  # index in `cached` after which the cached tail is kept
        page = 1
        while True:
            data = self._api(f"{endpoint}{sep}page={page}")
            items = (data or {}).get(key, []) if key else (data or [])
            fresh.extend(items)
            if len(items) < per_page:
                break
            if allow_stop and any((i["id"], i.get("updated_at")) in known for i in items):
                tail_from = max(pos[i["id"]] for i in items if i["id"] in pos)
                break
            page += 1
        merged = _dedupe(fresh)
        if tail_from is not None:
            seen = {i["id"] for i in merged}
            merged += [i for i in cached[tail_from + 1 :] if i["id"] not in seen]
        if key == "workflow_runs":
            # Runs are listed by creation time, so a run still going when it was cached can
            # sit below the stop point; refetch each such run so its result is current.
            fetched = {i["id"] for i in fresh}
            stale = [
                i["id"] for i in merged if i["id"] not in fetched and i.get("status") != "completed"
            ]
            # At most MAX_RUN_REFRESH per collect; the rest wait for the next run or full walk.
            todo = set(stale[:MAX_RUN_REFRESH])
            merged = [
                r
                for r in (self._refetch_run(i) if i["id"] in todo else i for i in merged)
                if r is not None
            ]
        kept = {i["id"] for i in merged}
        self.dropped[endpoint] = [i for i in cached if i["id"] not in kept]
        full_at = iso(datetime.now(UTC)) if tail_from is None else record.get("full_at")
        self._write(path, {"stamp": None, "data": merged, "full_at": full_at})
        return merged

    def _refetch_run(self, run: dict) -> dict | None:
        """The current state of one unfinished run, or None if it can't be read (deleted)."""
        try:
            return self._api(f"actions/runs/{run['id']}")
        except RuntimeError:
            log.debug("run %s could not be refreshed; dropping it", run["id"])
            return None

    def _api(self, endpoint: str, paginate: bool = False):
        """One `gh api` call (all pages slurped with `paginate`), parsed from JSON."""
        args = ["gh", "api", f"repos/{self.repo}/{endpoint}"]
        if paginate:
            args += ["--paginate", "--slurp"]
        env = dict(os.environ)
        env.setdefault("GH_TOKEN", env.get("git_api_key_0307", ""))
        for attempt in range(RETRIES + 1):
            try:
                out = subprocess.run(  # noqa: S603
                    args, capture_output=True, text=True, env=env, check=False, timeout=TIMEOUT
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"gh api {endpoint} timed out after {TIMEOUT} s") from exc
            with self._lock:
                self.calls += 1
            limited = out.returncode != 0 and RATE_LIMITED.search(out.stderr or "")
            if not limited or attempt == RETRIES:
                break
            # GitHub rate limits (429, or a 403 that says rate limit); jitter so parallel workers spread out
            time.sleep(BACKOFF * 2**attempt * random.uniform(0.5, 1.5))  # noqa: S311
        if out.returncode != 0:
            # gh's stderr never contains the token; keep only its first line.
            first = (out.stderr.strip().splitlines() or ["unknown error"])[0]
            raise RuntimeError(f"gh api {endpoint} failed: {first}")
        return json.loads(out.stdout or "null")


def _pr_files(gh: GitHub, n: int) -> list[Path]:
    return [
        gh._path(e)
        for e in (
            f"pulls/{n}",
            f"pulls/{n}/commits?per_page=100",
            f"pulls/{n}/reviews?per_page=100",
            f"issues/{n}/comments?per_page=100",
        )
    ]


def _pr_detail(gh: GitHub, p: dict) -> tuple[int, dict] | None:
    """One PR's detail, or None when GitHub no longer has it (404)."""
    n, stamp = p["number"], p["updated_at"]
    try:
        return n, {
            "pr": gh.get(f"pulls/{n}", stamp, paginate=False),
            "commits": gh.get(f"pulls/{n}/commits?per_page=100", stamp),
            "reviews": gh.get(f"pulls/{n}/reviews?per_page=100", stamp),
            "comments": gh.get(f"issues/{n}/comments?per_page=100", stamp),
        }
    except RuntimeError as exc:
        if NOT_FOUND.search(str(exc).partition(" failed: ")[2]):  # gh's stderr, not the path
            log.debug("PR #%s is gone on GitHub; skipping", n)
            for f in _pr_files(gh, n):  # deleted content must not linger on disk
                f.unlink(missing_ok=True)
            return None
        raise


def fetch_github(gh: GitHub, workers: int = 4) -> dict:
    pulls_ep = "pulls?state=all&sort=updated&direction=desc&per_page=100"
    pulls = _dedupe(gh.get_list(pulls_ep), "number")
    for p in gh.dropped.get(pulls_ep, []):  # deleted content must not linger on disk
        for f in _pr_files(gh, p["number"]):
            f.unlink(missing_ok=True)
    # Up to 4 PRs' details in flight; each endpoint writes only its own cache file.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        detail = dict(r for r in pool.map(lambda p: _pr_detail(gh, p), pulls) if r)
    issues = gh.get_list("issues?state=all&sort=updated&direction=desc&per_page=100")
    runs = gh.get_list("actions/runs?per_page=100", key="workflow_runs")
    return {"detail": detail, "issues": issues, "runs": runs}


# ---------- git ----------


def git_ledger(root: Path, ref: str) -> list[dict]:
    """Commits reachable from `ref`, newest first, with parents, author, time and message."""
    fmt = "%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%B%x1e"
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "log", ref, f"--format={fmt}"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=TIMEOUT,
    ).stdout
    commits = []
    for rec in out.split("\x1e"):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, parents, name, email, when, body = rec.split("\x1f", 5)
        commits.append(
            {
                "sha": sha,
                "merge": len(parents.split()) > 1,
                "author": name,
                "email": email,
                "at": when,
                "subject": body.splitlines()[0] if body else "",
                "ai": bool(AI_TRAILER.search(body)),
            }
        )
    return commits


# ---------- Claude Code session logs ----------


def price(model: str, usage: dict) -> float:
    inp, out, read = PRICES.get(model, PRICES["claude-opus-5-5"])
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens", 0)
    w5m = cc.get("ephemeral_5m_input_tokens", usage.get("cache_creation_input_tokens", 0) - w1h)
    return (
        usage.get("input_tokens", 0) * inp
        + usage.get("output_tokens", 0) * out
        + usage.get("cache_read_input_tokens", 0) * read
        + w5m * inp * 1.25
        + w1h * inp * 2.0
    ) / 1e6


def slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", path)


def project_dirs(root: Path, base: Path) -> list[Path]:
    """Claude Code project folders for every worktree of this repository."""
    out = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "worktree", "list", "--porcelain"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT,
    ).stdout
    paths = [line[9:] for line in out.splitlines() if line.startswith("worktree ")]
    wanted = {slug(p) for p in paths} or {slug(str(root.resolve()))}
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and d.name in wanted)


def claude_usage(dirs: list[Path], branch_to_pr: dict) -> dict:
    """Session counts and token/cost totals, folded into running sums while scanning the logs.

    Keeps only per-session counts and the totals, never message text or usage records.
    """
    seen: set[str] = set()  # message ids already counted: logs repeat a message per block
    sessions: dict[str, dict] = {}
    by_model: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "output": 0})
    by_day: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
    by_pr: dict[int, dict] = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
    kinds = Counter()
    linked = unlinked = 0.0
    for d in dirs:
        for f in d.rglob("*.jsonl"):
            sub = "subagents" in f.parts
            s = sessions.setdefault(f.stem, {"subagent": sub, "prompts": 0})
            with f.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    msg = rec.get("message") or {}
                    if (
                        rec.get("type") == "user"
                        and not sub
                        and isinstance(msg.get("content"), str)
                    ):
                        s["prompts"] += 1
                    if rec.get("type") != "assistant" or not msg.get("usage"):
                        continue
                    mid = msg.get("id") or rec.get("uuid")
                    if mid in seen:
                        continue
                    seen.add(mid)
                    model, u = msg.get("model", "unknown"), msg["usage"]
                    if model.startswith("<"):
                        continue
                    tok = (
                        u.get("input_tokens", 0)
                        + u.get("output_tokens", 0)
                        + u.get("cache_read_input_tokens", 0)
                        + u.get("cache_creation_input_tokens", 0)
                    )
                    cost = price(model, u)
                    kinds["input"] += u.get("input_tokens", 0)
                    kinds["output"] += u.get("output_tokens", 0)
                    kinds["cache_read"] += u.get("cache_read_input_tokens", 0)
                    kinds["cache_write"] += u.get("cache_creation_input_tokens", 0)
                    by_model[model]["tokens"] += tok
                    by_model[model]["cost"] += cost
                    by_model[model]["output"] += u.get("output_tokens", 0)
                    when = ts(rec.get("timestamp"))
                    if when:
                        by_day[day_key(when)]["tokens"] += tok
                        by_day[day_key(when)]["cost"] += cost
                    pr = branch_to_pr.get(rec.get("gitBranch") or "")
                    if pr:
                        by_pr[pr]["tokens"] += tok
                        by_pr[pr]["cost"] += cost
                        linked += cost
                    else:
                        unlinked += cost
    total = sum(m["cost"] for m in by_model.values())
    return {
        "sessions": sum(1 for s in sessions.values() if not s["subagent"]),
        "subagents": sum(1 for s in sessions.values() if s["subagent"]),
        "prompts": sum(s["prompts"] for s in sessions.values()),
        "tokens": sum(m["tokens"] for m in by_model.values()),
        "cost": round(total, 2),
        "kinds": dict(kinds),
        "by_model": [
            {"model": k, "tokens": v["tokens"], "cost": round(v["cost"], 2)}
            for k, v in sorted(by_model.items(), key=lambda kv: -kv[1]["cost"])
        ],
        "by_day": [
            {"day": k, "tokens": v["tokens"], "cost": round(v["cost"], 2)}
            for k, v in sorted(by_day.items())
        ],
        "by_pr": {
            str(k): {"tokens": v["tokens"], "cost": round(v["cost"], 2)} for k, v in by_pr.items()
        },
        "linked_cost": round(linked, 2),
        "unlinked_cost": round(unlinked, 2),
        "unlinked_share": round(unlinked / total, 3) if total else None,
    }


# ---------- metrics ----------


def review_findings(body: str) -> int:
    """Findings in one Cursor review comment: bullet items outside "No issues found" lanes."""
    return sum(1 for line in body.splitlines() if line.startswith("- **"))


def build_prs(gh_data: dict, runs: list[dict], now: datetime) -> list[dict]:
    ci_by_branch: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        if r.get("name") == CI_WORKFLOW and r.get("event") == "pull_request":
            ci_by_branch[r.get("head_branch") or ""].append(r)
    prs = []
    for n, d in sorted(gh_data["detail"].items(), key=lambda kv: int(kv[0])):
        p = d["pr"]
        opened, merged, closed = ts(p["created_at"]), ts(p.get("merged_at")), ts(p.get("closed_at"))
        branch = p["head"]["ref"]
        ci = sorted(ci_by_branch.get(branch, []), key=lambda r: r["created_at"])
        ci = [r for r in ci if ts(r["created_at"]) >= opened - timedelta(minutes=1)]
        if merged:
            ci = [r for r in ci if ts(r["created_at"]) <= merged]
        failed = [r for r in ci if r.get("conclusion") == "failure"]
        first_fail = ts(failed[0]["created_at"]) if failed else None
        commit_times = [ts(c["commit"]["committer"]["date"]) for c in d["commits"]]
        fix_pushes = sum(1 for t in commit_times if first_fail and t and t > first_fail)
        done = [r for r in ci if r.get("conclusion") in ("success", "failure")]
        last = done[-1]["conclusion"] if done else None
        green_at = max(
            (ts(r["updated_at"]) for r in ci if r.get("conclusion") == "success"), default=None
        )
        reviews = [c for c in d["comments"] if c["body"].startswith(REVIEW_MARK)]
        human_reviews = [
            r
            for r in d["reviews"]
            if not is_bot(r["user"]["login"]) and r["user"]["login"] != p["user"]["login"]
        ]
        body = p.get("body") or ""
        prs.append(
            {
                "number": int(n),
                "title": p["title"],
                "author": p["user"]["login"],
                "bot": is_bot(p["user"]["login"]),
                "branch": branch,
                "state": "merged" if merged else ("closed" if closed else "open"),
                "opened": iso(opened),
                "merged": iso(merged),
                "closed": iso(closed),
                "cycle_min": minutes(opened, merged),
                "age_min": minutes(opened, now) if not closed else None,
                "merged_by": (p.get("merged_by") or {}).get("login"),
                "additions": p.get("additions", 0),
                "deletions": p.get("deletions", 0),
                "lines": p.get("additions", 0) + p.get("deletions", 0),
                "files": p.get("changed_files", 0),
                "commits": p.get("commits", 0),
                "ai_commits": sum(
                    1 for c in d["commits"] if AI_TRAILER.search(c["commit"]["message"])
                ),
                "ci_runs": len(ci),
                "ci_failures": len(failed),
                "fix_pushes": fix_pushes,
                "merged_red": bool(merged and last == "failure"),
                "wait_after_green_min": minutes(green_at, merged)
                if green_at and merged and green_at <= merged
                else None,
                "review_rounds": len(reviews),
                "review_findings": sum(review_findings(c["body"]) for c in reviews),
                "human_reviews": len(human_reviews),
                "approved": any(r["state"] == "APPROVED" for r in human_reviews),
                "changes_requested": any(r["state"] == "CHANGES_REQUESTED" for r in human_reviews),
                "comments": p.get("comments", 0) + p.get("review_comments", 0),
                "closes": sorted({int(x) for x in CLOSES.findall(body)}),
                "url": p["html_url"],
            }
        )
    return prs


def build_issues(issues: list[dict], prs: list[dict]) -> list[dict]:
    closed_by = defaultdict(list)
    for p in prs:
        if p["state"] == "merged":
            for i in p["closes"]:
                closed_by[i].append(p["number"])
    out = []
    for i in issues:
        if i.get("pull_request"):
            continue
        out.append(
            {
                "number": i["number"],
                "title": i["title"],
                "state": i["state"],
                "author": i["user"]["login"],
                "labels": [lab["name"] for lab in i.get("labels", [])],
                "assignees": [a["login"] for a in i.get("assignees", [])],
                "opened": i["created_at"],
                "closed": i.get("closed_at"),
                "comments": i.get("comments", 0),
                "closed_by_pr": closed_by.get(i["number"], []),
                "reason": i.get("state_reason"),
                "url": i["html_url"],
            }
        )
    return sorted(out, key=lambda x: x["number"])


def build_ci(runs: list[dict]) -> dict:
    workflows = defaultdict(Counter)
    per_hour: dict[str, Counter] = defaultdict(Counter)
    ci_runs = []
    for r in runs:
        name = "Dependabot updates" if r.get("event") == "dynamic" else r["name"]
        workflows[name][r.get("conclusion") or r.get("status")] += 1
        if r["name"] == CI_WORKFLOW:
            ci_runs.append(r)
            if r.get("conclusion") in ("success", "failure"):
                per_hour[hour_key(ts(r["created_at"]))][r["conclusion"]] += 1
    concl = Counter(r.get("conclusion") for r in ci_runs)
    main = [r for r in ci_runs if r.get("head_branch") == "main" and r.get("event") == "push"]
    decided = concl["success"] + concl["failure"]
    return {
        "runs_total": len(runs),
        "workflows": [
            {"name": k, **v, "total": sum(v.values())} for k, v in sorted(workflows.items())
        ],
        "ci_runs": len(ci_runs),
        "ci_passed": concl["success"],
        "ci_failed": concl["failure"],
        "ci_other": len(ci_runs) - decided,
        "ci_rate": round(concl["success"] / decided, 3) if decided else None,
        "main_runs": len(main),
        "main_failed": sum(1 for r in main if r.get("conclusion") == "failure"),
        "per_hour": [
            {"hour": k, "passed": v["success"], "failed": v["failure"]}
            for k, v in sorted(per_hour.items())
        ],
        "list": [
            {
                "id": r["id"],
                "workflow": r["name"],
                "branch": r.get("head_branch"),
                "event": r.get("event"),
                "conclusion": r.get("conclusion") or r.get("status"),
                "at": r["created_at"],
                "minutes": round(
                    minutes(ts(r.get("run_started_at") or r["created_at"]), ts(r["updated_at"]))
                    or 0,
                    1,
                ),
                "url": r["html_url"],
            }
            for r in sorted(runs, key=lambda r: r["created_at"], reverse=True)
            if r["name"] in (CI_WORKFLOW, REVIEW_WORKFLOW)
        ],
    }


def build_commits(ledger: list[dict]) -> dict:
    seen, authored = set(), []
    for c in ledger:
        key = (c["subject"], c["author"], c["at"])
        if c["merge"] or key in seen:
            continue
        seen.add(key)
        authored.append(c)
    by_person = defaultdict(lambda: {"commits": 0, "ai": 0, "hours": set(), "hod": Counter()})
    by_day = defaultdict(lambda: {"commits": 0, "ai": 0})
    types = Counter()
    for c in authored:
        when = ts(c["at"])
        who = by_person[c["author"]]
        who["commits"] += 1
        who["ai"] += c["ai"]
        who["hours"].add(hour_key(when))
        who["hod"][when.astimezone(UK).hour] += 1
        by_day[day_key(when)]["commits"] += 1
        by_day[day_key(when)]["ai"] += c["ai"]
        m = PREFIX.match(c["subject"])
        types[m.group(1).lower() if m else "other"] += 1
    total, ai = len(authored), sum(c["ai"] for c in authored)
    return {
        "ledger": len(ledger),
        "merges": sum(c["merge"] for c in ledger),
        "authored": total,
        "ai": ai,
        "ai_floor": round(ai / total, 3) if total else None,
        "first": min((c["at"] for c in ledger), default=None),
        "by_person": [
            {
                "name": k,
                "commits": v["commits"],
                "ai": v["ai"],
                "active_hours": len(v["hours"]),
                "hod": [v["hod"].get(h, 0) for h in range(24)],
            }
            for k, v in sorted(by_person.items(), key=lambda kv: -kv[1]["commits"])
        ],
        "by_day": [{"day": k, **v} for k, v in sorted(by_day.items())],
        "types": [{"type": k, "n": v} for k, v in types.most_common()],
    }


def series(prs: list[dict], commits: list[dict]) -> dict:
    opened, merged = Counter(), Counter()
    events = []
    for p in prs:
        o = ts(p["opened"])
        opened[hour_key(o)] += 1
        events.append((o, 1))
        end = ts(p["merged"] or p["closed"])
        if p["merged"]:
            merged[hour_key(end)] += 1
        if end:
            events.append((end, -1))
    hours = sorted(set(opened) | set(merged))
    inflight, level, peak = [], 0, (0, None)
    for when, delta in sorted(events, key=lambda e: (e[0], e[1])):
        level += delta
        inflight.append({"at": iso(when), "n": level})
        if level > peak[0]:
            peak = (level, iso(when))
    done = sorted((p for p in prs if p["merged"]), key=lambda p: p["merged"])
    cum = [{"at": p["merged"], "n": i} for i, p in enumerate(done, 1)]
    return {
        "per_hour": [{"hour": h, "opened": opened[h], "merged": merged[h]} for h in hours],
        "inflight": inflight,
        "peak": {"n": peak[0], "at": peak[1]},
        "cumulative": cum,
        "active_hours": len(set(hours) | {hour_key(ts(c["at"])) for c in commits}),
    }


def summarize(prs: list[dict], issues: list[dict], now: datetime) -> dict:
    merged = [p for p in prs if p["state"] == "merged"]
    human = [p for p in merged if not p["bot"]]
    cyc = [p["cycle_min"] for p in merged]
    lines = [p["lines"] for p in merged]
    by_person = defaultdict(
        lambda: {"opened": 0, "merged": 0, "open": 0, "closed": 0, "lines": 0, "cycles": []}
    )
    for p in prs:
        who = by_person[p["author"]]
        who["opened"] += 1
        who[{"merged": "merged", "open": "open", "closed": "closed"}[p["state"]]] += 1
        if p["state"] == "merged":
            who["lines"] += p["lines"]
            who["cycles"].append(p["cycle_min"])
    groups = []

    def group(kind: str, label: str, sel: list[dict]) -> None:
        c = [p["cycle_min"] for p in sel]
        groups.append(
            {"group": kind, "label": label, "n": len(sel), "median": median(c), "p90": pctl(c, 0.9)}
        )

    group("Size", "small (<100 lines)", [p for p in merged if p["lines"] < 100])
    group("Size", "medium (100-499)", [p for p in merged if 100 <= p["lines"] < 500])
    group("Size", "large (500-1,999)", [p for p in merged if 500 <= p["lines"] < 2000])
    group("Size", "very large (2,000+)", [p for p in merged if p["lines"] >= 2000])
    group("CI failures", "CI never failed", [p for p in merged if p["ci_failures"] == 0])
    group("CI failures", "CI failed at least once", [p for p in merged if p["ci_failures"] > 0])
    group("Review", "Cursor review posted", [p for p in merged if p["review_rounds"] > 0])
    group("Review", "no Cursor review", [p for p in merged if p["review_rounds"] == 0])
    group("Author", "person", human)
    group("Author", "bot (Dependabot)", [p for p in merged if p["bot"]])
    factors = {
        "Cursor review rounds": "review_rounds",
        "commits on the pull request": "commits",
        "fix pushes after a red CI run": "fix_pushes",
        "CI failures": "ci_failures",
        "lines changed": "lines",
        "files changed": "files",
    }
    open_issues = [i for i in issues if i["state"] == "open"]
    return {
        "prs": len(prs),
        "merged": len(merged),
        "merged_human": len(human),
        "merged_bot": len(merged) - len(human),
        "open": sum(p["state"] == "open" for p in prs),
        "closed_unmerged": sum(p["state"] == "closed" for p in prs),
        "cycle_median": median(cyc),
        "cycle_p90": pctl(cyc, 0.9),
        "cycle_max": max(cyc, default=None),
        "lines_median": median(lines),
        "files_median": median([p["files"] for p in merged]),
        "commits_median": median([p["commits"] for p in merged]),
        "additions": sum(p["additions"] for p in merged),
        "deletions": sum(p["deletions"] for p in merged),
        "file_changes": sum(p["files"] for p in merged),
        "big_prs": sum(p["lines"] >= 1000 for p in merged),
        "reviewed_cursor": sum(p["review_rounds"] > 0 for p in human),
        "review_rounds": sum(p["review_rounds"] for p in prs),
        "review_findings": sum(p["review_findings"] for p in prs),
        "reviewed_human": sum(p["human_reviews"] > 0 for p in merged),
        "approved": sum(p["approved"] for p in merged),
        "changes_requested": sum(p["changes_requested"] for p in merged),
        "self_merged": sum(p["merged_by"] == p["author"] for p in human),
        "merged_red": sum(p["merged_red"] for p in merged),
        "ever_red": sum(p["ci_failures"] > 0 for p in merged),
        "fix_pushes": sum(p["fix_pushes"] for p in merged),
        "wait_after_green": median(
            [p["wait_after_green_min"] for p in human if p["wait_after_green_min"] is not None]
        ),
        "issues": len(issues),
        "issues_open": len(open_issues),
        "issues_closed": len(issues) - len(open_issues),
        "issues_unassigned": sum(not i["assignees"] for i in open_issues),
        "issues_closed_no_pr": sum(
            i["state"] == "closed" and not i["closed_by_pr"] for i in issues
        ),
        "by_person": [
            {
                "name": k,
                "opened": v["opened"],
                "merged": v["merged"],
                "open": v["open"],
                "closed": v["closed"],
                "lines": v["lines"],
                "cycle_median": median(v["cycles"]),
                "bot": is_bot(k),
            }
            for k, v in sorted(by_person.items(), key=lambda kv: -kv[1]["opened"])
        ],
        "cycle_groups": groups,
        "correlations": [
            {
                "factor": label,
                "r": pearson([float(p[k]) for p in merged], [p["cycle_min"] for p in merged]),
            }
            for label, k in factors.items()
        ],
        "hist_lines": histogram(
            lines,
            [10, 50, 100, 250, 500, 1000, 2000],
            ["<10", "10-49", "50-99", "100-249", "250-499", "500-999", "1,000-1,999", "2,000+"],
        ),
        "hist_files": histogram(
            [p["files"] for p in merged],
            [2, 3, 6, 11, 21, 51],
            ["1", "2", "3-5", "6-10", "11-20", "21-50", "51+"],
        ),
        "hist_commits": histogram(
            [p["commits"] for p in merged],
            [2, 3, 6, 11, 21],
            ["1", "2", "3-5", "6-10", "11-20", "21+"],
        ),
        "hist_cycle": histogram(
            cyc,
            [5, 15, 30, 60, 120, 240, 480, 1440],
            [
                "<5 min",
                "5-14 min",
                "15-29 min",
                "30-59 min",
                "1-2 h",
                "2-4 h",
                "4-8 h",
                "8-24 h",
                "1 day+",
            ],
        ),
        "hist_reviews": histogram(
            [p["review_rounds"] for p in human], [1, 2, 3, 4, 6], ["0", "1", "2", "3", "4-5", "6+"]
        ),
        "hist_fix": histogram(
            [p["fix_pushes"] for p in merged], [1, 2, 4, 8], ["0", "1", "2-3", "4-7", "8+"]
        ),
    }


def docs_inputs(root: Path) -> dict:
    """ADRs (decisions) and roadmap items, by title only."""
    adrs = []
    for f in sorted((root / "docs" / "adr").glob("[0-9][0-9][0-9][0-9]-*.md")):
        text = f.read_text(encoding="utf-8")
        title = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
            f.stem,
        )
        status = re.search(r"^\**Status\**:?\**\s*(.+)$", text, re.M | re.I)
        adrs.append(
            {
                "id": f.stem[:4],
                "title": title,
                "status": status.group(1).strip() if status else None,
                "path": f"docs/adr/{f.name}",
            }
        )
    roadmap = []
    rm = root / "docs" / "ROADMAP.md"
    if rm.exists():
        section = None
        for line in rm.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                section = line[3:].strip()
            m = re.match(r"^\s*[-*]\s+(\[( |x|X)\]\s+)?(.+)", line)
            if m and section:
                roadmap.append(
                    {
                        "section": section,
                        "done": (m.group(2) or "").lower() == "x",
                        "text": m.group(3)[:160],
                    }
                )
    return {"adrs": adrs, "roadmap": roadmap}


def collect(
    root: Path, repo: str, cache: Path, offline: bool = False, logs: Path = CLAUDE_PROJECTS
) -> dict:
    now = datetime.now(UTC)
    gh = GitHub(repo, cache, offline=offline)
    raw = fetch_github(gh)
    ref = "origin/main"
    if not offline:
        subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "fetch", "-q", "origin", "main"],  # noqa: S607
            check=False,
            timeout=TIMEOUT,
        )
    ledger = git_ledger(root, ref)
    prs = build_prs(raw, raw["runs"], now)
    issues = build_issues(raw["issues"], prs)
    branch_to_pr = {p["branch"]: p["number"] for p in prs}
    claude = claude_usage(project_dirs(root, logs), branch_to_pr)
    commits = build_commits(ledger)
    return {
        "schema": 1,
        "repo": repo,
        "generated_at": iso(now),
        "window": {"start": commits["first"], "end": iso(now)},
        "refresh": {"on_merge": True, "minutes": 360},
        "summary": summarize(prs, issues, now),
        "series": series(prs, ledger),
        "commits": commits,
        "ci": build_ci(raw["runs"]),
        "claude": claude,
        "docs": docs_inputs(root),
        "prs": prs,
        "issues": issues,
        "prices": {
            k: {"input": v[0], "output": v[1], "cache_read": v[2]} for k, v in PRICES.items()
        },
        "sources": {"github_calls": gh.calls, "ref": ref},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, default=Path(".report") / "report-data.json")
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--offline", action="store_true", help="use only the cache, no network")
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = collect(args.root, args.repo, args.cache, offline=args.offline)
    args.out.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
    s = data["summary"]
    print(
        f"wrote {args.out}: {s['prs']} PRs, {s['issues']} issues, {data['ci']['runs_total']} runs"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
