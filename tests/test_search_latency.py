"""Search latency: a search over 100+ sessions returns in < 200 ms (US-20 acceptance).

Stories: US-20.

The index is built once from synthetic, seeded transcripts (no real recordings); only the
query is timed, as the median of a few runs. The 200 ms target is never weakened on macOS or
Linux. Windows runners are slower and noisier, so there the budget is multiplied by a
documented headroom factor (``WINDOWS_HEADROOM``); ``RCC_SEARCH_LATENCY_FACTOR`` overrides it
for any other slow runner.
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta

import pytest

from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.search import TranscriptIndex, has_fts5
from recandcc.transcribe.output import write_outputs

SESSIONS = 120
SEGMENTS = 150  # about a 30 minute meeting
TARGET_S = 0.200
WINDOWS_HEADROOM = 3.0
RUNS = 7
T0 = datetime(2026, 1, 5, 9, 0, 0)

EN = [
    "the",
    "we",
    "meeting",
    "project",
    "budget",
    "deadline",
    "release",
    "team",
    "customer",
    "update",
    "review",
    "plan",
    "next",
    "week",
    "call",
    "today",
    "need",
    "think",
    "agree",
    "issue",
    "ship",
    "test",
    "design",
    "data",
    "report",
    "decision",
    "question",
    "answer",
    "follow",
    "action",
    "owner",
    "risk",
    "status",
    "launch",
    "roadmap",
    "priority",
]
RU = [
    "мы",
    "встреча",
    "проект",
    "бюджет",
    "срок",
    "релиз",
    "команда",
    "клиент",
    "обновление",
    "план",
    "неделя",
    "звонок",
    "сегодня",
    "нужно",
    "думаю",
    "согласен",
    "вопрос",
    "ответ",
    "задача",
    "решение",
    "риск",
    "статус",
    "запуск",
]
RARE_EN = "zebrafish"  # planted in exactly one session
RARE_RU = "квазар"


def _factor() -> float:
    env = os.environ.get("RCC_SEARCH_LATENCY_FACTOR")
    if env:
        return float(env)
    return WINDOWS_HEADROOM if sys.platform == "win32" else 1.0


def _segments(rng: random.Random, n: int) -> list[Segment]:
    out: list[Segment] = []
    for i in range(SEGMENTS):
        ru = n % 3 == 0 and i % 2 == 0
        words = rng.choices(RU if ru else EN, k=rng.randint(6, 18))
        if n == 7 and i == 40:
            words.append(RARE_EN)
        if n == 9 and i == 41:
            words.append(RARE_RU)
        if n == 11 and i == 42:
            words[:3] = ["launch", "date", "moved"]
        out.append(
            Segment(
                i * 12.0,
                i * 12.0 + 10,
                " ".join(words),
                "Me" if i % 2 else "Others",
                "ru" if ru else "en",
            )
        )
    return out


@pytest.fixture(scope="module", params=[True, False], ids=["fts5", "like"])
def index(request, tmp_path_factory: pytest.TempPathFactory) -> TranscriptIndex:
    fts = request.param
    if fts and not has_fts5():
        pytest.skip("this SQLite has no FTS5")
    root = tmp_path_factory.mktemp(f"latency-{'fts' if fts else 'like'}")
    out = root / "out"
    rng = random.Random(74)  # noqa: S311 - deterministic test data
    for n in range(SESSIONS):
        started = T0 + timedelta(hours=n)
        d = out / f"{started:%Y-%m-%d}" / f"{started:%H%M%S}_zoom-us-{n}"
        d.mkdir(parents=True)
        s = SessionInfo(f"s{n}", d, started, TriggerKind.CALL, "zoom.us", started)
        s.save()
        write_outputs(s, _segments(rng, n), 1800.0, "fake")
    idx = TranscriptIndex(root / "idx" / "search.sqlite3", fts=fts)
    assert idx.sync(out) == SESSIONS
    return idx


def _median_s(idx: TranscriptIndex, query: str) -> tuple[float, int]:
    idx.search(query)  # warm up: first connection and page cache
    times = []
    hits = 0
    for _ in range(RUNS):
        t = time.perf_counter()
        hits = len(idx.search(query))
        times.append(time.perf_counter() - t)
    return statistics.median(times), hits


@pytest.mark.parametrize(
    ("query", "expect_hits"),
    [
        pytest.param("meeting", True, id="common-word"),
        pytest.param("budget deadline", True, id="two-common-words"),
        pytest.param(RARE_EN, True, id="rare-word"),
        pytest.param("launch date moved", True, id="phrase"),
        pytest.param("бюдж", True, id="russian-prefix"),
        pytest.param(RARE_RU, True, id="rare-russian"),
        pytest.param("nonexistentwordxyz", False, id="no-hit"),
    ],
)
def test_search_over_100_sessions_is_fast(index: TranscriptIndex, query: str, expect_hits: bool):
    median, hits = _median_s(index, query)
    assert bool(hits) == expect_hits
    budget = TARGET_S * _factor()
    assert median < budget, f"{query!r}: median {median * 1000:.1f} ms >= {budget * 1000:.0f} ms"
