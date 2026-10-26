"""Language choice, track merging and echo removal. Pure functions, no model needed."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence

from ..models import Segment

ECHO_SIMILARITY = 0.6
ECHO_TOLERANCE_S = 1.5  # extra slack when checking time overlap (echo delay, timestamp jitter)
ME, OTHERS = "Me", "Others"

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)


def pick_language(probs: Iterable[tuple[str, float]], allowed: Sequence[str]) -> str:
    """Most probable language among `allowed` (any language if `allowed` is empty)."""
    return language_confidence(probs, allowed)[0]


def language_confidence(
    probs: Iterable[tuple[str, float]], allowed: Sequence[str]
) -> tuple[str, float]:
    """(best language among `allowed`, its probability renormalised over `allowed`)."""
    probs = list(probs)
    pool = [p for p in probs if p[0] in allowed] if allowed else probs
    total = sum(p for _, p in pool)
    if not pool or total <= 0:
        return (allowed[0] if allowed else "en"), 0.0
    lang, p = max(pool, key=lambda x: x[1])
    return lang, p / total


def smooth_languages(
    regions: list[tuple[float, float]],
    detections: list[tuple[str, float]],
    min_conf: float = 0.75,
    short_s: float = 2.0,
    max_gap_s: float = 2.0,
) -> list[str]:
    """Per-region languages where short, uncertain regions ("yeah", "да") take the language of
    the nearest neighbouring region within `max_gap_s` (previous preferred)."""
    langs = [lang for lang, _ in detections]
    weak = [
        (e - s) < short_s and c < min_conf
        for (s, e), (_, c) in zip(regions, detections, strict=True)
    ]
    for i in range(len(regions)):
        if not weak[i]:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if not weak[j]), None)
        nxt = next((j for j in range(i + 1, len(regions)) if not weak[j]), None)
        if prev is not None and regions[i][0] - regions[prev][1] <= max_gap_s:
            langs[i] = langs[prev]
        elif nxt is not None and regions[nxt][0] - regions[i][1] <= max_gap_s:
            langs[i] = langs[nxt]
    return langs


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def similarity(a: str, b: str) -> float:
    """difflib ratio of normalised texts; if `a` is a phrase (>= 3 words) found inside a longer
    `b`, the best matching word window of `b` is used instead."""
    a, b = normalise(a), normalise(b)
    if not a or not b:
        return 0.0
    best = difflib.SequenceMatcher(None, a, b).ratio()
    aw, bw = a.split(), b.split()
    if len(aw) >= 3 and len(bw) > len(aw):
        for i in range(len(bw) - len(aw) + 1):
            window = " ".join(bw[i : i + len(aw)])
            best = max(best, difflib.SequenceMatcher(None, a, window).ratio())
    return best


def dedupe_echo(
    me: list[Segment], others: list[Segment], threshold: float = ECHO_SIMILARITY
) -> list[Segment]:
    """Drop Me segments that overlap in time with Others segments and say the same thing."""
    kept = []
    for seg in me:
        near = [
            o
            for o in others
            if o.start < seg.end + ECHO_TOLERANCE_S and seg.start < o.end + ECHO_TOLERANCE_S
        ]
        if near:
            joined = " ".join(o.text for o in sorted(near, key=lambda s: s.start))
            if (
                max([similarity(seg.text, o.text) for o in near] + [similarity(seg.text, joined)])
                >= threshold
            ):
                continue
        kept.append(seg)
    return kept


def merge_tracks(me: list[Segment], others: list[Segment], dedupe: bool = True) -> list[Segment]:
    """Both tracks' segments sorted by start time, optionally without mic echo."""
    if dedupe:
        me = dedupe_echo(me, others)
    return sorted([*me, *others], key=lambda s: (s.start, s.end))
