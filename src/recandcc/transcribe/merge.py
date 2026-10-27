"""Language choice, track merging and echo removal. Pure functions, no model needed."""

from __future__ import annotations

import bisect
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


# -- splitting a segment where the speaker changes ---------------------------------------------

MIN_RUN_S = 1.0  # a run of one speaker shorter than this ...
MIN_RUN_WORDS = 3  # ... and with fewer words than this is flicker, absorbed by its neighbours


def _word_speakers(words: Sequence[tuple[float, float, str]], turns: Sequence) -> list[int]:
    """Diarized speaker of each word: most overlap, else the turn nearest its middle.

    Turns are sorted once and bisected, so a word only looks at the turns around it (ties go to
    the turn that comes first in `turns`, as with a plain scan)."""
    if not turns:
        return []
    order = sorted(range(len(turns)), key=lambda i: turns[i].start)
    ordered = [turns[i] for i in order]
    starts = [t.start for t in ordered]
    reach: list[float] = []  # reach[k]: latest end among ordered[: k + 1]
    for t in ordered:
        reach.append(max(reach[-1], t.end) if reach else t.end)
    first_by_end: dict[float, int] = {}  # earliest index in `turns` per end / start value
    first_by_start: dict[float, int] = {}
    for i, t in enumerate(turns):
        first_by_end.setdefault(t.end, i)
        first_by_start.setdefault(t.start, i)
    out = []
    for ws, we, _ in words:
        hi = bisect.bisect_left(starts, we)  # turns from `hi` on start at/after the word ends
        hits: list[int] = []
        for k in range(hi - 1, -1, -1):
            if reach[k] <= ws:  # nothing earlier reaches the word
                break
            if min(we, ordered[k].end) - max(ws, ordered[k].start) > 0:
                hits.append(order[k])
        if hits:
            overlap: dict[int, float] = {}
            for i in sorted(hits):  # input order, so ties resolve like a plain scan
                t = turns[i]
                o = min(we, t.end) - max(ws, t.start)
                overlap[t.speaker] = overlap.get(t.speaker, 0.0) + o
            out.append(max(overlap, key=lambda k: overlap[k]))
            continue
        mid = (ws + we) / 2
        cands = []  # nothing overlaps: the nearest turn ends before the word or starts after it
        if hi > 0 and reach[hi - 1] > ws:  # zero-length word inside a turn: rare, scan all
            cands = list(range(len(turns)))
        elif hi > 0:
            cands.append(first_by_end[reach[hi - 1]])
        if hi < len(starts):
            cands.append(first_by_start[starts[hi]])
        best = min(cands, key=lambda i: (max(turns[i].start - mid, mid - turns[i].end, 0.0), i))
        out.append(turns[best].speaker)
    return out


def _runs(
    words: Sequence[tuple[float, float, str]], who: Sequence[int]
) -> list[list]:  # [speaker, first, last+1]
    runs: list[list] = []
    for i, k in enumerate(who):
        if runs and runs[-1][0] == k:
            runs[-1][2] = i + 1
        else:
            runs.append([k, i, i + 1])
    while len(runs) > 1:

        def length(r):
            return words[r[2] - 1][1] - words[r[1]][0]

        def is_weak(r):
            return length(r) < MIN_RUN_S and r[2] - r[1] < MIN_RUN_WORDS

        # a weak run is absorbed only into a strong neighbour, never into another weak run
        # ("Yes." / "No." from two speakers stay apart); each pass removes a run, so it ends
        absorbable = [
            r
            for r in runs
            if length(r) < MIN_RUN_S
            and r[2] - r[1] < MIN_RUN_WORDS
            and any(
                0 <= n < len(runs) and not is_weak(runs[n])
                for n in (runs.index(r) - 1, runs.index(r) + 1)
            )
        ]
        if not absorbable:
            break
        r = min(absorbable, key=length)
        i = runs.index(r)
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs) and not is_weak(runs[j])]
        j = max(nbrs, key=lambda n: length(runs[n]))
        runs[j][1], runs[j][2] = min(runs[j][1], r[1]), max(runs[j][2], r[2])
        del runs[i]
        # coalesce neighbours that now share a speaker
        for a in range(len(runs) - 2, -1, -1):
            if runs[a][0] == runs[a + 1][0]:
                runs[a][2] = runs[a + 1][2]
                del runs[a + 1]
    return runs


def _join(raw: Sequence[str]) -> str:
    """Words back to text exactly: faster-whisper words carry their own leading space (none for
    CJK, sub-word tokens or punctuation), so they are concatenated as they are."""
    return "".join(raw).strip()


def split_on_speaker_change(
    segments: Sequence[Segment], turns: Sequence, speaker: str = OTHERS
) -> list[Segment]:
    """Split each `speaker` ("Others") segment that has word timings wherever the diarized
    speaker of consecutive words changes, at the word boundary. Runs shorter than MIN_RUN_S and
    MIN_RUN_WORDS words are absorbed by their neighbours. The pieces keep the language and
    rebuild the text from the words; segments without words, "Me" segments and segments with
    one speaker are returned unchanged. `turns`: diarization turns (start, end, speaker)."""
    if not turns:
        return list(segments)
    out: list[Segment] = []
    for seg in segments:
        runs = (
            _runs(seg.words, _word_speakers(seg.words, turns))
            if seg.speaker == speaker and len(seg.words) > 1
            else []
        )
        if len(runs) < 2:
            out.append(seg)
            continue
        for _, a, b in runs:
            ws = seg.words[a:b]
            out.append(
                Segment(
                    start=ws[0][0],
                    end=ws[-1][1],
                    text=_join([w[2] for w in ws]),
                    speaker=seg.speaker,
                    language=seg.language,
                    words=list(ws),
                )
            )
    return out
