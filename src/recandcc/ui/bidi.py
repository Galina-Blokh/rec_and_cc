"""Visual reordering of right-to-left text for widgets that do no bidi (Tk on Windows).

Pure and platform-independent. ``visual_rtl`` turns logical text of a right-to-left line
(Hebrew with embedded English words and numbers) into the string to hand to a widget that
paints characters left to right as given: Hebrew runs reversed, LTR runs and numbers kept in
reading order, runs laid out right to left, brackets mirrored. Unicode control characters
(isolates, LRM/RLM, embeddings) are dropped, since Tk would draw them as boxes.

This is a small subset of UAX #9 for one paragraph with an RTL base direction: strong
characters, numbers (with the separators inside them) and neutrals; no explicit embeddings.
The macOS panel and the transcript files use the platform's own bidi support instead.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable

_R, _L, _N, _NEUTRAL = 1, 2, 3, 0  # kinds: RTL letter, LTR letter, number, everything else
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")


def _kind(ch: str) -> int | None:
    """Kind of ``ch``, or None for characters that are removed (format controls)."""
    if unicodedata.category(ch) == "Cf":  # LRM/RLM, isolates, embeddings, ZWJ...
        return None
    bidi = unicodedata.bidirectional(ch)
    if bidi in ("R", "AL"):
        return _R
    if bidi == "L":
        return _L
    if bidi in ("EN", "AN"):
        return _N
    if bidi == "BN":
        return None
    return _NEUTRAL if bidi != "NSM" else -1  # -1: combining mark, follows its base


def visual_rtl(text: str) -> str:
    """Logical ``text`` of an RTL paragraph -> the string to display left to right."""
    chars: list[str] = []
    kinds: list[int] = []
    bidis: list[str] = []
    for ch in text:
        k = _kind(ch)
        if k is None:
            continue
        if k == -1 and chars:  # combining mark: stays glued to its base through the reversal
            chars[-1] += ch
            continue
        if k == -1:
            k = _NEUTRAL
        chars.append(ch)
        kinds.append(k)
        bidis.append(unicodedata.bidirectional(ch))
    n = len(chars)
    if n == 0:
        return ""
    # Separators inside a number (1,000 3.5 12:30 10/11) and signs/units next to one are numbers.
    for i, b in enumerate(bidis):
        if kinds[i] != _NEUTRAL:
            continue
        before = kinds[i - 1] == _N if i else False
        after = kinds[i + 1] == _N if i + 1 < n else False
        if (b in ("CS", "ES") and before and after) or (b == "ET" and (before or after)):
            kinds[i] = _N
    # Neutral runs take the direction of both neighbours if they agree (numbers count as RTL
    # for this), else the paragraph direction (RTL). Line ends count as RTL.
    levels = [0] * n
    i = 0
    while i < n:
        if kinds[i] != _NEUTRAL:
            levels[i] = 1 if kinds[i] == _R else 2
            i += 1
            continue
        j = i
        while j < n and kinds[j] == _NEUTRAL:
            j += 1
        left_l = i > 0 and kinds[i - 1] == _L
        right_l = j < n and kinds[j] == _L
        lvl = 2 if left_l and right_l else 1
        for m in range(i, j):
            levels[m] = lvl
        i = j
    # Rule L2: from the highest level down to 1, reverse every maximal run at that level or above.
    for lvl in range(max(levels), 0, -1):
        i = 0
        while i < n:
            if levels[i] < lvl:
                i += 1
                continue
            j = i
            while j < n and levels[j] >= lvl:
                j += 1
            chars[i:j] = chars[i:j][::-1]
            levels[i:j] = levels[i:j][::-1]
            i = j
    return "".join(
        c.translate(_MIRROR) if lv == 1 else c for c, lv in zip(chars, levels, strict=True)
    )


def wrap_logical(text: str, max_width: float, measure: Callable[[str], float] = len) -> list[str]:
    """Split logical ``text`` into lines that each measure at most ``max_width``.

    Breaks only at whitespace, so numbers and English words stay whole; a single word wider
    than the limit is hard-split by characters as a last resort. ``measure`` is e.g. Tk's
    ``font.measure``; the default counts characters. Wrap *before* ``visual_rtl``: reordering
    a long line first and letting the widget wrap it puts the logical end on the first line.
    """
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if measure(cand) <= max_width:
            cur = cand
            continue
        if cur:
            lines.append(cur)
            cur = ""
        while measure(word) > max_width and len(word) > 1:
            k = len(word) - 1
            while k > 1 and measure(word[:k]) > max_width:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        cur = word
    if cur or not lines:
        lines.append(cur)
    return lines
