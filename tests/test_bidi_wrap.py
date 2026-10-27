"""Wrapping long right-to-left captions before reordering for Tk.

Stories: US-29.
"""

from __future__ import annotations

from recandcc.ui.bidi import visual_rtl, wrap_logical
from recandcc.ui.captions import CaptionLine, tk_parts

W = "שלום"  # shalom
SENT = f"{W} {W} hello {W} 3.5 {W} {W} 1,000 {W} {W} {W}"


def _line(text: str) -> CaptionLine:
    return CaptionLine(0.0, "Others", text, "he")


def test_wraps_at_words_and_keeps_order():
    lines = wrap_logical(SENT, 16)
    assert len(lines) > 2 and " ".join(lines) == SENT
    assert all(len(x) <= 16 for x in lines)


def test_numbers_and_english_never_split():
    lines = wrap_logical(SENT, 11)
    words = [w for x in lines for w in x.split()]
    assert words == SENT.split()


def test_each_visual_line_is_reordering_of_its_slice():
    parts = tk_parts(_line(SENT), 20)
    logical = wrap_logical(SENT, 20 - len("  Others"))
    visual = [p for p, lab in parts if not lab and p != "\n"]
    assert visual == [visual_rtl(c) for c in logical]
    assert parts[1] == ("  Others", True)  # label on the first line only
    assert sum(1 for _, lab in parts if lab) == 1
    assert [p for p, _ in parts].count("\n") == len(logical) - 1


def test_ltr_unchanged_and_no_wrap_without_width():
    assert tk_parts(CaptionLine(0.0, "Me", "Hello team", ""), 3) == [
        ("Me\t", True),
        ("Hello team", False),
    ]
    assert tk_parts(_line(SENT)) == [(visual_rtl(SENT), False), ("  Others", True)]


def test_long_word_hard_split():
    lines = wrap_logical("x" * 25 + " ab", 10)
    assert lines == ["x" * 10, "x" * 10, "x" * 5 + " ab"]


def test_measure_callable_and_no_control_chars():
    lines = wrap_logical(f"{W} ‏{W} {W}", 9, lambda s: len(s) * 2)
    assert lines[0] == f"{W}"
    parts = tk_parts(_line(f"{W}‏ {W} ⁧{W}⁩ {W} {W}"), 14)
    assert all(not any(ord(c) in range(0x200B, 0x2070) for c in p) for p, _ in parts)
