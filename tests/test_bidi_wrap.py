"""Wrapping long right-to-left captions before reordering for Tk.

Stories: US-29.
"""

from __future__ import annotations

import unicodedata

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


def test_text_area_width_uses_real_padding_and_margins():
    from recandcc.ui.captions import text_area_width

    assert text_area_width(760, 10, 0, 0, 10) == 730
    assert text_area_width(760, 10, 2, 1, 10) == 724
    assert text_area_width(1, 10, 0, 0, 10) is None  # unmapped widget


def test_whitespace_collapses_and_marks_drop():
    assert wrap_logical(f"{W}‏   {W}\n{W}", 100) == [f"{W}\u200f {W} {W}"]


def test_geometry_width_fallback_for_a_withdrawn_window():
    from recandcc.ui.captions import geometry_width

    assert geometry_width("760x190+100+200") == 760
    assert geometry_width("640x480") == 640
    assert geometry_width("1x1+0+0") == 1
    assert geometry_width("") == 0


def test_hard_split_keeps_nikud_with_its_letter():
    word = "שָׁ" * 6  # three marks per letter-ish; marks must not start a line
    lines = wrap_logical(word, 4)
    assert "".join(lines) == word
    assert all(not unicodedata.combining(x[0]) for x in lines)


def test_brackets_mirror_per_wrapped_line():
    parts = tk_parts(_line(f"({W} {W}) {W} {W}"), 14)
    shown = [p for p, lab in parts if not lab and p != "\n"]
    assert any(")" in x or "(" in x for x in shown)
    assert all(not any(ord(c) < 32 for c in x) for x in shown)


def test_label_measured_with_its_own_font():
    parts = tk_parts(_line(SENT), 40, len, lambda s: len(s) * 5)
    assert sum(1 for p, _ in parts if p == "\n") >= 1
    assert parts[1] == ("  Others", True)
