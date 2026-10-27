"""Right-to-left reordering for the Tk captions (pure; no Tk window needed).

Stories: US-29, US-19.
"""

import pytest

from recandcc.ui.bidi import visual_rtl
from recandcc.ui.captions import CaptionLine, tk_parts

# Visual strings are what a left-to-right painter must draw so that the line reads correctly.
SHALOM = "שלום"
OLAM = "עולם"


def rev(s):
    return s[::-1]


@pytest.mark.parametrize(
    ("logical", "visual"),
    [
        (f"{SHALOM} {OLAM}", f"{rev(OLAM)} {rev(SHALOM)}"),  # Hebrew only
        (f"{SHALOM} world {OLAM}", f"{rev(OLAM)} world {rev(SHALOM)}"),  # one English word
        (f"{SHALOM} Hello World {OLAM}", f"{rev(OLAM)} Hello World {rev(SHALOM)}"),  # LTR phrase
        (f"{SHALOM} 25 {OLAM}", f"{rev(OLAM)} 25 {rev(SHALOM)}"),  # number
        (f"{SHALOM} 3.5 {OLAM}", f"{rev(OLAM)} 3.5 {rev(SHALOM)}"),  # decimal stays whole
        (f"{SHALOM} 1,000 {OLAM}", f"{rev(OLAM)} 1,000 {rev(SHALOM)}"),
        (f"ב-3.5 {OLAM}", f"{rev(OLAM)} 3.5-ב"),  # prefix + number
        (f"{SHALOM}, ({OLAM})", f"({rev(OLAM)}) ,{rev(SHALOM)}"),  # brackets mirrored
        (f"{SHALOM}⁧ x ⁩{OLAM}", f"{rev(OLAM)} x {rev(SHALOM)}"),  # controls dropped
        ("", ""),
        ("   ", "   "),
    ],
)
def test_visual_rtl_table(logical, visual):
    assert visual_rtl(logical) == visual


def test_no_control_characters_left():
    out = visual_rtl("‏שלום ⁦abc⁩ ‎12‬")
    assert not any(ord(c) in range(0x2066, 0x206A) or c in "‎‏‬" for c in out)


def test_ltr_text_is_not_reordered_by_a_language_that_is_not_rtl():
    line = CaptionLine(0.0, "Others", "Привет, team 42", "ru")
    assert tk_parts(line) == [("Others\t", True), ("Привет, team 42", False)]
    line = CaptionLine(0.0, "Me", "Hello team", "en")
    assert tk_parts(line) == [("Me\t", True), ("Hello team", False)]


def test_rtl_line_puts_label_on_the_right_after_the_visual_text():
    line = CaptionLine(0.0, "Me", f"{SHALOM} world {OLAM}", "he")
    parts = tk_parts(line)
    assert parts == [(f"{rev(OLAM)} world {rev(SHALOM)}", False), ("  Me", True)]


def test_rtl_detected_from_text_without_language():
    parts = tk_parts(CaptionLine(0.0, "Others", f"{SHALOM} {OLAM}", ""))
    assert parts[-1] == ("  Others", True)
    assert parts[0][0] == f"{rev(OLAM)} {rev(SHALOM)}"


def test_empty_and_whitespace_rtl_lines():
    assert tk_parts(CaptionLine(0.0, "Me", "", "he")) == [("", False), ("  Me", True)]
    assert tk_parts(CaptionLine(0.0, "Me", "  ", "he")) == [("  ", False), ("  Me", True)]


def test_combining_marks_stay_glued_to_their_letter():
    # Niqqud / shin dot must still follow its letter after the run is reversed.
    shin, dot, lamed = "\u05e9", "\u05c1", "\u05dc"
    assert visual_rtl(f"{shin}{dot}{lamed}") == f"{lamed}{shin}{dot}"
