"""Live captions: platform-neutral logic (formatting, ordering, trimming, header, timers).

Stories: US-11.
"""

from recandcc.models import Segment
from recandcc.ui.captions import (
    FINISHED_TEXT,
    CaptionBuffer,
    CaptionLine,
    clean_text,
    fmt_mmss,
    format_line,
    header_text,
    is_at_bottom,
    should_auto_hide,
)


def seg(start, text, speaker="Others", lang="en"):
    return Segment(start, start + 1.0, text, speaker, lang)


def test_fmt_mmss():
    assert fmt_mmss(0) == "00:00"
    assert fmt_mmss(-5) == "00:00"
    assert fmt_mmss(65.9) == "01:05"
    assert fmt_mmss(3600 + 2 * 60 + 3) == "1:02:03"


def test_header_text_live_and_finished():
    assert header_text("Zoom", 192) == "Live — Zoom · 🔴 03:12"
    assert header_text("", None) == "Live — recording"
    assert header_text("Zoom", 10, finished=True) == f"{FINISHED_TEXT} · Zoom"


def test_format_line_pads_label():
    line = CaptionLine(0.0, "Me", "Привет")
    assert format_line(line) == "Me      Привет"
    assert format_line(CaptionLine(0.0, "Others", "Hi"), width=6) == "Others Hi"


def test_clean_text_collapses_whitespace():
    assert clean_text("  Hello\n  team \t ok ") == "Hello team ok"
    assert clean_text("") == ""


def test_buffer_orders_by_start_and_skips_empty_and_duplicates():
    b = CaptionBuffer()
    assert b.add([seg(2.0, "second", "Me"), seg(0.5, "first")])
    assert not b.add([seg(3.0, "   "), seg(2.0, "second", "Me")])  # empty + duplicate
    b.add([seg(1.0, "late Others segment")])  # arrives after a later Me line
    assert [ln.text for ln in b.lines] == ["first", "late Others segment", "second"]
    assert b.plain_text().splitlines() == [
        "Others first",
        "Others late Others segment",
        "Me     second",
    ]


def test_buffer_trims_to_max_lines():
    b = CaptionBuffer(max_lines=3)
    b.add([seg(float(i), f"line {i}") for i in range(10)])
    assert len(b) == 3
    assert [ln.text for ln in b.lines] == ["line 7", "line 8", "line 9"]
    # a trimmed line may be re-added later (its dedupe key was dropped too)
    b.add([seg(10.0, "line 10")])
    assert b.lines[-1].text == "line 10" and len(b) == 3


def test_buffer_keeps_language_and_unknown_speaker():
    b = CaptionBuffer()
    b.add([Segment(0, 1, "Привет", "", "ru")])
    (ln,) = b.lines
    assert ln.label == "?" and ln.language == "ru"
    b.clear()
    assert len(b) == 0 and b.plain_text() == ""


def test_is_at_bottom():
    assert is_at_bottom(500, 510)  # within slack
    assert not is_at_bottom(300, 510)  # user scrolled up
    assert is_at_bottom(100, 80)  # content shorter than viewport


def test_should_auto_hide():
    assert not should_auto_hide(None, 100.0)
    assert not should_auto_hide(95.0, 100.0)
    assert should_auto_hide(90.0, 100.0)
