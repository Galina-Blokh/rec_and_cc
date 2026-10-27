"""Live captions: platform-neutral logic (formatting, ordering, trimming, header, timers).

Stories: US-11, US-19.
"""

import sys

import pytest

from recandcc.models import Segment
from recandcc.ui.captions import (
    FINISHED_TEXT,
    PDI,
    RLI,
    CaptionBuffer,
    CaptionLine,
    clean_text,
    default_frame,
    fmt_mmss,
    format_line,
    header_text,
    is_at_bottom,
    should_auto_hide,
    speaker_hex,
    speaker_rgb,
    tk_label_tag,
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


def test_default_frame_is_bottom_center_and_fits_the_screen():
    assert default_frame(None) == (200.0, 120.0, 760.0, 190.0)
    x, y, w, h = default_frame((0.0, 25.0, 1440.0, 875.0))
    assert (w, h) == (760.0, 190.0)
    assert x == (1440.0 - 760.0) / 2
    assert y == 25.0 + 90.0
    x, _, w, _ = default_frame((100.0, 0.0, 800.0, 600.0))  # narrow screen: 80% of width
    assert w == 640.0
    assert x == 100.0 + 80.0


def test_speaker_colors():
    assert speaker_rgb("Me") == (0.55, 0.86, 0.62)
    assert speaker_rgb("Others") == (0.52, 0.74, 1.0)
    assert speaker_rgb("Speaker 1") != speaker_rgb("Speaker 2")
    assert speaker_rgb("Speaker 1") not in (speaker_rgb("Me"), speaker_rgb("Others"))
    assert speaker_rgb("Speaker 7") == speaker_rgb("Speaker 1")  # the palette cycles
    assert speaker_rgb("?") == (0.85, 0.75, 0.5)


def test_hebrew_lines_are_right_to_left():
    b = CaptionBuffer()
    b.add([seg(0.0, "Hello", "Me"), seg(1.0, "שלום, אני ב-Zoom", lang="he")])
    en, he = b.lines
    assert not en.rtl and en.alignment == "left"
    assert he.rtl and he.alignment == "right"
    assert format_line(he) == f"Others  {RLI}שלום, אני ב-Zoom{PDI}"
    assert CaptionLine(0.0, "Me", "שלום").rtl  # no language: decided by the text


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit captions panel")
def test_mac_panel_lays_out_hebrew_right_to_left():
    pytest.importorskip("AppKit")
    from AppKit import NSParagraphStyleAttributeName, NSWritingDirectionRightToLeft

    from recandcc.ui.captions_mac import CaptionsPanel

    panel = CaptionsPanel()  # no window is created until show()
    panel.buffer.add([seg(0.0, "Hello", "Me"), seg(1.0, "שלום", lang="he")])
    text = panel._attributed()
    first = text.attribute_atIndex_effectiveRange_(NSParagraphStyleAttributeName, 0, None)[0]
    at = text.string().index("שלום")
    style = text.attribute_atIndex_effectiveRange_(NSParagraphStyleAttributeName, at, None)[0]
    assert first.baseWritingDirection() != NSWritingDirectionRightToLeft
    assert style.baseWritingDirection() == NSWritingDirectionRightToLeft


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit captions panel")
def test_mac_panel_without_a_window_survives_end_hide_and_tick():
    """Regression: a session ending while captions are off (no NSPanel) raised
    AttributeError in the UI tick."""
    pytest.importorskip("AppKit")
    from recandcc.ui.captions_mac import CaptionsPanel

    panel = CaptionsPanel()
    panel.show("Zoom", display=False)  # collects text, creates no window
    assert panel._panel is None and panel.active
    panel.append([seg(0.0, "Hello", "Me")])
    panel.tick(elapsed=3.0)  # live, no window: no-op
    panel.end()  # used to raise on _panel.setTitle_
    assert not panel.active
    panel.hide()
    panel._ended -= 60  # past the auto-hide delay
    panel.tick()
    assert panel._started is None and panel._ended is None
    panel.end()  # no session: no-op


@pytest.mark.skipif(sys.platform != "darwin", reason="AppKit captions panel")
def test_mac_panel_late_segment_after_auto_hide_keeps_captions_off():
    pytest.importorskip("AppKit")
    from recandcc.ui.captions_mac import CaptionsPanel

    panel = CaptionsPanel()
    panel.show("Zoom", display=False)
    panel.end()
    panel._ended -= 60
    panel.tick()  # session state reset
    assert panel._started is None
    panel.append([seg(0.0, "late", "Me")])  # implicit start must honour display=False
    assert panel._panel is None and panel.active


def test_tk_label_tags_never_reconfigure_me_and_others():
    """Stories: US-33. "Me"/"Others" keep their set-up tags; Speaker N gets its hex colour."""

    class FakeText:
        def __init__(self):
            self.calls = []

        def tag_configure(self, tag, **kw):
            self.calls.append((tag, kw))

    t, known = FakeText(), set()
    font = ("f", 10)
    assert tk_label_tag(t, known, "Me", font) == "Me"
    assert tk_label_tag(t, known, "Others", font) == "Others"
    assert t.calls == []
    tag = tk_label_tag(t, known, "Speaker 2", font)
    assert tag == "spk:" + speaker_hex("Speaker 2")
    assert t.calls == [(tag, {"foreground": speaker_hex("Speaker 2"), "font": font})]
    assert tk_label_tag(t, known, "Speaker 2", font) == tag
    assert len(t.calls) == 1  # configured once
