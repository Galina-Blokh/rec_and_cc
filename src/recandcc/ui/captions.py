"""Platform-neutral logic for the live captions window (no UI imports, unit-tested).

The macOS panel (``captions_mac.py``) and the Windows window (``tray.py``) both keep a
``CaptionBuffer`` and render its lines; everything that can be tested without a window
lives here: line formatting, ordering, trimming, header text, the auto-hide timer and the
writing direction of each line (Hebrew lines are right-aligned and laid out right to left).
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from dataclasses import dataclass

from ..models import Segment, is_rtl

MAX_LINES = 200
AUTO_HIDE_AFTER_S = 10.0
FINISHED_TEXT = "Recording finished — transcript saved"
# Unicode right-to-left isolate (RLI .. PDI) for plain-text renderings of RTL lines.
RLI, PDI = "\u2067", "\u2069"
# Within this many points of the bottom counts as "at the bottom" (keep auto-scrolling).
SCROLL_SLACK_PT = 24.0
PLACEHOLDER_TEXT = "Listening… captions appear a moment after each phrase."
DEFAULT_SIZE = (760.0, 190.0)  # panel width, height in points
BOTTOM_MARGIN_PT = 90.0
FALLBACK_ORIGIN = (200.0, 120.0)  # when there is no main screen
# (r, g, b) per speaker label; anything else gets the "other" color.
SPEAKER_RGB = {"Me": (0.55, 0.86, 0.62), "Others": (0.52, 0.74, 1.0)}
OTHER_RGB = (0.85, 0.75, 0.5)
# "Speaker N" (live captions) cycles through these; "Me" and "Others" keep their colours above.
SPEAKER_PALETTE = (
    (0.98, 0.78, 0.40),  # amber
    (0.86, 0.62, 0.95),  # violet
    (0.45, 0.88, 0.85),  # teal
    (0.98, 0.60, 0.60),  # coral
    (0.78, 0.88, 0.45),  # lime
    (0.95, 0.70, 0.85),  # pink
)


@dataclass(frozen=True)
class CaptionLine:
    start: float  # seconds from session start
    speaker: str  # "Me" / "Others"
    text: str
    language: str = ""

    @property
    def label(self) -> str:
        return speaker_label(self.speaker)

    @property
    def rtl(self) -> bool:
        """Right-to-left line (Hebrew): the window right-aligns it with RTL base direction."""
        return is_rtl(self.language, self.text)

    @property
    def alignment(self) -> str:
        """``"right"`` for RTL lines, else ``"left"`` (the widget-neutral justification)."""
        return "right" if self.rtl else "left"


def speaker_label(speaker: str) -> str:
    return (speaker or "").strip() or "?"


def clean_text(text: str) -> str:
    """Collapse whitespace/newlines; captions are one line per segment."""
    return " ".join((text or "").split())


def format_line(line: CaptionLine, width: int = 7) -> str:
    """Plain-text form: ``'Others  Hello team'`` (label padded to ``width``). An RTL text is
    wrapped in a right-to-left isolate so it doesn't reorder the label next to it."""
    text = f"{RLI}{line.text}{PDI}" if line.rtl else line.text
    return f"{line.label:<{width}} {text}"


def fmt_mmss(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def header_text(title: str | None, elapsed: float | None, finished: bool = False) -> str:
    """Window title: ``'Live — Zoom · 🔴 03:12'`` or the finished message."""
    name = (title or "").strip() or "recording"
    if finished:
        return f"{FINISHED_TEXT} · {name}"
    clock = f" · 🔴 {fmt_mmss(elapsed)}" if elapsed is not None else ""
    return f"Live — {name}{clock}"


def is_at_bottom(visible_bottom: float, content_height: float, slack: float = SCROLL_SLACK_PT):
    """True when the viewport shows the end of the content (so new text should auto-scroll).
    ``visible_bottom`` = scroll offset + viewport height, in the same units as the content."""
    return visible_bottom >= content_height - slack


def speaker_rgb(label: str) -> tuple[float, float, float]:
    if label in SPEAKER_RGB:
        return SPEAKER_RGB[label]
    head, _, num = label.partition(" ")
    if head == "Speaker" and num.isdigit() and int(num) >= 1:
        return SPEAKER_PALETTE[(int(num) - 1) % len(SPEAKER_PALETTE)]
    return OTHER_RGB


def speaker_hex(label: str) -> str:
    """``speaker_rgb`` as ``#rrggbb`` (Tk)."""
    r, g, b = speaker_rgb(label)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def default_frame(
    visible: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    """Initial panel frame ``(x, y, w, h)``: bottom-center of the screen's visible frame
    (AppKit coordinates, origin bottom-left), at most 80% of its width."""
    w, h = DEFAULT_SIZE
    if visible is None:
        return (*FALLBACK_ORIGIN, w, h)
    vx, vy, vw, _vh = visible
    w = min(w, vw * 0.8)
    return (vx + (vw - w) / 2, vy + BOTTOM_MARGIN_PT, w, h)


def should_auto_hide(ended_at: float | None, now: float, after: float = AUTO_HIDE_AFTER_S):
    return ended_at is not None and now - ended_at >= after


class CaptionBuffer:
    """Caption lines in time order, capped at ``max_lines`` (oldest dropped).

    Segments from the two tracks are transcribed independently, so an ``Others`` segment can
    arrive after a later ``Me`` one; lines are inserted by start time. Empty texts and exact
    duplicates (same speaker, start and text) are ignored.
    """

    def __init__(self, max_lines: int = MAX_LINES):
        self.max_lines = max_lines
        self._lines: list[CaptionLine] = []
        self._keys: set[tuple] = set()

    def __len__(self) -> int:
        return len(self._lines)

    @property
    def lines(self) -> list[CaptionLine]:
        return list(self._lines)

    def clear(self) -> None:
        self._lines.clear()
        self._keys.clear()

    def add(self, segments: Iterable[Segment]) -> bool:
        """Add segments; returns True if anything changed."""
        changed = False
        for seg in segments:
            text = clean_text(seg.text)
            if not text:
                continue
            line = CaptionLine(float(seg.start), speaker_label(seg.speaker), text, seg.language)
            key = (line.start, line.speaker, line.text)
            if key in self._keys:
                continue
            self._keys.add(key)
            starts = [ln.start for ln in self._lines]
            self._lines.insert(bisect.bisect_right(starts, line.start), line)
            changed = True
        if len(self._lines) > self.max_lines:
            for old in self._lines[: len(self._lines) - self.max_lines]:
                self._keys.discard((old.start, old.speaker, old.text))
            del self._lines[: len(self._lines) - self.max_lines]
        return changed

    def plain_text(self) -> str:
        width = max((len(ln.label) for ln in self._lines), default=0)
        return "\n".join(format_line(ln, width) for ln in self._lines)
