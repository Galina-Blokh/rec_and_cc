"""Floating live-captions panel for macOS (AppKit via PyObjC). Main thread only.

A non-activating HUD ``NSPanel``: it floats above other windows, joins every Space (so it
shows over full-screen video/Zoom), never takes focus from the app you're using, and
remembers where you put it. The text logic lives in ``captions.py``; this module only
renders a ``CaptionBuffer`` into an ``NSTextView``. Right-to-left (Hebrew) lines get their own
paragraph style: right-aligned with RTL base direction, so the label sits on the right and
the text (with any embedded English) reads right to left.

Lifecycle: ``show(title)`` (new session) → ``append(segments)``… → ``end()`` (text stays,
title says the transcript is saved, auto-hides after ~10 s) / ``hide()``. Call ``tick()``
from a main-thread timer to update the clock and run the auto-hide. Closing the window
with its close button hides it for the rest of that session only.
"""

from __future__ import annotations

# PyObjC builds AppKit/Foundation symbols at runtime and ships no type stubs, and it is not
# installed off macOS (CI type-checks on Linux), so its imports and attributes are unknown.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import logging
import time
from collections.abc import Iterable
from typing import Any

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBox,
    NSBoxCustom,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSFontAttributeName,
    NSFontWeightRegular,
    NSFontWeightSemibold,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSMutableAttributedString,
    NSMutableParagraphStyle,
    NSNoTitle,
    NSPanel,
    NSParagraphStyleAttributeName,
    NSScreen,
    NSScrollView,
    NSTextAlignmentNatural,
    NSTextAlignmentRight,
    NSTextTab,
    NSTextView,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskHUDWindow,
    NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskUtilityWindow,
    NSWritingDirectionRightToLeft,
)
from Foundation import NSObject

from ..models import Segment
from .captions import (
    PLACEHOLDER_TEXT,
    CaptionBuffer,
    default_frame,
    is_at_bottom,
    should_auto_hide,
    speaker_rgb,
)
from .captions import header_text as make_header

log = logging.getLogger(__name__)

AUTOSAVE_NAME = "RecAndCCLiveCaptions"
TEXT_PT = 16.0
LABEL_PT = 13.0
LABEL_COLUMN_PT = 64.0  # wrapped lines are indented to this tab stop
BIG = 1.0e7
BACKGROUND_TINT = 0.55  # black overlay alpha on the HUD material
FONT, COLOR, PARA = (
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSParagraphStyleAttributeName,
)


def _rgb(rgb, alpha: float = 1.0):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(rgb[0], rgb[1], rgb[2], alpha)


class _PanelDelegate(NSObject):
    """windowWillClose: → tell the owning CaptionsPanel the user closed it."""

    def initWithCallback_(self, cb):
        self = objc.super(_PanelDelegate, self).init()
        if self is None:
            return None
        self._cb = cb
        return self

    def windowWillClose_(self, _notification):
        try:
            self._cb()
        except Exception:
            log.exception("captions close callback failed")


class CaptionsPanel:
    def __init__(self, max_lines: int = 200):
        self.buffer = CaptionBuffer(max_lines)
        self.title = ""
        self._started: float | None = None  # monotonic, for the clock
        self._ended: float | None = None  # monotonic, for auto-hide
        self._user_closed = False
        self._programmatic_close = False
        # PyObjC objects (untyped), created by _ensure_panel()
        self._panel: Any = None
        self._text: Any = None
        self._scroll: Any = None
        self._delegate = None
        self._last_header = None

    # ------------------------------------------------------------------ public API
    @property
    def is_visible(self) -> bool:
        return bool(self._panel is not None and self._panel.isVisible())

    @property
    def active(self) -> bool:
        """A session is live (shown and not ended)."""
        return self._started is not None and self._ended is None

    def show(self, title: str, display: bool = True) -> None:
        """Start a new session: clear old text and bring the panel up (without focusing).
        ``display=False`` only starts collecting text (captions turned off); ``reveal()``
        shows it later."""
        self.buffer.clear()
        self.title = title or ""
        self._started = time.monotonic()
        self._ended = None
        self._user_closed = False
        if not display:
            self.hide()
            return
        self._ensure_panel()
        self._render_text(force_scroll=True)
        self._render_header(force=True)
        self._panel.orderFrontRegardless()

    def reveal(self) -> None:
        """Show the current/last session again (e.g. the menu toggle was switched on)."""
        if self._started is None:
            return
        self._user_closed = False
        self._ensure_panel()
        self._render_text(force_scroll=True)
        self._render_header(force=True)
        self._panel.orderFrontRegardless()

    def append(self, segments: Iterable[Segment]) -> None:
        if self._started is None:  # segments without live_start: start implicitly
            self.show(self.title)
        if self.buffer.add(segments) and self._panel is not None:
            self._render_text()

    def end(self) -> None:
        """Session over: keep the text, show 'finished', auto-hide after ~10 s."""
        if self._started is None:
            return
        self._ended = time.monotonic()
        self._render_header(force=True)

    def hide(self) -> None:
        if self._panel is not None and self._panel.isVisible():
            self._programmatic_close = True
            try:
                self._panel.orderOut_(None)
            finally:
                self._programmatic_close = False

    def tick(self, elapsed: float | None = None) -> None:
        """Main-thread timer hook: refresh the clock; auto-hide a finished session.
        ``elapsed`` overrides the panel's own clock (e.g. the engine's recording time)."""
        if self._panel is None or self._started is None:
            return
        if should_auto_hide(self._ended, time.monotonic()):
            self.hide()
            self._started = self._ended = None
            return
        self._render_header(elapsed=elapsed)

    def header(self, elapsed: float | None = None) -> str:
        if elapsed is None and self._started is not None:
            elapsed = time.monotonic() - self._started
        return make_header(self.title, elapsed, finished=self._ended is not None)

    # ------------------------------------------------------------------ AppKit
    def _on_user_close(self) -> None:
        if not self._programmatic_close:
            self._user_closed = True

    def _ensure_panel(self) -> None:
        if self._panel is not None:
            return
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskResizable
            | NSWindowStyleMaskUtilityWindow
            | NSWindowStyleMaskHUDWindow
            | NSWindowStyleMaskNonactivatingPanel
        )
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            self._default_frame(), style, NSBackingStoreBuffered, False
        )
        panel.setReleasedWhenClosed_(False)
        panel.setFloatingPanel_(True)
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setHidesOnDeactivate_(False)  # we're an accessory app: never "active"
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setWorksWhenModal_(True)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setMovableByWindowBackground_(True)
        panel.setMinSize_((320.0, 90.0))
        panel.setTitle_("Live captions")
        # Restores the user's last position/size if there is one, then keeps saving it.
        panel.setFrameAutosaveName_(AUTOSAVE_NAME)
        panel.setFrameUsingName_(AUTOSAVE_NAME)

        # The HUD material is translucent; a dark tint on top keeps white text readable over
        # bright video while still showing a hint of what's behind.
        view = panel.contentView()
        content = view.bounds()
        tint = NSBox.alloc().initWithFrame_(content)
        tint.setBoxType_(NSBoxCustom)
        tint.setBorderWidth_(0.0)
        tint.setTitlePosition_(NSNoTitle)
        tint.setFillColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, BACKGROUND_TINT))
        tint.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        view.addSubview_(tint)
        scroll = NSScrollView.alloc().initWithFrame_(content)
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setDrawsBackground_(False)
        scroll.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        size = scroll.contentSize()
        tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, size.width, size.height))
        tv.setEditable_(False)
        tv.setSelectable_(True)
        tv.setRichText_(True)
        tv.setDrawsBackground_(False)
        tv.setTextContainerInset_((10.0, 8.0))
        tv.setMinSize_((0.0, size.height))
        tv.setMaxSize_((BIG, BIG))
        tv.setVerticallyResizable_(True)
        tv.setHorizontallyResizable_(False)
        tv.setAutoresizingMask_(NSViewWidthSizable)
        tv.textContainer().setContainerSize_((size.width, BIG))
        tv.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(tv)
        panel.contentView().addSubview_(scroll)

        self._delegate = _PanelDelegate.alloc().initWithCallback_(self._on_user_close)
        panel.setDelegate_(self._delegate)
        self._panel, self._scroll, self._text = panel, scroll, tv

    @staticmethod
    def _default_frame():
        """Bottom-center of the main screen."""
        screen = NSScreen.mainScreen()
        visible = None
        if screen is not None:
            vf = screen.visibleFrame()
            visible = (vf.origin.x, vf.origin.y, vf.size.width, vf.size.height)
        return NSMakeRect(*default_frame(visible))

    def _render_header(self, elapsed: float | None = None, force: bool = False) -> None:
        text = self.header(elapsed)
        if force or text != self._last_header:
            self._last_header = text
            self._panel.setTitle_(text)

    def _at_bottom(self) -> bool:
        clip = self._scroll.contentView().bounds()
        doc = self._text.frame()
        return is_at_bottom(clip.origin.y + clip.size.height, doc.size.height)

    @staticmethod
    def _paragraph(rtl: bool = False):
        """Label column + hanging indent. TextKit measures tab stops and the head indent from
        the leading edge, so the RTL style mirrors the layout (label at the right edge)."""
        para = NSMutableParagraphStyle.alloc().init()
        align = NSTextAlignmentNatural if rtl else 0  # 0 = left, as before
        para.setTabStops_(
            [NSTextTab.alloc().initWithTextAlignment_location_options_(align, LABEL_COLUMN_PT, {})]
        )
        para.setDefaultTabInterval_(LABEL_COLUMN_PT)
        para.setHeadIndent_(LABEL_COLUMN_PT)
        para.setParagraphSpacing_(5.0)
        if rtl:
            para.setBaseWritingDirection_(NSWritingDirectionRightToLeft)
            para.setAlignment_(NSTextAlignmentRight)
        return para

    def _attributed(self):
        para, rtl_para = self._paragraph(), self._paragraph(rtl=True)
        text_font = NSFont.systemFontOfSize_weight_(TEXT_PT, NSFontWeightRegular)
        label_font = NSFont.systemFontOfSize_weight_(LABEL_PT, NSFontWeightSemibold)
        text_color = NSColor.colorWithCalibratedWhite_alpha_(0.96, 1.0)
        out = NSMutableAttributedString.alloc().init()
        lines = self.buffer.lines
        if not lines:
            dim = NSColor.colorWithCalibratedWhite_alpha_(0.7, 1.0)
            return NSMutableAttributedString.alloc().initWithString_attributes_(
                PLACEHOLDER_TEXT,
                {FONT: text_font, COLOR: dim, PARA: para},
            )
        for i, ln in enumerate(lines):
            p = rtl_para if ln.rtl else para
            color = _rgb(speaker_rgb(ln.label), 0.85)
            label = NSMutableAttributedString.alloc().initWithString_attributes_(
                ln.label + "\t",
                {FONT: label_font, COLOR: color, PARA: p},
            )
            body = NSMutableAttributedString.alloc().initWithString_attributes_(
                ln.text + ("\n" if i < len(lines) - 1 else ""),
                {FONT: text_font, COLOR: text_color, PARA: p},
            )
            out.appendAttributedString_(label)
            out.appendAttributedString_(body)
        return out

    def _render_text(self, force_scroll: bool = False) -> None:
        follow = force_scroll or self._at_bottom()
        storage = self._text.textStorage()
        storage.setAttributedString_(self._attributed())
        if follow:
            self._text.scrollRangeToVisible_((storage.length(), 0))
