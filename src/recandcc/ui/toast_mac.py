"""In-app notifications ("toasts") for macOS.

Posting notifications with ``osascript -e 'display notification …'`` makes macOS attribute
them to Script Editor, which then pops up an empty Script Editor window. The Python child of
RecAndCC.app isn't an app bundle, so it can't post real User Notifications of its own either.
Instead this shows a small floating, non-activating panel in the top-right corner that fades
out by itself. It never takes focus and shows over full-screen apps.

Main thread only (the menu-bar app's 0.5 s timer calls ``tick``).
"""

from __future__ import annotations

# PyObjC builds AppKit/Foundation symbols at runtime and ships no type stubs, and it is not
# installed off macOS (CI type-checks on Linux), so its imports and attributes are unknown.
# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
import time
from typing import Any

from AppKit import (
    NSBackingStoreBuffered,
    NSBox,
    NSBoxCustom,
    NSColor,
    NSFloatingWindowLevel,
    NSFont,
    NSMakeRect,
    NSNoTitle,
    NSPanel,
    NSScreen,
    NSTextField,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)

WIDTH, HEIGHT, MARGIN = 340.0, 70.0, 16.0
SHOW_SECONDS = 6.0


class ToastPanel:
    def __init__(self) -> None:
        # PyObjC objects (untyped), created on first use
        self._panel: Any = None
        self._title: Any = None
        self._message: Any = None
        self._hide_at: float | None = None

    def _label(self, frame, font, color):
        field = NSTextField.labelWithString_("")
        field.setFrame_(frame)
        field.setFont_(font)
        field.setTextColor_(color)
        field.setLineBreakMode_(4)  # NSLineBreakByTruncatingTail
        return field

    def _ensure_panel(self) -> None:
        if self._panel is not None:
            return
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WIDTH, HEIGHT),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        panel.setLevel_(NSFloatingWindowLevel)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )
        panel.setHidesOnDeactivate_(False)
        panel.setOpaque_(False)
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)  # purely informational; clicks pass through
        panel.setBackgroundColor_(NSColor.clearColor())
        # A custom NSBox draws the rounded dark background (no Quartz/CGColor needed).
        content = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, WIDTH, HEIGHT))
        content.setBoxType_(NSBoxCustom)
        content.setTitlePosition_(NSNoTitle)
        content.setBorderWidth_(0.0)
        content.setCornerRadius_(12.0)
        content.setFillColor_(NSColor.colorWithWhite_alpha_(0.12, 0.92))
        content.setContentViewMargins_((0, 0))
        panel.setContentView_(content)
        self._title = self._label(
            NSMakeRect(16, HEIGHT - 32, WIDTH - 32, 20),
            NSFont.boldSystemFontOfSize_(14),
            NSColor.whiteColor(),
        )
        self._message = self._label(
            NSMakeRect(16, 12, WIDTH - 32, 20),
            NSFont.systemFontOfSize_(13),
            NSColor.colorWithWhite_alpha_(0.85, 1.0),
        )
        content.contentView().addSubview_(self._title)
        content.contentView().addSubview_(self._message)
        self._panel = panel

    def show(self, title: str, message: str) -> None:
        """Show (or replace) the toast; it hides itself after a few seconds."""
        self._ensure_panel()
        self._title.setStringValue_(title)
        self._message.setStringValue_(message)
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + screen.size.width - WIDTH - MARGIN
        y = screen.origin.y + screen.size.height - HEIGHT - MARGIN
        self._panel.setFrameOrigin_((x, y))
        self._panel.orderFrontRegardless()
        self._hide_at = time.monotonic() + SHOW_SECONDS

    def tick(self) -> None:
        if self._hide_at is not None and time.monotonic() >= self._hide_at:
            self._hide_at = None
            if self._panel is not None:
                self._panel.orderOut_(None)
