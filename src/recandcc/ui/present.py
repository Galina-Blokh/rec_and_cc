"""Platform-neutral presenter logic for the menu-bar shells (no UI imports, unit-tested).

``mac.py`` (rumps), ``toast_mac.py`` (AppKit) and ``tray.py`` (pystray) call these to decide
*what* to show: menu item titles and check marks, notification texts, the ask-mode dialog's
AppleScript and how to read its answer, where a toast goes and when it hides, and the
transcript-search results (rows, notices, what a chosen match opens). The
shells only apply the result to their widgets, on the main thread.
"""

from __future__ import annotations

from pathlib import Path

from ..models import Mode, SessionInfo, TriggerKind
from .common import (
    ONBOARDING_TEXT,
    PERM_LABELS,
    PERM_TITLES,
    PROMPT_BUTTONS,
    as_str,
    clock,
    transcript_label,
)

MODE_LABELS = ((Mode.OFF, "Off"), (Mode.ASK, "Ask"), (Mode.AUTO, "Auto"))
MIN_DIALOG_TIMEOUT_S = 5
ONBOARDING_GRANT = "Grant now"
TOAST_SIZE = (340.0, 70.0)  # width, height
TOAST_MARGIN = 16.0
TOAST_SHOW_S = 6.0

Rect = tuple[float, float, float, float]  # x, y, width, height (AppKit: origin bottom-left)


def check(on: bool) -> int:
    """A menu item's ``state``: 1 shows a check mark."""
    return 1 if on else 0


def record_item_title(recording: bool, shortcut: str = "", sep: str = "  ") -> str:
    """The Start/Stop item, with the global shortcut appended when one is active."""
    title = "Stop recording" if recording else "Start recording"
    return f"{title}{sep}{shortcut}" if shortcut else title


def retention_label(days: int) -> str:
    """Item of the "Delete audio after" submenu (0 = keep audio forever)."""
    if days == 0:
        return "Never (keep audio)"
    return f"{days} day" + ("s" if days > 1 else "")


def permission_title(kind: str, status: str) -> str:
    """``'Microphone: Granted'``; an unknown status is shown as is."""
    return f"{PERM_TITLES[kind]}: {PERM_LABELS.get(status, status)}"


def prompt_notice(app: str, kind: TriggerKind) -> tuple[str, str]:
    """Toast shown when ask mode starts a recording and asks whether to keep it."""
    what = "a call" if kind == TriggerKind.CALL else "audio"
    return "Recording — keep it?", f"{app}: {what} detected"


def transcribed_notice(session: SessionInfo, result: object) -> tuple[str, str]:
    """Toast for a finished transcription (``result`` is the exception when it failed)."""
    if isinstance(result, Exception):
        return "Transcription failed", f"{session.directory.name}: {result}"
    return "Transcript ready", session.trigger_app or session.directory.name


def live_transcript_target(session_dir: str | Path) -> Path:
    """What "Open live transcript" opens: the transcript once written, else the folder."""
    d = Path(session_dir)
    md = d / "transcript.md"
    return md if md.exists() else d


def dialog_timeout(ask_timeout_s: float) -> int:
    """Seconds before the keep-recording dialog gives up (never under 5)."""
    return max(MIN_DIALOG_TIMEOUT_S, int(ask_timeout_s))


def prompt_dialog_script(message: str, timeout_s: int) -> str:
    """AppleScript for the ask-mode dialog (buttons from ``PROMPT_BUTTONS``)."""
    buttons = ", ".join(f'"{label}"' for label, _ in PROMPT_BUTTONS)
    return (
        f'display dialog {as_str(message)} with title "RecAndCC" '
        f'buttons {{{buttons}}} default button "Keep" '
        f"with icon note giving up after {timeout_s}"
    )


def prompt_dialog_choice(output: str | None) -> str | None:
    """Engine choice for the dialog's ``osascript`` output, or None when it timed out, was
    cancelled or killed (the engine's own timeout then decides)."""
    out = output or ""
    if "gave up:true" in out:
        return None
    for label, choice in PROMPT_BUTTONS:
        if f"button returned:{label}" in out:
            return choice
    return None


def onboarding_script() -> str:
    return (
        f'display dialog {as_str(ONBOARDING_TEXT)} with title "Welcome to RecAndCC" '
        f'buttons {{"Later", "{ONBOARDING_GRANT}"}} default button "{ONBOARDING_GRANT}" '
        f"with icon note"
    )


def onboarding_accepted(output: str | None) -> bool:
    return f"button returned:{ONBOARDING_GRANT}" in (output or "")


def toast_origin(visible: Rect, size: tuple[float, float] = TOAST_SIZE) -> tuple[float, float]:
    """Bottom-left origin that puts a toast in the top-right corner of the visible screen."""
    x, y, w, h = visible
    return (x + w - size[0] - TOAST_MARGIN, y + h - size[1] - TOAST_MARGIN)


def toast_expired(hide_at: float | None, now: float) -> bool:
    return hide_at is not None and now >= hide_at


def search_notice(hits: object) -> tuple[str, str] | None:
    """Notification for a search that has nothing to list (it failed or found nothing), or
    None when there are hits to show. Never contains the query: notices may be logged."""
    if isinstance(hits, Exception):
        return "Search failed", str(hits)
    return None if hits else ("Search", "No matches")


def search_hit_row(hit) -> tuple[str, str, str, str]:
    """(session, time in the recording, speaker, snippet) columns for a search.Hit."""
    return (
        transcript_label(hit.transcript),
        clock(hit.start),
        hit.speaker,
        " ".join(hit.snippet.split()),
    )


def search_status(query: str, hits: object) -> str:
    """Status line above a search window's results (``hits`` is the exception if it failed)."""
    if isinstance(hits, Exception):
        return f"Search failed: {hits}"
    n = len(hits) if isinstance(hits, list) else 0
    if not n:
        return f"No matches for “{query}”"
    return f"{n} match{'es' if n != 1 else ''} for “{query}”"


def search_open_target(hit) -> Path:
    """What choosing a match opens: its transcript, else its session folder (call off the UI
    thread: the check may touch an unreachable volume)."""
    try:
        return hit.transcript if hit.transcript.exists() else hit.folder
    except OSError:
        return hit.folder
