"""Meeting notes: list, rename and delete past sessions (no UI imports, unit-tested).

``scan_notes`` reads the session folders under ``output_dir`` (blocking: call it off the UI
thread) and the shells in ``mac.py`` / ``tray.py`` show the rows ``note_columns`` /
``note_label`` give. Deleting moves the folder to the system Trash where that needs no new
dependency (Foundation's ``NSFileManager`` on macOS, ``SHFileOperationW`` on Windows), with
no ``.discarded`` marker, so a folder restored from the Trash comes back as it was. Without
a Trash, or to finish a partial move, it uses ``discard_session`` (marker first, so a
half-deleted folder is never transcribed or listed and is retried at the next start).
Either way the folder leaves ``output_dir``; the caller then drops it from the search index.
"""

from __future__ import annotations

import ctypes
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..models import SessionInfo, TriggerKind, discard_session
from .common import as_str, fmt_elapsed

log = logging.getLogger(__name__)

NOTES_LIMIT = 200  # newest sessions listed
EXCERPT_CHARS = 120
TRANSCRIPT_MD, TRANSCRIPT_JSON, SUMMARY_MD = "transcript.md", "transcript.json", "summary.md"
ACTIONS = ("Open transcript", "Open summary", "Reveal folder", "Rename…", "Delete…")


@dataclass
class Note:
    directory: Path
    title: str
    started_at: datetime
    duration_s: float | None
    transcribed: bool
    speakers: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    has_transcript: bool = False
    has_summary: bool = False
    excerpt: str = ""  # first lines of summary.md, for display only (never logged)


def session_title(info: SessionInfo) -> str:
    """The user's title, else the app that triggered the recording, else what kind it was."""
    if info.title:
        return info.title
    if info.trigger_app:
        return info.trigger_app
    return "Manual recording" if info.trigger_kind == TriggerKind.MANUAL else info.directory.name


def summary_excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """The first text of a summary.md as one line (headings and list marks dropped)."""
    words: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("#*->").strip()
        if line:
            words += line.split()
        if sum(len(w) + 1 for w in words) > limit:
            break
    out = " ".join(words)
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"


def _transcript_facts(path: Path) -> tuple[list[str], list[str], float | None]:
    """(speakers in order of appearance, languages, duration) from a transcript.json."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        speakers = list(dict.fromkeys(str(s["speaker"]) for s in data.get("segments", [])))
        dur = data.get("duration")
        return speakers, [str(x) for x in data.get("languages", [])], dur and float(dur)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return [], [], None


def load_note(directory: Path) -> Note | None:
    """One session folder as a Note, or None if it is discarded or has no readable
    session.json."""
    d = Path(directory)
    try:
        if (d / SessionInfo.DISCARD_MARKER).exists():
            return None
        info = SessionInfo.load(d)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    speakers, languages, duration = _transcript_facts(d / TRANSCRIPT_JSON)
    if duration is None and info.ended_at is not None:
        duration = max(0.0, (info.ended_at - info.started_at).total_seconds())
    note = Note(d, session_title(info), info.started_at, duration, info.transcribed)
    note.speakers, note.languages = speakers, languages
    note.has_transcript = (d / TRANSCRIPT_MD).exists()
    try:
        note.excerpt = summary_excerpt((d / SUMMARY_MD).read_text(encoding="utf-8"))
        note.has_summary = True
    except (OSError, ValueError):
        pass
    return note


def scan_notes(output_dir: Path, limit: int = NOTES_LIMIT) -> list[Note]:
    """The newest ``limit`` sessions under <output_dir>/<day>/<session>/, newest first.
    Blocking (reads every listed session's files): call it off the UI thread."""
    try:
        metas = list(Path(output_dir).glob(f"*/*/{SessionInfo.FILENAME}"))
    except OSError:
        return []
    # Folder names sort by time (<day>/<HHMMSS>_<app>); read only the newest ones.
    metas.sort(key=lambda f: (f.parent.parent.name, f.parent.name), reverse=True)
    notes = [n for f in metas[: limit * 2] if (n := load_note(f.parent))]
    notes.sort(key=lambda n: n.started_at.timestamp(), reverse=True)
    return notes[:limit]


def note_columns(note: Note) -> tuple[str, str, str, str, str, str]:
    """(title, date and time, duration, speakers, languages, status) for a list row."""
    return (
        note.title,
        note.started_at.strftime("%Y-%m-%d %H:%M"),
        fmt_elapsed(note.duration_s) if note.duration_s is not None else "—",
        ", ".join(note.speakers) or "—",
        ", ".join(note.languages) or "—",
        ("transcribed" if note.transcribed else "not yet")
        + (", summary" if note.has_summary else ""),
    )


def note_label(note: Note) -> str:
    """One line per session for a plain list (the macOS dialog)."""
    title, when, dur, speakers, langs, status = note_columns(note)
    return f"{when} · {title} · {dur} · {speakers} · {langs} · {status}"


def note_actions(note: Note) -> list[str]:
    """What can be done with a session (``ACTIONS``, minus opening files it doesn't have)."""
    skip = {"Open transcript"} if not note.has_transcript else set()
    if not note.has_summary:
        skip.add("Open summary")
    return [a for a in ACTIONS if a not in skip]


def action_target(note: Note, action: str) -> Path:
    """The file or folder an "Open…" / "Reveal folder" action opens."""
    names = {"Open transcript": TRANSCRIPT_MD, "Open summary": SUMMARY_MD}
    return note.directory / names[action] if action in names else note.directory


def rename_session(directory: Path, title: str) -> SessionInfo:
    """Store a display title in session.json ("" goes back to the default title)."""
    info = SessionInfo.load(directory)
    info.title = " ".join(title.split())
    info.save()
    return info


def move_to_trash(path: Path) -> bool:
    """Move a file or folder to the system Trash; False if unsupported here or it failed."""
    try:
        if sys.platform == "darwin":
            from Foundation import (  # pyright: ignore[reportMissingImports]
                NSURL,  # pyright: ignore[reportAttributeAccessIssue]
                NSFileManager,  # pyright: ignore[reportAttributeAccessIssue]
            )

            url = NSURL.fileURLWithPath_(str(path))
            ok, _, err = NSFileManager.defaultManager().trashItemAtURL_resultingItemURL_error_(
                url, None, None
            )
            return bool(ok) and err is None
        if sys.platform == "win32":
            return _win_recycle(path)
    except Exception as e:  # noqa: BLE001 - fall back to deleting
        log.warning("could not move %s to the Trash: %s", path, type(e).__name__)
    return False


def _win_recycle(path: Path) -> bool:
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND),
            ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR),
            ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_ushort),
            ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    # FO_DELETE with FOF_ALLOWUNDO (Recycle Bin) | NOCONFIRMATION | SILENT | NOERRORUI
    op = SHFILEOPSTRUCTW(wFunc=3, pFrom=str(path) + "\0", fFlags=0x40 | 0x10 | 0x4 | 0x400)
    shell32 = ctypes.windll.shell32  # pyright: ignore[reportAttributeAccessIssue]
    return shell32.SHFileOperationW(ctypes.byref(op)) == 0 and not op.fAnyOperationsAborted


def can_trash() -> bool:
    return sys.platform in ("darwin", "win32")


def delete_session(directory: Path) -> str:
    """Delete a session: "trashed", "deleted" (permanently) or "failed" (left as it was).
    No Trash here, or a Trash move that took only part of the folder (a locked file on
    Windows) -> ``discard_session`` finishes it. Blocking: call it off the UI thread."""
    d = Path(directory)
    if can_trash() and move_to_trash(d) and not d.exists():
        return "trashed"
    if can_trash() and (d / SessionInfo.FILENAME).exists():
        return "failed"
    return "deleted" if discard_session(d) else "failed"


def delete_prompt(note: Note) -> str:
    where = "go to the Trash" if can_trash() else "are deleted permanently (no Trash here)"
    when = note.started_at.strftime("%Y-%m-%d %H:%M")
    return f"Delete “{note.title}” ({when})? Its audio, transcript and summary {where}."


# --- macOS: AppleScript dialogs (run with common.osascript on a worker thread)
def list_script(notes: list[Note]) -> str:
    items = ", ".join(as_str(f"{i}. {note_label(n)}"[:200]) for i, n in enumerate(notes, 1))
    return (
        f'choose from list {{{items}}} with title "Meeting notes" '
        f'with prompt "Newest first. Choose a meeting:" OK button name "Choose"'
    )


def actions_script(note: Note) -> str:
    prompt = note.title + (f"\n\n{note.excerpt}" if note.excerpt else "")
    items = ", ".join(map(as_str, note_actions(note)))
    return (
        f'choose from list {{{items}}} with title "Meeting notes" '
        f'with prompt {as_str(prompt)} OK button name "OK"'
    )


def rename_script(note: Note) -> str:
    return (
        f'display dialog "New title (empty: the default):" default answer {as_str(note.title)} '
        f'with title "Rename meeting" buttons {{"Cancel", "Rename"}} default button "Rename"'
    )


def delete_script(note: Note) -> str:
    return (
        f'display dialog {as_str(delete_prompt(note))} with title "Delete meeting" '
        f'buttons {{"Cancel", "Delete"}} default button "Cancel" with icon caution'
    )


def chosen_index(output: str, count: int) -> int | None:
    """Index of the row picked in ``list_script``'s dialog, or None (cancelled)."""
    number = output.partition(".")[0]
    return int(number) - 1 if number.isdigit() and 0 < int(number) <= count else None
