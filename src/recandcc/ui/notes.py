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

NOTES_LIMIT = 100  # newest sessions listed (each one's transcript.json is read)
EXCERPT_CHARS = 120
TITLE_CHARS = 200
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
        if line.lstrip().startswith("#"):
            continue  # headings ("# Summary", "## Action items")
        line = line.strip().lstrip("*-").strip()
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
    except Exception:  # noqa: BLE001 - one broken session.json must not hide the rest
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
    notes: list[Note] = []
    for f in metas:  # stop once enough are read (discarded / broken folders are skipped)
        if len(notes) >= limit:
            break
        if note := load_note(f.parent):
            notes.append(note)
    notes.sort(key=lambda n: n.started_at.timestamp(), reverse=True)
    return notes


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


def rename_session(directory: Path, title: str) -> None:
    """Store a display title in session.json ("" goes back to the default title); only the
    ``title`` key changes, so metrics written meanwhile are kept."""
    SessionInfo.set_title(directory, " ".join(title.split())[:TITLE_CHARS])


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
        log.warning("could not move %s to the Trash: %s", Path(path).name, type(e).__name__)
    return False


def shfileop_struct() -> type[ctypes.Structure]:
    """SHFILEOPSTRUCTW (shellapi.h); built on demand, wintypes is Windows-oriented."""
    from ctypes import wintypes

    class SHFILEOPSTRUCTW(ctypes.Structure):
        # shellapi.h packs it to 1 byte on 32-bit only; 64-bit uses natural alignment.
        _pack_ = 1 if ctypes.sizeof(ctypes.c_void_p) == 4 else 8
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

    return SHFILEOPSTRUCTW


def _win_recycle(path: Path) -> bool:
    from ctypes import wintypes

    op_type = shfileop_struct()
    # FO_DELETE with FOF_ALLOWUNDO (Recycle Bin) | NOCONFIRMATION | SILENT, plus
    # FOF_WANTNUKEWARNING: if the bin can't take it (too big, no bin on that drive) Windows
    # asks before deleting for good instead of doing it silently; "No" aborts.
    # pFrom is a list ending in an empty string: the buffer adds the second NUL.
    buf = ctypes.create_unicode_buffer(str(Path(path).resolve()) + "\0")
    flags = 0x40 | 0x10 | 0x4 | 0x4000
    op = op_type(wFunc=3, pFrom=ctypes.cast(buf, wintypes.LPCWSTR), fFlags=flags)
    fn = ctypes.windll.shell32.SHFileOperationW  # pyright: ignore[reportAttributeAccessIssue]
    fn.argtypes, fn.restype = [ctypes.POINTER(op_type)], ctypes.c_int
    return fn(ctypes.byref(op)) == 0 and not op.fAnyOperationsAborted


def can_trash() -> bool:
    return sys.platform in ("darwin", "win32")


def delete_session(directory: Path, output_dir: Path) -> str:
    """Delete a session folder (blocking: call it off the UI thread). Returns
    - "trashed": all of it is in the Trash;
    - "deleted": deleted for good (no Trash here, or ``discard_session`` finished a move
      that took only part of the folder, e.g. a locked file on Windows);
    - "partial": marked discarded, but some files are in use; the rest goes at next start;
    - "refused": a link or not inside ``output_dir``; "failed": the Trash refused it and
      the folder is still there."""
    d = Path(directory)
    if d.is_symlink() or Path(output_dir).resolve() not in d.resolve().parents:
        log.warning("refusing to delete %s: not a folder inside the output folder", d.name)
        return "refused"  # never follow a link out of output_dir
    if can_trash():
        moved = move_to_trash(d)
        if not d.exists():
            return "trashed"
        if not moved and (d / SessionInfo.FILENAME).exists():
            return "failed"
    return "deleted" if discard_session(d) else "partial"


DELETE_NOTICES = {  # delete_session result -> (title, message); none for a clean delete
    "partial": ("Meeting partly deleted", "Some files are in use; they go at the next start."),
    "refused": (
        "Could not delete the meeting",
        "Its folder is a link or outside the output folder.",
    ),
    "failed": ("Could not delete the meeting", "The Trash refused it; nothing was deleted."),
}


def delete_notice(result: str) -> tuple[str, str] | None:
    """The notice for a delete_session result (None: it went where the prompt said)."""
    if result == "deleted" and can_trash():
        return "Meeting deleted for good", "Part of it couldn't go to the Trash."
    return DELETE_NOTICES.get(result)


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


def run_dialogs(notes: list[Note], ask, act, open_) -> None:
    """The macOS flow (worker thread): pick a meeting, then an action. ``ask(script)`` runs
    AppleScript and returns its output, ``act(directory, action, title="")`` renames or
    deletes (``Controller.note_action_async``), ``open_(path)`` opens a file or folder."""
    i = chosen_index(ask(list_script(notes)), len(notes))
    if i is None:
        return
    note = notes[i]
    action = ask(actions_script(note))
    if action == "Rename…":
        out = ask(rename_script(note))  # "button returned:Rename, text returned:<title>"
        if out.startswith("button returned:Rename"):
            act(note.directory, "rename", out.partition("text returned:")[2])
    elif action == "Delete…":
        if ask(delete_script(note)).startswith("button returned:Delete"):
            act(note.directory, "delete")
    elif action in note_actions(note):
        open_(action_target(note, action))


def chosen_index(output: str, count: int) -> int | None:
    """Index of the row picked in ``list_script``'s dialog, or None (cancelled)."""
    number = output.partition(".")[0]
    return int(number) - 1 if number.isdigit() and 0 < int(number) <= count else None
