"""Export a finished session to DOCX and PDF (``transcript.docx`` / ``transcript.pdf``).

Reads ``transcript.json`` (plain text; the bidi isolates are only in md/srt) and writes one
paragraph per segment: ``[hh:mm:ss] Speaker:`` and the text. Hebrew segments are laid out
right to left: DOCX gets a bidi paragraph and an rtl run; the PDF is shaped by HarfBuzz and
reordered with fpdf2's bidi support. The PDF needs a system font that covers Latin, Cyrillic
and Hebrew (Arial, Arial Unicode, DejaVu Sans, ...); without one it raises ExportError. The
libraries are imported lazily, so the app start-up doesn't pay for them. Offline; transcript
text is never logged.

If the folder has a ``summary.md`` (see ``summary.py``), its summary and action items come
first, before the transcript; without it the files are exactly as before.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..models import Segment, is_rtl
from .output import JSON_NAME, fmt_hms, languages_of
from .summary import PENDING_MARKER, SOURCE_NAME, SUMMARY_NAME, transcript_fingerprint

log = logging.getLogger(__name__)

DOCX_NAME, PDF_NAME = "transcript.docx", "transcript.pdf"
FORMATS = ("docx", "pdf")
_FONT_CANDIDATES = (  # smaller faces first: Arial Unicode is ~22 MB
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
)


class ExportError(Exception):
    """The session can't be exported (message is safe to show: no transcript text)."""


def rtl_segment(seg: Segment) -> bool:
    """Hebrew by language, or (no language) by the first strong character."""
    return is_rtl(seg.language, seg.text)


def _clean(text: str) -> str:
    """Control characters (they are invalid in DOCX XML) become spaces."""
    return "".join(" " if unicodedata.category(c) == "Cc" else c for c in text).strip()


_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_BINARY = getattr(os, "O_BINARY", 0)  # Windows: no newline translation on raw reads
# POSIX can pin the checked folder with a directory descriptor; Windows can't (residual risk).
# CPython lists os.rename (not os.replace) in supports_dir_fd; both use renameat.
_DIR_FD = all(f in os.supports_dir_fd for f in (os.open, os.rename, os.unlink, os.stat))
_MOVED = "The session folder moved during the export; not writing through it"


@dataclass(frozen=True)
class _Pin:
    """The session folder as resolved once, when the export started: its real path, its
    identity (device, inode) and, on POSIX, a descriptor that every read and write goes
    through, so a later swap of the folder or of a parent can't redirect them."""

    path: Path
    ident: tuple[int, int]
    dfd: int | None


def _pin_folder(folder: Path) -> _Pin:
    """Resolve ``folder`` once and capture its identity in the same step: the resolved path
    is opened (POSIX) or stat'ed (Windows), then resolved again; if an ancestor was swapped
    in between, the second resolve differs and the export is refused. A folder that was a
    link or junction from the start is fine: it is the target that is pinned."""
    try:
        path = Path(folder).resolve(strict=True)
    except FileNotFoundError:
        raise ExportError(f"no {JSON_NAME} in {folder}") from None
    except (OSError, RuntimeError) as e:
        raise ExportError(f"can't open the session folder ({type(e).__name__})") from None
    dfd = None
    try:
        if _DIR_FD:
            dfd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | _NOFOLLOW)
            st = os.fstat(dfd)
        else:
            st = os.stat(path)
        if Path(folder).resolve() != path:
            raise ExportError(_MOVED)
    except OSError as e:
        if dfd is not None:
            os.close(dfd)
        raise ExportError(f"can't open the session folder ({type(e).__name__})") from None
    except BaseException:
        if dfd is not None:
            os.close(dfd)
        raise
    return _Pin(path, (st.st_dev, st.st_ino), dfd)


@contextlib.contextmanager
def _pinned(folder: Path, pin: _Pin | None = None) -> Iterator[_Pin]:
    """``pin`` if the caller already has one, else the folder pinned for this call only."""
    if pin is not None:
        yield pin
        return
    own = _pin_folder(folder)
    try:
        yield own
    finally:
        if own.dfd is not None:
            os.close(own.dfd)


def _check_path_pin(pin: _Pin) -> None:
    """Windows (no dir_fd): the folder must still resolve to, and be, the pinned one."""
    try:
        st = os.stat(pin.path)
    except OSError as e:
        raise ExportError(f"can't open the session folder ({type(e).__name__})") from None
    if (st.st_dev, st.st_ino) != pin.ident:
        raise ExportError(_MOVED)


def _open_read(pin: _Pin, name: str) -> int:
    """A read-only descriptor for ``name`` in the pinned folder; a linked file is refused."""
    flags = os.O_RDONLY | _NOFOLLOW | _BINARY
    if pin.dfd is not None:
        return os.open(name, flags, dir_fd=pin.dfd)
    _check_path_pin(pin)
    path = pin.path / name
    if not _NOFOLLOW and (path.is_symlink() or path.is_junction()):
        raise OSError("linked file")  # best available without O_NOFOLLOW
    return os.open(path, flags)


def _stat_in(pin: _Pin, name: str) -> os.stat_result:
    if pin.dfd is not None:
        return os.stat(name, dir_fd=pin.dfd, follow_symlinks=False)
    return os.stat(pin.path / name, follow_symlinks=False)


def load_session(folder: Path, pin: _Pin | None = None) -> tuple[dict, list[Segment]]:
    path = Path(folder) / JSON_NAME
    try:
        with _pinned(folder, pin) as p, os.fdopen(_open_read(p, JSON_NAME), "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        segs = [
            Segment(
                float(s["start"]),
                float(s["end"]),
                _clean(str(s["text"])),
                _clean(str(s.get("speaker", ""))),
                str(s.get("language", "")),
            )
            for s in data["segments"]
        ]
    except FileNotFoundError:
        raise ExportError(f"no {JSON_NAME} in {folder}") from None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ExportError(f"can't read {path}: {type(e).__name__}") from None
    return data, [s for s in segs if s.text]


_BIDI_ISOLATES = dict.fromkeys(map(ord, "\u2066\u2067\u2068\u2069"))  # summary.md's RLI/PDI
TRANSCRIPT_HEADING = "Transcript"
_MAX_SUMMARY_BYTES = 1_000_000
_BULLET = re.compile(r"[-*+] +")
_NUMBERED = re.compile(r"\d{1,3}[.)] +")


def load_summary(
    folder: Path, transcript_mtime: float | None = None, pin: _Pin | None = None
) -> list[tuple[str, str]]:
    """The blocks of ``summary.md`` as (kind, text): ``h1`` / ``h2`` headings, ``bullet`` items,
    ``p`` paragraphs and ``note`` (the italic "generated by" line). Empty if there is no
    readable summary. Bidi isolates are dropped: the exports set the direction themselves.
    ``transcript_mtime`` is the transcript's mtime when its snapshot was read (default: now).
    It is read from the pinned folder (see ``_pin_folder``): a folder that was a link from the
    start keeps its summary, one swapped since is not read through."""
    try:
        with _pinned(folder, pin) as p:
            # O_NOFOLLOW refuses a linked summary.md atomically; read one byte past the cap
            with os.fdopen(_open_read(p, SUMMARY_NAME), "rb") as f:
                blob = f.read(_MAX_SUMMARY_BYTES + 1)
            if len(blob) > _MAX_SUMMARY_BYTES:
                return []
            raw = blob.decode("utf-8-sig")  # a Notepad BOM must not hide "# Summary"
            stale = _summary_is_stale(p, transcript_mtime)
        if stale:
            log.debug("summary skipped: %s", stale)
            return []
    except ExportError as e:
        log.debug("summary skipped: %s", type(e).__name__)
        return []
    except (OSError, ValueError) as e:
        if not isinstance(e, FileNotFoundError):
            log.debug("summary skipped: %s", type(e).__name__)
        return []
    blocks: list[tuple[str, str]] = []
    for line in raw.splitlines():
        text = _clean(line.translate(_BIDI_ISOLATES))
        if not text or text in ("-", "*", "+", "#", "##", "###"):
            continue
        m = _BULLET.match(text)
        num = None if m else _NUMBERED.match(text)
        if text.startswith("### "):
            blocks.append(("h2", text[4:].strip()))
        elif text.startswith("## "):
            blocks.append(("h2", text[3:].strip()))
        elif text.startswith("# "):
            blocks.append(("h1", text[2:].strip()))
        elif m:
            blocks.append(("bullet", text[m.end() :].strip()))
        elif num:
            body = text[num.end() :].strip()
            blocks.append(("num", f"{text[: num.end()].strip()[:-1]}. {body}" if body else ""))
        elif len(text) > 2 and text[0] == text[-1] == "_":
            blocks.append(("note", text[1:-1]))
        else:
            blocks.append(("p", text))
    return [b for b in blocks if b[1]]


def _summary_is_stale(pin: _Pin, transcript_mtime: float | None = None) -> str:
    """Why the summary must not head the transcript ("" if it may): it is being redone
    (marker present) or made from other text than the transcript has now (re-transcribed).
    A summary with a fingerprint survives speaker renames; an older one uses the mtime rule."""
    with contextlib.suppress(OSError):
        _stat_in(pin, PENDING_MARKER)
        return "pending marker"
    try:
        with os.fdopen(_open_read(pin, SOURCE_NAME), "rb") as f:
            recorded = f.read().decode("utf-8").strip()
        if not recorded:
            raise ValueError("empty fingerprint")
        with os.fdopen(_open_read(pin, JSON_NAME), "rb") as f:
            segs = json.loads(f.read().decode("utf-8")).get("segments", [])
        return "" if transcript_fingerprint(segs) == recorded else "transcript text changed"
    except (OSError, ValueError, AttributeError, TypeError):
        pass  # no (or unreadable) fingerprint: an old session
    try:
        mtime = transcript_mtime
        if mtime is None:
            mtime = _stat_in(pin, JSON_NAME).st_mtime
        return "older than the transcript" if _stat_in(pin, SUMMARY_NAME).st_mtime < mtime else ""
    except OSError:
        return ""


def _majority_rtl(text: str) -> bool:
    """Right to left if strong RTL letters outnumber strong LTR ones (a Hebrew line that
    starts with a Latin name is still Hebrew); ties fall back to the first strong letter."""
    rtl = ltr = 0
    for ch in text:
        kind = unicodedata.bidirectional(ch)
        if kind in ("R", "AL"):
            rtl += 1
        elif kind == "L":
            ltr += 1
    return rtl > ltr if rtl != ltr else is_rtl("", text)


def _title(data: dict) -> str:
    started = str(data.get("started_at", ""))[:16].replace("T", " ")
    return f"Transcript {started}".strip()


def _meta(data: dict, segments: list[Segment]) -> str:
    app = data.get("trigger_app") or "manual"
    langs = ", ".join(languages_of(segments)) or "-"
    speakers = ", ".join(dict.fromkeys(s.speaker for s in segments if s.speaker)) or "-"
    return (
        f"Date: {str(data.get('started_at', ''))[:19].replace('T', ' ')} | Trigger: {app} | "
        f"Duration: {fmt_hms(float(data.get('duration') or 0))} | Languages: {langs} | "
        f"Speakers: {speakers}"
    )


_REPLACE_TRIES, _REPLACE_WAIT_S = 20, 0.1  # ~2 s: Word, a viewer or sync may hold the file
_export_lock = threading.Lock()  # one export at a time in this process


def _create_exclusive(tmp: Path) -> int:
    """Create the temp file (0600, O_EXCL). A stale file of the same name must not block it;
    on Windows a scanner may hold it for a moment, so removal and creation are retried."""
    for attempt in range(_REPLACE_TRIES):
        try:
            tmp.unlink(missing_ok=True)
            return os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except (PermissionError, FileExistsError):
            if attempt + 1 == _REPLACE_TRIES:
                raise
            time.sleep(_REPLACE_WAIT_S)
    raise AssertionError("unreachable")


def _write(path: Path, data: bytes, pin: _Pin | None = None) -> Path:
    """Write ``data`` to a temp file and rename it into place. The temp file is created
    owner-only (0600) and written through that same descriptor (no reopen that a scanner
    could block), with a name unique per process and thread (on Windows the output ACLs are
    inherited from the owner-only session folder, #42). Windows may hold the target open, so
    only the rename is retried. Only exception types are kept: library messages can quote the
    transcript.

    ``pin`` is the session folder pinned when the export started (``_pin_folder``: resolved
    once, with its device and inode captured in the same step). The write is refused if the
    folder path now resolves elsewhere (it, or a parent, was swapped for a link or junction
    while the document was rendered); a folder that was already a link at the start is fine.
    On POSIX the data then goes through the pinned directory descriptor, so no swap after the
    pin can redirect it. On Windows the folder's identity is compared again just before the
    path-based write; a swap in the moment between that check and the write remains possible.
    Without ``pin`` any link or junction is refused."""
    parent = path.parent
    if pin is None:
        if parent.is_symlink() or parent.is_junction():
            raise ExportError("The session folder is a link; not writing through it")
    else:
        try:
            moved = parent.resolve() != pin.path
        except (OSError, RuntimeError):
            moved = True
        if moved:
            raise ExportError(_MOVED)
    tmp_name = f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    if pin is not None and pin.dfd is not None:
        _write_in_dir(pin.dfd, path.name, tmp_name, data)
        return path
    if pin is not None:
        _check_path_pin(pin)
        parent = pin.path
    tmp, target = parent / tmp_name, parent / path.name
    try:
        try:
            fd = _create_exclusive(tmp)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception as e:
            raise ExportError(f"could not write {path.name} ({type(e).__name__})") from None
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, target)
                return path
            except PermissionError:
                if attempt + 1 < _REPLACE_TRIES:
                    time.sleep(_REPLACE_WAIT_S)
            except OSError as e:
                raise ExportError(f"could not write {path.name} ({type(e).__name__})") from None
        raise ExportError(f"{path.name} may be open in another app; close it and try again")
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _write_in_dir(dfd: int, name: str, tmp_name: str, data: bytes) -> None:
    """POSIX: create, write and rename inside the pinned folder through its directory
    descriptor (no path lookups at all), so the bytes land in the folder that was pinned."""
    try:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name, dir_fd=dfd)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW
        fd = os.open(tmp_name, flags, 0o600, dir_fd=dfd)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, name, src_dir_fd=dfd, dst_dir_fd=dfd)
    except OSError as e:
        raise ExportError(f"could not write {name} ({type(e).__name__})") from None
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name, dir_fd=dfd)


STALE_TEMP_AGE_S = 3600.0


def _sweep_stale_temps(pin: _Pin) -> None:
    """Remove export temp files (never the transcript's own ``.transcript.md.tmp`` etc.) that
    a crashed export left behind: ``.transcript.docx.<pid>.<tid>.tmp`` / ``.pdf``, untouched
    for an hour (so another process's export in flight is left alone)."""
    # No cross-process lock is needed: only temps untouched for an hour are removed, and an
    # export in flight (here or in another process) writes its temp within seconds.
    now = time.time()
    prefixes = tuple(f".{n}." for n in (DOCX_NAME, PDF_NAME))
    try:
        names = os.listdir(pin.dfd if pin.dfd is not None else pin.path)
    except OSError:
        return
    for name in names:
        if not (name.startswith(prefixes) and name.endswith(".tmp")):
            continue
        with contextlib.suppress(OSError):
            if now - _stat_in(pin, name).st_mtime > STALE_TEMP_AGE_S:
                if pin.dfd is not None:
                    os.unlink(name, dir_fd=pin.dfd)
                else:
                    (pin.path / name).unlink()


def _type_only(fn):
    """Library failures become ExportError with the exception type only: their messages can
    quote the transcript."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(
                f"{fn.__name__.removeprefix('export_').upper()} failed ({type(e).__name__})"
            ) from None

    return wrapper


_AFTER_BIDI = (  # CT_PPr children that follow w:bidi in the schema sequence
    "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
    "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
    "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
    "w:pPrChange",
)  # fmt: skip


def _set_bidi(paragraph) -> None:
    """Mark the paragraph right to left, at the schema-correct place in w:pPr (after pStyle,
    numPr, ...; before spacing, ind and jc) so Word never sees a misordered element."""
    from docx.oxml import OxmlElement

    ppr = paragraph._p.get_or_add_pPr()
    ppr.insert_element_before(OxmlElement("w:bidi"), *_AFTER_BIDI)


@_type_only
def export_docx(folder: Path, loaded=None, summary=None, *, pin: _Pin | None = None) -> Path:
    """Write ``transcript.docx``. ``pin`` is the folder pinned by ``export_session``; called
    on its own, the folder is pinned once here, before anything is read."""
    with _pinned(folder, pin) as p:
        return _export_docx(folder, p, loaded, summary)


def _export_docx(folder: Path, pin: _Pin, loaded, summary) -> Path:
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.shared import Pt

    data, segments = loaded or load_session(folder, pin)
    doc = Document()
    doc.core_properties.title = _title(data)
    doc.add_heading(_title(data), level=1)
    doc.add_paragraph(_meta(data, segments))
    if summary is None:
        summary = load_summary(folder, pin=pin)
    for kind, text in summary:
        if kind in ("h1", "h2"):
            p = doc.add_heading(text, level=2 if kind == "h1" else 3)
        elif kind in ("bullet", "num"):
            p = doc.add_paragraph(style="List Bullet" if kind == "bullet" else None)
            p.add_run(text)
        else:
            p = doc.add_paragraph()
            p.add_run(text).italic = kind == "note"
        if _majority_rtl(text):
            _set_bidi(p)
            for r in p.runs:
                rpr = r._r.get_or_add_rPr()
                if r.italic:
                    rpr.append(OxmlElement("w:iCs"))  # complex-script italic for RTL runs
                rpr.append(OxmlElement("w:rtl"))
    if summary:
        doc.add_heading(TRANSCRIPT_HEADING, level=2)
    if not segments:
        doc.add_paragraph("No speech detected.")
    for seg in segments:
        label = f"[{fmt_hms(seg.start)}] {seg.speaker}:"
        if rtl_segment(seg):
            # the LTR label gets its own paragraph, or the bidi paragraph would mirror it
            lp = doc.add_paragraph()
            lp.add_run(label).bold = True
            lp.paragraph_format.space_after = Pt(0)
            p = doc.add_paragraph()
            _set_bidi(p)
            p.add_run(seg.text)._r.get_or_add_rPr().append(OxmlElement("w:rtl"))
        else:
            p = doc.add_paragraph()
            p.add_run(label + " ").bold = True
            p.add_run(seg.text)
        p.paragraph_format.space_after = Pt(6)
    buf = io.BytesIO()
    doc.save(buf)
    return _write(Path(folder) / DOCX_NAME, buf.getvalue(), pin)


# Latin (with an accent), Cyrillic (with ё), Hebrew letters, niqqud, gershayim and geresh,
# and common punctuation: a font with only a few glyphs of each script is not enough.
_PROBE = "aZé\u0416\u0451\u044f\u05d0\u05ea\u05b7\u05f3\u05f4—…«»“”"


def _covers_scripts(path: Path) -> bool:
    """True if the font has glyphs for Latin, Cyrillic and Hebrew (fontTools comes with fpdf2)."""
    try:
        from fontTools.ttLib import TTFont

        with TTFont(str(path), lazy=True, fontNumber=0) as f:
            cmap = f.getBestCmap() or {}
        return all(ord(c) in cmap for c in _PROBE)
    except Exception:  # noqa: BLE001 - an unreadable font is just not a candidate
        return False


@functools.lru_cache(maxsize=1)
def find_font() -> Path:
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    for c in (*map(Path, _FONT_CANDIDATES), windir / "Fonts" / "arial.ttf"):
        if c.is_file() and _covers_scripts(c):
            return c
    raise ExportError(
        "no system font with Latin, Cyrillic and Hebrew found for the PDF (install Arial or "
        "DejaVu Sans); DOCX export still works"
    )


@_type_only
def export_pdf(folder: Path, loaded=None, summary=None, *, pin: _Pin | None = None) -> Path:
    """Write ``transcript.pdf``; ``pin`` as for ``export_docx``."""
    with _pinned(folder, pin) as p:
        return _export_pdf(folder, p, loaded, summary)


def _export_pdf(folder: Path, pin: _Pin, loaded, summary) -> Path:
    from fpdf import FPDF
    from fpdf.enums import WrapMode

    data, segments = loaded or load_session(folder, pin)
    font = find_font()
    pdf = FPDF(format="A4")
    pdf.set_title(_title(data))
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 18)
    pdf.add_font("body", "", str(font))
    pdf.set_text_shaping(True)
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin

    def para(text: str, size: int, rtl: bool = False, gap: float = 2) -> None:
        pdf.set_font("body", size=size)
        pdf.set_text_shaping(True, direction="rtl" if rtl else "ltr")
        pdf.set_x(pdf.l_margin)
        long_token = any(len(t) > 40 for t in text.split())
        pdf.multi_cell(
            width,
            size * 0.5,
            text,
            align="R" if rtl else "L",
            wrapmode=WrapMode.CHAR if long_token else WrapMode.WORD,
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(gap)

    para(_title(data), 16, gap=3)
    para(_meta(data, segments), 9, gap=5)
    if summary is None:
        summary = load_summary(folder, pin=pin)
    for kind, text in summary:
        rtl = _majority_rtl(text)
        if kind == "h1":
            para(text, 13, rtl=rtl, gap=2)
        elif kind == "h2" or kind == "num":
            para(text, 11, rtl=rtl, gap=1)
        elif kind == "bullet":
            para(f"\u2022 {text}", 11, rtl=rtl, gap=1)
        else:
            para(text, 9 if kind == "note" else 11, rtl=rtl, gap=3)
    if summary:
        para(TRANSCRIPT_HEADING, 13, gap=3)
    if not segments:
        para("No speech detected.", 11)
    for seg in segments:
        label = f"[{fmt_hms(seg.start)}] {seg.speaker}:"
        if rtl_segment(seg):
            # the LTR label on its own line keeps it intact above the right-to-left body
            para(label, 11, gap=0)
            para(seg.text, 11, rtl=True, gap=3)
        else:
            para(f"{label} {seg.text}", 11, gap=3)
    return _write(Path(folder) / PDF_NAME, bytes(pdf.output()), pin)


def export_session(folder: Path, fmt: str = "both", skipped: list[str] | None = None) -> list[Path]:
    """Write the requested files next to the transcript; returns their paths. Both files are
    built from one read of transcript.json. With ``both``, a PDF that can't be made (no
    font) doesn't lose the DOCX: the reason is appended to ``skipped`` if one is given.

    The folder is pinned once, before anything is read (``_pin_folder``); both reads and both
    writes use that one pin, so a swap of the folder or a parent mid-export can't redirect
    the PDF after the DOCX, nor write somewhere other than the folder that was read."""
    if fmt not in (*FORMATS, "both"):
        raise ExportError(f"unknown format {fmt!r}")
    with _export_lock, _pinned(folder) as pin:
        _sweep_stale_temps(pin)
        try:
            mtime = _stat_in(pin, JSON_NAME).st_mtime
        except OSError:
            mtime = None
        loaded = load_session(folder, pin)
        summary = load_summary(folder, mtime, pin)  # one decision for both files
        out = []
        if fmt in ("docx", "both"):
            out.append(export_docx(folder, loaded, summary, pin=pin))
        if fmt in ("pdf", "both"):
            try:
                out.append(export_pdf(folder, loaded, summary, pin=pin))
            except Exception as e:
                err = (
                    e
                    if isinstance(e, ExportError)
                    else ExportError(f"PDF failed ({type(e).__name__})")
                )
                if fmt == "pdf":
                    raise err from None
                if skipped is not None:
                    skipped.append(str(err))
    log.info("exported session %s (%s)", Path(folder).name, fmt)
    return out
