"""Export a finished session to DOCX and PDF (``transcript.docx`` / ``transcript.pdf``).

Reads ``transcript.json`` (plain text; the bidi isolates are only in md/srt) and writes one
paragraph per segment: ``[hh:mm:ss] Speaker:`` and the text. Hebrew segments are laid out
right to left: DOCX gets a bidi paragraph and an rtl run; the PDF is shaped by HarfBuzz and
reordered with fpdf2's bidi support. The PDF needs a system font that covers Latin, Cyrillic
and Hebrew (Arial, Arial Unicode, DejaVu Sans, ...); without one it raises ExportError. The
libraries are imported lazily, so the app start-up doesn't pay for them. Offline; transcript
text is never logged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from ..models import Segment, is_rtl
from .output import JSON_NAME, fmt_hms, languages_of

log = logging.getLogger(__name__)

DOCX_NAME, PDF_NAME = "transcript.docx", "transcript.pdf"
FORMATS = ("docx", "pdf")
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


class ExportError(Exception):
    """The session can't be exported (message is safe to show: no transcript text)."""


def rtl_segment(seg: Segment) -> bool:
    """Hebrew by language, or (no language) by the first strong character."""
    return is_rtl(seg.language, seg.text)


def load_session(folder: Path) -> tuple[dict, list[Segment]]:
    path = Path(folder) / JSON_NAME
    try:
        data = json.loads(path.read_text("utf-8"))
        segs = [
            Segment(
                float(s["start"]),
                float(s["end"]),
                str(s["text"]).strip(),
                str(s.get("speaker", "")),
                str(s.get("language", "")),
            )
            for s in data["segments"]
        ]
    except FileNotFoundError:
        raise ExportError(f"no {JSON_NAME} in {folder}") from None
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ExportError(f"can't read {path}: {type(e).__name__}") from None
    return data, [s for s in segs if s.text]


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


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _finish(tmp: Path, path: Path) -> Path:
    if sys.platform != "win32":
        os.chmod(tmp, 0o600)  # owner-only, like the other session outputs
    tmp.replace(path)
    return path


def export_docx(folder: Path) -> Path:
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.shared import Pt

    data, segments = load_session(folder)
    doc = Document()
    doc.core_properties.title = _title(data)
    doc.add_heading(_title(data), level=1)
    doc.add_paragraph(_meta(data, segments))
    if not segments:
        doc.add_paragraph("No speech detected.")
    for seg in segments:
        p = doc.add_paragraph()
        label = p.add_run(f"[{fmt_hms(seg.start)}] {seg.speaker}: ")
        label.bold = True
        body = p.add_run(seg.text)
        p.paragraph_format.space_after = Pt(6)
        if rtl_segment(seg):
            p._p.get_or_add_pPr().append(OxmlElement("w:bidi"))
            for run in (label, body):
                run._r.get_or_add_rPr().append(OxmlElement("w:rtl"))
    path = Path(folder) / DOCX_NAME
    tmp = _tmp(path)
    doc.save(str(tmp))
    return _finish(tmp, path)


def find_font() -> Path:
    windir = Path(os.environ.get("WINDIR", "C:/Windows"))
    for c in (*map(Path, _FONT_CANDIDATES), windir / "Fonts" / "arial.ttf"):
        if c.is_file():
            return c
    raise ExportError(
        "no system font with Latin, Cyrillic and Hebrew found for the PDF (install Arial or "
        "DejaVu Sans); DOCX export still works"
    )


def export_pdf(folder: Path) -> Path:
    from fpdf import FPDF

    data, segments = load_session(folder)
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
        pdf.multi_cell(
            width,
            size * 0.5,
            text,
            align="R" if rtl else "L",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(gap)

    para(_title(data), 16, gap=3)
    para(_meta(data, segments), 9, gap=5)
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
    path = Path(folder) / PDF_NAME
    tmp = _tmp(path)
    pdf.output(str(tmp))
    return _finish(tmp, path)


def export_session(folder: Path, fmt: str = "both") -> list[Path]:
    """Write the requested files next to the transcript; returns their paths."""
    if fmt not in (*FORMATS, "both"):
        raise ExportError(f"unknown format {fmt!r}")
    out = []
    if fmt in ("docx", "both"):
        out.append(export_docx(folder))
    if fmt in ("pdf", "both"):
        out.append(export_pdf(folder))
    log.info("exported session %s (%s)", Path(folder).name, fmt)
    return out
