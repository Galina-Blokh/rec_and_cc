"""DOCX / PDF export of a session transcript (synthetic transcript.json only).

Stories: US-25.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from docx import Document
from docx.oxml.ns import qn

from recandcc import cli
from recandcc.models import Segment
from recandcc.transcribe import export

SEGS = [
    {"start": 0.0, "end": 2.0, "text": "Hello team", "speaker": "Me", "language": "en"},
    {"start": 3.0, "end": 5.0, "text": "Привет, давайте", "speaker": "Others", "language": "ru"},
    {"start": 6.0, "end": 8.0, "text": "שלום לכולם", "speaker": "Speaker 1", "language": "he"},
]


@pytest.fixture
def session(tmp_path):
    d = tmp_path / "2026-10-26" / "140312_zoom-us"
    d.mkdir(parents=True)
    data = {
        "session_id": "x",
        "started_at": "2026-10-26T14:03:12",
        "trigger_app": "zoom-us",
        "duration": 9.0,
        "segments": SEGS,
    }
    (d / "transcript.json").write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    return d


def have_font() -> bool:
    try:
        export.find_font()
    except export.ExportError:
        return False
    return True


def test_docx_has_text_speakers_timestamps_and_rtl(session):
    path = export.export_docx(session)
    assert path == session / "transcript.docx"
    doc = Document(str(path))
    texts = [p.text for p in doc.paragraphs]
    assert "Transcript 2026-10-26 14:03" in texts[0]
    assert "[00:00:00] Me: Hello team" in texts
    assert "[00:00:03] Others: Привет, давайте" in texts
    assert "[00:00:06] Speaker 1: שלום לכולם" in texts
    bidi = [
        p._p.pPr is not None and p._p.pPr.find(qn("w:bidi")) is not None for p in doc.paragraphs
    ]
    assert bidi[-1] and not any(bidi[:-1])


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_pdf_is_written(session):
    path = export.export_pdf(session)
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF") and b"%%EOF" in raw[-32:]


def test_export_session_formats_and_private_mode(session):
    out = export.export_session(session, "docx")
    assert [p.name for p in out] == ["transcript.docx"]
    if sys.platform != "win32":
        assert (session / "transcript.docx").stat().st_mode & 0o077 == 0
    assert not list(session.glob(".*.tmp"))
    with pytest.raises(export.ExportError):
        export.export_session(session, "odt")


def test_missing_or_broken_transcript(tmp_path):
    with pytest.raises(export.ExportError):
        export.export_docx(tmp_path)
    (tmp_path / "transcript.json").write_text("{nope", "utf-8")
    with pytest.raises(export.ExportError):
        export.export_docx(tmp_path)


def test_rtl_detection_by_text_when_language_unknown():
    assert export.rtl_segment(Segment(0, 1, "שלום", "Me", ""))
    assert not export.rtl_segment(Segment(0, 1, "hello", "Me", ""))


def test_cli_export(session, capsys):
    assert cli.main(["export", str(session), "--format", "docx"]) == 0
    assert (session / "transcript.docx").exists()
    assert str(session / "transcript.docx") in capsys.readouterr().out
    assert cli.main(["export", str(session / "nope"), "--format", "docx"]) == 1


def test_cli_rejects_unknown_format(session):
    with pytest.raises(SystemExit):
        cli.main(["export", str(session), "--format", "odt"])


def test_importing_the_app_does_not_load_export_libraries():
    code = (
        "import sys, recandcc, recandcc.cli, recandcc.transcribe.export;"
        "sys.exit(any(m in sys.modules for m in ('docx', 'fpdf')))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
