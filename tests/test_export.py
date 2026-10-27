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


def _controller():
    import queue
    import threading

    from recandcc.ui.common import Controller

    ctrl = Controller.__new__(Controller)  # only the export wiring, no runtime
    ctrl.events, ctrl._export_lock = queue.Queue(), threading.Lock()
    return ctrl


def test_controller_export_posts_a_notice(session):
    ctrl = _controller()
    ctrl.export_async(session / "transcript.md")
    kind, title, msg = ctrl.events.get(timeout=10)
    assert (kind, title) == ("notice", "Export") and msg.startswith("Saved transcript.docx")
    assert (session / "transcript.docx").exists()


def test_controller_export_logs_only_the_error_type(session, monkeypatch, caplog):
    def boom(*_a):
        raise RuntimeError("secret words from the call")

    monkeypatch.setattr(export, "export_session", boom)
    ctrl = _controller()
    caplog.set_level("DEBUG")
    ctrl.export_async(session / "transcript.md")
    assert ctrl.events.get(timeout=10)[2].startswith("Export failed")
    assert "RuntimeError" in caplog.text and "secret words" not in caplog.text


def test_controller_ignores_a_second_export_while_one_runs(session):
    ctrl = _controller()
    ctrl._export_lock.acquire()
    ctrl.export_async(session / "transcript.md")
    assert ctrl.events.empty()


def test_library_errors_become_export_errors_without_text(session, monkeypatch):
    import docx

    def boom(self, _path):
        raise ValueError("Привет, секрет")

    monkeypatch.setattr(docx.document.Document, "save", boom)
    with pytest.raises(export.ExportError) as e:
        export.export_docx(session)
    assert "ValueError" in str(e.value) and "секрет" not in str(e.value)
    assert not list(session.glob(".*.tmp"))


def test_missing_font_keeps_the_docx_when_both(session, monkeypatch):
    def nofont():
        raise export.ExportError("no system font")

    monkeypatch.setattr(export, "find_font", nofont)
    skipped: list[str] = []
    out = export.export_session(session, "both", skipped)
    assert [p.name for p in out] == ["transcript.docx"] and skipped == ["no system font"]
    with pytest.raises(export.ExportError):
        export.export_session(session, "pdf")


def test_both_formats_share_one_snapshot(session, monkeypatch):
    calls = []
    real = export.load_session
    monkeypatch.setattr(export, "load_session", lambda f: calls.append(f) or real(f))
    export.export_session(session, "both")
    assert len(calls) == 1


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_pdf_lays_out_each_segment_with_its_direction(session, monkeypatch):
    from fpdf import FPDF

    seen = []
    real = FPDF.multi_cell

    def spy(self, w, h=None, text="", *a, **kw):
        seen.append((text, kw.get("align")))
        return real(self, w, h, text, *a, **kw)

    monkeypatch.setattr(FPDF, "multi_cell", spy)
    export.export_pdf(session)
    assert ("[00:00:00] Me: Hello team", "L") in seen
    assert ("[00:00:03] Others: Привет, давайте", "L") in seen
    assert ("[00:00:06] Speaker 1:", "L") in seen and ("שלום לכולם", "R") in seen


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_pdf_file_has_title_and_one_page(session):
    import re

    raw = export.export_pdf(session).read_bytes()
    assert b"Transcript 2026-10-26 14:03" in raw  # the document title (metadata)
    assert len(re.findall(rb"/Type /Page\b", raw)) == 1


def test_concurrent_exports_of_one_session_are_serialised(session):
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda _: export.export_session(session, "docx"), range(4)))
    assert all(r == [session / "transcript.docx"] for r in results)
    Document(str(session / "transcript.docx"))  # a whole, readable file
    assert not list(session.glob(".*.tmp"))
