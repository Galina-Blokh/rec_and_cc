"""DOCX / PDF export of a session transcript (synthetic transcript.json only).

Stories: US-25.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from docx import Document
from docx.oxml.ns import qn

from recandcc import cli
from recandcc.models import Segment, SessionInfo
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
    # a Hebrew segment: the LTR label has its own paragraph, the text a bidi one
    assert texts[-2:] == ["[00:00:06] Speaker 1:", "שלום לכולם"]
    bidi = [
        p._p.pPr is not None and p._p.pPr.find(qn("w:bidi")) is not None for p in doc.paragraphs
    ]
    assert bidi[-1] and not any(bidi[:-1])
    assert doc.paragraphs[-2].runs[0].bold


def test_docx_bidi_comes_before_spacing_in_schema_order(session):
    ppr = Document(str(export.export_docx(session))).paragraphs[-1]._p.pPr
    tags = [c.tag for c in ppr]
    assert tags.index(qn("w:bidi")) < tags.index(qn("w:spacing"))


def test_control_characters_are_cleaned(tmp_path):
    (tmp_path / "transcript.json").write_text(
        json.dumps(
            {"segments": [{"start": 0, "end": 1, "text": "a\x00b\x0bc", "speaker": "M\x01e"}]}
        ),
        "utf-8",
    )
    _data, segs = export.load_session(tmp_path)
    assert (segs[0].text, segs[0].speaker) == ("a b c", "M e")
    Document(str(export.export_docx(tmp_path)))  # valid XML


def test_temp_file_is_private_from_the_start(session, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX modes")
    modes, real = [], export.os.replace

    def spy(src, dst):
        modes.append(os.stat(src).st_mode & 0o777)
        return real(src, dst)

    monkeypatch.setattr(export.os, "replace", spy)
    export._write(session / "x.bin", b"data")
    assert modes == [0o600] and (session / "x.bin").read_bytes() == b"data"


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_pdf_text_reads_back_in_all_three_languages(session):
    from pypdf import PdfReader

    text = "\n".join(pg.extract_text() for pg in PdfReader(str(export.export_pdf(session))).pages)
    assert "Transcript 2026-10-26 14:03" in text
    assert "[00:00:00] Me: Hello team" in text and "Привет, давайте" in text
    he = "שלום לכולם"
    assert he in text or he[::-1] in text  # logical or visual order, depending on the extractor


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_a_very_long_unbroken_token_does_not_break_the_pdf(tmp_path):
    (tmp_path / "transcript.json").write_text(
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "x" * 400, "speaker": "Me"}]}),
        "utf-8",
    )
    assert export.export_pdf(tmp_path).read_bytes().startswith(b"%PDF")


def test_font_without_hebrew_is_not_picked(monkeypatch, tmp_path):
    f = tmp_path / "a.ttf"
    f.write_bytes(b"not a font")
    export.find_font.cache_clear()
    monkeypatch.setattr(export, "_FONT_CANDIDATES", (str(f),))
    monkeypatch.setenv("WINDIR", str(tmp_path / "none"))
    with pytest.raises(export.ExportError):
        export.find_font()


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
    assert subprocess.run([sys.executable, "-c", code], check=False, timeout=60).returncode == 0


def _controller(output_dir=None):
    import queue
    import threading

    from recandcc.ui.common import Controller

    ctrl = Controller.__new__(Controller)  # only the export wiring, no runtime
    ctrl.events, ctrl._export_lock = queue.Queue(), threading.Lock()
    ctrl.cfg = type("C", (), {"output_dir": output_dir})()
    return ctrl


@pytest.mark.parametrize("fmt", ["docx", "pdf", "both"])
def test_controller_export_posts_a_notice(session, monkeypatch, fmt):
    seen = []

    def fake(folder, f, skipped):
        seen.append(f)
        (folder / "transcript.docx").write_text("x")
        return [folder / "transcript.docx"]

    monkeypatch.setattr(export, "export_session", fake)
    ctrl = _controller(session.parent.parent)
    ctrl.export_async(session / "transcript.md", fmt)
    kind, title, msg = ctrl.events.get(timeout=10)
    assert (kind, title) == ("notice", "Export") and msg.startswith("Saved transcript.docx (")
    assert seen == [fmt]


def test_controller_releases_the_lock_when_the_thread_cannot_start(session, monkeypatch):
    import threading

    def no_start(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_start)
    ctrl = _controller(session.parent.parent)
    with pytest.raises(RuntimeError):
        ctrl.export_async(session / "transcript.md")
    assert ctrl._export_lock.acquire(blocking=False)


def test_controller_export_logs_only_the_error_type(session, monkeypatch, caplog):
    def boom(*_a):
        raise RuntimeError("secret words from the call")

    monkeypatch.setattr(export, "export_session", boom)
    ctrl = _controller(session.parent.parent)
    caplog.set_level("DEBUG")
    ctrl.export_async(session / "transcript.md")
    assert ctrl.events.get(timeout=10)[2].startswith("Export failed")
    assert "RuntimeError" in caplog.text and "secret words" not in caplog.text


def test_controller_ignores_a_second_export_while_one_runs(session):
    ctrl = _controller(session.parent.parent)
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


def test_a_pdf_failure_other_than_the_font_keeps_the_docx(session, monkeypatch):
    def boom(folder, loaded=None):
        raise RuntimeError("Привет, секрет")

    monkeypatch.setattr(export, "export_pdf", boom)
    skipped: list[str] = []
    out = export.export_session(session, "both", skipped)
    assert [p.name for p in out] == ["transcript.docx"]
    assert skipped == ["PDF failed (RuntimeError)"]


def test_replace_is_retried_while_the_file_is_held_open(session, monkeypatch):
    real, calls = export.os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("in use")
        return real(src, dst)

    monkeypatch.setattr(export.os, "replace", flaky)
    monkeypatch.setattr(export.time, "sleep", lambda _s: None)
    export.export_docx(session)
    assert len(calls) == 3
    Document(str(session / "transcript.docx"))
    assert not list(session.glob(".*.tmp"))


def test_replace_that_never_succeeds_is_an_export_error(session, monkeypatch):
    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(export.os, "replace", locked)
    monkeypatch.setattr(export.time, "sleep", lambda _s: None)
    with pytest.raises(export.ExportError):
        export.export_docx(session)
    assert not list(session.glob(".*.tmp"))


def test_latest_finished_session_skips_live_ones_and_ignores_mtime(tmp_path):
    from recandcc.ui.common import latest_finished_session

    def sess(day, name, done):
        d = tmp_path / day / name
        d.mkdir(parents=True)
        (d / "transcript.md").write_text("x")
        if done:
            (d / "transcript.json").write_text("{}")
        return d / "transcript.md"

    assert latest_finished_session(tmp_path) is None
    for i in range(12):  # many newer live sessions
        sess("2026-10-27", f"{i:02d}0000_zoom", False)
    new = sess("2026-10-26", "140312_zoom", True)
    old = sess("2026-10-25", "100000_meet", True)
    os.utime(old, (9e9, 9e9))  # re-transcribed just now: newest mtime, oldest session
    assert latest_finished_session(tmp_path) == new
    (new.parent / SessionInfo.DISCARD_MARKER).write_text("")  # discarded: the next one counts
    assert latest_finished_session(tmp_path) == old
    assert latest_finished_session(tmp_path / "missing") is None


def test_controller_export_of_nothing_finished_posts_a_notice(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl.export_async()
    assert ctrl.events.get(timeout=10) == ("notice", "Export", "No finished transcripts yet")


def test_a_held_open_file_gets_a_clear_notice(session, monkeypatch):
    def locked(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(export.os, "replace", locked)
    monkeypatch.setattr(export.time, "sleep", lambda _s: None)
    with pytest.raises(export.ExportError, match="open in another app"):
        export.export_docx(session)


def test_export_refuses_a_session_linked_from_outside_the_folder(tmp_path):
    if sys.platform == "win32":
        pytest.skip("symlinks need privileges")
    out, elsewhere = tmp_path / "out", tmp_path / "elsewhere"
    (out / "2099-12-31").mkdir(parents=True)
    elsewhere.mkdir()
    (out / "2099-12-31" / "000000_x").symlink_to(elsewhere)
    ctrl = _controller(out)
    ctrl.export_async(out / "2099-12-31" / "000000_x" / "transcript.md")
    assert "outside" in ctrl.events.get(timeout=10)[2]
    assert not list(elsewhere.iterdir())


def test_only_old_export_temps_are_swept(session):
    old = session / ".transcript.docx.99999.1.tmp"
    fresh = session / ".transcript.pdf.99999.1.tmp"  # another process, in flight
    others = [session / f".transcript.{e}.tmp" for e in ("md", "json", "srt")]
    for f in (old, fresh, *others):
        f.write_text("x")
    os.utime(old, (1, 1))
    export.export_session(session, "docx")
    assert not old.exists() and fresh.exists() and all(f.exists() for f in others)


def test_a_write_error_is_not_reported_as_a_file_in_use(session, monkeypatch):
    real = export.os.open

    def denied(path, *a, **kw):
        if str(path).endswith(".tmp"):
            raise PermissionError("read-only folder")
        return real(path, *a, **kw)

    monkeypatch.setattr(export.os, "open", denied)
    monkeypatch.setattr(export.time, "sleep", lambda _s: None)
    with pytest.raises(export.ExportError, match=r"could not write.*PermissionError"):
        export.export_docx(session)


def test_a_failing_cleanup_does_not_hide_the_export_error(session, monkeypatch):
    from pathlib import Path

    def locked(src, dst):
        raise PermissionError("in use")

    calls = []

    def no_unlink(self, missing_ok=False):
        calls.append(1)
        if len(calls) > 1:  # the pre-create removal works; the final cleanup is blocked
            raise PermissionError("held by antivirus")

    monkeypatch.setattr(export.os, "replace", locked)
    monkeypatch.setattr(export.time, "sleep", lambda _s: None)
    monkeypatch.setattr(Path, "unlink", no_unlink)
    with pytest.raises(export.ExportError, match="open in another app"):
        export.export_docx(session)


def test_a_long_meeting_exports(tmp_path):
    segs = [
        {
            "start": i,
            "end": i + 1,
            "text": f"line {i} שלום привет",
            "speaker": "Me",
            "language": "he",
        }
        for i in range(300)
    ]
    (tmp_path / "transcript.json").write_text(
        json.dumps({"segments": segs}, ensure_ascii=False), "utf-8"
    )
    assert len(Document(str(export.export_docx(tmp_path))).paragraphs) > 600
    if have_font():
        assert export.export_pdf(tmp_path).read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(not have_font(), reason="no Latin+Cyrillic+Hebrew system font")
def test_a_wrapped_rtl_paragraph_keeps_its_line_order():
    from fpdf import FPDF

    words = [f"מילה{chr(0x5D0 + i % 26)}{chr(0x5D0 + i // 26)}" for i in range(60)]
    pdf = FPDF(format="A4")
    pdf.add_font("body", "", str(export.find_font()))
    pdf.set_text_shaping(True, direction="rtl")
    pdf.add_page()
    pdf.set_font("body", size=11)
    lines = pdf.multi_cell(150, 6, " ".join(words), align="R", dry_run=True, output="LINES")
    assert len(lines) > 1  # it wrapped
    assert " ".join(lines).split() == words  # logical order kept across the wrap points


def test_a_session_folder_swapped_for_a_link_is_not_written_through(tmp_path):
    if sys.platform == "win32":
        pytest.skip("symlinks need privileges")
    real, link = tmp_path / "real", tmp_path / "link"
    real.mkdir()
    link.symlink_to(real)
    with pytest.raises(export.ExportError, match="link"):
        export._write(link / "x.bin", b"data")
    assert not list(real.iterdir())
