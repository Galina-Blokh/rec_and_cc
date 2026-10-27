"""Writing transcript.md / transcript.json / transcript.srt."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import fields
from pathlib import Path

from ..models import Segment, SessionInfo, is_rtl

PARAGRAPH_GAP_S = 2.0
MD_NAME, JSON_NAME, SRT_NAME = "transcript.md", "transcript.json", "transcript.srt"
# Unicode bidi controls for right-to-left (Hebrew) segments. transcript.json keeps the raw text.
RLI, PDI, RLM = "\u2067", "\u2069", "\u200f"


def fmt_hms(t: float) -> str:
    """00:01:23"""
    t = max(0, int(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def fmt_srt_time(t: float) -> str:
    """00:01:23,456"""
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def languages_of(segments: list[Segment]) -> list[str]:
    """Languages in order of first appearance."""
    return list(dict.fromkeys(s.language for s in segments if s.language))


def md_text(seg: Segment) -> str:
    """Markdown body of a segment: right-to-left text goes in a right-to-left isolate (RLI ..
    PDI), so the LTR "[time] Speaker:" prefix stays put and the line's punctuation and
    embedded English words land on the correct side."""
    text = seg.text.strip()
    return f"{RLI}{text}{PDI}" if text and is_rtl(seg.language, text) else text


def srt_text(seg: Segment) -> str:
    """SRT line of a segment. A right-to-left one starts with RLM, so players lay the whole
    subtitle out right to left (label on the right, trailing punctuation on the left); RLM is
    understood by older players that don't know isolates."""
    line = f"{seg.speaker}: {seg.text.strip()}"
    return RLM + line if is_rtl(seg.language, seg.text) else line


def paragraphs(segments: list[Segment], gap_s: float = PARAGRAPH_GAP_S) -> list[Segment]:
    """Join consecutive same-speaker segments separated by less than `gap_s` (but not across a
    change of writing direction, so each paragraph reads one way). A paragraph is `overlap`
    if any of its segments is."""
    out: list[Segment] = []
    for s in segments:
        p = out[-1] if out else None
        if (
            p
            and p.speaker == s.speaker
            and s.start - p.end < gap_s
            and is_rtl(p.language, p.text) == is_rtl(s.language, s.text)
        ):
            text = f"{p.text} {s.text}".strip()
            out[-1] = Segment(
                p.start,
                max(p.end, s.end),
                text,
                p.speaker,
                p.language,
                p.overlap or s.overlap,
                p.track,
            )
        else:
            out.append(
                Segment(s.start, s.end, s.text.strip(), s.speaker, s.language, s.overlap, s.track)
            )
    return out


def format_markdown(
    session: SessionInfo, segments: list[Segment], duration: float, live: bool = False
) -> str:
    """``live``: the session is still recording (partial transcript, rewritten as it grows)."""
    started = session.started_at
    trigger = session.trigger_app or "manual"
    lines = [
        f"# Transcript {started:%Y-%m-%d %H:%M}",
        "",
        f"- **Date:** {started:%Y-%m-%d %H:%M:%S}",
        f"- **Trigger:** {trigger} ({session.trigger_kind})",
        f"- **Duration:** {fmt_hms(duration)}",
        f"- **Languages:** {', '.join(languages_of(segments)) or '-'}",
        "",
    ]
    if live:
        lines += ["_Live transcript — recording in progress, updated every few seconds._", ""]
    if not segments and not live:
        lines.append("_No speech detected._")
    for p in paragraphs(segments):
        lines += [f"**[{fmt_hms(p.start)}] {p.speaker}:** {md_text(p)}", ""]
    return "\n".join(lines).rstrip() + "\n"


def format_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{fmt_srt_time(s.start)} --> {fmt_srt_time(s.end)}\n{srt_text(s)}\n"
        for i, s in enumerate(segments, 1)
    ]
    return "\n".join(blocks)


def format_json(session: SessionInfo, segments: list[Segment], duration: float, model: str) -> str:
    data = {
        "session_id": session.id,
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat() if session.ended_at else None,
        "trigger_kind": str(session.trigger_kind),
        "trigger_app": session.trigger_app,
        "duration": round(duration, 3),
        "languages": languages_of(segments),
        "model": model,
        "segments": [_segment_json(s) for s in segments],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def _segment_json(s: Segment) -> dict:
    """Segment fields for JSON: never the in-memory word timings (no deep copy of them), and
    `overlap` only where diarization found overlapped speech, `track` only where it is known."""
    data = {f.name: getattr(s, f.name) for f in fields(s) if f.name != "words"}
    if not s.overlap:
        del data["overlap"]
    if not s.track:
        del data["track"]
    return data


def write_outputs(
    session: SessionInfo, segments: list[Segment], duration: float, model: str
) -> list[Path]:
    """Write the three transcript files atomically (tmp + rename); returns their paths."""
    d = Path(session.directory)
    contents = {
        MD_NAME: format_markdown(session, segments, duration),
        JSON_NAME: format_json(session, segments, duration, model),
        SRT_NAME: format_srt(segments),
    }
    return [atomic_write(d / name, text) for name, text in contents.items()]


def write_live_markdown(session: SessionInfo, segments: list[Segment], duration: float) -> Path:
    """Rewrite transcript.md of a session that is still recording (atomically)."""
    return atomic_write(
        Path(session.directory) / MD_NAME,
        format_markdown(session, segments, duration, live=True),
        tries=1,  # on the live worker: never sleep, the next update rewrites it
    )


def atomic_write(path: Path, text: str, tries: int = 5) -> Path:
    """Write via a temp file + rename, so readers never see a half-written file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(tries):
            try:
                tmp.replace(path)
                break
            except PermissionError:  # Windows: a reader or antivirus holds the file
                if attempt + 1 == tries:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        with contextlib.suppress(OSError):  # a scanner may still hold it; the real error wins
            tmp.unlink(missing_ok=True)
    return path
