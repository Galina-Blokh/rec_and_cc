"""Writing transcript.md / transcript.json / transcript.srt."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..models import Segment, SessionInfo

PARAGRAPH_GAP_S = 2.0
MD_NAME, JSON_NAME, SRT_NAME = "transcript.md", "transcript.json", "transcript.srt"


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


def paragraphs(segments: list[Segment], gap_s: float = PARAGRAPH_GAP_S) -> list[Segment]:
    """Join consecutive same-speaker segments separated by less than `gap_s`."""
    out: list[Segment] = []
    for s in segments:
        p = out[-1] if out else None
        if p and p.speaker == s.speaker and s.start - p.end < gap_s:
            out[-1] = Segment(
                p.start, max(p.end, s.end), f"{p.text} {s.text}".strip(), p.speaker, p.language
            )
        else:
            out.append(Segment(s.start, s.end, s.text.strip(), s.speaker, s.language))
    return out


def format_markdown(session: SessionInfo, segments: list[Segment], duration: float) -> str:
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
    if not segments:
        lines.append("_No speech detected._")
    for p in paragraphs(segments):
        lines += [f"**[{fmt_hms(p.start)}] {p.speaker}:** {p.text}", ""]
    return "\n".join(lines).rstrip() + "\n"


def format_srt(segments: list[Segment]) -> str:
    blocks = [
        f"{i}\n{fmt_srt_time(s.start)} --> {fmt_srt_time(s.end)}\n{s.speaker}: {s.text.strip()}\n"
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
        "segments": [asdict(s) for s in segments],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


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
    paths = []
    for name, text in contents.items():
        tmp = d / f".{name}.tmp"
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(d / name)
        paths.append(d / name)
    return paths
