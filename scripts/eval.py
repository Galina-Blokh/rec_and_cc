"""Accuracy and speed of the transcription pipeline on a local, consented speech set.

    uv run python scripts/eval.py ~/RecAndCC-eval                  # table + eval-results.json
    uv run python scripts/eval.py ~/RecAndCC-eval --model small --json out.json

The set lives outside git (see docs/EVALUATION.md): ``manifest.json`` lists WAV clips with
reference utterances ``{start, end, speaker, language, text}``. Each clip is transcribed through
the real session pipeline (VAD, language ID, chunked Whisper) on a temporary copy, and its
speaker turns come from the real diarizer. Reported per language and overall:

- WER: word edits / reference words, after normalisation (case, punctuation, ё -> е).
  Hypothesis segments are matched to the reference utterance they overlap most.
- language-ID accuracy: share of utterances whose overlapping speech was tagged with the
  reference language (by duration).
- DER: missed + false-alarm + confused speaker time / reference speech time, 10 ms frames,
  no collar, best one-to-one speaker mapping.
- speed: transcription real-time factor and diarization seconds per clip.

Runs offline once the models are on disk. Prints scores only, never transcript text.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from recandcc.models import Segment, SessionInfo, Track, TriggerKind

FRAME_S = 0.01


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    speaker: str
    language: str
    text: str


@dataclass(frozen=True)
class Clip:
    id: str
    audio: Path
    track: Track
    utterances: list[Utterance]


def load_manifest(root: Path) -> list[Clip]:
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for c in data["clips"]:
        utts = [Utterance(**u) for u in c["utterances"]]
        if any(u.end <= u.start for u in utts):
            raise ValueError(f"clip {c['id']}: utterance with end <= start")
        clips.append(Clip(c["id"], root / c["audio"], Track(c.get("track", "others")), utts))
    return clips


# --------------------------------------------------------------------------------- WER
def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = "".join(ch if ch.isalnum() or ch in "'-" or ch.isspace() else " " for ch in text)
    return [w.strip("'-") for w in text.split() if w.strip("'-")]


def word_edits(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein distance over words (substitutions + deletions + insertions)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def match_segments(utts: list[Utterance], segs: list[Segment]) -> list[list[Segment]]:
    """Each hypothesis segment goes to the utterance it overlaps most (unmatched -> nearest)."""
    out: list[list[Segment]] = [[] for _ in utts]
    if not utts:
        return out
    for s in segs:
        scores = [_overlap(u.start, u.end, s.start, s.end) for u in utts]
        best = max(range(len(utts)), key=scores.__getitem__)
        if scores[best] == 0.0:
            mid = (s.start + s.end) / 2
            best = min(range(len(utts)), key=lambda i: abs((utts[i].start + utts[i].end) / 2 - mid))
        out[best].append(s)
    return out


def text_scores(utts: list[Utterance], segs: list[Segment]) -> dict[str, dict[str, float]]:
    """Per language: reference words, word edits, utterances, utterances with correct language."""
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(("words", "edits", "utts", "lid_ok"), 0)
    )
    for u, matched in zip(utts, match_segments(utts, segs), strict=True):
        ref = normalize(u.text)
        hyp = [w for s in sorted(matched, key=lambda s: s.start) for w in normalize(s.text)]
        a = acc[u.language]
        a["words"] += len(ref)
        a["edits"] += word_edits(ref, hyp)
        a["utts"] += 1
        by_lang: dict[str, float] = defaultdict(float)
        for s in matched:
            by_lang[s.language] += max(s.end - s.start, 1e-3)
        a["lid_ok"] += bool(by_lang) and max(by_lang, key=by_lang.__getitem__) == u.language
    return dict(acc)


# --------------------------------------------------------------------------------- DER
def _frames(turns: list[tuple[float, float, str]], n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s, e, spk in turns:
        mask = out.setdefault(spk, np.zeros(n, dtype=bool))
        mask[int(round(s / FRAME_S)) : int(round(e / FRAME_S))] = True
    return out


def der(ref: list[tuple[float, float, str]], hyp: list[tuple[float, float, str]]) -> dict:
    """Frame-level diarization error rate with the best one-to-one speaker mapping."""
    n = int(round(max((e for _, e, _ in ref + hyp), default=0.0) / FRAME_S)) + 1
    r, h = _frames(ref, n), _frames(hyp, n)
    r_count = sum((m.astype(int) for m in r.values()), np.zeros(n, dtype=int))
    h_count = sum((m.astype(int) for m in h.values()), np.zeros(n, dtype=int))
    total = int(r_count.sum())
    miss = int(np.maximum(r_count - h_count, 0).sum())
    false_alarm = int(np.maximum(h_count - r_count, 0).sum())
    rk, hk = list(r), list(h)
    overlap = {(a, b): int((r[a] & h[b]).sum()) for a in rk for b in hk}
    small, large = (rk, hk) if len(rk) <= len(hk) else (hk, rk)

    def score(pairs) -> int:
        return sum(overlap[(a, b) if small is rk else (b, a)] for a, b in pairs)

    if len(small) <= 6:
        best = max(
            (score(zip(small, p, strict=False)) for p in itertools.permutations(large, len(small))),
            default=0,
        )
    else:  # greedy: an upper bound on DER for many speakers
        used, best = set(), 0
        for (a, b), v in sorted(overlap.items(), key=lambda kv: -kv[1]):
            if a not in used and b not in used:
                used |= {a, b}
                best += v
    confusion = int(np.minimum(r_count, h_count).sum()) - best
    err = miss + false_alarm + confusion
    return {
        "der": round(err / total, 4) if total else 0.0,
        "miss_s": miss * FRAME_S,
        "false_alarm_s": false_alarm * FRAME_S,
        "confusion_s": round(confusion * FRAME_S, 2),
        "speech_s": round(total * FRAME_S, 2),
    }


# ----------------------------------------------------------------------------- pipeline
def run_clip(clip: Clip, transcriber, diarizer) -> dict:
    """Transcribe (and for "others", diarize) a temporary copy of the clip."""
    with tempfile.TemporaryDirectory(prefix="rcc-eval-") as tmp:
        session = SessionInfo(
            id=f"eval-{clip.id}",
            directory=Path(tmp),
            started_at=datetime.now(),
            trigger_kind=TriggerKind.MANUAL,
            tracks={clip.track: f"{clip.track}.wav"},
        )
        shutil.copyfile(clip.audio, Path(tmp) / f"{clip.track}.wav")
        session.save()
        segs = transcriber.transcribe_session(session)
        result = {"id": clip.id, "track": str(clip.track)}
        result.update(text=text_scores(clip.utterances, segs))
        result["rtf"] = session.metrics.get("transcription", {}).get("rtf")
        if clip.track is Track.OTHERS and diarizer is not None:
            t0 = time.monotonic()
            turns = diarizer.diarize_file(Path(tmp) / f"{clip.track}.wav")
            result["diarization_s"] = round(time.monotonic() - t0, 2)
            if turns is not None:
                ref = [(u.start, u.end, u.speaker) for u in clip.utterances]
                result["diarization"] = der(ref, [(t.start, t.end, str(t.speaker)) for t in turns])
    return result


def summarize(results: list[dict]) -> dict:
    langs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in results:
        for lang, a in r["text"].items():
            for k, v in a.items():
                langs[lang][k] += v
                langs["all"][k] += v
    per_lang = {
        lang: {
            "wer": round(a["edits"] / a["words"], 4) if a["words"] else None,
            "lid_accuracy": round(a["lid_ok"] / a["utts"], 4) if a["utts"] else None,
            "words": int(a["words"]),
            "utterances": int(a["utts"]),
        }
        for lang, a in sorted(langs.items())
    }
    d = [r["diarization"] for r in results if "diarization" in r]
    speech = sum(x["speech_s"] for x in d)
    errors = sum(x["der"] * x["speech_s"] for x in d)
    rtfs = [r["rtf"] for r in results if r.get("rtf") is not None]
    return {
        "languages": per_lang,
        "der": round(errors / speech, 4) if speech else None,
        "rtf_mean": round(sum(rtfs) / len(rtfs), 3) if rtfs else None,
        "clips": len(results),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("root", type=Path, help="eval set folder containing manifest.json")
    p.add_argument("--model", help="Whisper model (default: from config)")
    p.add_argument("--no-diarize", action="store_true", help="skip DER")
    p.add_argument("--json", type=Path, default=Path("eval-results.json"))
    args = p.parse_args(argv)

    from recandcc.config import load_config
    from recandcc.transcribe import Transcriber
    from recandcc.transcribe.diarize import Diarizer

    cfg = load_config().transcription
    cfg.diarize = False  # DER is measured on the diarizer directly
    if args.model:
        cfg.model = args.model
    clips = load_manifest(args.root.expanduser())
    transcriber = Transcriber(cfg)
    diarizer = None if args.no_diarize else Diarizer()
    try:
        results = [run_clip(c, transcriber, diarizer) for c in clips]
    finally:
        transcriber.close()
        if diarizer is not None:
            diarizer.close()
    report = {"model": cfg.model, "summary": summarize(results), "clips": results}
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"model {cfg.model}, {s['clips']} clips, mean RTF {s['rtf_mean']}, DER {s['der']}")
    for lang, v in s["languages"].items():
        print(f"  {lang:<4} WER {v['wer']}  language-ID {v['lid_accuracy']}  ({v['words']} words)")
    print(f"details: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
