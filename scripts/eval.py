"""Accuracy and speed of the transcription pipeline on a local, consented speech set.

    uv run python scripts/eval.py ~/RecAndCC-eval                  # table + eval-results.json
    uv run python scripts/eval.py ~/RecAndCC-eval --model small --json out.json

The set lives outside git (see docs/EVALUATION.md): ``manifest.json`` lists WAV clips with
reference utterances ``{start, end, speaker, language, text}``. Each clip is transcribed through
the real session pipeline (VAD, language ID, chunked Whisper) on a temporary copy, and its
speaker turns come from the real diarizer. Reported per language and overall:

- WER: word edits / reference words, after normalisation (case, punctuation, ё -> е).
  Hypothesis words are split between the reference utterances their segment overlaps.
- language-ID accuracy: share of utterances whose overlapping speech was tagged with the
  reference language (by overlap).
- DER: missed + false-alarm + confused speaker time / reference speech time, 10 ms frames,
  no collar, best one-to-one speaker mapping.
- speed: transcription real-time factor (total time / total audio) and diarization
  seconds per clip.
- live caption latency (``--live``): the clip is replayed through ``LiveTranscriber`` as
  capture feeds it (100 ms blocks); per emitted segment, seconds from the nominal capture
  time of the block holding its end (when that audio ended) to its emission. p50 / p90 / max
  (linear interpolation).

Runs offline once the models are on disk. Prints scores only, never transcript text.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind

FRAME_S = 0.01
BLOCK = SAMPLE_RATE // 10  # capture delivers ~100 ms blocks


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
    root = root.resolve()
    data = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    clips = []
    for n, c in enumerate(data["clips"]):
        try:
            cid, utts = c["id"], [Utterance(**u) for u in c["utterances"]]
            audio = (root / c["audio"]).resolve()
            utts = [replace(u, language=lang_code(u.language)) for u in utts]
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"clip {c.get('id', n)}: bad manifest entry ({e})") from None
        if not audio.is_relative_to(root):  # e.g. "../Documents/..." or an absolute path
            raise ValueError(f"clip {cid}: audio path is outside the eval folder")
        duration = sf.info(str(audio)).duration
        for u in utts:
            if not (math.isfinite(u.start) and math.isfinite(u.end) and 0 <= u.start < u.end):
                raise ValueError(f"clip {cid}: utterance times must be 0 <= start < end")
            if u.end > duration + 0.05:
                raise ValueError(f"clip {cid}: utterance ends after the audio ({duration:.2f} s)")
        clips.append(Clip(cid, audio, Track(c.get("track", "others")), utts))
    return clips


def lang_code(tag: str) -> str:
    """'ru-RU' / 'RU' -> 'ru', as the transcriber tags segments."""
    return tag.strip().lower().split("-")[0].split("_")[0]


# --------------------------------------------------------------------------------- WER
APOSTROPHES = str.maketrans("\u2019\u02bc\u2018\u00b4\u0060", "'" * 5)
BREVE = "\u0306"  # kept: й is и + breve


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е").translate(APOSTROPHES)
    # drop accents and stress marks (мо́локо -> молоко), but keep й
    text = "".join(
        ch
        for ch in unicodedata.normalize("NFD", text)
        if ch == BREVE or not unicodedata.category(ch).startswith("M")
    )
    text = unicodedata.normalize("NFC", text)
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


def _gap(u: Utterance, s: Segment) -> float:
    return max(u.start - s.end, s.start - u.end, 0.0)


def assign_words(utts: list[Utterance], segs: list[Segment]) -> list[tuple[list[str], dict]]:
    """Per utterance: the hypothesis words that belong to it and its language votes.

    A segment spanning several utterances has its words split between them in time order, in
    proportion to the overlap. A segment overlapping none goes to the nearest utterance (its
    words still count for WER) but doesn't vote on the language. Votes are overlap seconds.
    """
    words: list[list[tuple[float, int, str]]] = [[] for _ in utts]
    votes: list[dict[str, float]] = [defaultdict(float) for _ in utts]
    for s in segs:
        ws = normalize(s.text)
        ov = [_overlap(u.start, u.end, s.start, s.end) for u in utts]
        total = sum(ov)
        if not utts:
            break
        if total == 0.0:
            near = min(range(len(utts)), key=lambda i: _gap(utts[i], s))
            words[near] += [(s.start, k, w) for k, w in enumerate(ws)]
            continue
        cum, lo = 0.0, 0
        for i in sorted((i for i in range(len(utts)) if ov[i] > 0), key=lambda i: utts[i].start):
            cum += ov[i]
            hi = round(len(ws) * cum / total)
            words[i] += [(s.start, k, w) for k, w in enumerate(ws[lo:hi], lo)]
            votes[i][s.language] += ov[i]
            lo = hi
    return [([w for *_, w in sorted(ws)], dict(v)) for ws, v in zip(words, votes, strict=True)]


def text_scores(utts: list[Utterance], segs: list[Segment]) -> dict[str, dict[str, float]]:
    """Per language: reference words, word edits, utterances, utterances with correct language."""
    acc: dict[str, dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(("words", "edits", "utts", "lid_ok"), 0)
    )
    for u, (hyp, votes) in zip(utts, assign_words(utts, segs), strict=True):
        ref = normalize(u.text)
        a = acc[u.language]
        a["words"] += len(ref)
        a["edits"] += word_edits(ref, hyp)
        a["utts"] += 1
        a["lid_ok"] += bool(votes) and max(votes, key=votes.__getitem__) == u.language
    return dict(acc)


# --------------------------------------------------------------------------------- DER
def _frames(turns: list[tuple[float, float, str]], n: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for s, e, spk in turns:
        mask = out.setdefault(spk, np.zeros(n, dtype=bool))
        mask[int(round(s / FRAME_S)) : int(round(e / FRAME_S))] = True
    return out


def _best_mapping(m: list[list[int]]) -> int:
    """Largest total of a one-to-one row/column matching of matrix ``m``.

    Exact (dynamic programming over subsets of the smaller side) up to 12 speakers on that side;
    greedy above, which can only overstate DER.
    """
    if m and len(m) > len(m[0]):
        m = [list(col) for col in zip(*m, strict=True)]
    rows = len(m)
    if rows == 0:
        return 0
    if rows > 12:
        cells = sorted(
            ((v, i, j) for i, row in enumerate(m) for j, v in enumerate(row)), reverse=True
        )
        used_r, used_c, total = set(), set(), 0
        for v, i, j in cells:
            if i not in used_r and j not in used_c:
                used_r.add(i), used_c.add(j)
                total += v
        return total
    dp = {0: 0}  # bitmask of matched rows -> best total
    for j in range(len(m[0])):
        nxt = dict(dp)
        for mask, v in dp.items():
            for i in range(rows):
                if not mask & (1 << i):
                    k = mask | (1 << i)
                    nxt[k] = max(nxt.get(k, 0), v + m[i][j])
        dp = nxt
    return max(dp.values())


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
    best = _best_mapping([[overlap[(a, b)] for b in hk] for a in rk])
    confusion = int(np.minimum(r_count, h_count).sum()) - best
    err = miss + false_alarm + confusion
    return {
        "der": round(err / total, 4) if total else 0.0,
        "miss_s": miss * FRAME_S,
        "false_alarm_s": false_alarm * FRAME_S,
        "confusion_s": round(confusion * FRAME_S, 2),
        "speech_s": round(total * FRAME_S, 2),
        "error_s": round(err * FRAME_S, 2),
    }


# -------------------------------------------------------------------------- live latency
class ReplayClock:
    """Seconds since the replay started, as the live transcriber would see them in real time.

    While a region is being transcribed the replay runs in real time (the model's speed is what
    is measured). While nothing is in flight, waiting changes nothing, so it is skipped.
    """

    def __init__(self) -> None:
        self.t0, self.skipped = time.monotonic(), 0.0

    def __call__(self) -> float:
        return time.monotonic() - self.t0 + self.skipped

    def wait(self, seconds: float, heads: list[Future], pending: list[Future]) -> None:
        """`heads`: each track's oldest uncollected region (results are emitted in order);
        `pending`: all regions still being transcribed."""
        if any(f.done() for f in heads):
            return  # finished during the step: the worker's done-callback wakes it at once
        if pending:  # like the live worker: wake up early when a region is done
            futures_wait(pending, timeout=seconds, return_when=FIRST_COMPLETED)
        else:
            self.skipped += seconds


def arrival(end: float, samples: int) -> float:
    """When the capture block holding time `end` reaches the live transcriber."""
    block = math.ceil(round(end * SAMPLE_RATE) / BLOCK) * BLOCK
    return min(block, samples) / SAMPLE_RATE


def replay_live(
    audio: np.ndarray, track: Track, transcriber, session: SessionInfo, clock=None
) -> list[float]:
    """Replay `audio` through a LiveTranscriber; per emitted segment, its caption latency (s)."""
    from recandcc.transcribe.live import LiveTranscriber, live_vad

    transcriber.ensure_loaded()  # the app loads the models before the session starts
    live_vad(np.zeros(SAMPLE_RATE, dtype=np.float32))  # ... and has the VAD warm
    clock = clock or ReplayClock()
    emitted: list[tuple[float, float]] = []  # (segment end, emitted at)

    def on_segments(segs: list[Segment]) -> None:
        emitted.extend((s.end, clock()) for s in segs)

    # Drives the worker's private step and reads its job queues, as tests/test_live.py does;
    # test_replay_live_measures_caption_latency breaks if those internals change.
    live = LiveTranscriber(
        transcriber, session, on_segments, clock=clock, start=False, md_interval_s=math.inf
    )
    # Only the captions are measured: the batch run already wrote this session's transcript.
    live._start_diarization = live._write_final = lambda: None
    pos, n = 0, len(audio)
    while pos < n:  # one iteration per live worker step, driven here instead of its thread
        now = clock() * SAMPLE_RATE
        upto = n if now >= n else int(now) // BLOCK * BLOCK
        for i in range(pos, upto, BLOCK):
            live.feed(track, i, audio[i : min(i + BLOCK, upto)])
        pos = max(pos, upto)
        if pos >= n:
            break  # recording stops: flush() wakes the worker at once, no idle step first
        live._step(False)
        queues = [[j.future for j in tb.jobs if j.future] for tb in live._tracks.values()]
        heads = [q[0] for q in queues if q]
        clock.wait(live.step_s, heads, [f for q in queues for f in q if not f.done()])
    live._step(True)  # recording stopped: the tail is transcribed and emitted
    return [round(t - arrival(end, n), 3) for end, t in emitted]


def latency_summary(latencies: list[float]) -> dict:
    out: dict = {"segments": len(latencies)}
    if latencies:
        a = np.asarray(latencies, dtype=float)
        for k, q in (("p50_s", 50), ("p90_s", 90), ("max_s", 100)):
            out[k] = round(float(np.percentile(a, q)), 3)
    return out


# ----------------------------------------------------------------------------- pipeline
def run_clip(
    clip: Clip, transcriber, diarizer, root: Path | None = None, live: bool = False
) -> dict:
    """Transcribe (and for "others", diarize) a temporary copy of the clip."""
    with tempfile.TemporaryDirectory(prefix="clip-", dir=root) as tmp:
        session = SessionInfo(
            id=f"eval-{clip.id}",
            directory=Path(tmp),
            started_at=datetime.now(),
            trigger_kind=TriggerKind.MANUAL,
            tracks={clip.track: f"{clip.track}.wav"},
        )
        shutil.copyfile(clip.audio, Path(tmp) / f"{clip.track}.wav")
        session.save()
        result: dict = {"id": clip.id, "track": str(clip.track)}
        if clip.track is Track.OTHERS and diarizer is not None:  # before the transcriber
            t0 = time.monotonic()  # may drop the audio (keep_audio = false)
            turns = diarizer.diarize_file(Path(tmp) / f"{clip.track}.wav")
            result["diarization_s"] = round(time.monotonic() - t0, 2)
            result["diarization_ok"] = turns is not None
            if turns is not None:
                ref = [(u.start, u.end, u.speaker) for u in clip.utterances]
                result["diarization"] = der(ref, [(t.start, t.end, str(t.speaker)) for t in turns])
        segs = transcriber.transcribe_session(session)
        result["text"] = text_scores(clip.utterances, segs)
        m = session.metrics.get("transcription", {})
        result["audio_s"], result["elapsed_s"] = m.get("audio_s"), m.get("elapsed_s")
        if live:
            from recandcc.transcribe.chunking import load_track

            try:  # e.g. every live region failed: keep the other clips' results
                lat = replay_live(load_track(clip.audio), clip.track, transcriber, session)
            except Exception as e:  # noqa: BLE001 - reported as clips_without_live
                print(f"  warning: live replay of clip {clip.id} failed ({type(e).__name__})")
                result["live_ok"] = False
            else:
                result["live_ok"] = True
                result["live"] = {**latency_summary(lat), "latencies_s": lat}
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
    errors = sum(x["error_s"] for x in d)  # false alarms on clips without reference speech count
    live = [x for r in results if "live" in r for x in r["live"]["latencies_s"]]
    timed = [r for r in results if r.get("audio_s") and r.get("elapsed_s") is not None]
    audio = sum(r["audio_s"] for r in timed)
    return {
        "languages": per_lang,
        "der": round(errors / speech, 4) if speech else None,
        "rtf": round(sum(r["elapsed_s"] for r in timed) / audio, 3) if audio else None,
        "live": latency_summary(live) if any("live_ok" in r for r in results) else None,
        "clips": len(results),
        "clips_without_der": sum(1 for r in results if r.get("diarization_ok") is False),
        "clips_without_live": sum(1 for r in results if r.get("live_ok") is False),
    }


def live_line(v: dict) -> str:
    if not v["segments"]:
        return "no segments"
    return f"p50 {v['p50_s']} s, p90 {v['p90_s']} s, max {v['max_s']} s ({v['segments']} segments)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("root", type=Path, help="eval set folder containing manifest.json")
    p.add_argument("--model", help="Whisper model (default: from config)")
    p.add_argument("--no-diarize", action="store_true", help="skip DER")
    p.add_argument("--live", action="store_true", help="also measure live caption latency")
    p.add_argument("--json", type=Path, default=Path("eval-results.json"))
    args = p.parse_args(argv)

    from recandcc.config import load_config
    from recandcc.transcribe import Transcriber
    from recandcc.transcribe.diarize import Diarizer

    cfg = load_config().transcription
    cfg.diarize = False  # DER is measured on the diarizer directly
    cfg.keep_audio = True  # only a temporary copy; the clip itself is never touched
    if args.model:
        cfg.model = args.model
    clips = load_manifest(args.root.expanduser())
    transcriber = Transcriber(cfg)
    diarizer = None if args.no_diarize else Diarizer()
    try:
        if diarizer is not None:  # load the model now, not in the first clip's time
            diarizer.diarize(np.zeros(SAMPLE_RATE, dtype=np.float32))
        # one private (0700, created atomically) folder per run, removed by this run only
        with tempfile.TemporaryDirectory(prefix="rcc-eval-") as root:
            results = [run_clip(c, transcriber, diarizer, Path(root), args.live) for c in clips]
    finally:
        transcriber.close()
        if diarizer is not None:
            diarizer.close()
    report = {"model": cfg.model, "summary": summarize(results), "clips": results}
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    s = report["summary"]
    print(f"model {cfg.model}, {s['clips']} clips, RTF {s['rtf']}, DER {s['der']}")
    if s["clips_without_der"]:
        print(f"  warning: diarization failed on {s['clips_without_der']} clip(s), not in DER")
    for lang, v in s["languages"].items():
        print(f"  {lang:<4} WER {v['wer']}  language-ID {v['lid_accuracy']}  ({v['words']} words)")
    if s["live"] is not None:
        clips = [(r["id"], r["live"]) for r in results if "live" in r]
        for name, v in [("all", s["live"]), *clips]:
            print(f"  live {name}: {live_line(v)}")
        if s["clips_without_live"]:
            print(f"  warning: live replay failed on {s['clips_without_live']} clip(s)")
    print(f"details: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
