"""Voice-profile recall on synthetic sessions: is a person named once recognized later?

    uv run python scripts/voice_recall_eval.py

Issue #1: "A person named once is recognized by name in later sessions (>= 90% of their
segments on a test set)". Four macOS `say` voices (three English, one Russian) talk in
session 1; each is named once with ``voices.name_speaker`` (what ``recandcc name-speaker``
runs). Three later sessions use other sentences, other speaker mixes and orders, and two
voices that were never named. Each session's "Others" track goes through the real Diarizer
and the transcriber's labelling (``match_profiles`` against the stored profiles, as after a
call). Printed per speaker: the share of their later segments (and seconds) labelled with
their name; for the unnamed voices, the share wrongly given a name.

Synthetic and approximate: studio TTS is far cleaner than a call, and the segments are the
true sentences (one per turn) instead of Whisper's, so this overstates real-world recall.
Everything is written to a temp dir (audio, transcripts, profiles); the user's real voice
profiles are never read. Needs macOS, the voices below and the cached speaker model: nothing
is downloaded.

Stories: US-21.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "rcc_synth_voices", ROOT / "make_synthetic_evalset.py"
)
mk = importlib.util.module_from_spec(_spec)  # pyright: ignore[reportArgumentType]
sys.modules["rcc_synth_voices"] = mk
_spec.loader.exec_module(mk)  # pyright: ignore[reportOptionalMemberAccess]

# speaker id (make_synthetic_evalset.VOICES) -> the name given once in session 1
ENROLLED = {"en1": "Anna", "ru": "Boris", "en4": "Carl", "en7": "Dev"}
UNENROLLED = ("en5", "en6")  # Karen, Moira: never named
LETTERS = {"A": "en1", "B": "ru", "C": "en4", "D": "en7", "X": "en5", "Y": "en6"}
ENROLL_ORDER = "ABCDBADCCABD"  # session 1: three turns each
LATER = {"s2": "CAXBCXABXC", "s3": "DBYDXBYDXB", "s4": "ADYCDAYCAD"}  # each voice in two
EXTRA_RU = [
    "Я посмотрю документ сегодня вечером и напишу свои замечания.",
    "Нам нужно ещё немного времени, чтобы закончить проверку.",
    "Давайте перенесём демонстрацию на четверг.",
    "Клиенты в целом довольны новой версией.",
    "Спасибо всем, до встречи на следующей неделе.",
]


def sentences() -> dict[str, list[str]]:
    """Distinct en and ru sentences (four words or more), never repeated across sessions."""
    en = [t for c in mk.CLIPS.values() for s, t in c if s.startswith("en")]
    ru = [t for c in mk.CLIPS.values() for s, t in c if s == "ru"] + EXTRA_RU
    return {
        "en": [t for t in dict.fromkeys(mk.POOL + en) if len(t.split()) >= 4],
        "ru": list(dict.fromkeys(ru)),
    }


def plan() -> dict[str, list[tuple[str, str]]]:
    """Session id -> [(speaker id, text)], the enrollment session first."""
    pools = {lang: iter(texts) for lang, texts in sentences().items()}
    return {
        sid: [(LETTERS[c], next(pools[LETTERS[c][:2]])) for c in order]
        for sid, order in {"s1": ENROLL_ORDER, **LATER}.items()
    }


def _session(directory: Path, sid: str, seconds: float):
    from recandcc.models import SessionInfo, Track, TriggerKind

    directory.mkdir(parents=True)
    start = datetime(2026, 1, 1, 10) + timedelta(days=int(sid[1:]))
    s = SessionInfo(
        id=f"recall-{sid}",
        directory=directory,
        started_at=start,
        ended_at=start + timedelta(seconds=seconds),
        trigger_kind=TriggerKind.CALL,
        trigger_app="zoom.us",
        tracks={Track.OTHERS: "others.wav"},
    )
    s.save()
    return s


def run(workdir: Path, synth) -> tuple[dict[str, str], list[tuple[str, str, float]]]:
    """Enroll in session 1, label the later ones. Returns (speaker id -> the label it was
    named from, or "" if it could not be named) and (speaker id, label, seconds) per later
    segment. Profiles go to ``voices.VOICES_DIR``; point it at a temp dir first."""
    from recandcc.config import TranscriptionConfig
    from recandcc.models import Segment
    from recandcc.transcribe import Transcriber, voices
    from recandcc.transcribe.diarize import Diarizer
    from recandcc.transcribe.merge import OTHERS
    from recandcc.transcribe.output import write_outputs

    transcriber = Transcriber(TranscriptionConfig(speaker_names=True))
    diarizer = Diarizer()
    enrolled: dict[str, str] = {}
    later: list[tuple[str, str, float]] = []
    try:
        for sid, parts in plan().items():
            audio, utts = mk.build_clip(parts, synth, mk.meeting_gaps(parts))
            session = _session(workdir / sid, sid, len(audio) / mk.SR)
            turns = diarizer.diarize(audio)
            segs = [
                Segment(u["start"], u["end"], f"utterance {i}", OTHERS, u["language"])
                for i, u in enumerate(utts)
            ]
            labelled = transcriber._labelled(session, segs, turns)  # noqa: SLF001 - real path
            pairs = [
                (u["speaker"], s.speaker, s.end - s.start)
                for u, s in zip(utts, labelled, strict=True)
            ]
            if sid != "s1":
                later += pairs
                continue
            write_outputs(session, labelled, len(audio) / mk.SR, "synthetic")
            for spk, name in ENROLLED.items():
                label = Counter(lb for k, lb, _ in pairs if k == spk).most_common(1)[0][0]
                if label == OTHERS or label in enrolled.values():
                    enrolled[spk] = ""  # merged with another voice: can't be named apart
                    continue
                voices.name_speaker(session.directory, label, name)
                enrolled[spk] = label
    finally:
        diarizer.close()
        transcriber.close()
    return enrolled, later


def score(later: list[tuple[str, str, float]]) -> list[tuple[str, str, int, int, float, float]]:
    """(speaker id, expected, right segments, segments, right seconds, seconds) per voice;
    for the unnamed voices "right" counts the segments wrongly given an enrolled name."""
    names = set(ENROLLED.values())
    rows = []
    for spk in [*ENROLLED, *UNENROLLED]:
        mine = [(lb, d) for k, lb, d in later if k == spk]
        hit = [
            (lb, d) for lb, d in mine if (lb == ENROLLED[spk] if spk in ENROLLED else lb in names)
        ]
        rows.append(
            (
                spk,
                ENROLLED.get(spk, "(unnamed)"),
                len(hit),
                len(mine),
                sum(d for _, d in hit),
                sum(d for _, d in mine),
            )
        )
    return rows


def totals(rows) -> tuple[float, float, float]:
    """(segment recall, duration recall) of the enrolled voices, false-name rate of the rest."""
    named = [r for r in rows if r[0] in ENROLLED]
    other = [r for r in rows if r[0] not in ENROLLED]
    ratio = lambda a, b: a / b if b else 0.0  # noqa: E731
    return (
        ratio(sum(r[2] for r in named), sum(r[3] for r in named)),
        ratio(sum(r[4] for r in named), sum(r[5] for r in named)),
        ratio(sum(r[2] for r in other), sum(r[3] for r in other)),
    )


def missing_voices() -> list[str]:
    listing = subprocess.run(  # noqa: S603 - fixed argv
        ["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=True
    ).stdout
    need = {mk.VOICES[k] for k in LETTERS.values()}
    return sorted(need - mk.parse_voices(listing))


def main() -> int:
    from recandcc.transcribe import diarize, voices

    if sys.platform != "darwin":
        print("voice_recall_eval needs macOS (say, afconvert).", file=sys.stderr)
        return 2
    if missing := missing_voices():
        print(f"missing say voices: {', '.join(missing)}", file=sys.stderr)
        return 2
    if diarize.model_path(download=False) is None:
        print("speaker model not cached; run the app once (nothing is downloaded here).")
        return 2
    with tempfile.TemporaryDirectory(prefix="rcc-recall-") as tmp:
        work = Path(tmp)
        voices.VOICES_DIR = work / "voices"  # never the user's real profiles
        (work / "say").mkdir()
        enrolled, later = run(work / "sessions", mk.say_synth(work / "say"))
    print(f"{'voice':22} {'name':10} {'segments':>12} {'seconds':>16}")
    for spk, name, hit, n, hit_s, n_s in score(later):
        flag = "" if spk not in ENROLLED or enrolled[spk] else "  (not enrollable)"
        seg = f"{hit}/{n}"
        dur = f"{hit_s:.1f}/{n_s:.1f}"
        print(f"{mk.VOICES[spk]:22} {name:10} {seg:>12} {dur:>16}{flag}")
    seg_recall, dur_recall, false_rate = totals(score(later))
    print(f"recall: {seg_recall:.1%} of segments, {dur_recall:.1%} of speech (named voices)")
    print(f"false names: {false_rate:.1%} of the unnamed voices' segments")
    print("synthetic TTS with true segment boundaries: overstates real-world recall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
