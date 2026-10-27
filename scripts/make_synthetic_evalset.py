"""Build a small synthetic eval set with macOS text-to-speech, for scripts/eval.py.

    uv run python scripts/make_synthetic_evalset.py   # -> <temp dir>/rcc-synthetic-eval
    uv run python scripts/make_synthetic_evalset.py /tmp/rcc-synth
    uv run python scripts/eval.py /tmp/rcc-synth --languages en,ru,he --backend mlx

An interim stand-in for the consented set (docs/EVALUATION.md), not a replacement: `say`
voices are clean studio speech, so scores on this set overstate real-world accuracy. Covers
en, ru and he, code-switching (en+ru, en+he, ru+he) and a three-speaker English dialogue for
DER. The text is fixed, neutral content written below, not real data.

Each sentence is rendered with ``say -o file.aiff`` (writes a file, plays nothing) and
converted with ``afconvert`` to 16 kHz mono; its leading and trailing silence is trimmed, and
the clip is laid out with fixed pauses, so the manifest times come from the audio lengths.
A clip whose voice is not installed is skipped. macOS only. The set is written outside the
repo and outside ~/Documents.

Stories: none.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
LEAD_S, GAP_S, TAIL_S = 0.5, 0.7, 2.0  # eval.py: end clips with a couple of seconds of silence
SILENCE = 1e-3  # trim threshold (float samples)

# speaker id -> say voice; its language is the id's first two letters
VOICES = {
    "en1": "Samantha",
    "en2": "Reed (English (US))",
    "en3": "Flo (English (US))",
    "ru": "Milena",
    "he": "Carmit",
    # meetings only (--meetings): six clearly different English voices
    "en4": "Daniel",
    "en5": "Karen",
    "en6": "Moira",
    "en7": "Rishi",
}

# clip id -> [(speaker id, text)]
CLIPS: dict[str, list[tuple[str, str]]] = {
    "en-01": [
        ("en1", "Good morning, everyone, and thank you for joining the weekly planning meeting."),
        (
            "en1",
            "Today we will review the project schedule and the open tasks for the next release.",
        ),
        ("en1", "The design team finished the new settings page on Tuesday."),
        ("en1", "Testing started yesterday and the first results look stable."),
        ("en1", "Please send your questions before Friday so we can answer them in the next call."),
    ],
    "ru-01": [
        ("ru", "Добрый день, коллеги, давайте начнём нашу встречу."),
        ("ru", "Сегодня мы обсудим план работы на следующую неделю."),
        ("ru", "Отчёт по тестированию будет готов в четверг утром."),
        ("ru", "Если у вас есть вопросы, напишите их в общий чат."),
        ("ru", "Следующая встреча состоится в понедельник в десять часов."),
    ],
    "he-01": [
        ("he", "בוקר טוב לכולם, תודה שהצטרפתם לפגישה."),
        ("he", "היום נדבר על לוח הזמנים של הפרויקט."),
        ("he", "הצוות סיים את העבודה על הגרסה החדשה ביום שלישי."),
        ("he", "הבדיקות מתחילות מחר בבוקר."),
        ("he", "הפגישה הבאה תתקיים ביום ראשון בשעה עשר."),
    ],
    "en-ru-01": [
        ("en1", "Let's start with the budget for the second quarter."),
        ("ru", "Хорошо, бюджет уже согласован с финансовым отделом."),
        ("en1", "Great, then we can move on to the hiring plan."),
        ("ru", "Мы ищем двух инженеров и одного дизайнера."),
        ("en1", "Please share the job descriptions by the end of the week."),
        ("ru", "Договорились, я отправлю их в пятницу."),
    ],
    "en-he-01": [
        ("en2", "Can you hear me well? The connection seems a little slow today."),
        ("he", "כן, אני שומעת אותך טוב מאוד."),
        ("en2", "Good. Did you get the document I sent this morning?"),
        ("he", "קיבלתי אותו, ואקרא אותו אחרי הצהריים."),
        ("en2", "Thanks, let me know if anything is unclear."),
        ("he", "בסדר גמור, נדבר מחר."),
    ],
    "ru-he-01": [
        ("ru", "Привет, ты уже видела новое расписание поездов?"),
        ("he", "עוד לא, מתי הרכבת הראשונה יוצאת?"),
        ("ru", "Первый поезд отправляется в семь утра."),
        ("he", "זה מוקדם מדי בשבילי, אני אסע ברכבת של שמונה."),
        ("ru", "Хорошо, тогда встретимся на вокзале."),
        ("he", "מצוין, נתראה שם."),
    ],
    "en-3spk-01": [
        ("en1", "Welcome back. Let's go around the table with short updates."),
        ("en2", "I finished the database migration and the backups are running every night."),
        ("en3", "I am still working on the new onboarding screens."),
        ("en1", "How long do you think that will take?"),
        ("en3", "About two more days, if the review goes smoothly."),
        ("en2", "I can help with the review tomorrow afternoon."),
        ("en1", "Perfect, thank you both. That's all for today."),
    ],
}

# --meetings: longer turn-taking meetings for DER with 2, 3 and 6 speakers. Each is a string of
# speaker letters (A = en1 ... F = en7 skipping en3); turn i says POOL[i % len(POOL)], so
# lengths vary from one word to two sentences, and quick exchanges get a shorter gap (see `gaps`).
POOL = [
    "Good morning, everyone, and thanks for making the time today.",
    "Morning.",
    "Let's start with the status of the release, because that decides the rest of the agenda.",
    "Sure.",
    "The build has been green since Monday, and the remaining issues are all small.",
    "Which ones?",
    "Mostly wording on the settings page and one slow screen when the list is long.",
    "I can look at the slow screen today.",
    "Great, thank you.",
    "What about the documentation, is anyone working on that?",
    "I started the installation chapter last week, but I still need a reviewer.",
    "I'll review it.",
    "Please send it to me before lunch, and I will read it on the train this afternoon.",
    "Okay.",
    "On the planning side, we agreed to freeze the feature list on Thursday.",
    "Is that firm?",
    "Yes, unless something serious comes up.",
    "Then I need to finish the export work by Wednesday, which is tight but possible.",
    "Do you need help?",
    "Maybe with testing, once the first version is ready.",
    "Count me in.",
    "Let me write that down.",
    "Next topic: the customer feedback from the pilot group, which came in over the weekend.",
    "Most of it was positive.",
    "The main complaint was that the first start takes too long to explain.",
    "We could add a short guide on the first screen, instead of a long manual.",
    "Good idea, but who writes it?",
    "I can draft it tonight.",
    "Thanks.",
    "One more thing about the schedule: the office is closed on Friday afternoon.",
    "Right, so Friday is effectively half a day.",
    "Then we should do the demo on Thursday morning, not in the afternoon.",
    "Agreed.",
    "Does everyone have what they need to prepare?",
    "Yes.",
    "I need the final numbers from the test run.",
    "They will be ready tomorrow evening.",
    "Perfect.",
    "Before we close, let me summarize the decisions and the owners for each task.",
    "The slow screen goes to the first speaker, the guide to the second, the review to the third.",
    "And the demo is Thursday at ten.",
    "Any objections?",
    "None from me.",
    "Fine.",
    "Then we are done. Thank you all, and have a good week.",
    "Thanks, bye.",
    "Bye everyone.",
    "See you on Thursday.",
]
MEETINGS: dict[str, tuple[list[str], str]] = {
    # id -> (speaker ids, turn order); the 6-speaker meeting runs ~3.7 min; POOL repeats
    "meeting-2spk": (["en1", "en2"], "AB" * 12),
    "meeting-3spk": (["en1", "en2", "en4"], "ABCABACBCABCACBABCACBABCBA"),
    "meeting-6spk": (
        ["en1", "en2", "en4", "en5", "en6", "en7"],
        "ABCDEFADBECFAEBDCFABECDFAEBDCFAEDBFCAEDFBCADEBFCFEDCBAFBDACEFDBAECFBADECBFADC",
    ),
}
GAP_CYCLE_S = (0.7, 0.4, 1.0, 0.5)  # varied pauses between turns
QUICK_S = 0.15  # a short reply (up to 3 words) follows almost at once

Synth = Callable[[str, str], np.ndarray]  # (voice, text) -> float32 mono 16 kHz


def parse_voices(listing: str) -> set[str]:
    """Voice names from ``say -v '?'`` ("Reed (English (US))  en_US    # Hello! ...")."""
    names = set()
    for line in listing.splitlines():
        if m := re.match(r"^(.+?)\s+[a-z]{2,3}_[A-Za-z0-9]+\s+#", line):
            names.add(m[1].strip())
    return names


def meeting_parts(speakers: list[str], order: str) -> list[tuple[str, str]]:
    """(speaker id, text) per turn of a meeting: letter i of `order` speaks POOL[i]."""
    return [(speakers[ord(c) - ord("A")], POOL[i % len(POOL)]) for i, c in enumerate(order)]


def meeting_gaps(parts: list[tuple[str, str]]) -> list[float]:
    """Pause before each turn after the first: varied, and short before a quick reply."""
    return [
        QUICK_S if len(text.split()) <= 3 else GAP_CYCLE_S[i % len(GAP_CYCLE_S)]
        for i, (_, text) in enumerate(parts[1:])
    ]


def meeting_clips() -> dict[str, list[tuple[str, str]]]:
    return {cid: meeting_parts(*spec) for cid, spec in MEETINGS.items()}


def plan(
    installed: set[str], clips: dict[str, list[tuple[str, str]]] | None = None
) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """The clips whose voices are all installed, and the ids of the skipped ones."""
    keep, skipped = {}, []
    for cid, parts in (CLIPS if clips is None else clips).items():
        if all(VOICES[spk] in installed for spk, _ in parts):
            keep[cid] = parts
        else:
            skipped.append(cid)
    return keep, skipped


def trim(audio: np.ndarray, threshold: float = SILENCE) -> np.ndarray:
    """``audio`` without leading and trailing silence (empty if it is all silence)."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    return audio[loud[0] : loud[-1] + 1] if loud.size else audio[:0]


def layout(
    lengths: list[int], sr: int = SR, gaps: list[float] | None = None
) -> tuple[list[int], int]:
    """Start sample of each piece and the clip length: lead, pieces split by gaps (seconds,
    one fewer than pieces; default GAP_S each), tail."""
    gaps = gaps if gaps is not None else [GAP_S] * max(len(lengths) - 1, 0)
    pos, starts = round(LEAD_S * sr), []
    for i, n in enumerate(lengths):
        starts.append(pos)
        pos += n + (round(gaps[i] * sr) if i < len(gaps) else 0)
    return starts, pos + round(TAIL_S * sr)


def build_clip(
    parts: list[tuple[str, str]], synth: Synth, gaps: list[float] | None = None
) -> tuple[np.ndarray, list[dict]]:
    """The clip's audio and its manifest utterances (times from the rendered lengths)."""
    pieces = [trim(synth(VOICES[spk], text)) for spk, text in parts]
    starts, total = layout([len(p) for p in pieces], gaps=gaps)
    audio = np.zeros(total, dtype=np.float32)
    utts = []
    for (spk, text), p, s in zip(parts, pieces, starts, strict=True):
        audio[s : s + len(p)] = p
        utts.append(
            {
                "start": round(s / SR, 3),
                "end": round((s + len(p)) / SR, 3),
                "speaker": spk,
                "language": spk[:2],
                "text": text,
            }
        )
    return audio, utts


def build_set(
    out: Path,
    clips: dict[str, list[tuple[str, str]]],
    synth: Synth,
    gaps: dict[str, list[float]] | None = None,
) -> dict:
    """Write ``<id>.wav`` per clip and ``manifest.json`` into ``out``; returns the manifest.

    The old manifest goes first and the new one is written last, so a run that fails halfway
    leaves no manifest pointing at a mix of old and new clips."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    manifest: dict[str, list] = {"clips": []}
    for cid, parts in clips.items():
        audio, utts = build_clip(parts, synth, (gaps or {}).get(cid))
        sf.write(str(out / f"{cid}.wav"), audio, SR, subtype="PCM_16")
        multi = len({spk for spk, _ in parts}) > 1
        track = "others" if multi else "me"  # one voice: no DER to score
        manifest["clips"].append(
            {"id": cid, "audio": f"{cid}.wav", "track": track, "utterances": utts}
        )
        print(f"  {cid}: {len(audio) / SR:.1f} s, {len(utts)} utterances, track {track}")
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    (out / "manifest.json").write_text(text, encoding="utf-8")
    return manifest


def forbidden(out: Path) -> str | None:
    """Why ``out`` must not hold the set, else None: inside ~/Documents or any git checkout
    (this repo, its main checkout from a worktree, or another one), or a non-empty folder
    that is not an earlier set."""
    out = out.expanduser().resolve()
    if out.is_relative_to((Path.home() / "Documents").resolve()):
        return "refusing to write the set inside ~/Documents"
    if any((d / ".git").exists() for d in (out, *out.parents)):
        return "refusing to write the set inside a git checkout"
    if out.is_dir() and any(out.iterdir()) and not (out / "manifest.json").exists():
        return "refusing to write into a non-empty folder that holds no eval set"
    return None


def say_synth(tmp: Path) -> Synth:
    """Renders with ``say -o`` into ``tmp`` (nothing is played) and ``afconvert`` to 16 kHz."""
    count = iter(range(1_000_000))

    def synth(voice: str, text: str) -> np.ndarray:
        n = next(count)
        aiff, wav = tmp / f"{n}.aiff", tmp / f"{n}.wav"
        # fixed argv, no shell; -o writes a file instead of playing through the speakers
        subprocess.run(["/usr/bin/say", "-v", voice, "-o", str(aiff), text], check=True)  # noqa: S603
        conv = [
            "/usr/bin/afconvert",
            "-f",
            "WAVE",
            "-d",
            "LEI16@16000",
            "-c",
            "1",
            str(aiff),
            str(wav),
        ]
        subprocess.run(conv, check=True)  # noqa: S603 - fixed argv
        audio, sr = sf.read(str(wav), dtype="float32")
        if sr != SR or audio.ndim != 1:
            raise RuntimeError(f"afconvert gave {sr} Hz, {audio.ndim} channel dims")
        return audio

    return synth


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    default = Path(tempfile.gettempdir()) / "rcc-synthetic-eval"
    p.add_argument("out", type=Path, nargs="?", default=default, help=f"default: {default}")
    p.add_argument(
        "--meetings",
        action="store_true",
        help="write only the 2-, 3- and 6-speaker meetings (folder: rcc-synthetic-meetings)",
    )
    args = p.parse_args(argv)
    if args.meetings and args.out == default:
        args.out = default.with_name("rcc-synthetic-meetings")
    if sys.platform != "darwin":
        print(
            "make_synthetic_evalset needs macOS (say, afconvert); nothing written.", file=sys.stderr
        )
        return 2
    if why := forbidden(args.out):
        print(why, file=sys.stderr)
        return 2
    listing = subprocess.run(  # noqa: S603 - fixed argv
        ["/usr/bin/say", "-v", "?"], capture_output=True, text=True, check=True
    ).stdout
    source = meeting_clips() if args.meetings else CLIPS
    clips, skipped = plan(parse_voices(listing), source)
    for cid in skipped:
        need = sorted({VOICES[spk] for spk, _ in source[cid]})
        print(f"  skipping {cid}: needs voices {', '.join(need)}")
    if not clips:
        print("no clip has all its voices installed; nothing written.", file=sys.stderr)
        return 1
    out = args.out.expanduser()
    with tempfile.TemporaryDirectory(prefix="rcc-say-") as tmp:
        gaps = {c: meeting_gaps(parts) for c, parts in clips.items()} if args.meetings else None
        build_set(out, clips, say_synth(Path(tmp)), gaps)
    print(f"wrote {len(clips)} clips to {out} (synthetic TTS: scores overstate real accuracy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
