"""Write a deterministic synthetic 1-hour meeting transcript, to time `recandcc summarize`.

    uv run python scripts/make_synthetic_meeting_transcript.py   # -> <tmp>/rcc-synthetic-meeting
    uv run python scripts/make_synthetic_meeting_transcript.py /tmp/rcc-meeting --seed 7

Writes ``<out>/transcript.json`` in the app's format (about 8-9k English words over 3600 s,
four speakers: Me, Speaker 1-3), built from the fixed neutral templates below with a seeded
random generator: the same seed gives the same bytes. Not real data. The folder is written
outside the repo and outside ~/Documents; transcripts are never committed.

Stories: none.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

SPEAKERS = ["Me", "Speaker 1", "Speaker 2", "Speaker 3"]
DURATION_S = 3600.0
TARGET_WORDS = 8500
NAMES = ["Dana", "Omar", "Priya", "Lena", "Marco", "Yuki"]
TOPICS = [
    "the onboarding flow",
    "the billing migration",
    "the search latency",
    "the mobile release",
    "the partner integration",
    "the data retention policy",
    "the new dashboard",
    "the support backlog",
]
POOLS = {
    "chat": [
        "Good morning, can everyone hear me okay? The connection on my side is a bit slow today.",
        "Yes, we can hear you fine. Give it a second while the others join the call.",
        "Did anyone watch the match last night? I fell asleep before the second half, honestly.",
        "Let's get started, we have a lot to cover and only an hour, so I will keep us on time.",
        "Sorry, I was on mute again. That happens to me at least once in every single meeting.",
        "Thanks everyone, that is a good place to stop, and I will send the notes this afternoon.",
    ],
    "status": [
        "On {t} we finished the main work last week, and what is left is mostly testing.",
        "I think {t} is slipping by about three days, because the review took longer than planned.",
        "For {t} the numbers look better this month, but I do not trust them until we check logs.",
        "The risk with {t} is that two teams depend on the same change, so we must agree on order.",
        "Honestly {t} is the part I am most worried about, nobody here knows it well.",
        "Customers keep asking about {t}, so I would like us to give them a clear date this week.",
    ],
    "discuss": [
        "I see the point, but if we do that, we must update the docs and the tests.",
        "What would it cost to do it properly instead of the quick fix, in terms of days of work?",
        "Last time we skipped that step it caused an outage, so I would rather not skip it again.",
        "We could split it in two, ship the safe half now, and keep the risky half for later.",
        "I do not have strong feelings either way, as long as somebody owns it and writes it down.",
        "That makes sense to me, and it matches what the customers told us in the last interviews.",
        "Can we look at the actual data before deciding, since my memory of it is not very good?",
    ],
    "decision": [
        "So we decided to go with the smaller scope for {t} and revisit the rest next quarter.",
        "Okay, the decision is to move {t} to the next sprint and keep the other dates.",
        "We agree that {t} waits until the review is done, and nobody ships it before.",
    ],
    "action": [
        "{n} will write up the plan for {t} and share it with the group by Friday.",
        "{n}, can you check the numbers for {t} and report back on Tuesday morning, please?",
        "I will talk to the vendor about {t} tomorrow, and I will send an email with the answer.",
        "{n} promised to fix the open issues in {t} before the end of next week.",
    ],
}


def _line(rng: random.Random, kind: str) -> str:
    return rng.choice(POOLS[kind]).format(t=rng.choice(TOPICS), n=rng.choice(NAMES))


def build(seed: int = 0) -> dict:
    """The transcript.json content: deterministic for a seed."""
    rng = random.Random(seed)  # noqa: S311 - not cryptographic
    kinds, weights = list(POOLS), [1, 3, 6, 1, 2]
    texts: list[tuple[str, str]] = []
    words, last = 0, ""
    while words < TARGET_WORDS:
        kind = rng.choices(kinds, weights)[0]
        speaker = rng.choice([s for s in SPEAKERS if s != last])
        # a turn is one to three sentences of the same kind, by one speaker
        text = " ".join(_line(rng, kind) for _ in range(rng.randint(1, 3)))
        texts.append((speaker, text))
        words += len(text.split())
        last = speaker
    # spread the turns over the hour in proportion to their length
    scale = (DURATION_S - 5.0) / words
    segments, t = [], 2.0
    for speaker, text in texts:
        end = t + len(text.split()) * scale
        seg = {"start": round(t, 3), "end": round(end - 0.4, 3), "text": text}
        segments.append({**seg, "speaker": speaker, "language": "en"})
        t = end
    return {
        "session_id": f"synthetic-meeting-{seed}",
        "started_at": "2026-01-01T10:00:00",
        "ended_at": "2026-01-01T11:00:00",
        "trigger_kind": "manual",
        "trigger_app": None,
        "duration": DURATION_S,
        "languages": ["en"],
        "model": "synthetic",
        "segments": segments,
    }


def forbidden(out: Path) -> str | None:
    """Why ``out`` must not hold the transcript, else None: inside ~/Documents or any git
    checkout (this repo, its main checkout from a worktree, or another one), or a non-empty
    folder that is not an earlier synthetic meeting."""
    out = out.expanduser().resolve()
    if out.is_relative_to((Path.home() / "Documents").resolve()):
        return "refusing to write the transcript inside ~/Documents"
    if any((d / ".git").exists() for d in (out, *out.parents)):
        return "refusing to write the transcript inside a git checkout"
    if out.is_dir() and any(out.iterdir()) and not (out / "transcript.json").exists():
        return "refusing to write into a non-empty folder that holds no transcript"
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    default = Path(tempfile.gettempdir()) / "rcc-synthetic-meeting"
    p.add_argument("out", type=Path, nargs="?", default=default, help=f"default: {default}")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if why := forbidden(args.out):
        print(why, file=sys.stderr)
        return 2
    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    data = build(args.seed)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    (out / "transcript.json").write_text(text, encoding="utf-8")
    n = sum(len(s["text"].split()) for s in data["segments"])
    print(f"wrote {out / 'transcript.json'}: {len(data['segments'])} segments, {n} words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
