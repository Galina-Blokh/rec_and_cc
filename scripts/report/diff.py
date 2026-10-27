"""Print what moved between two report data files, in a few lines."""

from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = [
    ("summary", "merged", "merged pull requests"),
    ("summary", "open", "open pull requests"),
    ("summary", "issues_closed", "issues closed"),
    ("summary", "review_findings", "Cursor review findings"),
    ("ci", "ci_rate", "CI pass rate"),
    ("claude", "cost", "estimated Claude cost ($)"),
]


def changes(old: dict, new: dict) -> list[str]:
    out = []
    for group, key, label in KEYS:
        a, b = old.get(group, {}).get(key), new.get(group, {}).get(key)
        if a != b:
            out.append(f"{label}: {a} -> {b}")
    return out


def main(argv: list[str]) -> int:
    old, new = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv[:2])
    lines = changes(old, new)
    print("\n".join(lines) if lines else "no change in the headline figures")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
