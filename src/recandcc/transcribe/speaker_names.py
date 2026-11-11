"""Speaker names read from the conversation by the summary model, before each summary.

The rule-based steps (self-introductions, names a speaker is addressed by) only catch fixed
phrasings. Here the summary model reads the timed transcript and answers, as JSON, which
``Speaker N`` is whom, each with a short quote from the transcript as evidence. A model can
invent things, so nothing is trusted without these checks:

- only labels of the transcript that are still unnamed: ``Speaker N``, or ``Others`` when it is
  the only other label (a 1:1 call); ``Me`` (the owner) is never renamed;
- the name passes ``voices.check_name``, has at most three words, and every word of it is
  written in the transcript (no invented or "corrected" names);
- the evidence is an exact quote from the transcript (case and spacing aside) that contains
  the name, and it proves the claim by *who* said it: the speaker's own line must introduce
  them ("I'm James", "my name is Ahmed"); another speaker's line must address them by the name
  ("Sebastian, ...", "..., Ahmed?") right before they answer or right after they spoke. A
  mention ("the report from Galina") proves nothing;
- a name is not an existing label, nor one another speaker introduced themself with, and no
  two labels get the same name (both are dropped).

Nothing here logs or writes: the caller writes the checked plan (``names.apply_names``) and
logs only the number of names.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable

from ..models import Segment
from .intro_hints import introduced_names
from .merge import ME
from .output import fmt_hms, paragraphs
from .voices import NUMBERED, check_name

OTHERS = "Others"
LOCAL_CHARS = 24_000  # transcript sent to a local model (its context and RAM): the start
CLOUD_CHARS = 400_000
MIN_EVIDENCE = 8  # characters: shorter quotes ("Yes.") prove nothing
_WORD = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*")

Generate = Callable[[list[dict[str, str]], int], str]


def eligible_labels(segments: list[Segment]) -> list[str]:
    """Labels a name may be given to, in order of first speech."""
    labels = list(dict.fromkeys(s.speaker for s in sorted(segments, key=lambda s: s.start)))
    others = [label for label in labels if label != ME]
    if others == [OTHERS]:
        return [OTHERS]  # a 1:1 call: the other side is one person
    return [label for label in others if NUMBERED.fullmatch(label.casefold())]


def _lines(segments: list[Segment], max_chars: int) -> str:
    lines, size = [], 0
    for p in paragraphs(segments):
        if not p.text:
            continue
        line = f"[{fmt_hms(p.start)}] {p.speaker}: {p.text}"
        if size + len(line) + 1 > max_chars:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines)


def name_messages(segments: list[Segment], labels: list[str], max_chars: int) -> list[dict]:
    """Chat messages asking which label is whom."""
    wanted = ", ".join(f'"{label}"' for label in labels)
    system = (
        "You identify the speakers of a meeting transcript. Lines are "
        "'[HH:MM:SS] Label: text'. 'Me' is the person who recorded it; never name 'Me'. "
        "Use only names written in the transcript; never guess or invent a name."
    )
    ask = (
        f"For each of these labels: {wanted}, give the person's real name only when the "
        "transcript makes it clear, for example: they introduce themselves, they are addressed "
        "by name and then answer, or they are called on by name and then speak. Leave out "
        "every label you are not sure about. Answer with JSON only, in this form:\n"
        '{"Speaker 1": {"name": "<name as written>", "evidence": "<an exact quote from the '
        'transcript, a few words long, that contains the name and shows it>"}}\n'
        "Answer {} if no label is clear.\n\n"
        "Example (names invented): for the lines\n"
        "[00:00:05] Me: Orla, can you give us the status?\n"
        "[00:00:09] Speaker 1: Sure, the build is green.\n"
        "[00:00:15] Speaker 2: Hi all, I'm Pim from the mobile team.\n"
        'the answer is {"Speaker 1": {"name": "Orla", "evidence": "Orla, can you give us '
        'the status"}, "Speaker 2": {"name": "Pim", "evidence": "I\'m Pim from the mobile '
        'team"}}. Use the names of the real transcript below, never these.'
    )
    text = _lines(segments, max_chars)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{ask}\n\n---\n{text}\n---"},
    ]


def parse_answer(answer: str) -> dict[str, tuple[str, str]]:
    """label -> (name, evidence) from the model's answer: the last JSON object in it that
    maps labels to objects (text around it, e.g. a reasoning model's notes, is ignored)."""
    decoder = json.JSONDecoder()
    found: dict = {}
    for i, ch in enumerate(answer):
        if ch != "{":
            continue
        try:
            value, _ = decoder.raw_decode(answer[i:])
        except ValueError:
            continue
        if isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            found = value
    plan: dict[str, tuple[str, str]] = {}
    for label, item in found.items():
        name, evidence = item.get("name"), item.get("evidence")
        if isinstance(label, str) and isinstance(name, str) and isinstance(evidence, str):
            plan[label.strip()] = (name.strip(), evidence.strip())
    return plan


def _norm(text: str) -> str:
    return " ".join(text.replace("’", "'").casefold().split())


_SELF = r"(?:i'm|i am|my name is|my name's|this is|it's|меня зовут|это|я)\s+"
NEAR = 6  # segments searched for the answer (the addressing speaker may go on a little)


def _self_claim(quote: str, first: str) -> bool:
    return re.search(_SELF + re.escape(first) + r"\b", quote) is not None


def _addressed(quote: str, first: str) -> bool:
    """The name used as an address: at a clause start before a comma or question mark, or
    after a comma at the end of a clause ("Sebastian, ...", "..., Ahmed?", "work, Tobias.")."""
    name = re.escape(first)
    return (
        re.search(rf"(?:^|[.!?;]\s*)(?:\w+,?\s+)?{name}\s*[,?!]", quote) is not None
        or re.search(rf",\s*{name}\s*(?:[,.!?]|$)", quote) is not None
    )


def _proves(label: str, first: str, quote: str, ordered: list[Segment]) -> bool:
    """Whether ``quote`` (normalised) shows that ``label`` is called ``first``."""
    texts = [_norm(s.text) for s in ordered]
    for i, text in enumerate(texts):
        if quote not in text:
            # a quote across two lines starts in this one (else it belongs to the next line)
            nxt = texts[i + 1] if i + 1 < len(texts) else ""
            if not nxt or quote in nxt or quote not in f"{text} {nxt}":
                continue
        speaker = ordered[i].speaker
        if speaker == label:
            if _self_claim(quote, first):
                return True
            continue
        if not _addressed(quote, first):
            continue
        before = ordered[i - 1].speaker if i else None
        after = next((s.speaker for s in ordered[i + 1 : i + NEAR] if s.speaker != speaker), None)
        if label in (before, after):
            return True
    return False


def checked_names(plan: dict[str, tuple[str, str]], segments: list[Segment]) -> dict[str, str]:
    """The part of the model's plan that passes every check (see the module docstring)."""
    labels = set(eligible_labels(segments))
    ordered = sorted(segments, key=lambda s: s.start)
    corpus = _norm(" ".join(s.text for s in ordered))
    words = set(_WORD.findall(corpus))
    taken = {s.speaker.casefold() for s in segments}
    claims = introduced_names(segments)
    accepted: dict[str, str] = {}
    for label, (raw, evidence) in plan.items():
        if label not in labels:
            continue
        try:
            name = check_name(raw)
        except ValueError:
            continue
        parts = _WORD.findall(_norm(name))
        quote = _norm(evidence.strip("\"'“”«» "))
        if (
            not parts
            or len(parts) > 3
            or len(parts) != len(name.split())
            or not all(part in words for part in parts)
            or len(quote) < MIN_EVIDENCE
            or quote not in corpus
            or parts[0] not in _WORD.findall(quote)
            or not _proves(label, parts[0], quote, ordered)
            or _norm(name) in taken
            or any(
                _norm(name) in claimed or parts[0] in {c.split()[0] for c in claimed}
                for other, claimed in claims.items()
                if other != label
            )
        ):
            continue
        accepted[label] = name
    counts = Counter(_norm(n) for n in accepted.values())
    return {label: name for label, name in accepted.items() if counts[_norm(name)] == 1}


def infer_names(
    segments: list[Segment], generate: Generate, *, max_chars: int, max_tokens: int
) -> dict[str, str]:
    """label -> name for the unnamed speakers the model can tell from the conversation, after
    the checks. One model call; {} when no label is eligible."""
    labels = eligible_labels(segments)
    if not labels:
        return {}
    answer = generate(name_messages(segments, labels, max_chars), max_tokens)
    return checked_names(parse_answer(answer), segments)


__all__ = ["checked_names", "eligible_labels", "infer_names", "name_messages", "parse_answer"]
