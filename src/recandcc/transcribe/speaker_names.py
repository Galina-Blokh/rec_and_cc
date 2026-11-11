"""Speaker names read from the conversation by the summary model, before each summary.

The rule-based steps (self-introductions, names a speaker is addressed by) only catch fixed
phrasings. Here the summary model reads the timed transcript and answers, as JSON, which
``Speaker N`` is whom, each with a short quote from the transcript as evidence. A model can
invent things, so nothing is trusted without these checks:

- only labels of the transcript that are still unnamed: ``Speaker N``, or ``Others`` when it is
  the only other label (a 1:1 call) and doesn't introduce two names; ``Me`` (the owner) is
  never renamed;
- the name passes ``voices.check_name``, has at most three words, and every word of it is
  written in the transcript (no invented or "corrected" names);
- the evidence is an exact quote from the transcript (case and spacing aside) that contains
  the name, and it proves the claim by *who* said it: the speaker's own line must introduce
  them at a clause start ("I'm James", "my name is Ahmed"; not quoted speech); another
  speaker's line must call on them ("Sebastian, ...", "..., Ahmed?") just before they answer
  (within seconds), or thank/answer them ("Great work, Tobias.") just after they spoke. A
  mention ("the report from Galina") proves nothing, and the quote must hold the whole name;
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
_GREETING = (
    r"(?:hi|hello|hey|good morning|good afternoon|привет|здравствуйте)(?: everyone| all)?[,!]?"
)
_THANKS = r"thanks|thank you|great work|good work|well done|good job|nice work|cheers|спасибо"
# "<word>, you ..." / "<word>, can ...": calling on someone (fillers excluded below)
_CALL_ON = re.compile(
    r"(?:^|[.!?;]\s*)(?:\w+,?\s+)?([^\W\d_]+),\s+(?:you|can|could|would|will|please|go|are|do)\b"
)
_NOT_NAME_WORDS = "actually ok okay so well and but yes yeah no now then sorry also"
_NOT_NAMES = frozenset(_NOT_NAME_WORDS.split())
REPLY_S = 6.0  # an answer to an address starts within this many seconds of it
OVERLAP_S = 0.3  # ... and no earlier than this before the address ends


def _quoted(prefix: str) -> bool:
    """Whether text after ``prefix`` is inside an open quote or follows a colon (reported)."""
    return (
        prefix.count('"') % 2 == 1
        or prefix.count("“") > prefix.count("”")
        or prefix.count("«") > prefix.count("»")
        or ":" in prefix
    )


def _self_claim(text: str, name: str) -> bool:
    """``text`` (a whole normalised line) introduces its speaker as ``name`` at the start of a
    clause, not inside reported speech ('He said, "My name is Bob."')."""
    for m in re.finditer(_SELF + re.escape(name) + r"\b", text):
        prefix = text[: m.start()].rstrip()
        clause = re.split(r"[.!?]\s", prefix)[-1] if prefix else ""
        if _quoted(prefix):
            continue  # inside quotes or after "wrote:": someone else's words
        if not clause or re.fullmatch(_GREETING, clause) or prefix.endswith((".", "!", "?")):
            return True
    return False


def _direction(text: str, name: str) -> str | None:
    """How ``name`` is used in ``text`` (a whole normalised line): "forward" when someone is
    called on ("Alice, can you ...", "..., Ahmed?"), "back" when the one who just spoke is
    thanked or answered ("Great work, Tobias."), else None (a mention)."""
    n = re.escape(name)
    if re.search(rf"\b(?:{_THANKS})[,!]?\s+{n}\b", text):
        return "back"  # "Thanks, Alice, that helps.": to the one who just spoke
    if re.search(rf"(?:^|[.!?;]\s*)(?:\w+,?\s+)?{n}\s*[,?]", text) or re.search(
        rf",\s*{n}\s*\?", text
    ):
        return "forward"
    if re.search(rf",\s*{n}\s*(?:[.!]|$)", text):
        return "back"
    return None


def _calls_someone_else(text: str, name: str) -> bool:
    """ "Actually, Bob, you go first": the caller turns to another person before the answer."""
    first = name.split()[0]
    return any(
        m.group(1) not in _NOT_NAMES and m.group(1) != first for m in _CALL_ON.finditer(text)
    )


def _answer_words(ordered: list[Segment], texts: list[str], j: int) -> set[str]:
    """Words of the answer turn that starts at ``ordered[j]`` ("Alice is out today": an answer
    that says the name is someone else covering)."""
    words, end = set(_WORD.findall(texts[j])), ordered[j].end
    for k in range(j + 1, len(ordered)):
        if ordered[k].speaker != ordered[j].speaker or ordered[k].start - end > REPLY_S:
            break
        words |= set(_WORD.findall(texts[k]))
        end = max(end, ordered[k].end)
    return {re.sub(r"'s$", "", w) for w in words}


def _proves(label: str, name: str, quote: str, ordered: list[Segment]) -> bool:
    """Whether ``quote`` (normalised, containing ``name``) shows that ``label`` is ``name``,
    judged on the whole line it comes from."""
    texts = [_norm(s.text) for s in ordered]
    for i, text in enumerate(texts):
        if quote not in text:
            continue
        seg = ordered[i]
        if seg.speaker == label:
            if _self_claim(text, name):
                return True
            continue
        way = _direction(text, name)
        if way == "forward":
            end, nxt, j = seg.end, None, i  # the addressing speaker may go on before the answer
            later = [text[text.find(name) + len(name) :]]  # what the caller says after the name
            for j in range(i + 1, len(ordered)):
                other = ordered[j]
                if other.speaker != seg.speaker:
                    nxt = other
                    break
                if other.start - end > REPLY_S:
                    break
                end = max(end, other.end)
                later.append(texts[j])
            if (
                nxt is not None
                and nxt.speaker == label
                and not nxt.overlap
                and -OVERLAP_S <= nxt.start - end <= REPLY_S
                and not _calls_someone_else(" ".join(later), name)
                and name.split()[0] not in _answer_words(ordered, texts, j)
            ):
                return True
        elif way == "back" and i and ordered[i - 1].speaker == label:
            return True
    return False


def _self_names(ordered: list[Segment], label: str) -> set[str]:
    """Capitalised names ``label`` introduces itself with in any ``_SELF`` form ("It's Alice
    here"), casefolded."""
    found = set()
    for seg in ordered:
        if seg.speaker == label:
            for m in re.finditer(_SELF + r"([^\W\d_]+)", seg.text.replace("’", "'"), re.I):
                if m.group(1)[:1].isupper():
                    found.add(m.group(1).casefold())
    return found


def checked_names(plan: dict[str, tuple[str, str]], segments: list[Segment]) -> dict[str, str]:
    """The part of the model's plan that passes every check (see the module docstring)."""
    labels = set(eligible_labels(segments))
    ordered = sorted(segments, key=lambda s: s.start)
    corpus = _norm(" ".join(s.text for s in ordered))
    words = set(_WORD.findall(corpus))
    taken = {s.speaker.casefold() for s in segments}
    claims = introduced_names(segments)
    if len(claims.get(OTHERS, set()) | _self_names(ordered, OTHERS)) > 1:
        labels.discard(OTHERS)  # one track, two introduced names: not one person
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
            or " ".join(parts) not in quote  # the whole name, so no surname is invented
            or not _proves(label, " ".join(parts), quote, ordered)
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
