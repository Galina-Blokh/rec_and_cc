"""Self-introduction hints: "I'm Anna" in a speaker's first minutes suggests a name (#1 A.4).

Only a hint for ``recandcc speakers``: it reads the transcript, never renames, stores or logs
anything (the matched text is transcript content). The patterns are deliberately strict, so
a wrong suggestion is rarer than a missed one:

- English and Russian names must be capitalised (Whisper capitalises names). "I'm" and "Я"
  need the name to end the clause or be followed by "from", "here", "speaking", "из", ...:
  "I'm sure", "Я думаю" or "I'm Sorry, but" don't count. "This is" and "Это" count after a
  greeting ("Hi, this is Anna."), otherwise only with such a word ("This is Anna speaking"),
  so "This is Paris." or "Так, это Москва." don't. A title is kept: "I'm Dr. Smith".
- Hebrew has no capitals, and "אני" is followed by verbs and adjectives far more often than by
  names, so only "קוראים לי X", "שמי X" and "השם שלי X" count.
- Common words that follow these triggers (sure, late, рад, חושב, weekdays, ...) are never
  names, and neither are Me, Others or Speaker N.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..models import Segment
from .merge import ME
from .output import JSON_NAME
from .voices import NUMBERED, check_name

WINDOW = 120.0  # seconds from a speaker's first segment that are searched
_I = re.IGNORECASE
_HELLO = r"\b(?:hi|hello|hey|привет|здравствуйте)(?:\s+(?:everyone|all|guys|всем|все))?[,!]?\s+"
_CLAUSE = r"(?:^|[.!?,;:—–]\s*|" + _HELLO + ")"  # short triggers mid-sentence don't count
_SENTENCE = r"(?:^|[.!?]\s*)"
ANY, CLAUSE_END, FOLLOWED = 0, 1, 2  # what must come after the name
TRIGGERS = [
    (re.compile(r"\bmy name(?:'s|’s| is)\s+", _I), ANY),
    (re.compile(r"меня зовут\s+", _I), ANY),
    (re.compile(r"мо[её] имя\s+", _I), ANY),
    (re.compile(r"קוראים לי\s+"), ANY),
    (re.compile(r"(?:^|\s)(?:שמי|השם שלי)\s+"), ANY),
    (re.compile(_CLAUSE + r"(?:i'm|i’m|i am|я)\s+", _I), CLAUSE_END),
    (re.compile(_HELLO + r"(?:this is|это)\s+", _I), CLAUSE_END),
    (re.compile(_SENTENCE + r"(?:this is|это)\s+", _I), FOLLOWED),
]
FOLLOW = {"speaking", "here", "from", "and", "with", "из", "и", "говорит", "слушаю"}
TITLES = {"dr", "mr", "mrs", "ms", "prof", "professor"}
_STOP_WORDS = (
    # English
    "i a an the it this that me myself not just so sure sorry fine good great okay ok yes no "
    "here there back ready done happy glad afraid going speaking well right also still "
    "late early tired busy new excited online available calling joining sick away "
    "monday tuesday wednesday thursday friday saturday sunday zoom teams google microsoft "
    "english american russian israeli hebrew everyone everybody someone nobody speaker others "
    "dr mr mrs ms prof professor "
    # Russian
    "я это тут здесь рад рада да нет не думаю хорошо ладно тоже сейчас вот так понял "
    "поняла готов готова всё все уже просто очень согласен согласна "
    # Hebrew
    "חושב חושבת יודע יודעת לא כן פה כאן גם רוצה מצטער מצטערת בסדר שמח שמחה מבין מבינה "
    "זה רק עכשיו הולך הולכת צריך צריכה עייף עייפה אוהב אוהבת חושבים"
)
STOP = set(_STOP_WORDS.split())
_WORDS = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*|[^\w\s]")


@dataclass(frozen=True)
class Hint:
    name: str
    line: str  # the segment text it was said in


def _is_name(token: str) -> bool:
    parts = re.split(r"[-’']", token)
    if len(token) < 2 or token.casefold() in STOP or not all(p.isalpha() for p in parts):
        return False
    cased = token.lower() != token.upper()  # Latin/Cyrillic; Hebrew has no capitals
    return not cased or all(p[:1].isupper() and p[1:].islower() for p in parts)


def _name_after(text: str, needs: int) -> str | None:
    tokens = _WORDS.findall(text)
    title = []
    if tokens and tokens[0].casefold() in TITLES and tokens[0][:1].isupper():
        title, tokens = [tokens[0] + "."], tokens[2 if tokens[1:2] == ["."] else 1 :]
    if not tokens or not _is_name(tokens[0]):
        return None
    name, i = [tokens[0]], 1
    cased = tokens[0].lower() != tokens[0].upper()
    if cased and i < len(tokens) and tokens[i].casefold() not in FOLLOW and _is_name(tokens[i]):
        name, i = [*name, tokens[i]], 2  # a surname: "Anna Petrova"
    after = tokens[i].casefold() if i < len(tokens) else None
    if needs == FOLLOWED and after not in FOLLOW:
        return None  # "This is Paris.": a place as often as a person
    if needs == CLAUSE_END and after is not None and after.isalpha() and after not in FOLLOW:
        return None  # "I'm Anna Petrova Smith ..." or "Я Иду домой": not a clear name
    try:
        return check_name(" ".join(title + name))
    except ValueError:
        return None


def find_intro(text: str) -> str | None:
    """The name a speaker introduces themself with in `text`, or None."""
    for trigger, needs in TRIGGERS:
        for m in trigger.finditer(text):
            if name := _name_after(text[m.end() :], needs):
                return name
    return None


def intro_hints(segments: list[Segment], window: float = WINDOW) -> dict[str, Hint]:
    """`Speaker N` label -> the first self-introduction in its first `window` seconds of
    speech. `Me` and labels that are names already get none; a name that is already another
    label of the session isn't suggested."""
    taken = {s.speaker.casefold() for s in segments}
    first: dict[str, float] = {}
    hints: dict[str, Hint] = {}
    for s in sorted(segments, key=lambda s: s.start):
        label = s.speaker
        if label == ME or not NUMBERED.fullmatch(label.casefold()) or label in hints:
            continue
        if s.start - first.setdefault(label, s.start) > window:
            continue
        name = find_intro(s.text.strip())
        if name and name.casefold() not in taken:
            hints[label] = Hint(name, s.text.strip())
    return hints


def transcript_hints(directory: Path) -> dict[str, Hint]:
    """`intro_hints` of the session transcript in `directory` (reads it only)."""
    data = json.loads((Path(directory) / JSON_NAME).read_text(encoding="utf-8"))
    return intro_hints([Segment(**s) for s in data.get("segments", [])])


__all__ = ["WINDOW", "Hint", "find_intro", "intro_hints", "transcript_hints"]
