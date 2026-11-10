"""Names from direct address: "Sebastian, I have a question" and another speaker answers.

``addressed_names`` builds a conservative, session-only rename plan for the numbered labels
(``Speaker N``) of a transcript, for the post-transcription step next to ``confident_names``
(self-introductions). It never writes or logs anything: the matched text is transcript content.
A wrong name in a transcript is worse than a missing one, so the rules are strict:

- A name is *addressed* only at the start of a clause: "Sebastian, ...", "Okay, Anna, ..." or
  "and Donovan you were ...". "Nacho and Alex, I believe" and "Budget, scope and staffing"
  (lists), "I told Anna, ..." and reported speech after a colon are not. Only capitalised
  names count (no Hebrew), and never a common clause opener or filler, or a word that is
  mostly uncapitalised in the transcript ("Tomorrow, we ..." / "tomorrow").
- The addressee is the next *different* speaker, if they answer within a few seconds of the
  addressing speech. Only the last address of a turn counts.
- A label gets a name only when that name is the clear majority of the votes for the label
  (and the label the clear majority for the name, against *all* competing votes): two votes,
  or one vote when the name opens the turn and others capitalise it at least twice (a one-off
  garbled word like "Sustain, I believe." is not a name). One vote per answer: the last
  address before it. An answer that says the name itself ("Mira is absent") is no evidence.
  Ties, existing names and names another label already has are never assigned.
- ``Me`` is never renamed (it is the owner's mic track); it may address others.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection

from ..models import Segment
from .intro_hints import _is_name  # pyright: ignore[reportPrivateUsage]
from .voices import NUMBERED, check_name

REPLY_GAP = 4.0  # seconds from the end of the addressing speech to the answer
CONTINUE_MAX = 6.0  # the addressing speaker may keep talking this long before the answer
_TOKENS = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*|[^\w\s]")
_ALPHA = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*")
_END = frozenset(".!?;—–")  # not ":" (it introduces reported speech)
# Words that may sit between a clause start and the name: "Okay, Anna, ..."
_LEAD_WORDS = (
    "hey hi hello okay ok so well thanks thank you yes yeah right sorry perfect great good "
    "morning afternoon evening да ну так привет спасибо хорошо ладно"
)
_LEAD = frozenset(_LEAD_WORDS.split())
_ADDRESS_WORDS = "you your ты твой твоя твои вы ваш ваша ваши"
_AND_WORDS = "and or и или"
_AND = frozenset(_AND_WORDS.split())
_ADDRESS = frozenset(_ADDRESS_WORDS.split())
# Capitalised clause openers that are not names (the lowercase check catches most others)
_OPENER_WORDS = (
    "anyway anyways actually basically honestly generally typically personally frankly "
    "seriously obviously clearly exactly absolutely definitely certainly probably possibly "
    "maybe perhaps please finally firstly secondly lastly meanwhile however therefore thus "
    "also then now today tomorrow yesterday tonight later earlier again once still just only "
    "first second third next last sure fine okay alright cool nice awesome wow oh ah hmm "
    "yep yup yeah nope nah cheers bye goodbye ciao thanks thank okay "
    "tuesday monday wednesday thursday friday saturday sunday january february march april "
    "may june july august september october november december "
    "вообще кстати конечно например итак значит сегодня завтра вчера сейчас потом "
    "тогда затем наконец возможно наверное"
)
_OPENERS = frozenset(_OPENER_WORDS.split())


def _starts_clause(tokens: list[str], i: int, lead: frozenset[str]) -> bool:
    j = i - 1
    while j >= 0 and (tokens[j] == "," or tokens[j].casefold() in lead):
        j -= 1
    return j < 0 or tokens[j] in _END


def _is_list(tokens: list[str], i: int) -> bool:
    """Whether the words from ``i`` continue a list: "scope and ...", "Omar, and ..."."""
    rest = [t.casefold() for t in tokens[i : i + 3]]
    return bool(rest) and (rest[0] in _AND or (len(rest) > 1 and rest[1] in _AND))


def _opens_turn(tokens: list[str], i: int) -> bool:
    return all(t == "," or t.casefold() in _LEAD for t in tokens[:i])


def addresses(text: str, lowercase: Collection[str] = ()) -> list[tuple[str, bool]]:
    """(name, opens_the_turn) for each name addressed in ``text``, in order. ``lowercase``:
    words that are mostly uncapitalised in the transcript, which are not names."""
    tokens = _TOKENS.findall(text)
    found: list[tuple[str, bool]] = []
    for i, token in enumerate(tokens):
        folded = token.casefold()
        if (
            token.lower() == token.upper()  # no capitals (Hebrew, digits): can't tell a name
            or folded in _OPENERS
            or folded in _LEAD
            or folded in lowercase
            or folded.endswith("ly")
            or not _is_name(token)
        ):
            continue
        nxt = tokens[i + 1].casefold() if i + 1 < len(tokens) else ""
        if nxt == "," and _is_list(tokens, i + 2):
            continue  # "Budget, scope and staffing": a list, not an address
        if nxt == "," and _starts_clause(tokens, i, _LEAD):
            found.append((token, _opens_turn(tokens, i)))
        elif nxt in _ADDRESS and _starts_clause(tokens, i, _LEAD | {"and", "и"}):
            found.append((token, False))
    return found


def _replier(segments: list[Segment], i: int) -> int | None:
    """Index of the first segment of another speaker after ``segments[i]``, when it follows
    closely."""
    speaker, end = segments[i].speaker, segments[i].end
    for j in range(i + 1, min(i + 8, len(segments))):
        nxt = segments[j]
        if nxt.speaker == speaker:
            if nxt.start - end > CONTINUE_MAX:
                return None
            end = max(end, nxt.end)
            continue
        return None if nxt.overlap or nxt.start - end > REPLY_GAP else j
    return None


def addressed_names(
    segments: list[Segment],
    *,
    single_labels: Collection[str] = (),
    skip: Collection[str] = (),
    reserved: Collection[str] = (),
) -> dict[str, str]:
    """`Speaker N` label -> the name the others address it by. ``skip``: labels that are
    named already; ``reserved``: names that are taken. ``single_labels``: tracks known to
    have one speaker, which may be renamed like numbered labels."""
    ordered = sorted(segments, key=lambda s: s.start)
    taken = {s.speaker.casefold() for s in ordered} | {n.casefold() for n in reserved}
    capitalised: Counter[str] = Counter()
    capitalised_by: dict[str, Counter[str]] = defaultdict(Counter)
    plain: Counter[str] = Counter()
    for s in ordered:
        for word in _ALPHA.findall(s.text):
            folded = word.casefold()
            if word == word.lower():
                plain[folded] += 1
            else:
                capitalised[folded] += 1
                capitalised_by[folded][s.speaker] += 1
    lowercase = {w for w, n in plain.items() if n > capitalised[w]}  # mostly not a name
    # One vote per answer: the last address before it (a turn may be cut into many segments)
    asked: dict[int, tuple[str, bool]] = {}
    for i, seg in enumerate(ordered):
        found = addresses(seg.text.strip(), lowercase)
        reply = _replier(ordered, i) if found and not seg.overlap else None
        if reply is not None:
            asked[reply] = found[-1]  # the last address of a turn is the one that gets answered
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_name: dict[str, Counter[str]] = defaultdict(Counter)
    opening: set[tuple[str, str]] = set()
    shown: dict[str, str] = {}
    for j, (name, opens) in asked.items():
        label = ordered[j].speaker
        if label in skip or not (NUMBERED.fullmatch(label.casefold()) or label in single_labels):
            continue
        folded = name.casefold()
        if folded in {w.casefold() for w in _ALPHA.findall(ordered[j].text)}:
            continue  # "Mira is absent": an answer that says the name is not Mira answering
        shown.setdefault(folded, name)
        by_label[label][folded] += 1
        by_name[folded][label] += 1
        if opens:
            opening.add((label, folded))
    plan: dict[str, str] = {}
    for label, counts in by_label.items():
        (folded, votes), *rest = counts.most_common()
        if votes < 2 * sum(n for _, n in rest):
            continue  # a tie or no clear majority over all competing names
        said_by_others = capitalised[folded] - capitalised_by[folded][label]
        if votes < 2 and ((label, folded) not in opening or said_by_others < 2):
            continue
        if folded in taken or votes < 2 * sum(
            n for lbl, n in by_name[folded].items() if lbl != label
        ):
            continue
        try:
            plan[label] = check_name(shown[folded])
        except ValueError:
            continue
    # Two labels never get the same name (their votes were checked per name above)
    counts = Counter(name.casefold() for name in plan.values())
    return {label: name for label, name in plan.items() if counts[name.casefold()] == 1}


__all__ = ["addressed_names", "addresses"]
