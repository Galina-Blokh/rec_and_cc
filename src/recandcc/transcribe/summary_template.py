"""Summary templates (``summary.template`` in config.toml): the sections of ``summary.md``. Each
``## `` heading is one section; the lines under it are the instruction the model follows for that
section::

    # Meeting summary
    ## Decisions
    One bullet per decision made. Write NONE if there are none.
    ## Action items
    One bullet per task: "- <who>: <task>", with the deadline if one is said.

The first ``# `` line is the document's title. Other text before the first section is ignored,
so a note to yourself can go there. Empty (the default) is the built-in template: the six
sections of ``default_summary_template.md``, which ``DEFAULT_SUMMARY_PROMPT`` names in the same
order. A missing or invalid file logs one warning (the reason, never transcript text) and the
built-in template is used. Read at each summary. Offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from ..config import read_setting_file

log = logging.getLogger(__name__)

MAX_BYTES = 16_000  # a few lines of instructions; each section's prompt also carries the transcript
MAX_SECTIONS = 10  # one model call per section, so this bounds a summary's run time
NO_SECTION = "the template has no ## section"
UNNAMED = "a ## heading has no name"
TOO_MANY = f"the template has more than {MAX_SECTIONS} sections"
BUILTIN_FILE = "default_summary_template.md"


@dataclass(frozen=True)
class SummaryTemplate:
    title: str  # "" = the language's default title
    sections: tuple[tuple[str, str], ...]  # (heading, instruction) in order; "" = no instruction


def parse(text: str) -> SummaryTemplate:
    """The template in ``text``. ValueError (one of the fixed messages above) if it has no
    section, a section without a name, or more than MAX_SECTIONS sections."""
    title = ""
    sections: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line == "##" or line.startswith("## "):
            heading = line[2:].strip()
            if not heading:
                raise ValueError(UNNAMED)
            sections.append((heading, []))
        elif sections:
            sections[-1][1].append(line)
        elif line.startswith("# ") and not title:
            title = line[2:].strip()
    if not sections:
        raise ValueError(NO_SECTION)
    if len(sections) > MAX_SECTIONS:
        raise ValueError(TOO_MANY)
    return SummaryTemplate(
        title=title,
        sections=tuple((heading, " ".join(" ".join(body).split())) for heading, body in sections),
    )


@cache
def builtin() -> SummaryTemplate:
    """The built-in template: the package resource ``default_summary_template.md``."""
    text = resources.files("recandcc.transcribe").joinpath(BUILTIN_FILE).read_text("utf-8")
    return parse(text)


def load_template(setting: str | Path) -> SummaryTemplate:
    """The template named by ``summary.template``; "" is the built-in one. Any problem logs one
    warning and gives the built-in one."""
    if not str(setting).strip():
        return builtin()
    try:
        _, text = read_setting_file(setting, MAX_BYTES)
        return parse(text)
    except (OSError, ValueError, RuntimeError, UnicodeDecodeError) as e:
        # our own messages are fixed; any other error is named by its type (it can quote the file)
        reason = str(e) if type(e) is ValueError else type(e).__name__
        log.warning(
            "summary template %s not used (%s); using the built-in sections", setting, reason
        )
        return builtin()
