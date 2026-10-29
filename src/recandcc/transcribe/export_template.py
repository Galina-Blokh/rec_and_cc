"""Custom DOCX / PDF export templates (``export.template`` in config.toml).

A template is a small local TOML file::

    title    = "Meeting {date} {time} ({duration})"   # default: "Transcript {date} {time}"
    header   = "Confidential"                         # optional line under the title block
    footer   = "{app}, {languages}"                   # optional last line
    sections = ["action_items", "summary", "transcript"]   # which parts, in what order
    [headings]
    summary      = "Overview"
    action_items = "To do"
    transcript   = "Full text"

Placeholders are a plain ``{name}`` substitution (no attribute access, no format specs, no
eval): ``{date}``, ``{time}``, ``{duration}``, ``{title}``, ``{app}``, ``{speakers}``,
``{languages}``. Unknown names are left as written. A missing or invalid file logs one
warning (never transcript text) and the default layout is used. Offline.
"""

from __future__ import annotations

import logging
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..config import read_setting_file

log = logging.getLogger(__name__)

SECTIONS = ("summary", "action_items", "transcript")
_MAX_BYTES = 64_000
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class Template:
    title: str = "Transcript {date} {time}"
    header: str = ""
    footer: str = ""
    sections: tuple[str, ...] = SECTIONS
    headings: dict[str, str] = field(default_factory=dict)


def fill(text: str, values: dict[str, str]) -> str:
    """Replace known ``{name}`` placeholders; anything else stays literal."""
    return _PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def _strings(raw: dict, key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ValueError(key)
    return value


def parse(raw: dict) -> Template:
    """A Template from parsed TOML; ValueError if a key has the wrong type or an unknown
    section. Unknown top-level keys are ignored."""
    default = Template()
    sections = raw.get("sections", list(default.sections))
    if not isinstance(sections, list) or not sections or not all(s in SECTIONS for s in sections):
        raise ValueError("sections")
    headings = raw.get("headings", {})
    if not isinstance(headings, dict) or not all(
        k in SECTIONS and isinstance(v, str) for k, v in headings.items()
    ):
        raise ValueError("headings")
    return Template(
        title=_strings(raw, "title", default.title),
        header=_strings(raw, "header", ""),
        footer=_strings(raw, "footer", ""),
        sections=tuple(dict.fromkeys(sections)),
        headings={k: v.strip() for k, v in headings.items() if v.strip()},
    )


def load_template(setting: str | Path | None) -> Template | None:
    """The template named by ``export.template`` ("" / None = default layout). A relative
    path is taken from the config folder. Any problem logs one warning and returns None."""
    if not setting or not str(setting).strip():
        return None
    try:
        _, text = read_setting_file(setting, _MAX_BYTES)
        tpl = parse(tomllib.loads(text))
        if "transcript" not in tpl.sections:
            log.info("export template omits the transcript")
        return tpl
    except (
        OSError,
        ValueError,
        RuntimeError,
        UnicodeDecodeError,
    ) as e:  # TOMLDecodeError is a ValueError
        # the exception type only: a TOML error message can quote the file's text
        log.warning(
            "export template %s not used (%s); using the default layout",
            setting,
            type(e).__name__,
        )
        return None
