"""Summary templates (``summary.template``): parsing, the fallback to the default layout, one
model call per section, the layout of summary.md, and how the export reads the result.

Stories: US-27.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

import pytest

from recandcc import config as config_mod
from recandcc.config import Config, load_config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import summary
from recandcc.transcribe.export import _split_summary
from recandcc.transcribe.output import PDI, RLI, write_outputs
from recandcc.transcribe.summary_template import (
    MAX_BYTES,
    MAX_SECTIONS,
    NO_SECTION,
    TOO_MANY,
    UNNAMED,
    SummaryTemplate,
    load_template,
    parse,
)

TEMPLATE = (
    "# Meeting summary\n"
    "## Decisions\nOne bullet per decision.\n"
    "## Action items\nOne bullet per task.\n"
)
ASKED = re.compile(r"Write only the '(.+?)' part")


class FakeSections:
    """Answers each template section by its heading; counts the notes calls apart."""

    def __init__(self, answers: dict[str, str] | None = None):
        self.answers = answers or {}
        self.asked: list[str] = []
        self.notes = 0

    def __call__(self, messages, max_tokens):
        found = ASKED.search(messages[-1]["content"])
        if found is None:  # the notes pass of a long transcript
            self.notes += 1
            return "- a note"
        self.asked.append(found.group(1))
        return self.answers.get(found.group(1), "NONE")

    def close(self):
        pass


def seg(text: str, t: float = 0, speaker: str = "Me", lang: str = "en") -> Segment:
    return Segment(t, t + 2, text, speaker, lang)


def make_session(tmp_path: Path, segments: list[Segment]) -> Path:
    d = tmp_path / "2026-10-29" / "100000_zoom"
    d.mkdir(parents=True)
    s = SessionInfo("s1", d, datetime(2026, 10, 29, 10), TriggerKind.CALL, "zoom.us")
    s.save()
    write_outputs(s, segments, 60.0, "fake")
    return d


def test_parse_keeps_the_title_and_the_sections_in_order():
    assert parse(TEMPLATE) == SummaryTemplate(
        "Meeting summary",
        (("Decisions", "One bullet per decision."), ("Action items", "One bullet per task.")),
    )


def test_text_before_the_first_section_is_a_note_not_an_instruction():
    tpl = parse("Notes to myself.\n# Title\n### not a section\n## Summary\nKey points.\n")
    assert tpl == SummaryTemplate("Title", (("Summary", "Key points."),))


def test_lines_under_a_heading_form_one_instruction():
    tpl = parse("## Decisions\n- one\n\n   two  \n### sub\n")
    assert tpl.sections == (("Decisions", "- one two ### sub"),)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("# Only a title\n", NO_SECTION),
        ("## \nSomething\n", UNNAMED),
        ("## a\n" * (MAX_SECTIONS + 1), TOO_MANY),
    ],
)
def test_a_template_without_usable_sections_is_refused(text, reason):
    with pytest.raises(ValueError) as refused:
        parse(text)
    assert str(refused.value) == reason


def test_the_example_template_in_the_docs_parses():
    text = (Path(__file__).parents[1] / "docs" / "examples" / "summary-template.md").read_text(
        "utf-8"
    )
    tpl = parse(text)
    assert tpl.title == "Meeting summary" and len(tpl.sections) == 5


def test_no_setting_is_the_default_layout(caplog):
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_template("") is None and load_template("   ") is None
    assert not caplog.records


def test_a_relative_path_is_read_from_the_config_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    (tmp_path / "meeting.md").write_text(TEMPLATE, "utf-8")
    assert load_template("meeting.md") == parse(TEMPLATE)


def test_a_missing_file_warns_once_and_is_the_default(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_template(str(tmp_path / "nope.md")) is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "not used (not a file)" in warnings[0]


def test_a_bad_template_warns_without_quoting_it(tmp_path, caplog):
    path = tmp_path / "bad.md"
    path.write_text("secret words, no section\n", "utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_template(str(path)) is None
    assert NO_SECTION in caplog.text and "secret words" not in caplog.text


def test_an_oversized_template_is_refused(tmp_path, caplog):
    path = tmp_path / "big.md"
    path.write_text("## a\n" + "x" * MAX_BYTES, "utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_template(str(path)) is None
    assert "too large" in caplog.text


def test_a_template_that_is_not_utf8_warns_by_type_only(tmp_path, caplog):
    path = tmp_path / "latin.md"
    path.write_bytes(b"## Caf\xe9\n")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_template(str(path)) is None
    assert "UnicodeDecodeError" in caplog.text


def test_each_section_gets_one_call_on_the_same_transcript():
    gen = FakeSections({"Decisions": "- Ship on Friday", "Action items": "NONE"})
    language, answers = summary.summarize_sections(
        [seg("We ship on Friday.")], gen, parse(TEMPLATE).sections
    )
    assert gen.asked == ["Decisions", "Action items"] and gen.notes == 0
    assert answers == [("Decisions", "- Ship on Friday"), ("Action items", "NONE")]
    assert language == "en"


def test_a_long_transcript_is_reduced_before_the_sections(monkeypatch):
    monkeypatch.setattr(summary, "CHUNK_CHARS", 200)
    segs = [seg(f"point number {i} " * 4, t=i * 5) for i in range(30)]
    gen = FakeSections()
    summary.summarize_sections(segs, gen, parse(TEMPLATE).sections)
    assert gen.notes >= 5 and gen.asked == ["Decisions", "Action items"]


def test_an_empty_transcript_gets_no_calls(tmp_path):
    gen = FakeSections({"Decisions": "- Ship"})
    _, answers = summary.summarize_sections([], gen, parse(TEMPLATE).sections)
    assert gen.asked == [] and answers == [("Decisions", ""), ("Action items", "")]


@pytest.mark.parametrize(
    "lang, none",
    [("en", "None."), ("ru", "Нет."), ("he", f"{RLI}אין.{PDI}")],
)
@pytest.mark.parametrize("answer", ["NONE", "**none**", "Nothing."])
def test_a_none_answer_is_the_languages_none(lang, none, answer):
    out = summary.format_sections(lang, [("Decisions", answer)], "x/Qwen3.5-4B", "Meeting summary")
    assert f"## Decisions\n\n{none}\n" in out


def test_the_models_own_headings_are_not_repeated():
    out = summary.format_sections("en", [("Decisions", "## Decisions\n\n- Ship it")], "m", "T")
    assert out.count("## Decisions") == 1 and "- Ship it" in out


def test_right_to_left_list_marks_stay_outside_the_isolate():
    out = summary.format_sections("he", [("משימות", "- לשלוח הערות")], "m", "סיכום")
    assert f"## משימות\n\n- {RLI}לשלוח הערות{PDI}\n" in out


def test_without_a_title_the_language_default_is_used():
    out = summary.format_sections("ru", [("Решения", "- да")], "m")
    assert out.startswith("# Краткое содержание\n\n## Решения\n\n- да\n")


def test_summary_md_follows_the_template(tmp_path, caplog):
    d = make_session(tmp_path, [seg("We ship on Friday; Anna sends the deck.")])
    tpl = tmp_path / "meeting.md"
    tpl.write_text(TEMPLATE, "utf-8")
    cfg = Config()
    cfg.summary.template = str(tpl)
    gen = FakeSections({"Decisions": "- Ship on Friday", "Action items": "- Anna: send the deck"})
    with caplog.at_level(logging.INFO, logger="recandcc"):
        path = summary.summarize_session(d, cfg, gen)
    text = path.read_text("utf-8")
    assert text.startswith(
        "# Meeting summary\n\n## Decisions\n\n- Ship on Friday\n\n"
        "## Action items\n\n- Anna: send the deck\n\n"
    )
    assert gen.asked == ["Decisions", "Action items"]
    assert "by Qwen3.5-4B-MLX-4bit;" in text  # the footer names the model that wrote it
    assert "written: 2 section(s)" in caplog.text


def test_an_invalid_template_gives_the_default_summary(tmp_path, caplog):
    d = make_session(tmp_path, [seg("We agreed on the plan.")])
    bad = tmp_path / "bad.md"
    bad.write_text("no sections here\n", "utf-8")
    cfg = Config()
    cfg.summary.template = str(bad)

    def default_model(messages, max_tokens):
        user = messages[-1]["content"]
        if "List every task" in user:
            return "- Anna: send the deck"
        return "We agreed on the plan."

    with caplog.at_level(logging.WARNING, logger="recandcc"):
        path = summary.summarize_session(d, cfg, default_model)
    text = path.read_text("utf-8")
    assert text.startswith("# Summary\n\nWe agreed on the plan.\n\n## Action items\n\n- Anna")
    assert NO_SECTION in caplog.text


def test_export_takes_only_a_trailing_action_items_section_off_a_template():
    blocks = [
        ("h1", "Meeting summary"),
        ("h2", "Decisions"),
        ("bullet", "Ship"),
        ("h2", "Action items"),
        ("bullet", "Anna: deck"),
    ]
    parts = _split_summary(blocks)
    assert parts["summary"] == blocks[:3] and parts["action_items"] == blocks[3:]


def test_export_keeps_template_sections_after_an_action_items_one_in_the_summary():
    blocks = [
        ("h1", "Meeting summary"),
        ("h2", "Action items"),
        ("bullet", "Anna: deck"),
        ("h2", "Topics"),
        ("p", "Budget?"),
    ]
    assert _split_summary(blocks) == {"summary": blocks, "action_items": [], "note": []}


def test_the_default_layout_still_splits_at_its_action_items():
    note = ("note", "Generated on this computer")
    blocks = [("h1", "Краткое содержание"), ("p", "Итог."), ("h2", "Задачи"), ("bullet", "Анна")]
    parts = _split_summary([*blocks, note])
    assert parts == {"summary": blocks[:2], "action_items": blocks[2:], "note": [note]}


def test_a_summary_template_path_is_kept_as_written(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text('[summary]\ntemplate = "~/meeting.md"\n', "utf-8")
    assert load_config(path).summary.template == "~/meeting.md"
    path.write_text("[summary]\ntemplate = 3\n", "utf-8")
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert load_config(path).summary.template == ""
    assert "summary.template must be a path string" in caplog.text
