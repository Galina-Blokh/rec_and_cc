"""Self-introduction hints for speaker names ("I'm Anna" -> suggested: Anna), CLI only.

Stories: US-21.
"""

from __future__ import annotations

import logging

import pytest
from test_transcribe import make_session

from recandcc import cli
from recandcc.config import Config, TranscriptionConfig
from recandcc.models import Segment
from recandcc.transcribe import voices
from recandcc.transcribe.intro_hints import Hint, find_intro, intro_hints
from recandcc.transcribe.output import write_outputs


def seg(start, speaker, text, language="en"):
    return Segment(start, start + 2, text, speaker, language)


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("Hi, I'm Anna from sales.", "Anna"),
        ("Hello I am Bob.", "Bob"),
        ("My name is Anna Petrova and I run the team", "Anna Petrova"),
        ("Hi everyone, this is Carl speaking.", "Carl"),
        ("Меня зовут Анна.", "Анна"),
        ("Привет, я Анна, из отдела продаж.", "Анна"),
        ("Привет, это Борис.", "Борис"),
        ("Это Борис из бухгалтерии.", "Борис"),
        ("Hello. This is Anna speaking.", "Anna"),
        ("Hi all, I'm Dr. Smith.", "Dr. Smith"),
        ("I'm Prof Lee from the lab", "Prof. Lee"),
        ("קוראים לי דנה", "דנה"),
        ("שלום, קוראים לי יוסי.", "יוסי"),
        ("שמי מיכל ואני מהצוות", "מיכל"),
        ("השם שלי רון", "רון"),
    ],
)
def test_introductions_in_three_languages(text, name):
    assert find_intro(text) == name


@pytest.mark.parametrize(
    "text",
    [
        "I'm sure we can do that.",
        "I'm Sorry, but no.",
        "I am Back.",
        "I'm going to share my screen",
        "I think this is Anna's laptop.",
        "This is Great.",
        "my name is Speaker 2",
        "Я думаю, что да.",
        "Я Иду домой.",
        "Это хорошо.",
        "Это Понял?",
        "I'm Late.",
        "Sorry, I'm Busy.",
        "I'm Dr.",
        "This is Paris.",  # a place as often as a person
        "Так, это Москва.",
        "Это Анна.",
        "אני חושב שכן.",
        "שלום, אני חושב שכן.",
        "שלום, אני מדבר.",  # "אני" + a verb or adjective, far more often than a name
        "שלום, אני דנה.",
        "",
    ],
)
def test_ordinary_sentences_are_not_names(text):
    assert find_intro(text) is None


def test_one_hint_per_label_only_in_its_first_minutes_and_never_me():
    segments = [
        seg(0, "Me", "Hi, I'm Galina."),
        seg(1, "Speaker 1", "Good morning."),
        seg(5, "Speaker 1", "I'm Anna."),
        seg(9, "Speaker 1", "My name is Olga."),  # a second intro: the first one wins
        seg(300, "Speaker 2", "Hello."),
        seg(430, "Speaker 2", "My name is Bob."),  # 130 s after Speaker 2 first spoke
        seg(20, "Speaker 3", "Привет, это Анна."),  # hints are per label
        seg(30, "Carl", "I'm Carl."),  # already named
        seg(40, "Others", "I'm Dora."),  # not diarized
    ]
    hints = intro_hints(segments)
    assert hints == {
        "Speaker 1": Hint("Anna", "I'm Anna."),
        "Speaker 3": Hint("Анна", "Привет, это Анна."),
    }
    assert intro_hints(segments, window=200)["Speaker 2"].name == "Bob"


def test_a_name_that_is_already_a_label_is_not_suggested():
    assert intro_hints([seg(0, "Anna", "hi"), seg(1, "Speaker 1", "I'm Anna.")]) == {}


def _transcript(tmp_path, segments):
    session = make_session(tmp_path)
    write_outputs(session, segments, 60.0, "test")


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_shows_hints_stores_nothing_and_logs_no_text(
    tmp_path, monkeypatch, capsys, caplog, enabled
):
    out = tmp_path / "session"
    out.mkdir()
    _transcript(out, [seg(0, "Speaker 1", "Hi, I'm Anna from sales."), seg(3, "Me", "I'm Galina.")])
    cfg = Config(transcription=TranscriptionConfig(speaker_names=enabled))
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    caplog.set_level(logging.DEBUG)
    before = sorted(voices.VOICES_DIR.rglob("*"))
    files = sorted(out.iterdir())
    mtimes = [f.stat().st_mtime_ns for f in files]
    assert cli.main(["speakers", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "suggested: Anna (said: Hi, I'm Anna from sales.)" in printed
    assert "Galina" not in printed
    assert sorted(voices.VOICES_DIR.rglob("*")) == before  # no voice data, no profile
    assert sorted(out.iterdir()) == files and [f.stat().st_mtime_ns for f in files] == mtimes
    assert voices.load_profiles() == []
    assert not any("Anna" in r.getMessage() for r in caplog.records)
