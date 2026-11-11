"""Speaker names read from the conversation by the summary model, written into the transcript
before the summary (owner request: "extract names from the transcript, replace them in the
transcript, then use them in the summary").

Synthetic transcripts only (invented names), never fixtures from real recordings.
Stories: US-21, US-27.
"""

import json
import logging
from datetime import datetime

import pytest

from recandcc.config import Config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import summary
from recandcc.transcribe.output import write_outputs
from recandcc.transcribe.speaker_names import (
    checked_names,
    eligible_labels,
    infer_names,
    parse_answer,
)

KEY = "sk-test-do-not-leak"


def serve(monkeypatch, *replies) -> list:
    """Answer cloud requests in turn with these JSON bodies; returns the requests seen."""
    import contextlib
    import io
    from types import SimpleNamespace

    from recandcc.transcribe import summary_cloud

    seen, queue = [], list(replies)

    def fake_open(req, timeout=None):
        seen.append(req)
        return contextlib.nullcontext(io.BytesIO(json.dumps(queue.pop(0)).encode()))

    monkeypatch.setattr(summary_cloud, "_OPENER", SimpleNamespace(open=fake_open))
    return seen


def seg(start, speaker, text):
    return Segment(start, start + 2, text, speaker, "en")


MEETING = [
    seg(0, "Me", "Good morning. Let's start with the status of the release."),
    seg(4, "Speaker 1", "I can start. The build is green and the docs are in review."),
    seg(9, "Me", "Thanks. And the client side?"),
    seg(12, "Speaker 2", "This is Tobias, the mobile app is ready for the handover."),
    seg(18, "Speaker 1", "Great work, Tobias. I'm Mira, I will send the release notes today."),
]


def ok(name, quote):
    return (name, quote)


# --- parsing


def test_the_last_json_object_of_labels_is_read_and_notes_around_it_ignored():
    answer = (
        'Thinking about {"x": 1} first.\n'
        '{"Speaker 2": {"name": "Tobias", "evidence": "This is Tobias"}}\nDone.'
    )
    assert parse_answer(answer) == {"Speaker 2": ("Tobias", "This is Tobias")}


@pytest.mark.parametrize(
    "answer",
    ["no json here", "{}", '{"Speaker 1": "Mira"}', '{"Speaker 1": {"name": 3, "evidence": "x"}}'],
)
def test_answers_without_a_name_and_evidence_give_nothing(answer):
    assert parse_answer(answer) == {}


# --- checks


def test_a_name_with_a_real_quote_is_accepted():
    answer = {
        "Speaker 2": ok("Tobias", "This is Tobias, the mobile app"),
        "Speaker 1": ok("Mira", "I'm Mira, I will send the release notes"),
    }
    assert checked_names(answer, MEETING) == {"Speaker 2": "Tobias", "Speaker 1": "Mira"}


@pytest.mark.parametrize(
    "label, name, quote",
    [
        ("Speaker 1", "Sebastian", "I can start. The build is green"),  # name not in transcript
        ("Speaker 2", "Tobias", "Tobias said the app is ready"),  # quote not in transcript
        ("Speaker 1", "Mira", "The build is green and the docs"),  # quote without the name
        ("Speaker 2", "Tobias", "Tobias"),  # a quote too short to prove anything
        ("Me", "Mira", "I'm Mira, I will send the release notes"),  # the owner is never renamed
        ("Speaker 7", "Mira", "I'm Mira, I will send the release notes"),  # no such label
        ("Speaker 1", "Mira Tobias Good Morning", "I'm Mira, I will send"),  # too many words
        ("Speaker 1", "Speaker 3", "I'm Mira, I will send"),  # a reserved label
    ],
)
def test_names_that_cannot_be_checked_are_rejected(label, name, quote):
    assert checked_names({label: ok(name, quote)}, MEETING) == {}


@pytest.mark.parametrize(
    "label, name, quote",
    [
        # Speaker 1 only *mentions* Tobias: a mention is no evidence for its own speaker
        ("Speaker 1", "Tobias", "Great work, Tobias"),
        # addressed, but the one who answers / spoke just before is someone else
        ("Speaker 2", "Mira", "I'm Mira, I will send the release notes"),
    ],
)
def test_a_quote_must_show_that_this_speaker_has_the_name(label, name, quote):
    assert checked_names({label: ok(name, quote)}, MEETING) == {}


def test_an_address_counts_for_the_one_who_answers_or_just_spoke():
    segments = [
        seg(0, "Me", "Orla, can you give us the status?"),
        seg(3, "Speaker 1", "Sure, the build is green."),
        seg(6, "Speaker 2", "The app is ready too."),
        seg(9, "Me", "Thank you very much. Pim?"),
        seg(12, "Speaker 3", "Hello."),
        seg(15, "Speaker 1", "Remind me later about the report from Pim."),
    ]
    answer = {
        "Speaker 1": ok("Orla", "Orla, can you give us the status"),
        "Speaker 3": ok("Pim", "Thank you very much. Pim?"),
        "Speaker 2": ok("Pim", "the report from Pim"),  # a mention: rejected
    }
    assert checked_names(answer, segments) == {"Speaker 1": "Orla", "Speaker 3": "Pim"}


def test_two_labels_with_one_name_are_both_dropped():
    segments = [
        seg(0, "Speaker 1", "Hi, I'm Tobias from sales."),
        seg(3, "Speaker 2", "Hello, I'm Tobias from the mobile team."),
    ]
    answer = {
        "Speaker 1": ok("Tobias", "I'm Tobias from sales"),
        "Speaker 2": ok("Tobias", "I'm Tobias from the mobile team"),
    }
    assert checked_names(answer, segments) == {}


def test_a_name_another_speaker_introduced_themself_with_is_not_given_away():
    segments = [*MEETING, seg(30, "Speaker 3", "My name is Mira, I joined late.")]
    answer = {"Speaker 1": ok("Mira", "I'm Mira, I will send the release notes")}
    assert checked_names(answer, segments) == {}


def test_named_labels_are_left_alone_and_others_only_in_a_one_to_one_call():
    named = [seg(0, "Me", "Hi Anna, how are you?"), seg(3, "Anna", "Fine, thanks Anna.")]
    assert eligible_labels(named) == []
    one_to_one = [seg(0, "Me", "Hi Anna, how are you?"), seg(3, "Others", "Fine, thanks.")]
    assert eligible_labels(one_to_one) == ["Others"]
    assert checked_names({"Others": ok("Anna", "Hi Anna, how are you")}, one_to_one) == {
        "Others": "Anna"
    }
    mixed = [*one_to_one, seg(6, "Speaker 1", "Hello.")]
    assert eligible_labels(mixed) == ["Speaker 1"]


def test_infer_names_asks_once_and_skips_without_eligible_labels():
    asked = []

    def gen(messages, max_tokens):
        asked.append(messages)
        return '{"Speaker 2": {"name": "Tobias", "evidence": "This is Tobias"}}'

    assert infer_names(MEETING, gen, max_chars=10_000, max_tokens=300) == {"Speaker 2": "Tobias"}
    assert len(asked) == 1 and "[00:00:12] Speaker 2: This is Tobias" in asked[0][1]["content"]
    assert infer_names([seg(0, "Me", "Hello.")], gen, max_chars=10_000, max_tokens=300) == {}
    assert len(asked) == 1


# --- end to end: transcript renamed, then summarised with the names


def session(tmp_path, segments=MEETING):
    info = SessionInfo("names", tmp_path, datetime(2026, 11, 11, 10), TriggerKind.CALL, "teams")
    info.save()
    write_outputs(info, segments, 30.0, "fake")
    return tmp_path


NAMES = json.dumps(
    {
        "Speaker 1": {"name": "Mira", "evidence": "I'm Mira, I will send the release notes"},
        "Speaker 2": {"name": "Tobias", "evidence": "This is Tobias, the mobile app"},
    }
)


def test_names_are_written_into_the_transcript_and_the_summary_uses_them(tmp_path, caplog):
    d = session(tmp_path)
    prompts = []

    def gen(messages, max_tokens):
        prompts.append(messages[-1]["content"])
        return NAMES if len(prompts) == 1 else "- Mira: send the release notes [00:00:18]"

    cfg = Config(output_dir=tmp_path)
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary.summarize_session(d, cfg, gen).read_text("utf-8")
    data = json.loads((d / "transcript.json").read_text("utf-8"))
    assert {s["speaker"] for s in data["segments"]} == {"Me", "Mira", "Tobias"}
    assert "Tobias:" in (d / "transcript.md").read_text("utf-8")
    later = "\n".join(prompts[1:])
    assert "Mira:" in later and "Tobias:" in later and "Speaker 1" not in later
    assert "Mira" in text
    assert "2 speaker name(s) from the conversation" in caplog.text
    assert "Tobias" not in caplog.text  # names are transcript content: never logged


@pytest.mark.parametrize(
    "answer", ["I am not sure.", '{"Speaker 1": {"name": "Sebastian", "evidence": "nope"}}']
)
def test_an_unusable_answer_keeps_the_labels_and_still_summarises(tmp_path, answer):
    d = session(tmp_path)
    calls = []

    def gen(messages, max_tokens):
        calls.append(1)
        return answer if len(calls) == 1 else "- Speaker 1: send the notes"

    path = summary.summarize_session(d, Config(output_dir=tmp_path), gen)
    data = json.loads((d / "transcript.json").read_text("utf-8"))
    assert {s["speaker"] for s in data["segments"]} == {"Me", "Speaker 1", "Speaker 2"}
    assert path.exists()


def test_a_failing_naming_call_does_not_stop_the_summary(tmp_path, caplog):
    d = session(tmp_path)
    calls = []

    def gen(messages, max_tokens):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("model fell over")
        return "- Speaker 1: send the notes"

    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert summary.summarize_session(d, Config(output_dir=tmp_path), gen).exists()
    assert "speaker names for" in caplog.text and "RuntimeError" in caplog.text


def test_naming_is_off_with_names_from_transcript_false(tmp_path):
    d = session(tmp_path)
    prompts = []

    def gen(messages, max_tokens):
        prompts.append(messages[0]["content"])
        return NAMES

    cfg = Config(output_dir=tmp_path)
    cfg.transcription.names_from_transcript = False
    summary.summarize_session(d, cfg, gen)
    assert not any(p.startswith("You identify the speakers") for p in prompts)


def test_a_cloud_summary_names_speakers_in_a_first_request(tmp_path, monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    seen = serve(
        monkeypatch,
        {"output_text": NAMES},
        {"output_text": "## Decisions\n- Mira sends the release notes [00:00:18]."},
    )
    d = session(tmp_path)
    cfg = Config(output_dir=tmp_path)
    cfg.summary.provider = "openai"
    text = summary.summarize_session(d, cfg).read_text("utf-8")
    assert len(seen) == 2  # names, then the summary: both carry this session's transcript only
    second = json.loads(seen[1].data)
    assert "Mira:" in json.dumps(second) and "Speaker 1:" not in json.dumps(second)
    assert "Mira sends the release notes" in text
    data = json.loads((d / "transcript.json").read_text("utf-8"))
    assert {s["speaker"] for s in data["segments"]} == {"Me", "Mira", "Tobias"}


# --- regressions from the Codex review (gpt-6.1-sol): each was a wrong name before


def test_an_others_track_that_introduces_two_names_is_not_one_person():
    segments = [
        seg(0, "Me", "Welcome."),
        seg(3, "Others", "My name is Alice."),
        seg(6, "Others", "My name is Bob."),
    ]
    assert checked_names({"Others": ok("Alice", "My name is Alice")}, segments) == {}


def test_quoted_speech_is_not_a_self_introduction():
    segments = [seg(0, "Me", "Who called?"), seg(3, "Speaker 1", 'He said, "My name is Bob."')]
    assert checked_names({"Speaker 1": ok("Bob", "My name is Bob")}, segments) == {}


def test_the_quote_must_hold_the_whole_name():
    segments = [
        seg(0, "Me", "Smith sent the report."),
        seg(3, "Speaker 1", "For late arrivals, this is Alice from QA."),
    ]
    answer = {"Speaker 1": ok("Alice Smith", "this is Alice from QA")}
    assert checked_names(answer, segments) == {}


def test_a_call_on_someone_names_the_one_who_answers_not_the_one_before():
    segments = [
        seg(0, "Speaker 1", "The server is fixed."),
        seg(3, "Me", "Alice, can you give the client update?"),
        seg(6, "Speaker 2", "Sure, the client is ready."),
    ]
    quote = "Alice, can you give the client update"
    assert checked_names({"Speaker 1": ok("Alice", quote)}, segments) == {}
    assert checked_names({"Speaker 2": ok("Alice", quote)}, segments) == {"Speaker 2": "Alice"}
    late = [*segments[:2], seg(20, "Speaker 2", "Sure, the client is ready.")]
    assert checked_names({"Speaker 2": ok("Alice", quote)}, late) == {}  # no answer in time
    # the caller goes on talking first: the answer is timed from the end of their turn
    goes_on = [*segments[:2], seg(6, "Me", "And the dates too."), seg(10, "Speaker 2", "Sure.")]
    assert checked_names({"Speaker 2": ok("Alice", quote)}, goes_on) == {"Speaker 2": "Alice"}


def test_names_are_not_applied_to_a_transcript_that_changed_meanwhile(tmp_path):
    from recandcc.transcribe import names

    d = session(tmp_path)
    checked = [Segment(**s) for s in json.loads((d / "transcript.json").read_text())["segments"]]
    changed = [*MEETING[:3], seg(12, "Speaker 2", "This is Bob, from finance.")]
    write_outputs(SessionInfo.load(d), changed, 30.0, "fake")
    assert names.apply_names(d, {"Speaker 2": "Tobias"}, checked_on=checked) == {}
    data = json.loads((d / "transcript.json").read_text("utf-8"))
    assert "Tobias" not in {s["speaker"] for s in data["segments"]}


def test_the_fingerprint_is_of_the_text_the_summary_read(tmp_path):
    d = session(tmp_path)
    before = json.loads((d / "transcript.json").read_text("utf-8"))["segments"]
    calls = []

    def gen(messages, max_tokens):
        calls.append(1)
        if len(calls) == 2:  # the transcript changes while the summary is being written
            info = SessionInfo.load(d)
            write_outputs(info, [seg(0, "Me", "Cancel the release.")], 30.0, "fake")
        return "{}" if len(calls) == 1 else "- Ship on Monday."

    summary.summarize_session(d, Config(output_dir=tmp_path), gen)
    stored = (d / summary.SOURCE_NAME).read_text("utf-8")
    assert stored == summary.transcript_fingerprint(before)
