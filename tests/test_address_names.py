"""Names from direct address ("Tobias, I have a question" and another speaker answers).

Synthetic transcripts only (invented names), never fixtures from real recordings.
Stories: US-21, US-27.
"""

import json
from datetime import datetime

import pytest

from recandcc.config import Config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import names, summary
from recandcc.transcribe.address_names import addressed_names, addresses
from recandcc.transcribe.output import write_outputs


def seg(start, speaker, text, length=2.0):
    return Segment(start, start + length, text, speaker, "en")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Tobias, I have a question.", [("Tobias", True)]),
        ("Okay, Mira, you were there?", [("Mira", True)]),
        ("So we met. Mira, are you in?", [("Mira", False)]),
        ("and Mira you were with them", [("Mira", False)]),
        ("Nina and Omar, I believe.", []),  # a list, not an address
        ("I told Mira, then left.", []),
        ("Tomorrow, we ship it.", []),  # a clause opener
        ("Honestly, it works.", []),
        ("Okay, thank you.", []),  # fillers are not names
        ("Yep, project review.", []),
        ("Mira is there, Omar is there.", []),
        ("שלום דנה, מה שלומך", []),  # no capitals: can't tell a name from a word
    ],
)
def test_only_clause_initial_capitalised_names_are_addressed(text, expected):
    assert addresses(text) == expected


def test_words_seen_in_lowercase_are_not_names():
    assert addresses("Tomorrow, we ship it.", lowercase={"tomorrow"}) == []
    assert addresses("Sound, I mean.", lowercase={"sound"}) == []
    assert addresses("Tobias, hi.", lowercase={"sound"}) == [("Tobias", True)]


def test_a_name_written_in_lowercase_once_is_still_a_name():
    # Whisper capitalises inconsistently: "Mira" 3 times, "mira" once
    segments = [
        seg(0, "Me", "Mira, are you in?"),
        seg(3, "Speaker 1", "Yes."),
        seg(6, "Me", "and mira you were with them"),
        seg(9, "Speaker 1", "I was."),
        seg(12, "Me", "Thanks Mira."),
        seg(14, "Me", "Mira is great."),
    ]
    assert addressed_names(segments) == {"Speaker 1": "Mira"}


def test_the_addressed_speaker_who_answers_gets_the_name():
    segments = [
        seg(0, "Me", "Tobias, I have a question. What is it?"),
        seg(3, "Speaker 1", "It is the leadership event."),
        seg(6, "Speaker 2", "Ask Tobias anything."),
        seg(9, "Speaker 1", "Mira, you were there?"),
        seg(11, "Speaker 2", "Yes."),
        seg(13, "Speaker 1", "and Mira you were with them"),
        seg(15, "Speaker 2", "I was."),
    ]
    assert addressed_names(segments) == {"Speaker 1": "Tobias", "Speaker 2": "Mira"}


def test_only_the_last_address_of_a_turn_is_answered():
    segments = [
        seg(0, "Speaker 1", "Nina and Omar, I believe. Mira, you were there, I believe."),
        seg(3, "Speaker 2", "Yeah."),
        seg(6, "Speaker 1", "and Mira you were with them"),
        seg(8, "Speaker 2", "I was."),
    ]
    assert addressed_names(segments) == {"Speaker 2": "Mira"}


def test_one_vote_needs_the_name_to_open_the_turn_and_be_said_again():
    mention = seg(20, "Me", "I met Mira last week.")
    mid_turn = [seg(0, "Me", "So we met. Mira, are you in?"), seg(3, "Speaker 1", "Yes."), mention]
    opening = [seg(0, "Me", "Mira, are you in?"), seg(3, "Speaker 1", "Yes.")]
    assert addressed_names(mid_turn) == {}
    assert addressed_names(opening) == {}  # said once: could be a garbled word ("Sustain, I...")
    assert addressed_names([*opening, mention]) == {"Speaker 1": "Mira"}


@pytest.mark.parametrize(
    "segments",
    [
        # the answer comes too late, from the speaker themself, or overlapped
        [seg(0, "Me", "Mira, are you in?"), seg(12, "Speaker 1", "Yes.")],
        [seg(0, "Speaker 1", "Mira, are you in?"), seg(3, "Speaker 1", "Hello?")],
        [
            seg(0, "Me", "Mira, are you in?"),
            Segment(3, 5, "Yes.", "Speaker 1", "en", overlap=True),
        ],
        # two names tie for one label
        [
            seg(0, "Me", "Mira, are you in?"),
            seg(3, "Speaker 1", "Yes."),
            seg(6, "Me", "Omar, are you in?"),
            seg(9, "Speaker 1", "Yes."),
        ],
        # the owner is never renamed, whoever addresses them
        [seg(0, "Speaker 1", "Gali, are you in?"), seg(3, "Me", "Yes.")],
        # not a numbered label
        [seg(0, "Me", "Mira, are you in?"), seg(3, "Others", "Yes.")],
    ],
)
def test_weak_or_ambiguous_evidence_names_nobody(segments):
    assert addressed_names(segments) == {}


def test_taken_names_and_named_labels_are_left_alone():
    segments = [
        seg(0, "Me", "Mira, are you in?"),
        seg(3, "Speaker 1", "Yes."),
        seg(6, "Mira", "Hello."),
    ]
    assert addressed_names(segments) == {}
    plain = segments[:2]
    assert addressed_names(plain, reserved=["mira"]) == {}
    assert addressed_names(plain, skip=["Speaker 1"]) == {}


def test_two_labels_never_share_a_name():
    segments = [
        seg(0, "Me", "Mira, are you in?"),
        seg(3, "Speaker 1", "Yes."),
        seg(6, "Me", "Mira, are you in?"),
        seg(9, "Speaker 2", "Yes."),
    ]
    assert addressed_names(segments) == {}


def test_single_label_tracks_can_be_named():
    segments = [
        seg(0, "Me", "Mira, are you in?"),
        seg(3, "Others", "Yes."),
        seg(6, "Me", "We spoke to Mira before."),
    ]
    assert addressed_names(segments, single_labels={"Others"}) == {"Others": "Mira"}


def test_saved_transcript_and_summary_use_the_addressed_names(tmp_path, monkeypatch):
    segments = [
        seg(0, "Me", "Tobias, I have a question. What is it?"),
        seg(3, "Speaker 1", "It is the leadership event."),
        seg(6, "Speaker 2", "Ask Tobias anything."),
        seg(9, "Speaker 1", "Mira, you were there?"),
        seg(11, "Speaker 2", "Yes, I will send the notes tomorrow."),
        seg(14, "Speaker 1", "and Mira you were with them"),
        seg(16, "Speaker 2", "I was."),
    ]
    info = SessionInfo("addr", tmp_path, datetime(2026, 11, 10, 10), TriggerKind.CALL, "test")
    info.save()
    write_outputs(info, segments, 20.0, "fake")
    assert names.apply_introductions(tmp_path) == {"Speaker 1": "Tobias", "Speaker 2": "Mira"}
    data = json.loads((tmp_path / "transcript.json").read_text("utf-8"))
    assert {s["speaker"] for s in data["segments"]} == {"Me", "Tobias", "Mira"}
    assert "Mira:" in (tmp_path / "transcript.md").read_text("utf-8")

    prompts = []

    def generate(messages, max_tokens):
        prompts.append(messages[-1]["content"])
        return "- Mira: Send the notes."

    cfg = Config(output_dir=tmp_path)
    cfg.summary.enabled = True
    summary.summarize_session(tmp_path, cfg, generate=generate)
    text = "\n".join(prompts)
    assert "Mira:" in text and "Tobias:" in text
    assert "Speaker 1" not in text and "Speaker 2" not in text


def _controller(cfg):
    from recandcc.ui.common import Controller

    return Controller(
        cfg, monitor=object(), recorder=object(), tq=object(), transcriber=None, start_threads=False
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_summarizing_an_older_meeting_names_its_speakers_first(tmp_path, enabled):
    segments = [
        seg(0, "Me", "Mira, are you in?"),
        seg(3, "Speaker 1", "Yes."),
        seg(6, "Me", "and Mira you were with them"),
        seg(9, "Speaker 1", "I was."),
    ]
    info = SessionInfo("old", tmp_path, datetime(2026, 11, 2, 10), TriggerKind.CALL, "test")
    info.save()
    write_outputs(info, segments, 12.0, "fake")
    cfg = Config(output_dir=tmp_path)
    cfg.transcription.names_from_transcript = enabled
    _controller(cfg)._name_from_transcript(tmp_path)
    speakers = {
        s["speaker"] for s in json.loads((tmp_path / "transcript.json").read_text())["segments"]
    }
    assert speakers == ({"Me", "Mira"} if enabled else {"Me", "Speaker 1"})


# --- regressions from the Codex review (gpt-6.1-sol): each was a wrong name before


def test_an_introduction_claim_vetoes_the_same_name_for_another_label(tmp_path):
    segments = [
        seg(0, "Me", "My name is Mira."),
        seg(3, "Speaker 1", "Mira, do you have the notes?"),
        seg(6, "Speaker 2", "I can find them for you."),
        seg(9, "Speaker 1", "Mira, can you open the slides?"),
        seg(12, "Speaker 2", "I can do it instead."),
    ]
    info = SessionInfo("veto", tmp_path, datetime(2026, 11, 10, 10), TriggerKind.CALL, "test")
    info.save()
    write_outputs(info, segments, 15.0, "fake")
    assert names.apply_introductions(tmp_path) == {}
    data = json.loads((tmp_path / "transcript.json").read_text("utf-8"))
    assert {s["speaker"] for s in data["segments"]} == {"Me", "Speaker 1", "Speaker 2"}


def test_one_turn_cut_into_segments_gives_one_vote_the_last_address():
    segments = [
        seg(0, "Me", "Mira, are you there?"),
        seg(2, "Me", "Mira, can you hear us?"),
        seg(4, "Me", "Omar, please cover for her."),
        seg(7, "Speaker 1", "I will cover for her."),
        seg(10, "Me", "Omar, are you ready?"),
        seg(12, "Speaker 1", "Yes."),
        seg(14, "Me", "We met Omar before."),
    ]
    assert addressed_names(segments) == {"Speaker 1": "Omar"}


def test_an_answer_that_says_the_name_is_no_evidence():
    segments = [
        seg(0, "Me", "Mira, are you there?"),
        seg(3, "Speaker 1", "Mira is absent. I can cover for her."),
    ]
    assert addressed_names(segments) == {}
    # nor do the speaker's own mentions make a one-off name look real
    own = [
        seg(0, "Me", "Mira, are you there?"),
        seg(3, "Speaker 1", "Yes. We asked Mira earlier."),
        seg(6, "Speaker 1", "Mira will come."),
    ]
    assert addressed_names(own) == {}


@pytest.mark.parametrize(
    "turn",
    [
        "The email says: Mira, send the notes.",  # reported speech
        "Budget, scope and staffing are open.",  # a list that starts a sentence
        "Anna, Bob and Carl, can you join?",
    ],
)
def test_reported_speech_and_lists_are_not_addresses(turn):
    assert addresses(turn) == []
    segments = [
        seg(0, "Me", turn),
        seg(3, "Speaker 1", "We should discuss that."),
        seg(6, "Me", turn),
        seg(9, "Speaker 1", "We should discuss that too."),
    ]
    assert addressed_names(segments) == {}


def test_a_minority_of_all_votes_is_not_a_clear_majority():
    segments = []
    for k, name in enumerate(["Mira", "Mira", "Omar", "Nina", "Pavel"]):
        segments += [seg(6 * k, "Me", f"{name}, are you in?"), seg(6 * k + 3, "Speaker 1", "Yes.")]
    assert addressed_names(segments) == {}


# --- regressions from the second Codex pass


def test_speech_that_goes_on_after_a_comma_is_not_a_list():
    segments = [
        seg(0, "Me", "Mira, are you there? Omar, and can you cover for her?"),
        seg(3, "Speaker 1", "I can cover."),
        seg(6, "Me", "Mira, can you hear us? Omar, and please send her a message."),
        seg(9, "Speaker 1", "I will."),
    ]
    assert addressed_names(segments) == {"Speaker 1": "Omar"}


@pytest.mark.parametrize(
    "turn",
    [
        "Budget, scope, and staffing are open.",  # an Oxford comma
        "Budget, project scope and staffing are open.",  # several words per entry
    ],
)
def test_lists_with_a_final_conjunction_are_not_addresses(turn):
    assert addresses(turn) == []


def test_the_answer_veto_covers_the_whole_turn_and_possessives():
    split_reply = [
        seg(0, "Me", "Mira, are you there?"),
        seg(3, "Speaker 1", "Sorry."),
        seg(5, "Speaker 1", "Mira is absent."),
        seg(8, "Me", "Mira, can you hear us?"),
        seg(11, "Speaker 1", "I can cover."),
        seg(13, "Speaker 1", "Mira is still absent."),
    ]
    possessive = [
        seg(0, "Me", "Mira, are you there?"),
        seg(3, "Speaker 1", "Mira's absent."),
        seg(6, "Me", "Mira, can you hear us?"),
        seg(9, "Speaker 1", "Mira’s still absent."),
    ]
    assert addressed_names(split_reply) == {}
    assert addressed_names(possessive) == {}


def test_answers_by_me_count_as_competing_evidence():
    segments = [
        seg(0, "Speaker 1", "Mira, can you hear us?"),
        seg(3, "Me", "Yes."),
        seg(6, "Speaker 1", "Mira, can you open the notes?"),
        seg(9, "Me", "Sure."),
        seg(12, "Speaker 1", "Mira, are you ready?"),
        seg(15, "Speaker 2", "I will cover for her."),
    ]
    assert addressed_names(segments) == {}


def test_a_full_name_introduction_reserves_the_first_name():
    segments = [
        seg(0, "Me", "My name is Mira Stone."),
        seg(3, "Speaker 1", "Mira, are you in?"),
        seg(6, "Speaker 2", "She is absent. I can cover."),
        seg(9, "Speaker 1", "Mira, do you have the notes?"),
        seg(12, "Speaker 2", "She is not here."),
    ]
    assert addressed_names(segments, reserved={"mira stone"}) == {}
    assert addressed_names(segments) == {"Speaker 2": "Mira"}  # without the claim it would name


# --- regressions from the third Codex pass


def test_an_uncertain_last_address_cancels_the_turn_instead_of_falling_back():
    segments = [
        seg(0, "Me", "Mira, are you there? Omar, send the notes and slides.", 6),
        seg(7, "Speaker 1", "I will send both."),
        seg(12, "Me", "Mira, can you hear us? Omar, share your screen and notes.", 6),
        seg(19, "Speaker 1", "I will share both."),
        seg(22, "Me", "We spoke to Mira and Omar before."),
    ]
    assert addressed_names(segments) == {}


@pytest.mark.parametrize(
    "turn",
    [
        "Budget, the scope of this release, and staffing are open.",
        "Budget, scope, staffing.",
    ],
)
def test_long_and_bare_comma_lists_are_not_addresses(turn):
    assert addresses(turn) == []
    segments = [
        seg(0, "Me", turn),
        seg(3, "Speaker 1", "We should discuss that."),
        seg(6, "Me", turn),
        seg(9, "Speaker 1", "We should discuss that too."),
    ]
    assert addressed_names(segments) == {}


def test_cross_talk_that_started_before_the_question_ended_is_no_answer():
    segments = [
        seg(0, "Me", "I have a question about the build. Mira, are you there?", 8),
        seg(1, "Speaker 1", "The previous item also needs approval."),
        seg(12, "Me", "One more thing about the build. Mira, can you hear us?", 8),
        seg(13, "Speaker 1", "I was still explaining that earlier item."),
        seg(22, "Me", "We spoke to Mira before."),
    ]
    assert addressed_names(segments) == {}


def test_the_answer_veto_reads_a_long_turn_to_its_end():
    filler = [seg(3 + 2 * k, "Speaker 1", "The tests are passing.") for k in range(12)]
    absent = seg(3 + 2 * 12, "Speaker 1", "Mira is absent; I am covering for her.")
    segments = [
        seg(0, "Me", "Mira, can you give us the status?"),
        *filler,
        absent,
        seg(31, "Me", "Thanks, I will catch up with Mira later."),
    ]
    assert addressed_names(segments) == {}
