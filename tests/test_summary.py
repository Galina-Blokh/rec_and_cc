"""Local meeting summaries: prompts, map-reduce chunking, summary.md, opt-in, offline skip.

Stories: US-27.
"""

from __future__ import annotations

import logging
import queue
import re
import sys
import threading
import types
from datetime import datetime
from pathlib import Path

import pytest

from recandcc import cli
from recandcc.config import DEFAULT_SUMMARY_MODEL, Config, load_config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import gpu, summary
from recandcc.transcribe.output import PDI, RLI, write_outputs

SPOKEN = "the merger price is 42 million"


class FakeGen:
    """Records prompts; answers by task, so the output shows which step produced what."""

    def __init__(self, actions: str = "- Anna sends the deck by Friday\n- Me books the room"):
        self.calls: list[str] = []
        self.actions = actions

    def __call__(self, messages, max_tokens):
        user = messages[-1]["content"]
        task = "notes" if "notes" in user else "summary" if "Summarise" in user else "actions"
        self.calls.append(task)
        return {"notes": "- a note", "summary": "We agreed on the plan.", "actions": self.actions}[
            task
        ]

    def close(self):
        pass


def make_session(tmp_path: Path, segments: list[Segment]) -> Path:
    d = tmp_path / "2026-10-27" / "100000_zoom"
    d.mkdir(parents=True)
    s = SessionInfo("s1", d, datetime(2026, 10, 27, 10), TriggerKind.CALL, "zoom.us")
    s.save()
    write_outputs(s, segments, 60.0, "fake")
    return d


def seg(text: str, lang: str = "en", speaker: str = "Me", t: float = 0) -> Segment:
    return Segment(t, t + 2, text, speaker, lang)


@pytest.mark.parametrize(
    "lang, name",
    [("en", "English"), ("ru", "Russian"), ("he", "Hebrew"), ("de", "the language with code 'de'")],
)
def test_prompts_ask_for_the_meetings_language(lang, name):
    for task, phrase in [("notes", "notes"), ("summary", "Summarise"), ("actions", "NONE")]:
        system, user = summary.build_messages(task, "Me: hi", lang, "2 of 3")
        assert f"Always answer in {name}" in system["content"]
        assert phrase in user["content"] and "Me: hi" in user["content"]
        assert "(part 2 of 3)" in user["content"]


def test_main_language_is_the_one_with_the_most_speech_time():
    segs = [seg("hello", "en"), Segment(2, 9, "это фраза", "Me", "ru"), seg("ok", "en", t=9)]
    assert summary.main_language(segs) == "ru"  # 7 s of Russian, 4 s of English
    assert summary.main_language([seg("x", "")]) == "en"


GEMMA = "mlx-community/gemma-4-e4b-it-4bit"


def test_mixed_meeting_uses_one_language_for_model_prompt_and_headings(tmp_path, monkeypatch):
    """Long Hebrew speech with few characters, plus a wordy English line: Hebrew wins for
    the model and for the summary's language."""
    seen = []

    class Gen(FakeGen):
        def __init__(self, model, allow_unpinned=False, on_download=None):
            super().__init__()
            seen.append(model)

        def __call__(self, messages, max_tokens):
            seen.append(messages[0]["content"])
            return super().__call__(messages, max_tokens)

    monkeypatch.setattr(summary, "MlxGenerator", Gen)
    cfg = Config()
    cfg.summary.language_models = {"he": GEMMA}
    segs = [Segment(0, 30, "כן", "Me", "he"), seg("a much longer English sentence", t=40)]
    text = summary.summarize_session(make_session(tmp_path, segs), cfg).read_text("utf-8")
    assert seen[0] == GEMMA and "Always answer in Hebrew" in seen[1]
    assert text.startswith("# סיכום")


def test_routed_model_unavailable_falls_back_to_the_default(tmp_path, monkeypatch, caplog):
    made = []

    class Gen(FakeGen):
        def __init__(self, model, allow_unpinned=False, on_download=None):
            super().__init__()
            self.model = model
            made.append(model)

        def __call__(self, messages, max_tokens):
            if self.model == GEMMA:  # offline, or an mlx-lm without Gemma 4
                raise summary.ModelUnavailable("the summary model is not on disk")
            return super().__call__(messages, max_tokens)

    monkeypatch.setattr(summary, "MlxGenerator", Gen)
    cfg = Config()
    cfg.summary.language_models = {"he": GEMMA}
    d = make_session(tmp_path, [Segment(0, 20, SPOKEN, "Me", "he")])
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary.summarize_session(d, cfg).read_text("utf-8")
    assert made == [GEMMA, DEFAULT_SUMMARY_MODEL]
    assert "by Qwen2.5-3B-Instruct-4bit;" in text
    assert f"summary model {GEMMA} unavailable; using {DEFAULT_SUMMARY_MODEL}" in caplog.text
    assert SPOKEN not in caplog.text
    cfg.summary.language_models = {}  # the default itself unavailable: skipped as before
    monkeypatch.setattr(cfg.summary, "model", GEMMA)
    with pytest.raises(summary.ModelUnavailable):
        summary.summarize_session(d, cfg)


@pytest.mark.parametrize(
    ("models", "lang", "expected"),
    [
        ({}, "he", DEFAULT_SUMMARY_MODEL),  # no routing: the old behaviour
        ({"he": GEMMA}, "he", GEMMA),
        ({"he": GEMMA}, "en", DEFAULT_SUMMARY_MODEL),
    ],
)
def test_summary_model_is_routed_by_the_main_language(models, lang, expected):
    cfg = Config()
    cfg.summary.language_models = models
    segs = [seg("hello", "en", t=0), Segment(5, 6, "שלום", "Me", lang)]
    segs.append(Segment(10, 40, "x", "Speaker 1", lang))
    assert summary.summary_model(cfg, segs) == expected


def test_session_loads_only_the_routed_model_and_frees_it(tmp_path, monkeypatch):
    made = []

    class Gen(FakeGen):
        def __init__(self, model, allow_unpinned=False, on_download=None):
            super().__init__()
            self.model, self.closed = model, False
            made.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(summary, "MlxGenerator", Gen)
    cfg = Config()
    cfg.summary.language_models = {"he": GEMMA}
    he = make_session(tmp_path / "he", [Segment(0, 20, "שלום לכולם", "Me", "he")])
    text = summary.summarize_session(he, cfg).read_text("utf-8")
    en = make_session(tmp_path / "en", [seg("hello")])
    summary.summarize_session(en, cfg)
    assert [(g.model, g.closed) for g in made] == [(GEMMA, True), (DEFAULT_SUMMARY_MODEL, True)]
    assert "by gemma-4-e4b-it-4bit;" in text


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        (
            "<|channel>thought\nThe user wants a summary...<channel|>Anna fixes signing.",
            "Anna fixes signing.",
        ),
        ("<|channel>thought\na<channel|>x<|channel>thought\nb<channel|>\n Final. ", "Final."),
        ("  plain answer\n", "plain answer"),
    ],
)
def test_gemma_thinking_is_stripped(raw, answer):
    assert summary.strip_thinking(raw) == answer


def _fake_mlx_lm(monkeypatch, text: str | list[str], seen: list, thinking: bool | None):
    """Fake mlx_lm that streams ``text`` (or each of a list of texts, one per generation);
    returns a fake tokenizer (no real model loads)."""
    texts = [text] if isinstance(text, str) else list(text)

    def stream_generate(model, tokenizer, prompt, max_tokens, **kw):
        seen.append(max_tokens)
        t = texts.pop(0) if len(texts) > 1 else texts[0]
        return iter(types.SimpleNamespace(text=c) for c in [t[:5], t[5:]])

    def apply_chat_template(messages, add_generation_prompt, **kw):
        seen.append(kw)
        return [1]

    tok = types.SimpleNamespace(apply_chat_template=apply_chat_template)
    if thinking is not None:  # None: an mlx-lm without has_thinking
        tok.has_thinking = thinking
    if thinking:  # what mlx-lm detects for Gemma 4
        tok.think_start, tok.think_end = "<|channel>thought", "<channel|>"
    sample = types.SimpleNamespace(make_logits_processors=lambda **kw: [])
    monkeypatch.setitem(
        sys.modules, "mlx_lm", types.SimpleNamespace(stream_generate=stream_generate)
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample)
    return tok


RAW_GEMMA = "<|channel>thought\nplan<channel|>Answer."


@pytest.mark.parametrize(
    ("model", "thinking", "seen", "answer"),
    [
        (GEMMA, True, [{"enable_thinking": True}, 400 + summary.THINKING_TOKENS], "Answer."),
        (DEFAULT_SUMMARY_MODEL, False, [{}, 400], "Answer."),  # Qwen2.5: prompt as before
        # no has_thinking (another mlx-lm): no budget, but a thought never reaches summary.md
        (GEMMA, None, [{}, 400], "Answer."),
    ],
)
def test_generator_gives_a_thinking_model_room_and_strips_it(
    monkeypatch, model, thinking, seen, answer
):
    calls: list = []
    gen = summary.MlxGenerator(model)
    gen._model, gen._tokenizer = object(), _fake_mlx_lm(monkeypatch, RAW_GEMMA, calls, thinking)
    assert gen([{"role": "user", "content": "x"}], 400) == answer
    assert calls == seen


def test_a_cut_off_thought_is_retried_once_without_thinking(monkeypatch, caplog):
    cut = f"<|channel>thought\n{SPOKEN} and still thinking"
    budget = 400 + summary.THINKING_TOKENS
    for second, expected in [("Plain answer.", "Plain answer."), (cut, None)]:
        calls: list = []
        gen = summary.MlxGenerator(GEMMA)
        gen._model, gen._tokenizer = object(), _fake_mlx_lm(monkeypatch, [cut, second], calls, True)
        with caplog.at_level(logging.DEBUG, logger="recandcc"):
            if expected is None:  # cut off again: no answer at all
                with pytest.raises(summary.SummaryUnavailable, match="ran out of tokens"):
                    gen([{"role": "user", "content": "x"}], 400)
            else:
                assert gen([{"role": "user", "content": "x"}], 400) == expected
        assert calls == [{"enable_thinking": True}, budget, {"enable_thinking": False}, 400]
    assert "thought cut off; retrying without" in caplog.text
    assert SPOKEN not in caplog.text


def test_thought_cut_off_writes_no_summary(tmp_path, caplog):
    def cut_off(messages, max_tokens):
        return summary.strip_thinking(f"<|channel>thought\n{SPOKEN} and more")

    d = make_session(tmp_path, [seg(SPOKEN)])
    notices = []
    q = summary.SummaryQueue(Config(), notify=lambda *n: notices.append(n), generate=cut_off)
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        q.submit(d)
        q.join()
    assert not (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()
    assert notices == [("Summary skipped", summary.CUT_OFF)]
    assert SPOKEN not in caplog.text


def test_a_model_this_mlx_lm_cannot_load_is_skipped(monkeypatch):
    def load(path):
        raise ValueError("Model type gemma4 not supported.")

    monkeypatch.setattr(summary, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(gpu, "local_model_path", lambda *a: "/cache/m")
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=load))
    with pytest.raises(summary.SummaryUnavailable, match="can't load"):
        summary.MlxGenerator(GEMMA).load()


def test_offered_summary_models_are_pinned_and_documented():
    from recandcc.transcribe import revisions

    assert DEFAULT_SUMMARY_MODEL in revisions.OFFERED_SUMMARY_MODELS
    root = Path(__file__).resolve().parents[1]
    docs = [(root / d).read_text("utf-8") for d in ("README.md", "docs/CONFIGURATION.md")]
    for model in revisions.OFFERED_SUMMARY_MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", revisions.HF_REVISIONS[model]), model
        assert model in summary.MODEL_SIZES
        assert all(model.split("/")[1] in d for d in docs), model
    assert summary.download_notice(GEMMA) == "Downloading the summary model (~5.1 GB, once)"
    assert summary.download_notice("me/custom") == "Downloading the summary model (once)"


def test_load_config_reads_summary_language_models(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'[summary]\nlanguage_models = {{ iw = "{GEMMA}", xx = "m" }}\n', "utf-8")
    assert load_config(path).summary.language_models == {"he": GEMMA}
    assert Config().summary.language_models == {}
    cfg = load_config(path)
    cfg.save(tmp_path / "saved.toml")  # round-trips
    assert load_config(tmp_path / "saved.toml").summary.language_models == {"he": GEMMA}


def test_chunks_stay_within_the_limit_and_keep_everything():
    lines = ["a" * 30, "b" * 30, "c" * 95, "d" * 10]
    chunks = summary.chunk_lines(lines, max_chars=40)
    assert all(len(c) <= 40 for c in chunks)
    assert "".join(chunks).replace("\n", "") == "".join(lines)
    assert summary.chunk_lines(["x", "y"], 40) == ["x\ny"]


def test_short_transcript_is_summarised_in_one_pass():
    gen = FakeGen()
    lang, text, items = summary.summarize([seg("Let's ship on Monday.")], gen)
    assert gen.calls == ["summary", "actions"]
    assert (lang, text) == ("en", "We agreed on the plan.")
    assert items == ["Anna sends the deck by Friday", "Me books the room"]


def test_long_transcript_is_mapped_to_notes_then_reduced(monkeypatch):
    monkeypatch.setattr(summary, "CHUNK_CHARS", 200)
    segs = [seg(f"point number {i} " * 4, speaker=f"Speaker {i % 2}", t=i * 5) for i in range(30)]
    gen = FakeGen()
    summary.summarize(segs, gen)
    notes = gen.calls.count("notes")
    assert notes >= 5 and gen.calls[notes:] == ["summary", "actions"]


def test_two_chunks_at_the_limit_give_two_notes_calls():
    line = "Me: " + "x" * 3_995  # 3999 characters
    lines = [line] * 3  # two fit one 8000-character chunk, three don't
    assert [len(c) for c in summary.chunk_lines(lines)] == [7_999, 3_999]
    gen = FakeGen()
    summary._reduce(lines, gen, "en", 400)
    assert gen.calls == ["notes", "notes"]


def test_notes_longer_than_a_chunk_are_reduced_again(monkeypatch, caplog):
    monkeypatch.setattr(summary, "CHUNK_CHARS", 100)
    calls: list[int] = []

    def gen(messages, max_tokens):
        calls.append(1)
        return "- " + "n" * 60  # two notes don't fit one chunk: another round

    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary._reduce(["Me: " + "a" * 90] * 4, gen, "en", 400)
    # round 1: 4 chunks -> 4 notes (4 chunks of 62); rounds 2 and 3: 4 -> 4 again
    assert len(calls) == 12 and len(text) <= 100
    assert "after 3 rounds; cut" in caplog.text


def test_action_items_are_parsed_and_none_means_empty():
    answer = "Here they are:\n- one\n* two\n3. three\n4) four\n-\n- NONE"
    assert summary.action_items(answer) == ["one", "two", "three", "four"]
    assert summary.action_items("NONE") == []


def test_summary_md_next_to_the_transcript(tmp_path):
    d = make_session(tmp_path, [seg("Привет, обсудим бюджет и сроки проекта.", "ru")])
    path = summary.summarize_session(d, Config(), FakeGen(actions="NONE"))
    assert path == d / "summary.md"
    text = path.read_text("utf-8")
    assert text.startswith("# Краткое содержание\n\nWe agreed on the plan.\n\n## Задачи\n\nНет.")
    assert "Qwen2.5-3B-Instruct-4bit" in text


def test_hebrew_summary_is_right_to_left(tmp_path):
    d = make_session(tmp_path, [seg("שלום, נדבר על התקציב", "he")])
    text = summary.summarize_session(d, Config(), FakeGen()).read_text("utf-8")
    assert "# סיכום" in text and "## משימות" in text
    assert f"- {RLI}Anna sends the deck by Friday{PDI}" in text


def test_empty_transcript_needs_no_model(tmp_path):
    d = make_session(tmp_path, [])
    gen = FakeGen()
    assert "None." in summary.summarize_session(d, Config(), gen).read_text("utf-8")
    assert gen.calls == []


def test_summary_is_off_by_default(tmp_path):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.summary.enabled is False
    assert "[summary]" in (tmp_path / "config.toml").read_text("utf-8")


def _controller(cfg):
    from recandcc.ui.common import Controller

    ctrl = Controller.__new__(Controller)  # only the post-transcription wiring
    ctrl.events, ctrl.cfg, ctrl._lock = queue.Queue(), cfg, threading.Lock()
    ctrl.refresh_async = lambda force=False: None
    ctrl._index_session = lambda directory: None
    ctrl.engine = types.SimpleNamespace(state=types.SimpleNamespace(status="idle"))
    return ctrl


def test_controller_summarises_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "MlxGenerator", lambda *a, **k: FakeGen())
    d = make_session(tmp_path, [seg("Let's ship on Monday.")])
    ctrl = _controller(Config(output_dir=tmp_path))
    ctrl._on_transcribed(SessionInfo.load(d), [])
    assert getattr(ctrl, "_summaries", None) is None
    ctrl.cfg.summary.enabled = True
    ctrl._on_transcribed(SessionInfo.load(d), RuntimeError("failed"))
    assert getattr(ctrl, "_summaries", None) is None  # a failed transcription: nothing
    ctrl._on_transcribed(SessionInfo.load(d), [])
    ctrl._summaries.join()
    assert (d / "summary.md").exists()


def test_controller_summarises_after_the_final_speaker_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "MlxGenerator", lambda *a, **k: FakeGen())
    d = make_session(tmp_path, [seg("Let's ship on Monday.")])
    ctrl = _controller(Config(output_dir=tmp_path))
    ctrl.cfg.summary.enabled = True
    waiting = []  # a slow diarization: the summary waits for its relabelled transcript
    ctrl.transcriber = types.SimpleNamespace(after_labels=lambda _d, fn: waiting.append(fn))
    ctrl._on_transcribed(SessionInfo.load(d), [])
    assert getattr(ctrl, "_summaries", None) is None and len(waiting) == 1
    assert (d / summary.PENDING_MARKER).exists()  # owed even if we quit before the labels
    waiting[0]()
    ctrl._summaries.join()
    assert (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()


def _transcribed(d: Path) -> Path:
    s = SessionInfo.load(d)
    s.transcribed = True
    s.save()
    return d


def test_pending_marker_is_set_on_submit_and_cleared_when_done(tmp_path):
    d = make_session(tmp_path, [seg("hello")])
    started = []

    class Watch(FakeGen):
        def __call__(self, messages, max_tokens):
            started.append((d / summary.PENDING_MARKER).exists())
            return super().__call__(messages, max_tokens)

    q = summary.SummaryQueue(Config(), generate=Watch())
    q.submit(d)
    q.join()
    assert started and all(started)  # there while the summary runs
    assert (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()


def test_startup_requeues_pending_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "MlxGenerator", lambda *a, **k: FakeGen())
    d = _transcribed(make_session(tmp_path, [seg("hello")]))
    summary.mark_pending(d)  # the app quit mid-summary
    ctrl = _controller(Config(output_dir=tmp_path))
    ctrl._resubmit_summaries()  # summaries off: left alone
    assert getattr(ctrl, "_summaries", None) is None
    ctrl.cfg.summary.enabled = True
    other = tmp_path / "2026-10-27" / "110000_zoom"
    other.mkdir()
    (other / summary.PENDING_MARKER).touch()
    (other / SessionInfo.DISCARD_MARKER).touch()  # discarded: never summarised
    assert summary.pending_summaries(tmp_path) == [d]
    ctrl._resubmit_summaries()
    ctrl._summaries.join()
    assert (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()


def test_cli_name_speaker_only_marks_the_summary_stale(tmp_path, monkeypatch, capsys):
    def no_model(*a, **k):
        raise AssertionError("the CLI must not run the model (the app may be recording)")

    monkeypatch.setattr(summary, "summarize_session", no_model)
    d = make_session(tmp_path, [seg("hello")])
    cfg = Config()
    cli._refresh_summary(d, cfg)  # no summary.md: nothing to do
    assert not (d / summary.PENDING_MARKER).exists()
    (d / "summary.md").write_text("old", encoding="utf-8")
    cli._refresh_summary(d, cfg)
    assert (d / summary.PENDING_MARKER).exists()
    assert f"recandcc summarize {d}" in capsys.readouterr().out
    cfg.summary.enabled = True
    cli._refresh_summary(d, cfg)
    assert "redone by the app" in capsys.readouterr().out


def test_idle_refresh_queues_pending_summaries_but_not_in_flight_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "MlxGenerator", lambda *a, **k: FakeGen())
    d = _transcribed(make_session(tmp_path, [seg("hello")]))
    summary.mark_pending(d)  # e.g. `name-speaker` marked it stale
    ctrl = _controller(Config(output_dir=tmp_path))
    ctrl.cfg.summary.enabled = True
    ctrl._in_flight = set()
    ctrl.transcriber = types.SimpleNamespace(labels_pending=lambda _d: True)
    ctrl._resubmit_summaries()  # still waiting for its speaker labels: not yet
    ctrl._in_flight = {d}
    ctrl.transcriber = types.SimpleNamespace(labels_pending=lambda _d: False)
    ctrl._resubmit_summaries()  # still being transcribed: not yet
    assert getattr(ctrl, "_summaries", None) is None
    ctrl._in_flight = set()
    ctrl._resubmit_summaries()
    ctrl._summaries.join()
    assert (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()


def test_a_queued_session_is_not_summarised_twice(tmp_path):
    d = make_session(tmp_path, [seg("hello")])
    gate, gen = threading.Event(), FakeGen()

    class Slow(FakeGen):
        def __call__(self, messages, max_tokens):
            gate.wait(5)
            return gen(messages, max_tokens)

    q = summary.SummaryQueue(Config(), generate=Slow())
    q.submit(d)
    q.submit(Path(str(d) + "/."))  # e.g. startup's re-queue of the same session
    gate.set()
    q.join()
    assert gen.calls == ["summary", "actions"]  # one summary
    q.submit(d)  # done: a later stale marker runs again
    q.join()
    assert len(gen.calls) == 4


def test_discarded_session_is_skipped_and_its_marker_cleared(tmp_path):
    d = make_session(tmp_path, [seg("hello")])
    (d / SessionInfo.DISCARD_MARKER).touch()
    q = summary.SummaryQueue(Config(), generate=FakeGen())
    q.submit(d)
    q.join()
    assert not (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()


def test_missing_model_offline_is_skipped_with_one_notice(tmp_path, monkeypatch, caplog):
    def offline(repo, allow_unpinned=False):
        raise ConnectionError("no network")

    monkeypatch.setattr(summary, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(gpu, "local_model_path", offline)
    d = make_session(tmp_path, [seg(SPOKEN)])
    notices = []
    q = summary.SummaryQueue(Config(), notify=lambda *n: notices.append(n))
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        q.submit(d)
        q.submit(d)
        q.join()
    assert not (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()
    assert notices == [("Summary skipped", "the summary model is not on disk or can't be loaded")]
    assert "summary of 100000_zoom skipped" in caplog.text
    assert SPOKEN not in caplog.text


def test_other_platforms_skip_the_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    d = make_session(tmp_path, [seg("hello")])
    with pytest.raises(summary.SummaryUnavailable, match="Apple silicon"):
        summary.summarize_session(d, Config())
    assert not (d / "summary.md").exists()


def test_logs_never_contain_transcript_or_summary_text(tmp_path, caplog):
    class Leaky(FakeGen):
        def __call__(self, messages, max_tokens):
            if "Summarise" in messages[-1]["content"]:
                raise RuntimeError(messages[-1]["content"])  # an error quoting the prompt
            return SPOKEN

    d = make_session(tmp_path, [seg(SPOKEN)])
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        summary.summarize_session(d, Config(), FakeGen(actions=f"- {SPOKEN}"))
        q = summary.SummaryQueue(Config(), generate=Leaky())
        q.submit(d)
        q.join()
    assert "written: 1 action item(s)" in caplog.text
    assert "failed: RuntimeError" in caplog.text
    assert SPOKEN not in caplog.text


def test_summaries_wait_while_recording_and_pause_between_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "CHUNK_CHARS", 200)
    gen, polls = FakeGen(), []

    def busy():
        polls.append(len(gen.calls))
        # recording for the first 3 polls; a new recording after the first model call
        return len(polls) <= 3 or (len(gen.calls) == 1 and polls.count(1) < 3)

    d = make_session(tmp_path, [seg("word " * 30, t=i * 5) for i in range(10)])
    q = summary.SummaryQueue(Config(), busy=busy, generate=gen, poll_s=0.001)
    q.submit(d)
    q.join()
    assert polls[:4] == [0, 0, 0, 0]  # nothing ran while recording
    assert polls.count(1) == 3  # paused after the first chunk until the recording ended
    assert (d / "summary.md").exists()


def test_first_download_posts_a_notice(tmp_path, monkeypatch):
    def download(repo, allow_unpinned=False, on_download=None):
        on_download()
        raise ConnectionError("offline")

    monkeypatch.setattr(summary, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(gpu, "local_model_path", download)
    notices = []
    q = summary.SummaryQueue(Config(), notify=lambda *n: notices.append(n))
    q.submit(make_session(tmp_path, [seg("hello")]))
    q.join()
    assert notices[0] == ("Summary model", "Downloading the summary model (~1.7 GB, once)")


def test_cache_miss_calls_the_download_hook(monkeypatch):
    def snapshot_download(repo, revision=None, local_files_only=False):
        if local_files_only:
            raise FileNotFoundError(repo)
        return "/cache/model"

    fake = types.SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    hook = []
    path = gpu.local_model_path(DEFAULT_SUMMARY_MODEL, on_download=lambda: hook.append(1))
    assert (path, hook) == ("/cache/model", [1])


def test_cli_summarize(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_config", lambda: Config())
    d = make_session(tmp_path, [seg("hello")])
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    assert cli.main(["summarize", str(d)]) == 1
    assert "Apple silicon" in capsys.readouterr().err
    monkeypatch.setattr(summary, "MlxGenerator", lambda *a, **k: FakeGen())
    assert cli.main(["summarize", str(d)]) == 0
    assert capsys.readouterr().out.strip() == str(d / "summary.md")
    assert cli.main(["summarize", str(tmp_path / "missing")]) == 1
    (d / "transcript.json").write_text('{"segments": [{"text": "' + SPOKEN + '"}]}')
    capsys.readouterr()
    assert cli.main(["summarize", str(d)]) == 1  # a TypeError: its type only, never the text
    assert capsys.readouterr().err.strip() == "Can't summarize the session: TypeError"


@pytest.mark.slow
def test_real_model_summarises_en_and_ru(tmp_path):
    if not summary.mlx_lm_available():
        pytest.skip("needs Apple silicon with mlx-lm")
    from huggingface_hub import snapshot_download

    try:
        rev = gpu.revision(DEFAULT_SUMMARY_MODEL)
        snapshot_download(DEFAULT_SUMMARY_MODEL, revision=rev, local_files_only=True)
    except Exception:  # noqa: BLE001 - never download in tests
        pytest.skip("summary model not cached (run `recandcc summarize <session>` once)")
    for lang, lines in {
        "en": ["Let's ship the release on Monday.", "OK, I will write the release notes."],
        "ru": ["Давайте выпустим релиз в понедельник.", "Хорошо, я напишу заметки к релизу."],
    }.items():
        d = make_session(
            tmp_path / lang, [seg(t, lang, f"Speaker {i}") for i, t in enumerate(lines)]
        )
        text = summary.summarize_session(d, Config()).read_text("utf-8")
        heading = summary.HEADINGS[lang][1]
        assert f"## {heading}" in text and len(text) > 80
