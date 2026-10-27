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
from recandcc import config as config_mod
from recandcc.config import (
    DEFAULT_SUMMARY_MODEL,
    PREVIOUS_DEFAULT_SUMMARY_MODEL,
    QWEN_RESEARCH_SUMMARY_MODEL,
    SUMMARY_MODEL_KEPT,
    SUMMARY_MODEL_KEPT_PREVIOUS,
    SUMMARY_MODEL_SWITCHED,
    Config,
    SummaryConfig,
    load_config,
)
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


def test_overlapping_speech_counts_once_per_language():
    """Me and Others talking over each other in Russian for 6 s must not beat 8 s of English."""
    segs = [Segment(0, 6, "да", "Me", "ru"), Segment(0, 6, "да", "Speaker 1", "ru")]
    segs += [Segment(1, 5, "ну", "Speaker 2", "ru"), Segment(10, 18, "yes", "Me", "en")]
    segs.append(Segment(4, 8, "так", "Me", "ru"))  # the Russian span is now 0-8 s
    assert summary.main_language(segs[:4]) == "en"  # 6 s of ru (not 16) vs 8 s of en
    segs.append(Segment(20, 21, "и", "Me", "ru"))
    assert summary.main_language(segs) == "ru"  # 9 s vs 8 s


QWEN35 = "mlx-community/Qwen3.5-4B-MLX-4bit"


def test_unpinned_language_model_is_ignored_unless_allowed(tmp_path, caplog):
    cfg = Config()
    segs = [Segment(0, 9, "שלום", "Me", "he")]
    cfg.summary.language_models = {"he": "someone/unpinned-llm"}
    with caplog.at_level(logging.WARNING, logger="recandcc"):
        assert summary.summary_model(cfg, segs) == DEFAULT_SUMMARY_MODEL
    assert "ignoring unpinned summary.language_models model someone/unpinned-llm" in caplog.text
    cfg.summary.language_models = {"he": str(tmp_path)}  # a local model folder is fine
    assert summary.summary_model(cfg, segs) == str(tmp_path)
    cfg.summary.language_models = {"he": "someone/unpinned-llm"}
    cfg.transcription.allow_unpinned_models = True
    assert summary.summary_model(cfg, segs) == "someone/unpinned-llm"


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
    cfg.summary.language_models = {"he": PREVIOUS_DEFAULT_SUMMARY_MODEL}  # not the default
    segs = [Segment(0, 30, "כן", "Me", "he"), seg("a much longer English sentence", t=40)]
    text = summary.summarize_session(make_session(tmp_path, segs), cfg).read_text("utf-8")
    assert seen[0] == PREVIOUS_DEFAULT_SUMMARY_MODEL and "Always answer in Hebrew" in seen[1]
    assert text.startswith("# סיכום")


def test_routed_model_unavailable_falls_back_to_the_default(tmp_path, monkeypatch, caplog):
    made = []
    offline = {PREVIOUS_DEFAULT_SUMMARY_MODEL}

    class Gen(FakeGen):
        def __init__(self, model, allow_unpinned=False, on_download=None):
            super().__init__()
            self.model = model
            made.append(model)

        def __call__(self, messages, max_tokens):
            if self.model in offline:  # not on disk, or an mlx-lm that can't load it
                raise summary.ModelUnavailable("the summary model is not on disk")
            return super().__call__(messages, max_tokens)

    monkeypatch.setattr(summary, "MlxGenerator", Gen)
    cfg = Config()
    cfg.summary.language_models = {"he": PREVIOUS_DEFAULT_SUMMARY_MODEL}  # not the default
    d = make_session(tmp_path, [Segment(0, 20, SPOKEN, "Me", "he")])
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary.summarize_session(d, cfg).read_text("utf-8")
    assert made == [PREVIOUS_DEFAULT_SUMMARY_MODEL, DEFAULT_SUMMARY_MODEL]
    assert f"by {Path(DEFAULT_SUMMARY_MODEL).name};" in text
    assert (
        f"summary model {PREVIOUS_DEFAULT_SUMMARY_MODEL} unavailable; using {DEFAULT_SUMMARY_MODEL}"
    ) in caplog.text
    assert SPOKEN not in caplog.text
    cfg.summary.language_models = {}  # the default itself unavailable: skipped as before
    offline.add(DEFAULT_SUMMARY_MODEL)
    with pytest.raises(summary.ModelUnavailable):
        summary.summarize_session(d, cfg)


@pytest.mark.parametrize(
    ("models", "lang", "expected"),
    [
        ({}, "he", DEFAULT_SUMMARY_MODEL),  # no routing: the default
        ({"he": PREVIOUS_DEFAULT_SUMMARY_MODEL}, "he", PREVIOUS_DEFAULT_SUMMARY_MODEL),
        ({"he": PREVIOUS_DEFAULT_SUMMARY_MODEL}, "en", DEFAULT_SUMMARY_MODEL),
        ({"he": DEFAULT_SUMMARY_MODEL}, "he", DEFAULT_SUMMARY_MODEL),  # routed to it: no-op
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
    cfg.summary.language_models = {"he": PREVIOUS_DEFAULT_SUMMARY_MODEL}
    he = make_session(tmp_path / "he", [Segment(0, 20, "שלום לכולם", "Me", "he")])
    text = summary.summarize_session(he, cfg).read_text("utf-8")
    en = make_session(tmp_path / "en", [seg("hello")])
    summary.summarize_session(en, cfg)
    assert [(g.model, g.closed) for g in made] == [
        (PREVIOUS_DEFAULT_SUMMARY_MODEL, True),
        (DEFAULT_SUMMARY_MODEL, True),
    ]
    assert "by Qwen2.5-1.5B-Instruct-4bit;" in text


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        (
            "<think>\nThe user wants a summary...\n</think>\n\nAnna fixes signing.",
            "Anna fixes signing.",
        ),
        ("<think>\n\n</think>\n\n- Me: book the room", "- Me: book the room"),  # thinking off
        ("<|channel>thought\nplan<channel|>Final.", "Final."),  # Gemma 4's channel
        ("  plain answer\n", "plain answer"),
    ],
)
def test_thought_blocks_are_stripped(raw, answer):
    assert summary.strip_thinking(raw) == answer


@pytest.mark.parametrize("raw", ["<think>\nstill thinking", "<|channel>thought\nstill"])
def test_an_unfinished_thought_gives_no_answer(raw):
    with pytest.raises(summary.SummaryUnavailable, match="ran out of tokens"):
        summary.strip_thinking(raw)


def _fake_mlx_lm(monkeypatch, text: str, seen: list, thinking: bool | None):
    """Fake mlx_lm that streams ``text``; returns a fake tokenizer (no real model loads)."""

    def stream_generate(model, tokenizer, prompt, max_tokens, **kw):
        seen.append(max_tokens)
        return iter(types.SimpleNamespace(text=c) for c in [text[:5], text[5:]])

    def apply_chat_template(messages, add_generation_prompt, **kw):
        seen.append(kw)
        return [1]

    tok = types.SimpleNamespace(apply_chat_template=apply_chat_template)
    if thinking is not None:  # None: an mlx-lm without has_thinking
        tok.has_thinking = thinking
    if thinking:  # what mlx-lm detects for Qwen3.5
        tok.think_start, tok.think_end = "<think>", "</think>"
    sample = types.SimpleNamespace(make_logits_processors=lambda **kw: [])
    monkeypatch.setitem(
        sys.modules, "mlx_lm", types.SimpleNamespace(stream_generate=stream_generate)
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample)
    return tok


@pytest.mark.parametrize(
    ("model", "thinking", "raw", "seen"),
    [
        # Qwen3.5 (the default) is a thinking model: thinking off, within max_tokens
        (
            DEFAULT_SUMMARY_MODEL,
            True,
            "<think>\n\n</think>\n\nAnswer.",
            [{"enable_thinking": False}, 400],
        ),
        # an older model without thinking tokens (Qwen2.5): its prompt is unchanged
        (PREVIOUS_DEFAULT_SUMMARY_MODEL, False, "Answer.", [{}, 400]),
        # no has_thinking (another mlx-lm): template defaults, but no thought reaches summary.md
        (QWEN35, None, "<think>\nplan\n</think>\nAnswer.", [{}, 400]),
        (QWEN35, None, "<|channel>thought\nplan<channel|>Answer.", [{}, 400]),
    ],
)
def test_generator_turns_thinking_off_and_strips_thoughts(monkeypatch, model, thinking, raw, seen):
    calls: list = []
    gen = summary.MlxGenerator(model)
    gen._model, gen._tokenizer = object(), _fake_mlx_lm(monkeypatch, raw, calls, thinking)
    assert gen([{"role": "user", "content": "x"}], 400) == "Answer."
    assert calls == seen


def test_thought_cut_off_writes_no_summary(tmp_path, caplog):
    def cut_off(messages, max_tokens):
        return summary.strip_thinking(f"<think>\n{SPOKEN} and more")

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
        raise ValueError("Model type qwen3_5 not supported.")

    monkeypatch.setattr(summary, "mlx_lm_available", lambda: True)
    monkeypatch.setattr(gpu, "local_model_path", lambda *a: "/cache/m")
    monkeypatch.setitem(sys.modules, "mlx_lm", types.SimpleNamespace(load=load))
    with pytest.raises(summary.SummaryUnavailable, match="can't load"):
        summary.MlxGenerator(QWEN35).load()


def test_offered_summary_models_are_pinned_and_documented():
    from recandcc.transcribe import revisions

    assert DEFAULT_SUMMARY_MODEL in revisions.OFFERED_SUMMARY_MODELS
    root = Path(__file__).resolve().parents[1]
    config_doc = (root / "docs/CONFIGURATION.md").read_text("utf-8")  # documents every model
    for model in revisions.OFFERED_SUMMARY_MODELS:
        assert re.fullmatch(r"[0-9a-f]{40}", revisions.HF_REVISIONS[model]), model
        assert model in summary.MODEL_SIZES
        assert model in config_doc, model
    assert QWEN35 in (root / "README.md").read_text("utf-8")  # the Hebrew routing example
    assert summary.download_notice(QWEN35) == "Downloading the summary model (~3.0 GB, once)"
    assert summary.download_notice("me/custom") == "Downloading the summary model (once)"


def test_load_config_reads_summary_language_models(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'[summary]\nlanguage_models = {{ iw = "{QWEN35}", xx = "m" }}\n', "utf-8")
    assert load_config(path).summary.language_models == {"he": QWEN35}
    assert Config().summary.language_models == {}
    cfg = load_config(path)
    cfg.save(tmp_path / "saved.toml")  # round-trips
    assert load_config(tmp_path / "saved.toml").summary.language_models == {"he": QWEN35}


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


@pytest.mark.parametrize(
    "bullet",
    ["- None", "* None.", "1. NONE!", "- **None**", "- nothing.", "- Нет", "- Нет.", "- ничего!"],
)
def test_a_bullet_that_only_says_none_is_no_action_item(bullet):
    assert summary.action_items(f"Action items:\n{bullet}\n- Anna sends the deck") == [
        "Anna sends the deck"
    ]
    assert summary.action_items(bullet) == []


def test_a_task_that_mentions_none_is_kept():
    assert summary.action_items("- Nothing new: Anna books the room") == [
        "Nothing new: Anna books the room"
    ]


def test_summary_md_next_to_the_transcript(tmp_path):
    d = make_session(tmp_path, [seg("Привет, обсудим бюджет и сроки проекта.", "ru")])
    path = summary.summarize_session(d, Config(), FakeGen(actions="NONE"))
    assert path == d / "summary.md"
    text = path.read_text("utf-8")
    assert text.startswith("# Краткое содержание\n\nWe agreed on the plan.\n\n## Задачи\n\nНет.")
    assert "by Qwen3.5-4B-MLX-4bit;" in text  # the footer names the model that wrote it


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


def test_saved_old_default_model_follows_the_new_default(tmp_path):
    # Config.save writes every key, so configs saved before this change hold the 3B without
    # anyone choosing it: it is read as unset unless its licence was explicitly accepted.
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    cfg = load_config(path)
    assert cfg.summary.model == DEFAULT_SUMMARY_MODEL
    cfg.save(path)
    assert load_config(path).summary.model == DEFAULT_SUMMARY_MODEL


def test_saved_1_5b_default_moves_to_the_new_default(tmp_path):
    # the 1.5B was the default until now: Config.save wrote it into configs nobody chose it for
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(enabled=True, model=PREVIOUS_DEFAULT_SUMMARY_MODEL)).save(path)
    assert load_config(path).summary.model == DEFAULT_SUMMARY_MODEL  # CLI: in memory only
    assert load_config(path).summary.notice is None
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == DEFAULT_SUMMARY_MODEL
    assert cfg.summary.notice == SUMMARY_MODEL_SWITCHED  # summaries on: told once
    assert load_config(path, migrate=True).summary.notice is None
    text = path.read_text("utf-8")
    assert DEFAULT_SUMMARY_MODEL in text and PREVIOUS_DEFAULT_SUMMARY_MODEL not in text


def test_accepted_qwen_research_licence_keeps_the_3b(tmp_path):
    path = tmp_path / "config.toml"
    keep = SummaryConfig(model=QWEN_RESEARCH_SUMMARY_MODEL, accept_qwen_research_licence=True)
    Config(summary=keep).save(path)
    assert load_config(path).summary.model == QWEN_RESEARCH_SUMMARY_MODEL
    Config(summary=SummaryConfig(model="me/custom-llm")).save(path)
    assert load_config(path).summary.model == "me/custom-llm"  # other models untouched


def _cache_model(cache: Path, repo: str, rev: str | None = None) -> None:
    """A fake complete snapshot of ``repo`` (its pinned commit by default) in an HF cache."""
    root = cache / ("models--" + repo.replace("/", "--"))
    snap = root / "snapshots" / (rev or gpu.revision(repo) or "commit")
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(b"weights")  # a plain file: no symlink rights needed
    (snap / "config.json").write_text("{}")


def test_old_default_switch_is_saved_with_only_that_key(tmp_path):
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    path.write_text(path.read_text("utf-8") + "\n[extra]\nhand = 1\n", "utf-8")
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == DEFAULT_SUMMARY_MODEL
    text = path.read_text("utf-8")
    assert DEFAULT_SUMMARY_MODEL in text and QWEN_RESEARCH_SUMMARY_MODEL not in text
    assert "hand = 1" in text  # key-level save, not a full overwrite
    assert load_config(path, migrate=True).summary.model == DEFAULT_SUMMARY_MODEL


HAND_EDITED = f"""# my settings, edited by hand
output_dir = "/tmp/x"  # where transcripts go

[summary]
# summaries on for work calls
enabled = true
model = "{QWEN_RESEARCH_SUMMARY_MODEL}"  # the old default
max_tokens = 400

[transcription]
model = "base"  # must stay
"""


def test_switch_rewrites_only_the_model_line_and_keeps_comments(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(HAND_EDITED, "utf-8")
    assert load_config(path, migrate=True).summary.model == DEFAULT_SUMMARY_MODEL
    expected = HAND_EDITED.replace(QWEN_RESEARCH_SUMMARY_MODEL, DEFAULT_SUMMARY_MODEL)
    assert path.read_text("utf-8") == expected  # every comment and other line unchanged
    assert load_config(path).transcription.model == "base"


def test_model_line_falls_back_to_a_key_save_when_not_a_plain_line(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'summary = {{ model = "{QWEN_RESEARCH_SUMMARY_MODEL}" }}\n', "utf-8")
    config_mod.set_string_line(path, "summary", "model", DEFAULT_SUMMARY_MODEL)
    assert load_config(path).summary.model == DEFAULT_SUMMARY_MODEL


def test_following_the_switch_notice_keeps_the_3b(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(HAND_EDITED, "utf-8")
    assert load_config(path, migrate=True).summary.notice == SUMMARY_MODEL_SWITCHED
    # do what the notice says: set summary.model back to the 3B and accept its licence
    assert f'summary.model = "{QWEN_RESEARCH_SUMMARY_MODEL}"' in SUMMARY_MODEL_SWITCHED
    assert "summary.accept_qwen_research_licence = true" in SUMMARY_MODEL_SWITCHED
    text = path.read_text("utf-8").replace(DEFAULT_SUMMARY_MODEL, QWEN_RESEARCH_SUMMARY_MODEL)
    path.write_text(text.replace("max_tokens", "accept_qwen_research_licence = true\nmax_tokens"))
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == QWEN_RESEARCH_SUMMARY_MODEL and cfg.summary.notice is None
    assert QWEN_RESEARCH_SUMMARY_MODEL in path.read_text("utf-8")


def test_cli_loads_switch_in_memory_only(tmp_path):
    # a CLI command run first after the upgrade must not use up the app's one-time notice
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(enabled=True, model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    before = path.read_text("utf-8")
    cfg = load_config(path)
    assert cfg.summary.model == DEFAULT_SUMMARY_MODEL
    assert cfg.summary.notice is None
    assert path.read_text("utf-8") == before
    assert load_config(path, migrate=True).summary.notice == SUMMARY_MODEL_SWITCHED
    assert load_config(path, migrate=True).summary.notice is None  # once


def test_cached_3b_is_kept_while_the_new_default_is_missing(tmp_path):
    # offline users who already have the 3B must not lose summaries or face a surprise fetch
    _cache_model(config_mod.HF_CACHE_DIR, QWEN_RESEARCH_SUMMARY_MODEL)
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    before = path.read_text("utf-8")
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == QWEN_RESEARCH_SUMMARY_MODEL
    assert cfg.summary.notice is None  # summaries off: nothing to say
    assert path.read_text("utf-8") == before
    _cache_model(config_mod.HF_CACHE_DIR, PREVIOUS_DEFAULT_SUMMARY_MODEL)  # not the new default
    assert load_config(path, migrate=True).summary.model == QWEN_RESEARCH_SUMMARY_MODEL
    _cache_model(config_mod.HF_CACHE_DIR, DEFAULT_SUMMARY_MODEL)  # both on disk: switch
    assert load_config(path, migrate=True).summary.model == DEFAULT_SUMMARY_MODEL


def test_kept_cached_3b_is_noticed_once(tmp_path):
    # nothing downloads the 1.5B while the 3B is kept, so the user is told how to switch
    _cache_model(config_mod.HF_CACHE_DIR, QWEN_RESEARCH_SUMMARY_MODEL)
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(enabled=True, model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    assert load_config(path).summary.notice is None  # CLI: no notice used up
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == QWEN_RESEARCH_SUMMARY_MODEL
    assert cfg.summary.notice == SUMMARY_MODEL_KEPT
    assert load_config(path, migrate=True).summary.notice is None  # once


def test_cached_1_5b_is_kept_and_noticed_once_while_the_new_default_is_missing(tmp_path):
    # the previous default, offline-safe like the 3B: nothing downloads the 4B by surprise
    _cache_model(config_mod.HF_CACHE_DIR, PREVIOUS_DEFAULT_SUMMARY_MODEL)
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(enabled=True, model=PREVIOUS_DEFAULT_SUMMARY_MODEL)).save(path)
    before = path.read_text("utf-8")
    assert load_config(path).summary.notice is None  # CLI: no notice used up
    cfg = load_config(path, migrate=True)
    assert cfg.summary.model == PREVIOUS_DEFAULT_SUMMARY_MODEL
    assert cfg.summary.notice == SUMMARY_MODEL_KEPT_PREVIOUS
    assert load_config(path, migrate=True).summary.notice is None  # once
    assert path.read_text("utf-8") == before  # kept: config.toml is not rewritten
    _cache_model(config_mod.HF_CACHE_DIR, DEFAULT_SUMMARY_MODEL)  # now the new default is too
    assert load_config(path, migrate=True).summary.model == DEFAULT_SUMMARY_MODEL


def test_unpinned_4b_snapshot_does_not_replace_a_cached_3b(tmp_path):
    # another tool's copy of the new default at a different commit wouldn't load offline
    _cache_model(config_mod.HF_CACHE_DIR, QWEN_RESEARCH_SUMMARY_MODEL)
    _cache_model(config_mod.HF_CACHE_DIR, DEFAULT_SUMMARY_MODEL, rev="2" * 40)
    path = tmp_path / "config.toml"
    Config(summary=SummaryConfig(model=QWEN_RESEARCH_SUMMARY_MODEL)).save(path)
    assert load_config(path, migrate=True).summary.model == QWEN_RESEARCH_SUMMARY_MODEL


def test_config_does_not_import_transcribe():
    # AGENTS rule 8: config.py is a leaf; the summary pins live there and revisions.py uses them
    src = Path(config_mod.__file__).read_text("utf-8")
    assert "from .transcribe" not in src and "import recandcc.transcribe" not in src
    for model, rev in config_mod.SUMMARY_MODEL_REVISIONS.items():
        assert gpu.revision(model) == rev


@pytest.mark.parametrize("value", ['"true"', '"false"', '"yes"', "1", "1.0"])
def test_non_boolean_licence_acceptance_is_not_consent(tmp_path, value):
    path = tmp_path / "config.toml"
    path.write_text(
        f'[summary]\nmodel = "{QWEN_RESEARCH_SUMMARY_MODEL}"\n'
        f"accept_qwen_research_licence = {value}\n",
        "utf-8",
    )
    cfg = load_config(path)
    assert cfg.summary.model == DEFAULT_SUMMARY_MODEL
    assert cfg.summary.accept_qwen_research_licence is False
    assert cfg.summary.notice is None  # CLI loads: in memory only
    cfg = load_config(path, migrate=True)  # summaries off, but they tried to keep the 3B
    assert cfg.summary.notice is not None
    assert cfg.summary.notice.startswith(
        "accept_qwen_research_licence must be a TOML boolean true, not a "
    )
    assert cfg.summary.notice.endswith(SUMMARY_MODEL_SWITCHED)
    assert "notice" not in path.read_text("utf-8")  # runtime only, never saved


def test_only_the_pinned_snapshot_counts_for_a_pinned_repo():
    cache = config_mod.HF_CACHE_DIR
    _cache_model(cache, QWEN_RESEARCH_SUMMARY_MODEL, rev="0" * 40)  # some other commit
    assert config_mod.summary_model_on_disk(QWEN_RESEARCH_SUMMARY_MODEL) is False
    _cache_model(cache, "me/unpinned", rev="1" * 40)  # unpinned: any snapshot
    assert config_mod.summary_model_on_disk("me/unpinned") is True


def test_interrupted_download_does_not_count_as_on_disk(tmp_path):
    snap = config_mod.HF_CACHE_DIR / "models--a--b" / "snapshots" / "c"
    snap.mkdir(parents=True)
    try:
        (snap / "model.safetensors").symlink_to(tmp_path / "missing-blob")
    except OSError:
        pytest.skip("symlinks unavailable")
    assert config_mod.summary_model_on_disk("a/b") is False
    assert config_mod.summary_model_on_disk("no/such") is False


def test_cache_check_without_huggingface_hub_uses_the_default_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "HF_CACHE_DIR", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", None)  # not importable
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # every platform
    assert config_mod.summary_model_on_disk(QWEN_RESEARCH_SUMMARY_MODEL) is False
    _cache_model(tmp_path / ".cache" / "huggingface" / "hub", QWEN_RESEARCH_SUMMARY_MODEL)
    assert config_mod.summary_model_on_disk(QWEN_RESEARCH_SUMMARY_MODEL) is True


@pytest.mark.parametrize("enabled", [True, False])
def test_controller_notices_the_switch_once(tmp_path, monkeypatch, enabled):
    from recandcc.ui import common
    from recandcc.ui.common import Controller

    class Monitor:
        def snapshot(self):
            return []

    path = tmp_path / "config.toml"
    old = SummaryConfig(enabled=enabled, model=QWEN_RESEARCH_SUMMARY_MODEL)
    Config(output_dir=tmp_path, summary=old).save(path)
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    cfg = load_config(path, migrate=True)
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=object(), start_threads=False)
    notices = []
    while not ctrl.events.empty():
        ev = ctrl.events.get()
        if ev[0] == "notice":
            notices.append(ev[1:])
    expected = [("Summary model", SUMMARY_MODEL_SWITCHED)] if enabled else []
    assert notices == expected  # summaries off: the model doesn't matter, no toast


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
    assert notices[0] == ("Summary model", "Downloading the summary model (~3.0 GB, once)")


@pytest.mark.parametrize(
    ("model", "notice"),
    [
        (DEFAULT_SUMMARY_MODEL, "Downloading the summary model (~3.0 GB, once)"),
        (PREVIOUS_DEFAULT_SUMMARY_MODEL, "Downloading the summary model (~0.9 GB, once)"),
        (QWEN_RESEARCH_SUMMARY_MODEL, "Downloading the summary model (~1.7 GB, once)"),
        ("me/custom-llm", "Downloading the summary model (once)"),
    ],
)
def test_download_notice_names_the_chosen_models_size(model, notice):
    assert summary.download_notice(model) == notice


def test_default_summary_model_is_apache_and_the_older_ones_stay_pinned():
    # Apache-2.0 default (Qwen3.5 4B); the 1.5B and the non-commercial Qwen Research 3B stay
    # selectable offline
    assert DEFAULT_SUMMARY_MODEL == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert not Config().summary.enabled  # still opt-in
    assert gpu.revision(DEFAULT_SUMMARY_MODEL) == "32f3e8ecf65426fc3306969496342d504bfa13f3"
    assert gpu.revision(PREVIOUS_DEFAULT_SUMMARY_MODEL) == (
        "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
    )
    assert gpu.revision(QWEN_RESEARCH_SUMMARY_MODEL) == "4f83f8f146fdf28b512a06562b671d7af4fab457"


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


def test_a_numbered_room_session_says_the_owner_is_one_of_the_room_speakers():
    system, _ = summary.build_messages("summary", "Speaker 1: hi", "en", room=True)
    assert "one of the numbered speakers in the room" in system["content"]
    assert "'Me' is the owner" not in system["content"]
    asked = []

    def gen(messages, _max_tokens):
        asked.append(messages[0]["content"])
        return "- x"

    room = [seg("hi", speaker="Speaker 1"), seg("hello", speaker="Speaker 2", t=3)]
    summary.summarize(room, gen, room=True)
    assert asked and all("numbered speakers in the room" in a for a in asked)
    asked.clear()
    summary.summarize([seg("hi"), seg("ok", speaker="Speaker 1")], gen, room=True)  # lone Me
    assert asked and all("'Me' is the owner" in a for a in asked)


def test_summarize_session_reads_a_room_start_from_session_json(tmp_path):
    room = [seg("hi", speaker="Speaker 1"), seg("hello", speaker="Speaker 2", t=3)]
    d = make_session(tmp_path, room)
    s = SessionInfo.load(d)
    s.trigger_kind, s.started_as_room = TriggerKind.ROOM, True
    s.save()
    asked = []

    def gen(messages, _max_tokens):
        asked.append(messages[0]["content"])
        return "- x"

    summary.summarize_session(d, Config(), gen)
    assert asked and all("one of the numbered speakers in the room" in a for a in asked)
