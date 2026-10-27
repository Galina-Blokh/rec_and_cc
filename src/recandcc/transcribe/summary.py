"""Local meeting summaries: ``summary.md`` (short summary + action items) next to a transcript.

Opt-in (``summary.enabled``). A small instruct model runs through mlx-lm on Apple silicon,
pinned in revisions.py and loaded offline-first like Whisper (gpu.local_model_path). Long
transcripts are summarised map-reduce: notes per chunk, then the summary and the action list
from the notes. Everything runs on one background ``summary`` thread; logs never contain
transcript or summary text.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import platform
import queue
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from ..config import Config
from ..models import Segment, is_rtl
from .output import JSON_NAME, PDI, RLI, atomic_write, paragraphs

log = logging.getLogger("recandcc.transcribe")

SUMMARY_NAME = "summary.md"
CHUNK_CHARS = 12_000  # ~3-8k tokens (en..he): well inside Qwen2.5's 32k context
MAX_ROUNDS = 3  # reduce rounds before the notes are cut to fit
LANGUAGE_NAMES = {"en": "English", "ru": "Russian", "he": "Hebrew"}
# (summary heading, action items heading, "none") per language; English otherwise.
HEADINGS = {
    "en": ("Summary", "Action items", "None."),
    "ru": ("Краткое содержание", "Задачи", "Нет."),
    "he": ("סיכום", "משימות", "אין."),
}
NONE_MARK = "NONE"

Messages = list[dict[str, str]]


class Generator(Protocol):
    def __call__(self, messages: Messages, max_tokens: int) -> str: ...


class SummaryUnavailable(RuntimeError):
    """No summary model can run here (platform, missing model offline); the message is a
    fixed reason without any transcript text."""


def main_language(segments: list[Segment]) -> str:
    """The language with the most text ("en" if none is known)."""
    chars: dict[str, int] = {}
    for s in segments:
        if s.language:
            chars[s.language] = chars.get(s.language, 0) + len(s.text)
    return max(chars, key=lambda k: chars[k]) if chars else "en"


def transcript_lines(segments: list[Segment]) -> list[str]:
    return [f"{p.speaker}: {p.text}" for p in paragraphs(segments) if p.text]


def chunk_lines(lines: list[str], max_chars: int = 0) -> list[str]:
    """Join lines into chunks of at most ``max_chars`` (default CHUNK_CHARS; a longer line is
    split)."""
    max_chars = max_chars or CHUNK_CHARS
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        while len(line) > max_chars:
            if cur:
                chunks.append("\n".join(cur))
                cur, size = [], 0
            chunks.append(line[:max_chars])
            line = line[max_chars:]
        if cur and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def build_messages(task: str, text: str, language: str, part: str = "") -> Messages:
    """Chat messages for ``task`` ("notes", "summary" or "actions") in ``language``."""
    lang = LANGUAGE_NAMES.get(language, f"the language with code '{language}'")
    system = (
        "You summarise meeting transcripts. Lines are 'Speaker: text'; 'Me' is the owner. "
        f"Always answer in {lang}, even if parts of the input are in another language. "
        "Use only facts from the input; do not invent names, dates or tasks."
    )
    what = "meeting transcript" + (f" (part {part})" if part else "")
    if task == "notes":
        ask = (
            f"Write concise bullet-point notes of this {what}: key points, decisions, and every "
            "task with who will do it and when, if said."
        )
    elif task == "summary":
        ask = f"Summarise this {what} in 3-6 sentences: topic, key points and decisions."
    else:
        ask = (
            f"List every task that someone in this {what} was asked to do or promised to do. "
            "One line per task: '- <who>: <task>', where <who> is the name if said, else the "
            "speaker label, and add the deadline if one is said. "
            f"Write {NONE_MARK} only if there is no task at all."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{ask}\n\n---\n{text}\n---"},
    ]


def _reduce(lines: list[str], generate: Generator, language: str, max_tokens: int) -> str:
    """Map step: replace the text by chunk notes until it fits in one chunk."""
    chunks = chunk_lines(lines)
    for _ in range(MAX_ROUNDS):
        if len(chunks) <= 1:
            break
        n = len(chunks)
        notes = [
            generate(build_messages("notes", c, language, f"{i} of {n}"), max_tokens)
            for i, c in enumerate(chunks, 1)
        ]
        chunks = chunk_lines(notes)
    return chunks[0][:CHUNK_CHARS] if chunks else ""


def action_items(answer: str) -> list[str]:
    items = []
    for line in answer.splitlines():
        line = line.strip()
        if line[:1] in "-*•" and line[1:].strip():
            items.append(line[1:].strip())
        elif line[:1].isdigit() and line.lstrip("0123456789")[:1] in ".)":
            items.append(line.lstrip("0123456789")[1:].strip())
    return [i for i in items if i and i.upper() != NONE_MARK]


def summarize(
    segments: list[Segment], generate: Generator, max_tokens: int = 400
) -> tuple[str, str, list[str]]:
    """(language, summary, action items) of a transcript."""
    language = main_language(segments)
    text = _reduce(transcript_lines(segments), generate, language, max_tokens)
    if not text:
        return language, "", []
    summary = generate(build_messages("summary", text, language), max_tokens).strip()
    items = action_items(generate(build_messages("actions", text, language), max_tokens))
    return language, summary, items


def format_summary(language: str, summary: str, items: list[str], model: str) -> str:
    title, actions, none = HEADINGS.get(language, HEADINGS["en"])

    def line(text: str) -> str:  # right-to-left text in an isolate, as in transcript.md
        return f"{RLI}{text}{PDI}" if is_rtl(language, text) else text

    out = [f"# {title}", "", line(summary or none), "", f"## {actions}", ""]
    out += [f"- {line(i)}" for i in items] or [line(none)]
    out += ["", f"_Generated on this computer by {Path(model).name}; it can be wrong._"]
    return "\n".join(out) + "\n"


def summarize_session(directory: Path, cfg: Config, generate: Generator | None = None) -> Path:
    """Write ``summary.md`` for the final transcript in ``directory``; returns its path.
    Raises SummaryUnavailable if no model can run (nothing is written then)."""
    directory = Path(directory)
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data.get("segments", [])]
    t0 = time.monotonic()
    owned = None
    if generate is None:  # the real model, freed again after this session
        generate = owned = MlxGenerator(cfg.summary.model, cfg.transcription.allow_unpinned_models)
    try:
        language, summary, items = summarize(segments, generate, cfg.summary.max_tokens)
    finally:
        if owned is not None:
            owned.close()
    path = atomic_write(
        directory / SUMMARY_NAME, format_summary(language, summary, items, cfg.summary.model)
    )
    log.info(
        "summary of %s written: %d action item(s), %s, in %.1fs",
        directory.name,
        len(items),
        language,
        time.monotonic() - t0,
    )
    return path


def mlx_lm_available() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


class MlxGenerator:
    """The summary LLM through mlx-lm. Loaded on first use and used from one thread (the
    summary thread); each GPU step holds gpu.MLX_LOCK so Whisper can interleave."""

    def __init__(self, model: str, allow_unpinned: bool = False) -> None:
        self.model_name = model
        self.allow_unpinned = allow_unpinned
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if not mlx_lm_available():
            raise SummaryUnavailable("local summaries need Apple silicon with mlx-lm")
        from .gpu import MLX_LOCK, local_model_path

        try:
            path = local_model_path(self.model_name, self.allow_unpinned)
        except Exception as e:  # noqa: BLE001 - offline, unpinned, broken cache
            log.info("summary model %s unavailable: %s", self.model_name, type(e).__name__)
            raise SummaryUnavailable("the summary model is not on disk or can't be loaded") from e
        from mlx_lm import load  # pyright: ignore[reportMissingImports]

        with MLX_LOCK:
            self._model, self._tokenizer = load(path)[:2]

    def __call__(self, messages: Messages, max_tokens: int) -> str:
        if self._model is None:
            self.load()
        from mlx_lm import stream_generate  # pyright: ignore[reportMissingImports]
        from mlx_lm.sample_utils import (  # pyright: ignore[reportMissingImports]
            make_logits_processors,
        )

        from .gpu import MLX_LOCK

        model: Any = self._model
        tokenizer: Any = self._tokenizer
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        steps = stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            logits_processors=make_logits_processors(repetition_penalty=1.1),
        )
        out = []
        while True:
            with MLX_LOCK:  # one token at a time, so live captions keep flowing
                step = next(steps, None)
            if step is None:
                return "".join(out).strip()
            out.append(step.text)

    def close(self) -> None:
        self._model = self._tokenizer = None


class SummaryQueue:
    """Summarises finished sessions one at a time on a daemon ``summary`` thread.
    ``on_skip(reason)`` is called once per distinct reason when no model can run."""

    def __init__(
        self,
        cfg: Config,
        on_skip: Callable[[str], None] | None = None,
        generate: Generator | None = None,
    ) -> None:
        self.cfg = cfg
        self.on_skip = on_skip
        self.generate = generate
        self._q: queue.Queue[Path] = queue.Queue()
        self._skipped: set[str] = set()
        threading.Thread(target=self._loop, name="summary", daemon=True).start()

    def submit(self, directory: Path) -> None:
        self._q.put(Path(directory))

    def join(self) -> None:
        self._q.join()

    def _loop(self) -> None:
        while True:
            directory = self._q.get()
            try:
                summarize_session(directory, self.cfg, self.generate)
            except SummaryUnavailable as e:
                log.info("summary of %s skipped: %s", directory.name, e)
                if self.on_skip and str(e) not in self._skipped:
                    self._skipped.add(str(e))
                    self.on_skip(str(e))
            except Exception as e:  # noqa: BLE001 - never kill the thread
                # the type only: a model error could quote the transcript
                log.error("summary of %s failed: %s", directory.name, type(e).__name__)
            finally:
                self._q.task_done()
