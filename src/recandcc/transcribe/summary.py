"""Local meeting summaries: ``summary.md`` (short summary + action items) next to a transcript.

Opt-in (``summary.enabled``). A small instruct model runs through mlx-lm on Apple silicon,
pinned in revisions.py and loaded offline-first like Whisper (gpu.local_model_path). Long
transcripts are summarised map-reduce: notes per chunk, then the summary and the action list
from the notes. Everything runs on one background ``summary`` thread; logs never contain
transcript or summary text.
"""

from __future__ import annotations

import contextlib
import gc
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

from ..config import DEFAULT_SUMMARY_MODEL, QWEN_RESEARCH_SUMMARY_MODEL, Config
from ..models import Segment, SessionInfo, is_rtl
from .output import JSON_NAME, PDI, RLI, atomic_write, paragraphs

log = logging.getLogger("recandcc.transcribe")

SUMMARY_NAME = "summary.md"
# In a session folder while its summary is owed (queued, or stale after a speaker rename);
# startup re-queues these, so a summary cut short by quitting is redone.
PENDING_MARKER = ".summary-pending"
# ~2-5k tokens (en..he): well inside Qwen2.5's 32k context, and each chunk's prompt prefill
# (which holds gpu.MLX_LOCK) stays within ~1-2 s.
CHUNK_CHARS = 8_000
PREFILL_STEP = 512  # prompt tokens per prefill step (lower peak memory than mlx-lm's 2048)
# One-time download size of each pinned summary model, for the download notice.
DOWNLOAD_SIZES = {DEFAULT_SUMMARY_MODEL: "~0.9 GB", QWEN_RESEARCH_SUMMARY_MODEL: "~1.7 GB"}
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


def download_notice(model: str) -> str:
    """The one-time download notice, with the size when ``model`` is a pinned one."""
    size = DOWNLOAD_SIZES.get(model)
    return f"Downloading the summary model ({size + ', ' if size else ''}once)"


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
    if len(chunks) > 1 or (chunks and len(chunks[0]) > CHUNK_CHARS):
        log.debug("summary notes still %d chunks after %d rounds; cut", len(chunks), MAX_ROUNDS)
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


def mark_pending(directory: Path) -> None:
    """Note that ``directory`` still needs (a fresh) summary.md. Never recreates a deleted
    folder; never raises."""
    with contextlib.suppress(OSError):
        (Path(directory) / PENDING_MARKER).touch()


def clear_pending(directory: Path) -> None:
    try:
        (Path(directory) / PENDING_MARKER).unlink(missing_ok=True)
    except OSError:
        log.warning("could not clear the summary marker of %s", Path(directory).name)


def pending_summaries(output_dir: Path) -> list[Path]:
    """Transcribed session folders whose summary is still owed, oldest first (discarded and
    not yet transcribed ones are skipped: the latter get one after their transcription)."""
    out = []
    for marker in sorted(Path(output_dir).glob(f"*/*/{PENDING_MARKER}")):
        d = marker.parent
        if (d / SessionInfo.DISCARD_MARKER).exists() or not (d / JSON_NAME).exists():
            continue
        try:
            if SessionInfo.load(d).transcribed:
                out.append(d)
        except Exception:  # noqa: BLE001 - unreadable session.json: nothing to summarise
            log.warning("skipping unreadable session %s", d.name)
    return out


def mlx_lm_available() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


class MlxGenerator:
    """The summary LLM through mlx-lm. Loaded on first use and used from one thread (the
    summary thread). gpu.MLX_LOCK is held for each prompt's prefill (~1-2 s per chunk) and then
    for one generated token at a time, so Whisper can interleave."""

    def __init__(
        self,
        model: str,
        allow_unpinned: bool = False,
        on_download: Callable[[], None] | None = None,
    ) -> None:
        self.model_name = model
        self.allow_unpinned = allow_unpinned
        self.on_download = on_download
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if not mlx_lm_available():
            raise SummaryUnavailable("local summaries need Apple silicon with mlx-lm")
        from .gpu import MLX_LOCK, local_model_path

        try:
            path = local_model_path(self.model_name, self.allow_unpinned, self.on_download)
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
            prefill_step_size=PREFILL_STEP,
            logits_processors=make_logits_processors(repetition_penalty=1.1),
        )
        out = []
        while True:
            with MLX_LOCK:  # the prefill, then one token at a time
                step = next(steps, None)
            if step is None:
                return "".join(out).strip()
            out.append(step.text)

    def close(self) -> None:
        """Free the model, so its 1-2 GB don't stay resident between sessions."""
        if self._model is None:
            return
        self._model = self._tokenizer = None
        gc.collect()
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        from .gpu import MLX_LOCK

        with MLX_LOCK:
            clear = getattr(mx, "clear_cache", None) or mx.metal.clear_cache
            clear()


class SummaryQueue:
    """Summarises finished sessions one at a time on a daemon ``summary`` thread.

    It never competes with a recording for the GPU: while ``busy()`` is true (the engine is
    recording or asking) it waits before a session and between model calls. ``notify(title,
    message)`` gets a one-time notice per skip reason and before the model's first download.
    """

    def __init__(
        self,
        cfg: Config,
        notify: Callable[[str, str], None] | None = None,
        busy: Callable[[], bool] | None = None,
        generate: Generator | None = None,
        poll_s: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.notify = notify or (lambda title, message: None)
        self.busy = busy or (lambda: False)
        self.generate = generate
        self.poll_s = poll_s
        self._q: queue.Queue[Path] = queue.Queue()
        self._skipped: set[str] = set()
        self._queued: set[Path] = set()  # queued or running (resolved): never twice at once
        self._queued_lock = threading.Lock()
        threading.Thread(target=self._loop, name="summary", daemon=True).start()

    def submit(self, directory: Path) -> None:
        directory = Path(directory)
        mark_pending(directory)  # removed when done; startup re-queues it if we quit first
        key = directory.resolve()
        with self._queued_lock:
            if key in self._queued:
                return
            self._queued.add(key)
        self._q.put(directory)

    def join(self) -> None:
        self._q.join()

    def _wait_idle(self) -> None:
        while self.busy():
            time.sleep(self.poll_s)

    def _job(self, directory: Path) -> None:
        owned = None
        gen = self.generate
        if gen is None:
            gen = owned = MlxGenerator(
                self.cfg.summary.model,
                self.cfg.transcription.allow_unpinned_models,
                on_download=lambda: self.notify(
                    "Summary model", download_notice(self.cfg.summary.model)
                ),
            )

        def gated(messages: Messages, max_tokens: int) -> str:
            self._wait_idle()  # a recording started: pause between model calls
            return gen(messages, max_tokens)

        try:
            summarize_session(directory, self.cfg, gated)
        finally:
            if owned is not None:
                owned.close()

    def _loop(self) -> None:
        while True:
            directory = self._q.get()
            try:
                self._wait_idle()
                if not directory.exists():
                    continue  # deleted meanwhile
                if (directory / SessionInfo.DISCARD_MARKER).exists():
                    clear_pending(directory)
                    continue  # discarded: purge_discarded_sessions deletes it
                self._job(directory)
                clear_pending(directory)
            except SummaryUnavailable as e:
                clear_pending(directory)  # no model here: retrying at every start won't help
                log.info("summary of %s skipped: %s", directory.name, e)
                if str(e) not in self._skipped:
                    self._skipped.add(str(e))
                    self.notify("Summary skipped", str(e))
            except Exception as e:  # noqa: BLE001 - never kill the thread
                # the type only: a model error could quote the transcript
                log.error("summary of %s failed: %s", directory.name, type(e).__name__)
                clear_pending(directory)  # e.g. a broken transcript: don't retry every start
            finally:
                with self._queued_lock:
                    self._queued.discard(directory.resolve())
                self._q.task_done()
