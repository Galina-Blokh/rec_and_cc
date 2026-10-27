"""Local meeting summaries: ``summary.md`` (short summary + action items) next to a transcript.

Opt-in (``summary.enabled``). A small instruct model runs through mlx-lm on Apple silicon
(elsewhere on the CPU through onnxruntime-genai, see summary_cpu.py and create_generator),
pinned in revisions.py and loaded offline-first like Whisper (gpu.local_model_path); with
``summary.language_models`` the transcript's main spoken language picks it. Long
transcripts are summarised map-reduce: notes per chunk, then the summary and the action list
from the notes. Everything runs on one background ``summary`` thread; logs never contain
transcript or summary text.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
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

from ..config import (
    DEFAULT_SUMMARY_MODEL,
    PREVIOUS_DEFAULT_SUMMARY_MODEL,
    QWEN_RESEARCH_SUMMARY_MODEL,
    Config,
)
from ..models import Segment, SessionInfo, is_rtl
from .output import JSON_NAME, PDI, RLI, atomic_write, paragraphs

log = logging.getLogger("recandcc.transcribe")

SUMMARY_NAME = "summary.md"
# In a session folder while its summary is owed (queued, or stale after a speaker rename);
# startup re-queues these, so a summary cut short by quitting is redone.
PENDING_MARKER = ".summary-pending"
# Fingerprint of the transcript text a summary was made from. Unlike an mtime it survives a
# speaker rename or call-app names (they rewrite transcript.json but not what was said).
SOURCE_NAME = ".summary-source"
# ~2-5k tokens (en..he): well inside the context of every offered model, and each chunk's
# prompt prefill (which holds gpu.MLX_LOCK) stays within ~1-2 s.
CHUNK_CHARS = 8_000
PREFILL_STEP = 512  # prompt tokens per prefill step (lower peak memory than mlx-lm's 2048)
# One-time download size of each offered (pinned) summary model, for the download notice.
MODEL_SIZES = {
    DEFAULT_SUMMARY_MODEL: "~3.0 GB",
    PREVIOUS_DEFAULT_SUMMARY_MODEL: "~0.9 GB",
    QWEN_RESEARCH_SUMMARY_MODEL: "~1.7 GB",
}
# Thought-block markers of thinking models (Qwen3/3.5; Gemma 4's channel). Summaries run
# with thinking off; a thought block that still shows up is cut from every answer.
THOUGHT_MARKERS = (("<think>", "</think>"), ("<|channel>thought", "<channel|>"))
CUT_OFF = "the summary model ran out of tokens while thinking"
MAX_ROUNDS = 3  # reduce rounds before the notes are cut to fit
LANGUAGE_NAMES = {"en": "English", "ru": "Russian", "he": "Hebrew"}
# (summary heading, action items heading, "none") per language; English otherwise.
HEADINGS = {
    "en": ("Summary", "Action items", "None."),
    "ru": ("Краткое содержание", "Задачи", "Нет."),
    "he": ("סיכום", "משימות", "אין."),
}
NONE_MARK = "NONE"
# A bullet made only of one of these (any case, no punctuation) says there is no task.
NONE_WORDS = frozenset({NONE_MARK.casefold(), "nothing", "нет", "ничего", "אין"})

Messages = list[dict[str, str]]


class Generator(Protocol):
    def __call__(self, messages: Messages, max_tokens: int) -> str: ...


class ModelGenerator(Generator, Protocol):
    model_name: str

    def close(self) -> None: ...


def download_notice(model: str) -> str:
    """The one-time download notice, with the size when ``model`` is a pinned one (of the
    build this machine downloads: MLX, or the CPU one in summary_cpu)."""
    sizes = MODEL_SIZES
    if not mlx_lm_available():
        from . import summary_cpu

        if summary_cpu.cpu_available():
            sizes = summary_cpu.DOWNLOAD_SIZES
    size = sizes.get(model)
    return f"Downloading the summary model ({size + ', ' if size else ''}once)"


class SummaryUnavailable(RuntimeError):
    """No summary model can run here (platform, missing model offline); the message is a
    fixed reason without any transcript text."""


class ModelUnavailable(SummaryUnavailable):
    """This summary model can't be downloaded or loaded (another one may still work)."""


def main_language(segments: list[Segment]) -> str:
    """The language with the most speech time ("en" if none is known). It picks both the
    summary's language (prompt, headings) and its model (summary_model). Overlapping
    segments (Me and Others at once) count their shared time once."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for s in segments:
        if s.language and s.end > s.start:
            spans.setdefault(s.language, []).append((s.start, s.end))
    secs: dict[str, float] = {}
    for lang, intervals in spans.items():
        total, reach = 0.0, float("-inf")
        for start, end in sorted(intervals):
            if end > reach:
                total += end - max(start, reach)
                reach = end
        secs[lang] = total
    return max(secs, key=lambda k: secs[k]) if secs else "en"


def summary_model(cfg: Config, segments: list[Segment]) -> str:
    """``summary.language_models`` entry of the main language, else ``summary.model``. An
    entry that is neither pinned (revisions.py) nor a local path is ignored with a warning,
    unless ``transcription.allow_unpinned_models`` is on."""
    from .revisions import revision

    model = cfg.summary.language_models.get(main_language(segments))
    if model is None:
        return cfg.summary.model
    if revision(model) or Path(model).exists() or cfg.transcription.allow_unpinned_models:
        return model
    log.warning("ignoring unpinned summary.language_models model %s; using summary.model", model)
    return cfg.summary.model


def strip_thinking(text: str, markers=THOUGHT_MARKERS) -> str:
    """The answer after any thought block (``(start, end)`` in ``markers``). Raises
    SummaryUnavailable if a thought never ended: no answer, so no summary.md saying "none"."""
    for start, end in markers:
        if end in text:
            text = text.rsplit(end, 1)[1]
        elif start in text:
            raise SummaryUnavailable(CUT_OFF)
    return text.strip()


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


def build_messages(
    task: str, text: str, language: str, part: str = "", room: bool = False
) -> Messages:
    """Chat messages for ``task`` ("notes", "summary" or "actions") in ``language``.
    ``room``: a room session whose mic speakers are numbered, so no line is 'Me'."""
    lang = LANGUAGE_NAMES.get(language, f"the language with code '{language}'")
    owner = "the owner is one of the numbered speakers in the room" if room else "'Me' is the owner"
    system = (
        f"You summarise meeting transcripts. Lines are 'Speaker: text'; {owner}. "
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


def _reduce(
    lines: list[str], generate: Generator, language: str, max_tokens: int, room: bool = False
) -> str:
    """Map step: replace the text by chunk notes until it fits in one chunk."""
    chunks = chunk_lines(lines)
    for _ in range(MAX_ROUNDS):
        if len(chunks) <= 1:
            break
        n = len(chunks)
        notes = [
            generate(build_messages("notes", c, language, f"{i} of {n}", room), max_tokens)
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
    return [i for i in items if i and not _says_none(i)]


def _says_none(item: str) -> bool:
    """True if the bullet only says there is no task ("None.", "**NONE**", "Нет.", "אין.")."""
    return "".join(ch for ch in item if ch.isalnum()).casefold() in NONE_WORDS


def summarize(
    segments: list[Segment], generate: Generator, max_tokens: int = 400, room: bool = False
) -> tuple[str, str, list[str]]:
    """(language, summary, action items) of a transcript. ``room``: the session started as a
    room session; if its mic speakers were numbered (no 'Me' line), the prompts say so."""
    language = main_language(segments)
    room = room and not any(s.speaker == "Me" for s in segments)
    text = _reduce(transcript_lines(segments), generate, language, max_tokens, room)
    if not text:
        return language, "", []
    summary = generate(build_messages("summary", text, language, room=room), max_tokens).strip()
    items = action_items(generate(build_messages("actions", text, language, room=room), max_tokens))
    return language, summary, items


def transcript_fingerprint(segments: list[dict]) -> str:
    """Hash of each transcript.json segment's start, end and text (speaker labels excluded)."""
    h = hashlib.sha256()
    for seg in segments:
        h.update(f"{seg.get('start')}\x1f{seg.get('end')}\x1f{seg.get('text')}\x1e".encode())
    return h.hexdigest()


def format_summary(language: str, summary: str, items: list[str], model: str) -> str:
    title, actions, none = HEADINGS.get(language, HEADINGS["en"])

    def line(text: str) -> str:  # right-to-left text in an isolate, as in transcript.md
        return f"{RLI}{text}{PDI}" if is_rtl(language, text) else text

    out = [f"# {title}", "", line(summary or none), "", f"## {actions}", ""]
    out += [f"- {line(i)}" for i in items] or [line(none)]
    out += ["", f"_Generated on this computer by {Path(model).name}; it can be wrong._"]
    return "\n".join(out) + "\n"


def summarize_session(
    directory: Path,
    cfg: Config,
    generate: Generator | None = None,
    on_download: Callable[[str], None] | None = None,
    before_call: Callable[[], None] | None = None,
) -> Path:
    """Write ``summary.md`` for the final transcript in ``directory``; returns its path.
    Raises SummaryUnavailable if no model can run (nothing is written then). Without
    ``generate``, the routed model (summary_model) is loaded on first use and freed when done,
    so only one summary model is ever in memory; ``on_download(model)`` runs before its
    one-time download. A routed model that can't be downloaded or loaded falls back once to
    ``summary.model``. ``before_call`` runs before each model call."""
    directory = Path(directory)
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data.get("segments", [])]
    model = summary_model(cfg, segments)
    try:
        room = SessionInfo.load(directory).started_as_room
    except Exception as e:  # noqa: BLE001 - only a hint for the prompt
        log.debug("no session.json for the summary prompt (%s)", type(e).__name__)
        room = False
    t0 = time.monotonic()

    def run(gen: Generator) -> tuple[str, str, list[str]]:
        def call(messages: Messages, max_tokens: int) -> str:
            if before_call is not None:
                before_call()
            return gen(messages, max_tokens)

        return summarize(segments, call, cfg.summary.max_tokens, room)

    def run_model(name: str) -> tuple[str, str, list[str]]:
        notice = on_download
        hook = None if notice is None else lambda: notice(name)
        owned = create_generator(name, cfg.transcription.allow_unpinned_models, hook)
        try:  # the real model, freed again after this session
            return run(owned)
        finally:
            owned.close()

    if generate is not None:
        language, summary, items = run(generate)
    else:
        try:
            language, summary, items = run_model(model)
        except ModelUnavailable:
            if model == cfg.summary.model:
                raise
            log.info("summary model %s unavailable; using %s", model, cfg.summary.model)
            model = cfg.summary.model
            language, summary, items = run_model(model)
    with contextlib.suppress(OSError):  # an old fingerprint must not sit next to a new summary
        (directory / SOURCE_NAME).unlink(missing_ok=True)
    path = atomic_write(directory / SUMMARY_NAME, format_summary(language, summary, items, model))
    with contextlib.suppress(OSError):  # no fingerprint: export falls back to the mtime rule
        atomic_write(directory / SOURCE_NAME, transcript_fingerprint(data.get("segments", [])))
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


def create_generator(
    model: str, allow_unpinned: bool = False, on_download: Callable[[], None] | None = None
) -> ModelGenerator:
    """The summary model's generator on this machine: mlx-lm on Apple silicon, else the CPU
    one (onnxruntime-genai, summary_cpu.py, imported only here). With neither, MlxGenerator,
    whose load() says why summaries can't run."""
    if not mlx_lm_available():
        from . import summary_cpu

        if summary_cpu.cpu_available():
            return summary_cpu.CpuGenerator(model, allow_unpinned, on_download)
    return MlxGenerator(model, allow_unpinned, on_download)


def _template_reads_thinking_flag(tokenizer: Any) -> bool:
    """Whether the chat template reads ``enable_thinking`` (Qwen3.5's does, Qwen2.5's doesn't).
    Unlike ``has_thinking``, this doesn't depend on the mlx-lm version."""
    template = getattr(tokenizer, "chat_template", None)
    return isinstance(template, str) and "enable_thinking" in template


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
            raise ModelUnavailable("the summary model is not on disk or can't be loaded") from e
        from mlx_lm import load  # pyright: ignore[reportMissingImports]

        try:
            with MLX_LOCK:
                self._model, self._tokenizer = load(path)[:2]
        except Exception as e:  # noqa: BLE001 - e.g. a model type this mlx-lm doesn't know
            log.info("summary model %s failed to load: %s", self.model_name, type(e).__name__)
            raise ModelUnavailable("this mlx-lm can't load the summary model") from e
        thinks = getattr(self._tokenizer, "has_thinking", None)  # see __call__
        if thinks is None:
            log.warning("this mlx-lm doesn't say whether %s is a thinking model", self.model_name)
        log.debug("summary model %s thinks: %s", self.model_name, thinks)

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

        def generate(think: bool | None, budget: int) -> str:
            kw = {} if think is None else {"enable_thinking": think}
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kw)
            steps = stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=budget,
                prefill_step_size=PREFILL_STEP,
                logits_processors=make_logits_processors(repetition_penalty=1.1),
            )
            out = []
            while True:
                with MLX_LOCK:  # the prefill, then one token at a time
                    step = next(steps, None)
                if step is None:
                    return "".join(out)
                out.append(step.text)

        # A thinking model answers with thinking off: mlx-lm's TokenizerWrapper.has_thinking
        # (Qwen3.5) or a chat template that reads enable_thinking (Qwen3.5's does). Without the
        # flag the template's default turns thinking on, so a missing has_thinking (another
        # mlx-lm) must not be enough to lose it. Qwen2.5's template reads no flag: its prompt is
        # unchanged. Thought markers are stripped either way.
        thinking = getattr(tokenizer, "has_thinking", False) is True
        think = False if thinking or _template_reads_thinking_flag(tokenizer) else None
        markers = THOUGHT_MARKERS
        start, end = getattr(tokenizer, "think_start", None), getattr(tokenizer, "think_end", None)
        if thinking and isinstance(start, str) and isinstance(end, str):
            markers = (*markers, (start, end))
        return strip_thinking(generate(think, max_tokens), markers)

    def close(self) -> None:
        """Free the model, so its 1-3 GB don't stay resident between sessions."""
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
        summarize_session(  # reads cfg now: a config reload may have changed the model
            directory,
            self.cfg,
            self.generate,
            on_download=lambda model: self.notify("Summary model", download_notice(model)),
            before_call=self._wait_idle,  # a recording started: pause between model calls
        )

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
