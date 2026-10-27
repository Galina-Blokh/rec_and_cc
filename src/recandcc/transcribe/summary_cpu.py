"""CPU summary generator for machines without MLX (Windows, Linux): ONNX Runtime itself.

Same interface as summary.MlxGenerator; summary.create_generator picks it when mlx-lm can't
run. It runs on the ``summary`` thread only (like the MLX one), never on UI or audio threads.

Backend: ``onnxruntime.InferenceSession`` on its CPU provider, which the app already depends on
(cp312 wheels for Windows, Linux and Apple silicon; none for Intel Macs), with ``tokenizers``
for the text. The GenAI layer is not used: its wheels ship telemetry code. The decoding loop
(greedy, with a KV cache) is ours and sends nothing anywhere.

Model: the same Qwen2.5 1.5B Instruct (Apache-2.0) as the MLX default, as the int4 ONNX export
that Hugging Face's ``onnx-community`` publishes. Its files are pinned to one repo commit and
their sha256s (CPU_MODELS), downloaded once into the app's model cache, and checked by size
(and sha256 once per process) before every load. Once the files are there, nothing here touches
the network. Logs never contain prompt or answer text.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs

from ..config import APP_NAME, DEFAULT_SUMMARY_MODEL, PREVIOUS_DEFAULT_SUMMARY_MODEL
from .summary import Messages, ModelUnavailable, SummaryUnavailable

log = logging.getLogger("recandcc.transcribe")

MODELS_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
DOWNLOAD_TIMEOUT_S = 30.0
DECODER = "onnx/model_q4.onnx"  # the pinned file that holds the graph and its weights
TOKENIZER = "tokenizer.json"
EOS_IDS = (151645, 151643)  # <|im_end|>, <|endoftext|>: the generation_config.json of Qwen2.5
REPETITION_PENALTY = 1.1  # as in MlxGenerator
REPETITION_WINDOW = 20  # the last tokens the penalty applies to: mlx-lm's default
PREFILL_STEP = 512  # prompt tokens per run: the graph returns logits for every position
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NOT_ON_DISK = "the summary model is not on disk or can't be loaded"  # as in MlxGenerator
TOO_LONG = "the transcript part is too long for the CPU summary model"
CONTEXT_LENGTH = 32768  # Qwen2.5's
STALE_PART_S = 3600.0  # a .part.<pid> file untouched this long is dead, whatever its pid
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5
_STILL_ACTIVE = 259
_DTYPES: dict[str, type] = {  # the ONNX tensor types a Qwen2 export uses
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}


@dataclass(frozen=True)
class ModelFile:
    path: str  # in the repo, and under the model's folder
    sha256: str
    size: int


@dataclass(frozen=True)
class CpuModel:
    repo: str
    revision: str  # full commit id
    files: tuple[ModelFile, ...]

    @property
    def folder(self) -> Path:
        return MODELS_DIR / f"{self.repo.replace('/', '--')}-{self.revision[:12]}"

    def url(self, f: ModelFile) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{f.path}"


# Qwen2.5 1.5B Instruct, int4 (MatMulNBits) ONNX, ~1.8 GB. To update: take the new commit
# (``curl -s https://huggingface.co/api/models/<repo>/revision/main``, field ``sha``) and the
# files' sha256 (LFS ``oid`` in ``.../tree/<sha>``, or sha256sum of the download).
QWEN2_5_1_5B = CpuModel(
    repo="onnx-community/Qwen2.5-1.5B-Instruct",
    revision="d54957e3af1be04546436e874a60fe386041b244",
    files=(
        ModelFile(
            DECODER,
            "70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6",
            1_787_566_590,
        ),
        ModelFile(
            TOKENIZER,
            "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
            7_031_673,
        ),
        ModelFile(
            "tokenizer_config.json",
            "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430",
            7_849,
        ),
    ),
)

# summary.model (an MLX repo) -> its CPU build. Only the Qwen2.5 1.5B has one, so the current
# default (the Qwen3.5 4B, no CPU build) and the previous default both map to it. Other models
# have none: a local folder laid out like the pinned build can still be given as summary.model.
CPU_MODELS: dict[str, CpuModel] = {
    DEFAULT_SUMMARY_MODEL: QWEN2_5_1_5B,
    PREVIOUS_DEFAULT_SUMMARY_MODEL: QWEN2_5_1_5B,
}
DOWNLOAD_SIZES = {name: "~1.8 GB" for name in CPU_MODELS}

_lock = threading.Lock()  # one download/check at a time (app summary thread vs. CLI calls)
_verified: set[Path] = set()  # files whose sha256 was checked in this process
_mapped_logged: set[str] = set()  # summary models whose CPU mapping was logged (once a process)


def cpu_available() -> bool:
    """Whether onnxruntime and tokenizers (both installed with the app) are importable. Checked
    without importing them; the model itself may still be unavailable (offline, a bad file)."""
    names = ("onnxruntime", "tokenizers")
    return all(importlib.util.find_spec(name) is not None for name in names)


def _is_valid(path: Path, f: ModelFile) -> bool:
    """On disk with the right size and, checked once per process, the right sha256."""
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    if path in _verified:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != f.sha256:
        log.warning("summary model file %s is corrupt; removing it", f.path)
        path.unlink(missing_ok=True)
        return False
    _verified.add(path)
    return True


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError(f"refusing redirect to non-https URL {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str) -> Any:
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https model URL {url!r}")
    opener = urllib.request.build_opener(_HttpsOnlyRedirect)
    return opener.open(url, timeout=DOWNLOAD_TIMEOUT_S)


def _download(url: str, path: Path, f: ModelFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part.{os.getpid()}")
    digest, written = hashlib.sha256(), 0
    try:
        with _open(url) as r, tmp.open("wb") as out:
            while block := r.read(1 << 20):
                written += len(block)
                if written > f.size:  # a bad or hostile server must not fill the disk
                    raise ValueError(f"download larger than the expected {f.size} bytes")
                digest.update(block)
                out.write(block)
        if written != f.size or digest.hexdigest() != f.sha256:
            raise ValueError(f"size or sha256 mismatch for {f.path}")
        tmp.replace(path)
        _verified.add(path)
    finally:
        tmp.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Whether process ``pid`` exists, without signalling it: on Windows os.kill(pid, 0) would
    terminate the process, so that platform asks the kernel instead."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:  # ERROR_ACCESS_DENIED means it exists, another user's
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # it exists, another user's
        return True
    return True


def _remove_stale_parts(folder: Path) -> None:
    """At a download's start, delete the ``.part.<pid>`` files of downloads that died: untouched
    for an hour, or from a process that is gone, or from this one (it holds the lock, so none of
    its own downloads is running)."""
    now = time.time()
    for part in folder.rglob(".*.part.*"):
        try:
            pid: int | None = int(part.name.rsplit(".part.", 1)[1])
        except ValueError:  # not a name we write: judged by its age alone
            pid = None
        dead = pid is not None and (pid == os.getpid() or not _pid_alive(pid))
        try:
            if dead or now - part.stat().st_mtime > STALE_PART_S:
                part.unlink(missing_ok=True)
        except OSError:
            log.debug("a stale download part could not be removed")


def model_dir(model: CpuModel, on_download: Callable[[], None] | None = None) -> Path:
    """Folder with ``model``'s verified files, downloading missing ones once (``on_download``
    runs before the first one). Raises if they can't be had."""
    folder = model.folder
    with _lock:
        missing = [f for f in model.files if not _is_valid(folder / f.path, f)]
        if missing and on_download is not None:
            on_download()
        if missing:
            _remove_stale_parts(folder)
        for f in missing:
            log.info("downloading summary model file %s (%.0f MB, once)", f.path, f.size / 1e6)
            t0 = time.monotonic()
            _download(model.url(f), folder / f.path, f)
            log.info("downloaded %s in %.1fs", f.path, time.monotonic() - t0)
    return folder


def _is_local_build(folder: Path) -> bool:
    """A folder laid out like the pinned build (the decoder and its tokenizer) is used as is."""
    return (folder / DECODER).is_file() and (folder / TOKENIZER).is_file()


def resolve(model: str, on_download: Callable[[], None] | None = None) -> Path:
    """Local folder to load for ``summary.model`` on the CPU."""
    if _is_local_build(Path(model)):
        return Path(model)
    cpu = CPU_MODELS.get(model)
    if cpu is None:
        log.info("summary model %s has no CPU build", model)
        raise ModelUnavailable("the summary model has no CPU build")
    if model not in _mapped_logged:  # once a process; model names only, never transcript text
        _mapped_logged.add(model)
        log.info(
            "summary model %s runs on the CPU as %s (a smaller model: weaker for Hebrew, Russian "
            "and action items)",
            model,
            cpu.repo,
        )
    try:
        return model_dir(cpu, on_download)
    except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
        log.info("summary model %s unavailable: %s", cpu.repo, type(e).__name__)
        raise ModelUnavailable(NOT_ON_DISK) from e


def chatml(messages: Messages) -> str:
    """The Qwen2.5 chat template (tokenizer_config.json of the pinned build) for plain messages:
    ChatML turns, the default system prompt when the first message isn't one, and the assistant
    turn left open for the answer. The tests render the template itself and compare."""
    system = {"role": "system", "content": QWEN_DEFAULT_SYSTEM}
    turns = [*messages] if messages and messages[0]["role"] == "system" else [system, *messages]
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in turns)
    return text + "<|im_start|>assistant\n"


@dataclass(frozen=True)
class _Graph:
    """The decoder a session runs, as its export names things: the layers, the cache's head
    shape and dtype, each input's numpy type, and the outputs the loop reads (logits first, then
    every layer's present key and value)."""

    layers: int
    kv_heads: int
    head_dim: int
    kv_dtype: type
    input_dtypes: dict[str, type]
    outputs: tuple[str, ...]
    position_ids: bool


def _read_graph(session: Any) -> _Graph:
    """Read the Qwen2 key/value layout from the session's own inputs and outputs. Raises
    ValueError for any other graph, which then fails at load, before anything runs."""
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = {o.name for o in session.get_outputs()}
    layers = 0
    while f"past_key_values.{layers}.key" in inputs:
        layers += 1
    past = [f"past_key_values.{n}.{kv}" for n in range(layers) for kv in ("key", "value")]
    present = [f"present.{n}.{kv}" for n in range(layers) for kv in ("key", "value")]
    required = {"input_ids", "attention_mask", *past}
    if not layers or not required <= inputs.keys() or set(inputs) - {*required, "position_ids"}:
        raise ValueError("not a decoder with a key/value cache for each layer")
    if not ({"logits", *present} <= outputs):
        raise ValueError("the decoder doesn't return logits and each layer's present cache")
    key = inputs[past[0]]
    kv_heads, head_dim = key.shape[1], key.shape[3]
    if not isinstance(kv_heads, int) or not isinstance(head_dim, int):
        raise ValueError("the cache's head shape isn't fixed")
    return _Graph(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_dtype=_DTYPES[key.type],
        input_dtypes={name: _DTYPES[i.type] for name, i in inputs.items()},
        outputs=("logits", *present),
        position_ids="position_ids" in inputs,
    )


def _pick(logits: np.ndarray, recent: list[int]) -> int:
    """The greedy choice after mlx-lm's repetition penalty: each distinct token of the last
    REPETITION_WINDOW (the prompt's included) has its logit divided by the penalty if positive,
    multiplied if negative. Without a penalty this is plain argmax."""
    scores = np.array(logits, dtype=np.float32)
    seen = np.unique(np.asarray(recent[-REPETITION_WINDOW:], dtype=np.int64))
    picked = scores[seen]
    scores[seen] = np.where(picked < 0, picked * REPETITION_PENALTY, picked / REPETITION_PENALTY)
    return int(np.argmax(scores))


Cache = list[tuple[Any, Any]]  # (key, value) per layer, each [1, kv_heads, positions, head_dim]


class CpuGenerator:
    """The summary LLM on the CPU through ONNX Runtime. Loaded on first use, used from one thread
    (the summary thread), freed by close(). ``pause`` runs before each generated token, so an
    answer yields to a recording that starts mid-way (as ``before_call`` does between calls)."""

    def __init__(
        self,
        model: str,
        allow_unpinned: bool = False,  # unused: only pinned CPU builds are downloaded
        on_download: Callable[[], None] | None = None,
        pause: Callable[[], None] | None = None,
    ) -> None:
        self.model_name = model
        self.on_download = on_download
        self.pause = pause
        self._session: Any = None
        self._tokenizer: Any = None
        self._graph: Any = None

    def load(self) -> None:
        if not cpu_available():
            raise SummaryUnavailable("local summaries need Apple silicon or onnxruntime")
        try:  # before the download: a runtime that won't import fetches nothing
            ort: Any = importlib.import_module("onnxruntime")
            tokenizers: Any = importlib.import_module("tokenizers")
        except ImportError as e:
            log.info("the CPU summary runtime can't be imported: %s", type(e).__name__)
            raise ModelUnavailable("onnxruntime can't be imported here") from e
        path = resolve(self.model_name, self.on_download)
        try:
            session = ort.InferenceSession(str(path / DECODER), providers=["CPUExecutionProvider"])
            tokenizer = tokenizers.Tokenizer.from_file(str(path / TOKENIZER))
            graph = _read_graph(session)
        except Exception as e:  # noqa: BLE001 - a broken or unsupported model folder
            log.info("summary model %s failed to load: %s", self.model_name, type(e).__name__)
            raise ModelUnavailable(NOT_ON_DISK) from e
        self._session, self._tokenizer, self._graph = session, tokenizer, graph

    def _empty_cache(self) -> Cache:
        g = self._graph
        empty = np.zeros((1, g.kv_heads, 0, g.head_dim), dtype=g.kv_dtype)
        return [(empty, empty)] * g.layers

    def _run(self, ids: list[int], cache: Cache) -> tuple[np.ndarray, Cache]:
        """Run ``ids`` after the cached positions: the last position's logits (a float32 copy)
        and the cache grown by ``ids``."""
        g = self._graph
        start = int(cache[0][0].shape[2])
        feed: dict[str, Any] = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, start + len(ids)), dtype=np.int64),
        }
        if g.position_ids:
            feed["position_ids"] = np.arange(start, start + len(ids), dtype=np.int64)[None, :]
        for n, (key, value) in enumerate(cache):
            feed[f"past_key_values.{n}.key"] = key
            feed[f"past_key_values.{n}.value"] = value
        feed = {name: np.asarray(v, dtype=g.input_dtypes[name]) for name, v in feed.items()}
        outs = dict(zip(g.outputs, self._session.run(list(g.outputs), feed), strict=True))
        grown = [(outs[f"present.{n}.key"], outs[f"present.{n}.value"]) for n in range(g.layers)]
        return np.array(outs["logits"][0, -1], dtype=np.float32), grown

    def _prefill(self, ids: list[int], cache: Cache) -> tuple[np.ndarray, Cache]:
        """The prompt in PREFILL_STEP pieces, so the logits held at once stay small; the last
        piece's last logits are the first answer token's."""
        pieces = [ids[at : at + PREFILL_STEP] for at in range(0, len(ids), PREFILL_STEP)]
        for piece in pieces[:-1]:
            _, cache = self._run(piece, cache)
        return self._run(pieces[-1], cache)

    def __call__(self, messages: Messages, max_tokens: int) -> str:
        if self._session is None:
            self.load()
        ids = self._tokenizer.encode(chatml(messages), add_special_tokens=False).ids
        if len(ids) + max_tokens > CONTEXT_LENGTH:  # refused, never cut: a cut prompt misleads
            raise SummaryUnavailable(TOO_LONG)
        logits, cache = self._prefill(ids, self._empty_cache())
        recent = list(ids)
        answer: list[int] = []
        while len(answer) < max_tokens:
            if self.pause is not None:  # e.g. waits while a recording runs, as between calls
                self.pause()
            token = _pick(logits, recent)
            if token in EOS_IDS:
                break
            answer.append(token)
            recent.append(token)
            if len(answer) < max_tokens:  # the logits for the next token; none after the last
                logits, cache = self._run([token], cache)
        return self._tokenizer.decode(answer, skip_special_tokens=True).strip()

    def close(self) -> None:
        """Free the session (~2 GB of RAM) between sessions."""
        if self._session is None:
            return
        self._session = self._tokenizer = self._graph = None
        gc.collect()
