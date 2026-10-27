"""CPU summary generator for machines without MLX (Windows, Linux): onnxruntime-genai.

Same interface as summary.MlxGenerator; summary.create_generator picks it when mlx-lm can't
run. It runs on the ``summary`` thread only (like the MLX one), never on UI or audio threads.

Backend: onnxruntime-genai (Microsoft, MIT), which ships cp312 wheels on PyPI for Windows
(x64, arm64) and Linux and builds on the onnxruntime the app already uses. llama-cpp-python
was the first choice (a GGUF of the same model), but PyPI only has its source archive, so a
Windows install would need a C++ toolchain.

Model: the same Qwen2.5 1.5B Instruct (Apache-2.0) as the MLX default, as the int4 ONNX
export that Hugging Face's ``onnx-community`` publishes. Its files are pinned to one repo
commit and their sha256s (CPU_MODELS), downloaded once into the app's model cache, and checked
by size (and sha256 once per process) before every load; ``genai_config.json`` (not in that
repo) is written by us. Once the files are there, nothing here touches the network. Logs
never contain prompt or answer text.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import platformdirs

from ..config import APP_NAME
from .summary import Messages, ModelUnavailable, SummaryUnavailable

log = logging.getLogger("recandcc.transcribe")

MODELS_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
DOWNLOAD_TIMEOUT_S = 30.0
REPETITION_PENALTY = 1.1  # as in MlxGenerator
CONFIG_NAME = "genai_config.json"
NOT_ON_DISK = "the summary model is not on disk or can't be loaded"  # as in MlxGenerator


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
    config: dict[str, Any] = field(default_factory=dict)  # written as genai_config.json

    @property
    def folder(self) -> Path:
        return MODELS_DIR / f"{self.repo.replace('/', '--')}-{self.revision[:12]}"

    def url(self, f: ModelFile) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{f.path}"


def _qwen2_config(onnx_file: str, layers: int, hidden: int, heads: int, kv_heads: int) -> dict:
    """genai_config.json for a Qwen2.5 Instruct transformers.js export (separate key/value
    caches that grow each step, position_ids input, greedy search)."""
    names = {
        "past_key_names": "past_key_values.%d.key",
        "past_value_names": "past_key_values.%d.value",
    }
    return {
        "model": {
            "type": "qwen2",
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "vocab_size": 151936,
            "context_length": 32768,
            "decoder": {
                "filename": onnx_file,
                "head_size": hidden // heads,
                "hidden_size": hidden,
                "num_attention_heads": heads,
                "num_hidden_layers": layers,
                "num_key_value_heads": kv_heads,
                "inputs": {
                    "input_ids": "input_ids",
                    "attention_mask": "attention_mask",
                    "position_ids": "position_ids",
                    **names,
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present.%d.key",
                    "present_value_names": "present.%d.value",
                },
            },
        },
        "search": {"do_sample": False, "num_beams": 1, "past_present_share_buffer": False},
    }


# Qwen2.5 1.5B Instruct, int4 (MatMulNBits) ONNX, ~1.8 GB. To update: take the new commit
# (``curl -s https://huggingface.co/api/models/<repo>/revision/main``, field ``sha``) and the
# files' sha256 (LFS ``oid`` in ``.../tree/<sha>``, or sha256sum of the download).
QWEN2_5_1_5B = CpuModel(
    repo="onnx-community/Qwen2.5-1.5B-Instruct",
    revision="d54957e3af1be04546436e874a60fe386041b244",
    files=(
        ModelFile(
            "onnx/model_q4.onnx",
            "70c24509f760fed9a8f50391165c46a87660197028a42c4547763f1c173ddea6",
            1_787_566_590,
        ),
        ModelFile(
            "tokenizer.json",
            "a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf",
            7_031_673,
        ),
        ModelFile(
            "tokenizer_config.json",
            "6a14243f0ae04884e597b3f032c05e15de9f417318fdcac24ab4e2680e691430",
            7_849,
        ),
    ),
    config=_qwen2_config("onnx/model_q4.onnx", layers=28, hidden=1536, heads=12, kv_heads=2),
)

# summary.model (an MLX repo) -> its CPU build. Other models have none: a local folder with a
# genai_config.json can still be given as summary.model.
CPU_MODELS: dict[str, CpuModel] = {"mlx-community/Qwen2.5-1.5B-Instruct-4bit": QWEN2_5_1_5B}
DOWNLOAD_SIZES = {"mlx-community/Qwen2.5-1.5B-Instruct-4bit": "~1.8 GB"}

_lock = threading.Lock()  # one download/check at a time (app summary thread vs. CLI calls)
_verified: set[Path] = set()  # files whose sha256 was checked in this process


def cpu_available() -> bool:
    return importlib.util.find_spec("onnxruntime_genai") is not None


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


def model_dir(model: CpuModel, on_download: Callable[[], None] | None = None) -> Path:
    """Folder with ``model``'s verified files and genai_config.json, downloading missing files
    once (``on_download`` runs before the first one). Raises if they can't be had."""
    folder = model.folder
    with _lock:
        missing = [f for f in model.files if not _is_valid(folder / f.path, f)]
        if missing and on_download is not None:
            on_download()
        for f in missing:
            log.info("downloading summary model file %s (%.0f MB, once)", f.path, f.size / 1e6)
            t0 = time.monotonic()
            _download(model.url(f), folder / f.path, f)
            log.info("downloaded %s in %.1fs", f.path, time.monotonic() - t0)
        config = folder / CONFIG_NAME
        text = json.dumps(model.config, indent=2)
        if not config.is_file() or config.read_text("utf-8") != text:
            config.write_text(text, "utf-8")
    return folder


def resolve(model: str, on_download: Callable[[], None] | None = None) -> Path:
    """Local folder to load for ``summary.model`` on the CPU."""
    if (Path(model) / CONFIG_NAME).is_file():
        return Path(model)
    cpu = CPU_MODELS.get(model)
    if cpu is None:
        log.info("summary model %s has no CPU build", model)
        raise ModelUnavailable("the summary model has no CPU build")
    try:
        return model_dir(cpu, on_download)
    except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
        log.info("summary model %s unavailable: %s", cpu.repo, type(e).__name__)
        raise ModelUnavailable(NOT_ON_DISK) from e


class CpuGenerator:
    """The summary LLM through onnxruntime-genai on the CPU. Loaded on first use, used from one
    thread (the summary thread), freed by close()."""

    def __init__(
        self,
        model: str,
        allow_unpinned: bool = False,  # unused: only pinned CPU builds are downloaded
        on_download: Callable[[], None] | None = None,
    ) -> None:
        self.model_name = model
        self.on_download = on_download
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if not cpu_available():
            raise SummaryUnavailable("local summaries need Apple silicon or onnxruntime-genai")
        path = resolve(self.model_name, self.on_download)
        import onnxruntime_genai as og  # pyright: ignore[reportMissingImports]

        for mod in (og, importlib.import_module("onnxruntime")):  # ETW events on Windows
            off = getattr(mod, "disable_telemetry_events", None)
            if off is not None:
                off()
        try:
            self._model = og.Model(str(path))
            self._tokenizer = og.Tokenizer(self._model)
        except Exception as e:  # noqa: BLE001 - a broken or unsupported model folder
            self._model = self._tokenizer = None
            log.info("summary model %s failed to load: %s", self.model_name, type(e).__name__)
            raise ModelUnavailable(NOT_ON_DISK) from e

    def __call__(self, messages: Messages, max_tokens: int) -> str:
        if self._model is None:
            self.load()
        import onnxruntime_genai as og  # pyright: ignore[reportMissingImports]

        tokenizer: Any = self._tokenizer
        prompt = tokenizer.apply_chat_template(json.dumps(messages), add_generation_prompt=True)
        ids = tokenizer.encode(prompt)
        params = og.GeneratorParams(self._model)
        params.set_search_options(
            max_length=len(ids) + max_tokens,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
        )
        gen = og.Generator(self._model, params)
        gen.append_tokens(ids)
        stream = tokenizer.create_stream()
        out = []
        while not gen.is_done():
            gen.generate_next_token()
            out.append(stream.decode(gen.get_next_tokens()[0]))
        return "".join(out).strip()

    def close(self) -> None:
        """Free the model (~2 GB of RAM) between sessions."""
        if self._model is None:
            return
        self._model = self._tokenizer = None
        gc.collect()
