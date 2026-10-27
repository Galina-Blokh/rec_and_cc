"""Opt-in semantic search: a small multilingual sentence-embedding model run through ONNX Runtime.

``paraphrase-multilingual-MiniLM-L12-v2`` (50+ languages incl. en and ru, Apache-2.0, 384-d)
in its int8 ONNX export from ``Xenova/paraphrase-multilingual-MiniLM-L12-v2``. Both files are
pinned to an exact repo commit and a sha256, and downloaded once (only when
``[search] semantic = true``) into the platform cache dir; afterwards nothing here touches
the network. ONNX Runtime and ``tokenizers`` are already installed as faster-whisper
dependencies, so no new dependency is needed.

``Embedder.embed`` returns L2-normalised float32 rows (cosine = dot product), or None when the
model isn't available (offline and not yet downloaded; retried at most every ``RETRY_AFTER_S``).
It blocks: call it from a worker thread, never from the UI or an audio thread.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import platformdirs

from .config import APP_NAME

log = logging.getLogger("recandcc.semantic")

MODELS_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
REVISION = "2c4055b12046f11709e9df2c122e59ffbdc2f900"
_BASE = f"https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/{REVISION}/"
MAX_TOKENS = 128
BATCH = 32
DIM = 384
RETRY_AFTER_S = 600.0  # after a failed download, don't try again for this long
DOWNLOAD_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


MODEL = ModelFile(
    name=f"minilm-l12-multilingual-{REVISION[:8]}-int8.onnx",
    url=_BASE + "onnx/model_quantized.onnx",
    sha256="66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc",
    size=118_308_126,
)
TOKENIZER = ModelFile(
    name=f"minilm-l12-multilingual-{REVISION[:8]}-tokenizer.json",
    url=_BASE + "tokenizer.json",
    sha256="b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441",
    size=17_082_913,
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray | None: ...


_download_lock = threading.Lock()
_last_failure = 0.0


def model_path(model: ModelFile, download: bool = True) -> Path | None:
    """Local path of `model`, downloading (and verifying) it once if needed. None if it is not
    on disk and can't be downloaded now. A model on disk is used without any network access."""
    global _last_failure
    path = MODELS_DIR / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        return None
    with _download_lock:
        if path.is_file() and path.stat().st_size == model.size:
            return path
        if _last_failure and time.monotonic() - _last_failure < RETRY_AFTER_S:
            return None
        try:
            _download(model, path)
        except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
            _last_failure = time.monotonic()
            log.warning("semantic search unavailable: can't download %s (%s)", model.name, e)
            return None
    return path


def _download(model: ModelFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    log.info("downloading search model %s (%.0f MB, once)", model.name, model.size / 1e6)
    if not model.url.startswith("https://"):
        raise ValueError(f"refusing non-https model URL {model.url!r}")
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(model.url, timeout=DOWNLOAD_TIMEOUT_S) as r,  # noqa: S310 - https only, checked above
            tmp.open("wb") as f,
        ):
            while block := r.read(1 << 20):
                digest.update(block)
                f.write(block)
        if digest.hexdigest() != model.sha256:
            raise ValueError(f"sha256 mismatch ({digest.hexdigest()})")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


class OnnxEmbedder:
    """The sentence model on the CPU. Loaded on first use; thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any] | None = None

    def _load(self) -> tuple[Any, Any] | None:
        with self._lock:
            if self._loaded is None:
                m, t = model_path(MODEL), model_path(TOKENIZER)
                if m is None or t is None:
                    return None
                import onnxruntime as ort  # lazy: only when semantic search is on
                from tokenizers import Tokenizer

                tok = Tokenizer.from_file(str(t))
                tok.enable_truncation(MAX_TOKENS)
                tok.enable_padding()
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2  # background work: leave cores to transcription
                self._loaded = (ort.InferenceSession(str(m), opts), tok)
            return self._loaded

    def embed(self, texts: list[str]) -> np.ndarray | None:
        loaded = self._load()
        if loaded is None:
            return None
        session, tok = loaded
        out = []
        for i in range(0, len(texts), BATCH):
            enc = tok.encode_batch(texts[i : i + BATCH])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = session.run(None, feed)[0]
            pooled = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9))
        if not out:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)


_default: OnnxEmbedder | None = None
_default_lock = threading.Lock()


def default_embedder() -> OnnxEmbedder:
    global _default
    with _default_lock:
        if _default is None:
            _default = OnnxEmbedder()
        return _default
