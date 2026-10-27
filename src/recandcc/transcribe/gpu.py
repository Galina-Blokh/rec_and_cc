"""Apple-silicon GPU backend (MLX Whisper).

On an M-series Mac one Whisper pass on the GPU takes ~0.12 s (large-v3-turbo) versus ~2.5 s
on the CPU, which is what keeps live captions close to the speech. MLX isn't safe to drive
from several threads at once, so every GPU call goes through one dedicated thread; callers on
the CPU worker pool (VAD, language ID with the small CPU model, bookkeeping) just wait for
their result, so CPU and GPU work overlap. ``MLX_LOCK`` serialises those calls with the
other MLX user, the local summary model (summary.py), which takes it once per token.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import platform
import sys
import threading
import time
import types
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..config import TranscriptionConfig
from .merge import language_confidence
from .revisions import check_download_allowed, revision

log = logging.getLogger("recandcc.transcribe")

# Held around every MLX call, so Whisper and the summary model never run on the GPU at once.
MLX_LOCK = threading.Lock()

# faster-whisper model names -> MLX conversions on Hugging Face.
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large": "mlx-community/whisper-large-v3-mlx",  # faster-whisper alias of large-v3
}


def mlx_repo(model: str) -> str:
    """HF repo of the MLX version of ``model`` (a repo id or local path is used as is)."""
    if "/" in model:
        return model
    return MLX_REPOS.get(model, f"mlx-community/whisper-{model}-mlx")


def local_model_path(repo: str, allow_unpinned: bool = False) -> str:
    """Local snapshot path of ``repo``, downloading it only if it isn't on disk yet.

    Passing the local path to mlx_whisper (instead of the repo id) means loading never
    contacts Hugging Face once the model has been downloaded, so the app works offline. The
    repo's pinned commit (revisions.py) is used, both on disk and for the download; an
    unpinned repo is only downloaded with ``allow_unpinned`` (a cached copy loads either way).
    """
    if Path(repo).exists():
        return repo
    from huggingface_hub import snapshot_download

    rev = revision(repo)
    try:
        return snapshot_download(repo, revision=rev, local_files_only=True)
    except FileNotFoundError:
        # A missing or partial snapshot (LocalEntryNotFoundError, incl. IncompleteSnapshotError):
        # download it, or finish the interrupted download (a pinned one, or an opted-in one).
        check_download_allowed(repo, rev, allow_unpinned)
        log.info("model %s not on disk yet; downloading it once", repo)
    except Exception:
        # A snapshot that is present but unreadable or corrupt must not go online: surface it.
        log.warning(
            "model %s failed to load from the local cache %s; "
            "delete it to download the model again",
            repo,
            hf_cache_folder(repo),
        )
        raise
    return snapshot_download(repo, revision=rev)


def hf_cache_folder(repo: str) -> Path:
    """Hugging Face cache folder of ``repo`` (all its snapshots), for log messages."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        HF_HUB_CACHE = "~/.cache/huggingface/hub"
    return Path(HF_HUB_CACHE).expanduser() / ("models--" + repo.replace("/", "--"))


def gpu_available() -> bool:
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_whisper") is not None
    )


def use_gpu(cfg: TranscriptionConfig) -> bool:
    """Whether to transcribe with MLX: backend "mlx", or "auto" on Apple silicon."""
    if cfg.backend == "faster-whisper" or cfg.device == "cpu":
        return False
    if cfg.backend == "mlx":
        return True
    return gpu_available()


def _stub_word_timestamps(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(
        "word timestamps need numba and scipy, which RecAndCC does not install "
        "(ADR 0011, follow-up 4)"
    )


def import_mlx_whisper() -> Any:
    """Import mlx_whisper without numba and scipy, which pyproject.toml excludes.

    ``mlx_whisper/transcribe.py`` imports ``timing.add_word_timestamps``, and ``timing.py``
    imports numba and scipy, but only ``word_timestamps=True`` calls it. RecAndCC never asks
    for word timestamps, so when numba or scipy is missing a stub ``mlx_whisper.timing`` is
    registered first; it raises a clear error if word timestamps are ever requested. With
    both installed the real module is used. Every mlx_whisper import in RecAndCC must go
    through here first, since a direct one fails on the missing numba
    (tests/test_mlx_imports.py enforces that, tests/test_no_torch.py checks that neither
    library is imported).
    """
    if "mlx_whisper.timing" not in sys.modules and (
        importlib.util.find_spec("numba") is None or importlib.util.find_spec("scipy") is None
    ):
        stub = types.ModuleType("mlx_whisper.timing")
        stub.__dict__["add_word_timestamps"] = _stub_word_timestamps
        sys.modules["mlx_whisper.timing"] = stub
    return importlib.import_module("mlx_whisper")


class MlxWhisper:
    """One MLX Whisper model, driven from a single dedicated thread."""

    def __init__(self, model: str, allow_unpinned: bool = False) -> None:
        self.repo = mlx_repo(model)  # HF repo id; replaced by its local path in load()
        self.allow_unpinned = allow_unpinned
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="whisper-gpu")
        self._model = None

    def _get_model(self):
        # mlx / mlx_whisper are installed only on Apple Silicon (see pyproject.toml).
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        import_mlx_whisper()
        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        # Same cache mlx_whisper.transcribe() uses, so the weights are loaded once.
        return holder.get_model(self.repo, mx.float16)

    def load(self) -> None:
        def _load():
            t0 = time.monotonic()
            self.repo = local_model_path(self.repo, self.allow_unpinned)
            with MLX_LOCK:
                self._model = self._get_model()
            log.info("loaded MLX whisper %s on the GPU in %.1fs", self.repo, time.monotonic() - t0)

        self._thread.submit(_load).result()

    def transcribe(self, audio: np.ndarray, language: str) -> list[tuple[float, float, str]]:
        """(start, end, text) per segment, relative to the start of ``audio``."""

        def run():
            with MLX_LOCK:
                result = import_mlx_whisper().transcribe(
                    np.ascontiguousarray(audio, dtype=np.float32),
                    path_or_hf_repo=self.repo,
                    language=language,
                    condition_on_previous_text=False,
                    verbose=None,
                )
            segments = cast(list[dict[str, Any]], result["segments"])  # untyped in mlx_whisper
            return [(s["start"], s["end"], s["text"]) for s in segments]

        return self._thread.submit(run).result()

    def detect_language(self, audio: np.ndarray, allowed: Sequence[str]) -> tuple[str, float]:
        """Most likely language among ``allowed`` and its renormalised confidence."""

        def run():
            import mlx.core as mx  # pyright: ignore[reportMissingImports]

            import_mlx_whisper()
            from mlx_whisper.audio import (  # pyright: ignore[reportMissingImports]
                N_FRAMES,
                N_SAMPLES,
                log_mel_spectrogram,
                pad_or_trim,
            )

            with MLX_LOCK:
                model = self._model or self._get_model()
                mel = log_mel_spectrogram(
                    np.ascontiguousarray(audio, dtype=np.float32),
                    n_mels=model.dims.n_mels,
                    padding=N_SAMPLES,
                )
                segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
                _, probs = model.detect_language(segment)
            return list(probs.items())

        return language_confidence(self._thread.submit(run).result(), allowed)

    def close(self) -> None:
        self._thread.shutdown(wait=True)
        self._model = None
