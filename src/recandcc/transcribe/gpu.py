"""Apple-silicon GPU backend (MLX Whisper).

On an M-series Mac one Whisper pass on the GPU takes ~0.12 s (large-v3-turbo) versus ~2.5 s
on the CPU, which is what keeps live captions close to the speech. MLX isn't safe to drive
from several threads at once, so every GPU call goes through one dedicated thread; callers on
the CPU worker pool (VAD, language ID with the small CPU model, bookkeeping) just wait for
their result, so CPU and GPU work overlap.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import platform
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..config import TranscriptionConfig
from .merge import language_confidence
from .revisions import revision

log = logging.getLogger("recandcc.transcribe")

# faster-whisper model names -> MLX conversions on Hugging Face.
MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def mlx_repo(model: str) -> str:
    """HF repo of the MLX version of ``model`` (a repo id or local path is used as is)."""
    if "/" in model:
        return model
    return MLX_REPOS.get(model, f"mlx-community/whisper-{model}-mlx")


def local_model_path(repo: str) -> str:
    """Local snapshot path of ``repo``, downloading it only if it isn't on disk yet.

    Passing the local path to mlx_whisper (instead of the repo id) means loading never
    contacts Hugging Face once the model has been downloaded, so the app works offline. The
    repo's pinned commit (revisions.py) is used, both on disk and for the download.
    """
    if Path(repo).exists():
        return repo
    from huggingface_hub import snapshot_download

    rev = revision(repo)
    if rev is None:
        log.info("model %s has no pinned revision; using its latest version", repo)
    try:
        return snapshot_download(repo, revision=rev, local_files_only=True)
    except FileNotFoundError:  # cache miss (huggingface_hub's LocalEntryNotFoundError)
        log.info("model %s not on disk yet; downloading it once", repo)
    except Exception:
        # Anything else (corrupt cache, permissions, ...) must not go online: surface it.
        log.warning("model %s failed to load from the local cache", repo)
        raise
    return snapshot_download(repo, revision=rev)


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


class MlxWhisper:
    """One MLX Whisper model, driven from a single dedicated thread."""

    def __init__(self, model: str) -> None:
        self.repo = mlx_repo(model)  # HF repo id; replaced by its local path in load()
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="whisper-gpu")
        self._model = None

    def _get_model(self):
        import mlx.core as mx

        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        # Same cache mlx_whisper.transcribe() uses, so the weights are loaded once.
        return holder.get_model(self.repo, mx.float16)

    def load(self) -> None:
        def _load():
            t0 = time.monotonic()
            self.repo = local_model_path(self.repo)
            self._model = self._get_model()
            log.info("loaded MLX whisper %s on the GPU in %.1fs", self.repo, time.monotonic() - t0)

        self._thread.submit(_load).result()

    def transcribe(self, audio: np.ndarray, language: str) -> list[tuple[float, float, str]]:
        """(start, end, text) per segment, relative to the start of ``audio``."""

        def run():
            import mlx_whisper

            result = mlx_whisper.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32),
                path_or_hf_repo=self.repo,
                language=language,
                condition_on_previous_text=False,
                verbose=None,
            )
            return [(s["start"], s["end"], s["text"]) for s in result["segments"]]

        return self._thread.submit(run).result()

    def detect_language(self, audio: np.ndarray, allowed: Sequence[str]) -> tuple[str, float]:
        """Most likely language among ``allowed`` and its renormalised confidence."""

        def run():
            import mlx.core as mx
            from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim

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
