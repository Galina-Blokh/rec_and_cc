"""Apple-silicon GPU backend (MLX Whisper).

On an M-series Mac one Whisper pass on the GPU takes ~0.12 s (large-v3-turbo) versus ~2.5 s
on the CPU, which is what keeps live captions close to the speech. MLX isn't safe to drive
from several threads at once, so every GPU call goes through one dedicated thread; callers on
the CPU worker pool (VAD, language ID with the small CPU model, bookkeeping) just wait for
their result, so CPU and GPU work overlap. ``MLX_LOCK`` serialises those calls with the
other MLX user, the local summary model (summary.py), which holds it for a prompt prefill
and then once per token.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import platform
import re
import sys
import threading
import time
import types
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..config import TranscriptionConfig
from .merge import language_confidence
from .revisions import IVRIT_MLX_REPOS, check_download_allowed, revision

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
    if model in IVRIT_MLX_REPOS:  # a CTranslate2 build named in language_models
        return IVRIT_MLX_REPOS[model]
    if "/" in model or "\\" in model or Path(model).is_dir():  # repo id or local folder
        return model
    return MLX_REPOS.get(model, f"mlx-community/whisper-{model}-mlx")


def local_model_path(
    repo: str, allow_unpinned: bool = False, on_download: Callable[[], None] | None = None
) -> str:
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
        if on_download is not None:
            on_download()
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


# The mlx-whisper release mlx_timing.py was copied from; another version keeps words off.
VENDORED_TIMING_VERSION = "0.4.3"
_version_warned = False


def _stub_word_timestamps(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError(
        "word timestamps need numba and scipy, or mlx-whisper "
        f"{VENDORED_TIMING_VERSION} for RecAndCC's numpy aligner (ADR 0011, follow-up 4)"
    )


def _mlx_whisper_version() -> str | None:
    """Installed mlx-whisper version, without importing it. ``mlx_whisper/_version.py`` comes
    first: a PyInstaller bundle has no dist-info for importlib.metadata."""
    try:
        spec = importlib.util.find_spec("mlx_whisper")
    except (ImportError, ValueError):
        spec = None
    for folder in (spec and spec.submodule_search_locations) or []:
        try:
            text = (Path(folder) / "_version.py").read_text(encoding="utf-8")
        except OSError:
            continue
        if m := re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", text, re.MULTILINE):
            return m.group(1)
    try:
        return importlib.metadata.version("mlx-whisper")
    except importlib.metadata.PackageNotFoundError:
        return None


def _install_timing() -> None:
    """Register ``mlx_whisper.timing`` before mlx_whisper imports the real one (which needs
    numba and scipy): our numpy port ``mlx_timing`` for the version it was copied from, else
    a stub that turns word timestamps off. With numba and scipy installed, the real one."""
    if "mlx_whisper.timing" in sys.modules or (
        importlib.util.find_spec("numba") is not None
        and importlib.util.find_spec("scipy") is not None
    ):
        return
    version = _mlx_whisper_version()
    if version == VENDORED_TIMING_VERSION:
        from . import mlx_timing

        sys.modules["mlx_whisper.timing"] = mlx_timing
        return
    log.debug("mlx-whisper %s is not %s: stub word timing", version, VENDORED_TIMING_VERSION)
    stub = types.ModuleType("mlx_whisper.timing")
    stub.__dict__["add_word_timestamps"] = _stub_word_timestamps
    sys.modules["mlx_whisper.timing"] = stub


def mlx_word_timestamps() -> bool:
    """Whether ``MlxWhisper.transcribe(..., words=True)`` works (not with the stub). Called
    only when words are wanted (split_on_speaker_change), so a mismatch warns here, once."""
    global _version_warned
    _install_timing()
    # What mlx_whisper.transcribe really calls, once it is loaded; else what it will import.
    owner = sys.modules.get("mlx_whisper.transcribe") or sys.modules.get("mlx_whisper.timing")
    ok = getattr(owner, "add_word_timestamps", None) is not _stub_word_timestamps
    if not ok and not _version_warned:
        _version_warned = True
        log.warning(
            "mlx-whisper %s is not %s: MLX word timestamps (split_on_speaker_change) are off",
            _mlx_whisper_version(),
            VENDORED_TIMING_VERSION,
        )
    return ok


def import_mlx_whisper() -> Any:
    """Import mlx_whisper without numba and scipy, which pyproject.toml excludes.

    ``mlx_whisper/transcribe.py`` imports ``timing.add_word_timestamps``, and ``timing.py``
    imports numba and scipy (only ``word_timestamps=True`` calls it). So when either is
    missing, ``_install_timing`` first registers ``mlx_timing`` (the same functions on numpy
    kernels) as ``mlx_whisper.timing``, or a stub if the mlx-whisper version differs from the
    one it was copied from. Every mlx_whisper import in RecAndCC must go through here first,
    since a direct one fails on the missing numba (tests/test_mlx_imports.py enforces that,
    tests/test_no_torch.py checks that neither library is imported).
    """
    _install_timing()
    return importlib.import_module("mlx_whisper")


class MlxWhisper:
    """One MLX Whisper model, driven from a single dedicated thread."""

    def __init__(
        self, model: str, allow_unpinned: bool = False, share: MlxWhisper | None = None
    ) -> None:
        self.repo = mlx_repo(model)  # HF repo id; replaced by its local path in load()
        self.allow_unpinned = allow_unpinned
        # `share`: another model's GPU thread, so that two models (language_models) never
        # drive MLX from two threads at once.
        self._owns_thread = share is None
        self._thread = share._thread if share else ThreadPoolExecutor(1, "whisper-gpu")
        self._model = None

    def _get_model(self):
        # mlx / mlx_whisper are installed only on Apple Silicon (see pyproject.toml).
        import mlx.core as mx  # pyright: ignore[reportMissingImports]

        import_mlx_whisper()
        holder = importlib.import_module("mlx_whisper.transcribe").ModelHolder
        if self._model is not None:
            # mlx_whisper.transcribe() caches one model only: hand it ours (both stay in memory)
            # instead of reloading weights whenever the other language's model ran last.
            holder.model, holder.model_path = self._model, self.repo
        # Same cache mlx_whisper.transcribe() uses, so the weights are loaded once.
        return holder.get_model(self.repo, mx.float16)

    def load(self) -> None:
        t0 = time.monotonic()
        # Resolve (and, the first time, download) on the calling thread: only the weight load
        # goes to the GPU thread, so a slow download of a model that shares it never stalls
        # the other model's transcribe/detect_language calls (or its close()).
        self.repo = local_model_path(self.repo, self.allow_unpinned)

        def _load():
            with MLX_LOCK:
                self._model = self._get_model()

        self._thread.submit(_load).result()
        log.info("loaded MLX whisper %s on the GPU in %.1fs", self.repo, time.monotonic() - t0)

    def transcribe(self, audio: np.ndarray, language: str, words: bool = False) -> list[Any]:
        """(start, end, text) per segment, relative to the start of ``audio``. With ``words``
        (only when ``mlx_word_timestamps()``): (start, end, text, [(start, end, " word")])."""

        def run():
            with MLX_LOCK:
                self._get_model()  # make ours the model mlx_whisper.transcribe() uses
                result = import_mlx_whisper().transcribe(
                    np.ascontiguousarray(audio, dtype=np.float32),
                    path_or_hf_repo=self.repo,
                    language=language,
                    condition_on_previous_text=False,
                    verbose=None,
                    **({"word_timestamps": True} if words else {}),
                )
            segments = cast(list[dict[str, Any]], result["segments"])  # untyped in mlx_whisper
            if not words:
                return [(float(s["start"]), float(s["end"]), s["text"]) for s in segments]
            return [
                (
                    float(s["start"]),
                    float(s["end"]),
                    s["text"],
                    [(float(w["start"]), float(w["end"]), w["word"]) for w in s.get("words", [])],
                )
                for s in segments
            ]

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
        if self._owns_thread:
            self._thread.shutdown(wait=True)
        self._model = None
