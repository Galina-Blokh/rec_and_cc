"""Pinned Hugging Face model revisions: the one place to update them.

Every Whisper download (MLX and faster-whisper) asks for an exact commit of its repo instead
of whatever ``main`` points to today, so a changed or compromised upstream repo can't silently
swap the weights we run. A commit id fixes the exact contents of every file in the repo (the
weights are LFS files, addressed by their sha256), and the local cache keeps one folder per commit
(``~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<commit>``), so offline loading
with ``local_files_only=True`` finds exactly the pinned snapshot.

Every model name the docs offer (``OFFERED_MODELS``) is pinned for both backends. To update a
model: look up the new commit
(``curl -s https://huggingface.co/api/models/<repo>/revision/main``, field ``sha``), replace
the hash below, run ``uv run pytest -m slow`` once online. Repos not listed here (custom repo
ids) are only downloaded with ``transcription.allow_unpinned_models = true``; already-cached
copies load either way.
"""

from __future__ import annotations

import logging

log = logging.getLogger("recandcc.transcribe")

# Model names offered in README.md / docs/CONFIGURATION.md (tests check each is pinned).
OFFERED_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "turbo")

# Hugging Face repo id -> pinned commit.
HF_REVISIONS: dict[str, str] = {
    # MLX (Apple-silicon GPU) models, see gpu.py
    "mlx-community/whisper-large-v3-turbo": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    "mlx-community/whisper-large-v3-mlx": "49e6aa286ad60c14352c404340ded53710378a11",
    "mlx-community/whisper-medium-mlx": "7fc08c4eac4c316526498f147dfdee6f6303f975",
    "mlx-community/whisper-small-mlx": "45f3915923c7a79a5a5b5a7d909d39aeb0e5630e",
    "mlx-community/whisper-base-mlx": "1e3e249fb8d01c655324bd6841b1deadffd6d04c",
    "mlx-community/whisper-tiny-mlx": "6caf9c55601caafbe6508a8b0d216bdf4783c4e8",
    # faster-whisper (CPU / CUDA) models, see engine.py
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "Systran/faster-whisper-large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "Systran/faster-whisper-medium": "08e178d48790749d25932bbc082711ddcfdfbc4f",
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",
    "Systran/faster-whisper-base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "Systran/faster-whisper-tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
    # Hebrew-tuned ivrit.ai Whisper (transcription.language_models), faster-whisper builds
    "ivrit-ai/whisper-large-v3-turbo-ct2": "72ad623a37947395efcc3933132353790e5a12f5",
    "ivrit-ai/whisper-large-v3-ct2": "e9ed4a4a98d761b0f617d668303de2c514236c66",
    # ... and their MLX conversions
    "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx": "53ad8c6cd8b32eb0303f093a404ae13c1b1d567f",
    "mlx-community/ivrit-ai-whisper-large-v3-mlx": "097c0cb2bb4288a3c82f6d524a39c4ce09afa187",
    # Local summary LLM (mlx-lm, Apple silicon), see summary.py
    "mlx-community/Qwen2.5-3B-Instruct-4bit": "4f83f8f146fdf28b512a06562b671d7af4fab457",
    # ... and the Hebrew-capable ones offered for summary.language_models (Gemma 4, Apache-2.0
    # upstream). model.safetensors sha256 at these commits (HF API, LFS oid):
    # e4b 932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff (5.1 GB),
    # e2b 038e39a37a7667373d2c3991375446b10c96ae1d717a68674870343db376b76e (3.6 GB).
    "mlx-community/gemma-4-e4b-it-4bit": "475b9088d29754a3379866cf5aeb6b41acd313c2",
    "mlx-community/gemma-4-e2b-it-4bit": "238767527555cb75a05732a84dff5d6ba0dd6809",
}

# Summary models offered in the docs for summary.model / summary.language_models (tests
# check each is pinned).
OFFERED_SUMMARY_MODELS = (
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gemma-4-e2b-it-4bit",
)

# The same ivrit.ai model in both formats: CTranslate2 (faster-whisper) repo -> MLX repo. Either
# name in `transcription.language_models` loads the build that fits the backend in use.
IVRIT_MLX_REPOS: dict[str, str] = {
    "ivrit-ai/whisper-large-v3-turbo-ct2": "mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx",
    "ivrit-ai/whisper-large-v3-ct2": "mlx-community/ivrit-ai-whisper-large-v3-mlx",
}

# faster-whisper model names -> their repos (same mapping as faster_whisper.utils._MODELS,
# limited to the models pinned above).
FASTER_WHISPER_REPOS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "tiny": "Systran/faster-whisper-tiny",
}


def ct2_repo(model: str) -> str:
    """The faster-whisper build of ``model``: an ivrit.ai MLX repo maps to its CTranslate2
    repo, anything else is returned as is."""
    return next((ct2 for ct2, mlx in IVRIT_MLX_REPOS.items() if mlx == model), model)


class UnpinnedModelError(RuntimeError):
    """A model with no pinned revision is not on disk and unpinned downloads are off."""


def revision(repo: str) -> str | None:
    """Pinned commit of ``repo`` (a Hugging Face repo id), or None if it isn't pinned."""
    return HF_REVISIONS.get(repo)


def faster_whisper_revision(model: str) -> str | None:
    """Pinned commit for a faster-whisper model name or repo id, or None if not pinned."""
    return revision(FASTER_WHISPER_REPOS.get(model, model))


def check_download_allowed(model: str, rev: str | None, allow_unpinned: bool) -> None:
    """Raise UnpinnedModelError before downloading an unpinned ``model``, unless allowed."""
    if rev is not None:
        return
    if not allow_unpinned:
        log.error(
            "model %s has no pinned revision and is not on disk; refusing to download it. "
            "Use an offered model name, a local model path, or set "
            "transcription.allow_unpinned_models = true",
            model,
        )
        raise UnpinnedModelError(f"refusing to download unpinned model {model}")
    log.warning("model %s has no pinned revision; downloading its latest version", model)
