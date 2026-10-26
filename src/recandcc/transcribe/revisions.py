"""Pinned Hugging Face model revisions: the one place to update them.

Every Whisper download (MLX and faster-whisper) asks for an exact commit of its repo instead
of whatever ``main`` points to today, so a changed or compromised upstream repo can't silently
swap the weights we run. A commit id fixes the exact contents of every file in the repo (the
weights are LFS files, addressed by their sha256), and the local cache keeps one folder per commit
(``~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<commit>``), so offline loading
with ``local_files_only=True`` finds exactly the pinned snapshot.

To update a model: look up the new commit (``huggingface-cli download <repo>`` prints the
snapshot path, or see the repo's "Files and versions" history), replace the hash below, run
``uv run pytest -m slow`` once online. Models not listed here (custom repo ids, other sizes)
are downloaded unpinned, with a log message.
"""

from __future__ import annotations

# Hugging Face repo id -> pinned commit.
HF_REVISIONS: dict[str, str] = {
    # MLX (Apple-silicon GPU) main model, see gpu.py
    "mlx-community/whisper-large-v3-turbo": "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    # faster-whisper (CPU / CUDA) models, see engine.py
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "Systran/faster-whisper-small": "536b0662742c02347bc0e980a01041f333bce120",
    "Systran/faster-whisper-base": "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66",
    "Systran/faster-whisper-tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
}

# faster-whisper model names -> their repos (same mapping as faster_whisper.utils._MODELS,
# limited to the models pinned above).
FASTER_WHISPER_REPOS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "tiny": "Systran/faster-whisper-tiny",
}


def revision(repo: str) -> str | None:
    """Pinned commit of ``repo`` (a Hugging Face repo id), or None if it isn't pinned."""
    return HF_REVISIONS.get(repo)


def faster_whisper_revision(model: str) -> str | None:
    """Pinned commit for a faster-whisper model name or repo id, or None if not pinned."""
    return revision(FASTER_WHISPER_REPOS.get(model, model))
