"""The MLX (GPU) path never imports torch, so torch is excluded from the environment.

mlx-whisper declares torch, but only its unused ``torch_whisper.py`` imports it, so
pyproject.toml drops it with ``[tool.uv] exclude-dependencies`` (ADR 0011, follow-up 4).
These tests catch an mlx-whisper upgrade or a code change that starts loading torch.
Each check runs in a fresh interpreter so modules imported by other tests don't count, and
a meta-path hook turns any torch import into an ImportError that the check reports.

Skipped where mlx-whisper isn't installed (Windows, Linux, Intel Macs).

Stories: none.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None, reason="mlx-whisper is not installed"
)

# Blocks and records every attempt to import torch, then runs the body of a check.
PRELUDE = """
import sys

attempts = []

class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            attempts.append(name)
            raise ImportError("torch import blocked by test_no_torch")
        return None

sys.meta_path.insert(0, NoTorch())
"""

EPILOGUE = """
assert not attempts, f"torch import attempted: {attempts}"
assert "torch" not in sys.modules, "torch is in sys.modules"
print("no-torch-ok")
"""


def run_check(body: str, env: dict[str, str] | None = None) -> None:
    code = PRELUDE + textwrap.dedent(body) + EPILOGUE
    proc = subprocess.run(  # noqa: S603 - fixed argv: this interpreter and our own code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, **(env or {})},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "no-torch-ok" in proc.stdout


def test_torch_is_not_installed():
    assert importlib.util.find_spec("torch") is None


def test_gpu_module_and_mlx_whisper_import_without_torch():
    run_check(
        """
        import numpy as np
        import recandcc.transcribe.gpu as gpu
        import mlx_whisper
        import mlx_whisper.transcribe
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        assert gpu.gpu_available()
        mel = log_mel_spectrogram(np.zeros(16000, dtype=np.float32), n_mels=128, padding=N_SAMPLES)
        assert pad_or_trim(mel, 3000, axis=-2).shape == (3000, 128)
        """
    )


@pytest.mark.slow
def test_real_mlx_transcription_without_torch():
    from recandcc.transcribe.gpu import hf_cache_folder, mlx_repo

    if not hf_cache_folder(mlx_repo("large-v3-turbo")).is_dir():
        pytest.skip("MLX large-v3-turbo is not cached (this test never downloads)")
    run_check(
        """
        import numpy as np
        from recandcc.transcribe.gpu import MlxWhisper

        w = MlxWhisper("large-v3-turbo")
        try:
            w.load()
            audio = (0.01 * np.random.default_rng(0).standard_normal(32000)).astype(np.float32)
            w.detect_language(audio, ["en", "ru"])
            w.transcribe(audio, "en")
        finally:
            w.close()
        """,
        env={"HF_HUB_OFFLINE": "1"},
    )
