"""The MLX (GPU) path never imports torch, numba or scipy, so they are excluded.

mlx-whisper declares torch, but only its unused ``torch_whisper.py`` imports it; numba (with
llvmlite) and scipy are imported only by ``timing.py`` for word timestamps, which
``gpu.import_mlx_whisper`` replaces with the numpy port ``mlx_timing``. pyproject.toml drops all
of them with ``[tool.uv] exclude-dependencies`` (ADR 0011, follow-up 4). These tests catch
an mlx-whisper upgrade or a code change that starts loading them.
Each check runs in a fresh interpreter so modules imported by other tests don't count. A
meta-path hook turns any torch import into an ImportError that the check reports; numba,
llvmlite and scipy aren't hooked, because ``import_mlx_whisper`` probes for them with
``find_spec``, so the check asserts that none of them ended up in ``sys.modules``.

Skipped where mlx-whisper isn't installed (Windows, Linux, Intel Macs).

Stories: none.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap

import pytest

from recandcc.models import SAMPLE_RATE

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx_whisper") is None, reason="mlx-whisper is not installed"
)

EXCLUDED = ("torch", "numba", "llvmlite", "scipy")

# Blocks and records every attempt to import torch, then runs the body of a check.
PRELUDE = f"""
import sys

EXCLUDED = {EXCLUDED!r}
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
loaded = [m for m in sys.modules if m.split(".")[0] in EXCLUDED]
assert not loaded, f"excluded modules loaded: {loaded}"
print("no-torch-ok")
"""


def run_check(body: str, env: dict[str, str] | None = None) -> str:
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
    return proc.stdout


@pytest.mark.parametrize("name", EXCLUDED)
def test_excluded_package_is_not_installed(name):
    assert importlib.util.find_spec(name) is None


def test_gpu_module_and_mlx_whisper_import_without_excluded_packages():
    run_check(
        """
        import numpy as np
        import pytest
        import recandcc.transcribe.gpu as gpu

        mlx_whisper = gpu.import_mlx_whisper()
        import mlx_whisper.transcribe
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim

        assert gpu.gpu_available()
        mel = log_mel_spectrogram(np.zeros(16000, dtype=np.float32), n_mels=128, padding=N_SAMPLES)
        assert pad_or_trim(mel, 3000, axis=-2).shape == (3000, 128)

        from recandcc.transcribe import mlx_timing

        assert sys.modules["mlx_whisper.timing"] is mlx_timing and gpu.mlx_word_timestamps()
        """
    )


@pytest.mark.slow
def test_real_mlx_transcription_without_excluded_packages():
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


@pytest.mark.slow
def test_real_mlx_word_timestamps_without_excluded_packages(tmp_path):
    """Words from the numpy aligner: monotone, inside the audio, joined they give the segment
    text, and the text is no worse than without words. Counts and timing only are printed."""
    from recandcc.transcribe.gpu import hf_cache_folder, mlx_repo

    if not hf_cache_folder(mlx_repo("large-v3-turbo")).is_dir():
        pytest.skip("MLX large-v3-turbo is not cached (this test never downloads)")
    if not (shutil.which("say") and shutil.which("afconvert")):
        pytest.skip("needs macOS say/afconvert")
    reference = "The quarterly budget review is scheduled for next Thursday afternoon."
    aiff, wav = tmp_path / "s.aiff", tmp_path / "s.wav"
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), reference], check=True)
    fmt = f"LEF32@{SAMPLE_RATE}"
    subprocess.run(["afconvert", "-f", "WAVE", "-d", fmt, "-c", "1", aiff, wav], check=True)
    out = run_check(
        f"""
        import time
        import soundfile as sf
        from recandcc.transcribe.gpu import MlxWhisper
        from recandcc.transcribe.merge import similarity

        audio, _ = sf.read({str(wav)!r}, dtype="float32")
        duration = len(audio) / 16000
        w = MlxWhisper("large-v3-turbo")
        try:
            w.load()
            plain = w.transcribe(audio, "en")
            t0 = time.monotonic()
            worded = w.transcribe(audio, "en", words=True)
            took = time.monotonic() - t0
        finally:
            w.close()
        words = [x for s in worded for x in s[3]]
        assert words
        starts = [a for a, _, _ in words]
        assert starts == sorted(starts)
        assert all(0 <= a <= b <= duration + 0.05 for a, b, _ in words)
        for s in worded:
            assert "".join(x[2] for x in s[3]).split() == s[2].split()
        ref = {reference!r}
        text = " ".join(s[2] for s in worded)
        assert similarity(text, ref) >= similarity(" ".join(s[2] for s in plain), ref)
        print(f"words={{len(words)}} segments={{len(worded)}} audio={{duration:.2f}}s "
              f"words_pass={{took:.2f}}s first={{words[0][0]:.2f}} last={{words[-1][1]:.2f}}")
        """,
        env={"HF_HUB_OFFLINE": "1"},
    )
    print(out.splitlines()[-2])  # the summary line (-s shows it)
