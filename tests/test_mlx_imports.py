"""mlx_whisper is imported only through ``gpu.import_mlx_whisper``.

numba and scipy are excluded from the environment (ADR 0011, follow-up 4), so a direct
``import mlx_whisper`` fails unless ``import_mlx_whisper`` has registered its stub
``mlx_whisper.timing`` first. Only ``transcribe/gpu.py``, which calls it before each import,
may import mlx_whisper. Static, so it runs on every platform.

Stories: none.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "recandcc"
ALLOWED = SRC / "transcribe" / "gpu.py"
DIRECT_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+mlx_whisper\b|import_module\(\s*['\"]mlx_whisper", re.MULTILINE
)


def test_mlx_whisper_is_imported_only_in_gpu_module():
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if path != ALLOWED and DIRECT_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"import mlx_whisper via gpu.import_mlx_whisper(): {offenders}"


def test_pattern_catches_direct_imports():
    for line in (
        "import mlx_whisper",
        "    from mlx_whisper.audio import N_SAMPLES",
        'importlib.import_module("mlx_whisper.transcribe")',
    ):
        assert DIRECT_IMPORT.search(line), line
    assert not DIRECT_IMPORT.search("from .gpu import import_mlx_whisper")
    assert DIRECT_IMPORT.search(ALLOWED.read_text(encoding="utf-8"))
