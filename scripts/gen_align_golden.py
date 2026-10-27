"""Generate tests/fixtures/align_golden.npz from mlx-whisper's real word-alignment kernels.

Runs the upstream ``median_filter``, ``dtw_cpu``/``backtrace`` and ``dtw`` (scipy + numba),
extracted from ``mlx_whisper/timing.py``, on small seeded inputs. numba and scipy are excluded
from the project (ADR 0011), so run it in a throwaway venv, never the project's:

    python -m venv /tmp/v && /tmp/v/bin/pip install numpy scipy numba
    /tmp/v/bin/python scripts/gen_align_golden.py path/to/mlx_whisper/timing.py
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np

UPSTREAM = ("median_filter", "backtrace", "dtw_cpu", "dtw")
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "align_golden.npz"


def load_upstream(timing_py: Path) -> dict:
    tree = ast.parse(timing_py.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in UPSTREAM]
    if {f.name for f in funcs} != set(UPSTREAM):
        raise SystemExit(f"upstream functions not found in {timing_py}")
    ns = {
        "np": np,
        "numba": importlib.import_module("numba"),
        "signal": importlib.import_module("scipy.signal"),
    }
    # Runs the upstream functions as written; this is a dev-only fixture generator.
    code = compile(ast.Module(body=funcs, type_ignores=[]), str(timing_py), "exec")
    exec(code, ns)  # noqa: S102
    return ns


def main(timing_py: str) -> None:
    up = load_upstream(Path(timing_py))
    rng = np.random.default_rng(20261027)
    data: dict[str, np.ndarray] = {}
    for k in range(3):
        x = rng.standard_normal((2, 5, 40)).astype(np.float32)
        if k == 1:
            x = rng.integers(0, 3, x.shape).astype(np.float32)  # many equal values
        data[f"median_in_{k}"] = x
        data[f"median_out_{k}"] = up["median_filter"](x, 7)
    tie = rng.integers(0, 2, (12, 60)).astype(np.float32)  # tie-heavy
    nan_inf = rng.standard_normal((9, 50)).astype(np.float32)
    nan_inf[:, 7] = np.nan
    nan_inf[4, 20:30] = np.inf
    for k, x in enumerate((rng.standard_normal((7, 33)).astype(np.float32), tie, nan_inf)):
        data[f"dtw_in_{k}"] = x
        data[f"dtw_out_{k}"] = up["dtw"](x.copy())
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **data)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv[1])
