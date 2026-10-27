"""Numpy word-alignment kernels (``transcribe/_align_np.py``) against plain ports of
mlx-whisper's ``timing.py`` (``median_filter``, ``dtw_cpu`` + ``backtrace``).

Stories: US-12.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time

import numpy as np
import pytest

from recandcc.transcribe._align_np import dtw, median_filter

log = logging.getLogger(__name__)


def ref_median(x: np.ndarray, k: int) -> np.ndarray:
    """Naive per-element median over the reflect-padded last axis."""
    p = k // 2
    x = x.astype(np.float32)
    padded = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(p, p)], mode="reflect")
    out = np.empty_like(x)
    for idx in np.ndindex(x.shape):
        t = idx[-1]
        out[idx] = np.median(padded[idx[:-1]][t : t + k])
    return out


def ref_dtw(x: np.ndarray) -> np.ndarray:
    """Direct Python port of upstream ``dtw_cpu`` + ``backtrace``."""
    n, m = x.shape
    cost = np.ones((n + 1, m + 1), dtype=np.float32) * np.inf
    trace = -np.ones((n + 1, m + 1), dtype=np.float32)
    cost[0, 0] = 0
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            c0, c1, c2 = cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t
    i, j = n, m
    trace[0, :] = 2
    trace[:, 0] = 1
    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        if trace[i, j] == 0:
            i, j = i - 1, j - 1
        elif trace[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return np.array(result)[::-1, :].T


@pytest.mark.parametrize("f", [3, 7, 8, 100, 1500])
def test_median_filter_matches_naive_reference(f: int) -> None:
    rng = np.random.default_rng(f)
    x = rng.standard_normal((2, 3, f)).astype(np.float32)
    ints = rng.integers(0, 3, (2, 3, f)).astype(np.float64)  # many equal values
    for a in (x, ints):  # length <= pad width (f = 3) passes through, as upstream
        np.testing.assert_array_equal(median_filter(a, 7), a if f <= 3 else ref_median(a, 7))


def test_median_filter_one_dimensional_and_pass_through() -> None:
    x = np.random.default_rng(1).standard_normal(50)
    np.testing.assert_array_equal(median_filter(x, 7), ref_median(x[None], 7)[0])
    short = np.arange(6, dtype=np.float64).reshape(2, 3)  # length 3 <= pad 3: unchanged
    assert median_filter(short, 7) is short
    with pytest.raises(ValueError):
        median_filter(np.zeros((1, 10)), 4)


def test_median_filter_matches_scipy_upstream() -> None:
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(2).standard_normal((4, 5, 200)).astype(np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (3, 3)), mode="reflect")
    upstream = signal.medfilt(padded, kernel_size=(1, 1, 7))[..., 3:-3]
    np.testing.assert_array_equal(median_filter(x, 7), upstream)


def assert_valid_path(path: np.ndarray, n: int, m: int) -> None:
    assert path.shape[0] == 2
    assert tuple(path[:, 0]) == (0, 0)
    assert tuple(path[:, -1]) == (n - 1, m - 1)
    steps = np.diff(path, axis=1)
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(axis=0) >= 1).all()


@pytest.mark.parametrize("n", [1, 2, 10, 120])
@pytest.mark.parametrize("m", [1, 5, 300, 1500])
def test_dtw_matches_upstream_port(n: int, m: int) -> None:
    rng = np.random.default_rng(n * 10_000 + m)
    for x in (
        rng.standard_normal((n, m)).astype(np.float32),
        rng.integers(0, 3, (n, m)).astype(np.float32),  # integer costs force ties
    ):
        got = dtw(x)
        np.testing.assert_array_equal(got, ref_dtw(x))
        assert_valid_path(got, n, m)


def test_dtw_with_nan_and_inf_from_a_constant_column() -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 20, 60)).astype(np.float32)
    w[:, :, 10] = 1.0  # constant column: std 0 -> NaN after standardising
    with np.errstate(divide="ignore", invalid="ignore"):
        std = (w - w.mean(axis=-2, keepdims=True)) / w.std(axis=-2, keepdims=True)
    matrix = -median_filter(std, 7).mean(axis=0)
    assert np.isnan(matrix).any()
    np.testing.assert_array_equal(dtw(matrix), ref_dtw(matrix))
    inf = rng.standard_normal((8, 30))
    inf[3, :] = np.inf
    np.testing.assert_array_equal(dtw(inf), ref_dtw(inf))


def test_dtw_is_fast_enough() -> None:
    x = np.random.default_rng(4).standard_normal((150, 1500)).astype(np.float32)
    start = time.perf_counter()
    dtw(x)
    elapsed = time.perf_counter() - start
    log.info("dtw 150x1500: %.3f s", elapsed)
    assert elapsed < 0.5


def test_numba_and_scipy_are_never_imported() -> None:
    code = (
        "import sys, recandcc.transcribe._align_np\n"
        "print([m for m in ('numba', 'scipy') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
