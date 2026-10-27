"""Numpy word-alignment kernels (``transcribe/_align_np.py``) against plain ports of
mlx-whisper's ``timing.py`` (``median_filter``, ``dtw_cpu`` + ``backtrace``).

Stories: US-12.
"""

from __future__ import annotations

import logging
import math
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from recandcc.transcribe._align_np import dtw, median_filter
from recandcc.transcribe.mlx_timing import MAX_ALIGNMENT_HEADS

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
    for width in (4, 0, -3):
        with pytest.raises(ValueError):
            median_filter(np.zeros((1, 10)), width)
    with pytest.raises(ValueError):  # checked before the short-input pass-through
        median_filter(np.zeros((1, 2)), 4)


# Skipped in CI and the dev env because scipy is excluded (ADR 0011); verified manually
# with scipy 1.18.1 installed in a throwaway environment.
def test_median_filter_matches_scipy_upstream() -> None:
    signal = pytest.importorskip("scipy.signal")
    x = np.random.default_rng(2).standard_normal((4, 5, 200)).astype(np.float32)
    padded = np.pad(x, ((0, 0), (0, 0), (3, 3)), mode="reflect")
    upstream = signal.medfilt(padded, kernel_size=(1, 1, 7))[..., 3:-3]
    np.testing.assert_array_equal(median_filter(x, 7), upstream)


def test_median_filter_spreads_a_nan_across_its_window() -> None:
    x = np.arange(40, dtype=np.float32)
    x[20] = np.nan
    out = median_filter(x, 7)
    assert np.isnan(out[17:24]).all()
    assert not np.isnan(out[:17]).any() and not np.isnan(out[24:]).any()


def assert_valid_path(path: np.ndarray, n: int, m: int) -> None:
    assert path.shape[0] == 2
    assert tuple(path[:, 0]) == (0, 0)
    assert tuple(path[:, -1]) == (n - 1, m - 1)
    steps = np.diff(path, axis=1)
    assert ((steps >= 0) & (steps <= 1)).all() and (steps.sum(axis=0) >= 1).all()


@pytest.mark.parametrize(
    ("n", "m"),
    [(1, 1), (1, 300), (2, 5), (10, 1), (10, 300), (40, 150), (120, 5), (120, 1500)],
)
def test_dtw_matches_upstream_port(n: int, m: int) -> None:
    rng = np.random.default_rng(n * 10_000 + m)
    for x in (
        rng.standard_normal((n, m)).astype(np.float32),
        rng.integers(0, 3, (n, m)).astype(np.float32),  # integer costs force ties
    ):
        got = dtw(x)
        if n * m <= 40 * 150:  # the Python reference is slow; golden vectors cover upstream
            np.testing.assert_array_equal(got, ref_dtw(x))
        assert_valid_path(got, n, m)


@pytest.mark.slow
def test_dtw_matches_upstream_port_at_whisper_scale() -> None:
    """Exact equality at a real window's shape (120 tokens x 1500 frames); slow, since the
    Python reference walks every cell."""
    rng = np.random.default_rng(120_1500)
    for x in (
        rng.standard_normal((120, 1500)).astype(np.float32),
        rng.integers(0, 3, (120, 1500)).astype(np.float32),
    ):
        np.testing.assert_array_equal(dtw(x), ref_dtw(x))


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


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_dtw_rejects_an_empty_matrix(shape: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        dtw(np.zeros(shape, dtype=np.float32))


def test_matches_golden_vectors_from_the_real_upstream_kernels() -> None:
    """Fixtures made by the real scipy/numba kernels: see scripts/gen_align_golden.py."""
    golden = np.load(Path(__file__).parent / "fixtures" / "align_golden.npz")
    for k in range(3):
        got = median_filter(golden[f"median_in_{k}"], 7)
        np.testing.assert_array_equal(got, golden[f"median_out_{k}"])
        np.testing.assert_array_equal(dtw(golden[f"dtw_in_{k}"]), golden[f"dtw_out_{k}"])


# Speed budgets: generous locally, times CI_HEADROOM on CI runners (any OS) and on Windows;
# RCC_ALIGN_FACTOR overrides the factor for any other slow runner (as in
# test_search_latency.py). Each check warms up once and times the median of a few runs.
CI_HEADROOM = 3.0
DTW_BUDGET_S = 0.5  # measured ~15 ms for 150x1500 on an M-series Mac
# Per head count, for heads x 150 x 1500, width 7. 64 = mlx_timing.MAX_ALIGNMENT_HEADS, the
# most the MLX word pass ever filters (measured ~0.25 s for 20 and ~0.80 s for 64).
MEDIAN_BUDGET_S = {20: 2.0, MAX_ALIGNMENT_HEADS: 2.4}


def _factor() -> float:
    default = CI_HEADROOM if os.environ.get("CI") or sys.platform == "win32" else 1.0
    env = os.environ.get("RCC_ALIGN_FACTOR")
    if not env:
        return default
    try:
        value = float(env)
    except ValueError:
        value = 0.0
    if not (value > 0 and math.isfinite(value)):
        log.warning("ignoring invalid RCC_ALIGN_FACTOR; using %s", default)
        return default
    return value


def _median_time(fn: Callable[[], object], runs: int) -> float:
    fn()  # warm-up
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def test_dtw_is_fast_enough() -> None:
    x = np.random.default_rng(4).standard_normal((150, 1500)).astype(np.float32)
    elapsed = _median_time(lambda: dtw(x), runs=5)
    log.info("dtw 150x1500: median %.3f s", elapsed)
    assert elapsed < DTW_BUDGET_S * _factor()


@pytest.mark.parametrize("heads", [20, MAX_ALIGNMENT_HEADS])
def test_median_filter_speed_and_memory_at_a_whisper_shape(heads: int) -> None:
    tokens, frames, width = 150, 1500, 7
    x = np.random.default_rng(5).standard_normal((heads, tokens, frames)).astype(np.float32)
    elapsed = _median_time(lambda: median_filter(x, width), runs=3)
    log.info("median_filter %dx%dx%d: median %.3f s", heads, tokens, frames, elapsed)
    assert elapsed < MEDIAN_BUDGET_S[heads] * _factor()
    # Temporaries beyond the output stay bounded by one head's windows, so collapsing the
    # per-head loop (about heads x as much) is caught.
    limit = 4 * tokens * frames * width * x.itemsize  # 4x one head's windows
    peak = _peak_bytes(lambda: median_filter(x, width))
    log.info("median_filter peak %.1f MB, limit %.1f MB + output", peak / 1e6, limit / 1e6)
    assert peak - x.nbytes < limit
    if heads != 20:
        return  # the canary below needs all heads' windows at once (~400 MB at 64)
    # Canary: all heads at once, as a collapsed loop would do. Proves tracemalloc sees numpy
    # buffers here, so the guard above can fail.
    padded = np.pad(x, ((0, 0), (0, 0), (width // 2, width // 2)), mode="reflect")
    canary = _peak_bytes(lambda: np.median(sliding_window_view(padded, width, axis=-1), axis=-1))
    log.info("all-heads canary %.1f MB", canary / 1e6)
    assert canary - x.nbytes > 3 * limit


def _peak_bytes(fn: Callable[[], object]) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_numba_and_scipy_are_never_imported() -> None:
    code = (
        "import sys, recandcc.transcribe._align_np\n"
        "print([m for m in ('numba', 'scipy') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
