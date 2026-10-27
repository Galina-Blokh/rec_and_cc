"""Pure-numpy word-alignment kernels for the MLX Whisper path.

Reimplements ``median_filter``, ``backtrace``, ``dtw_cpu`` and ``dtw`` from mlx-whisper's
``mlx_whisper/timing.py`` (algorithm from OpenAI Whisper / mlx-examples, MIT licence), which
use ``scipy.signal.medfilt`` and numba. RecAndCC excludes numba and scipy (ADR 0011), so these
give the same results with numpy only. Wired into word timestamps in a follow-up.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def median_filter(x: np.ndarray, filter_width: int) -> np.ndarray:
    """Median filter of odd width ``filter_width`` along the last axis, reflect-padded."""
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        return x  # reflect padding needs pad_width < length, as upstream
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")
    x32 = np.asarray(x, dtype=np.float32)
    # One (rows, length) block per leading index (a head) bounds the window temporary.
    blocks = x32.reshape(-1, *x32.shape[-2:]) if x32.ndim >= 2 else x32.reshape(1, 1, -1)
    out = np.empty_like(blocks)
    for h, block in enumerate(blocks):
        padded = np.pad(block, ((0, 0), (pad_width, pad_width)), mode="reflect")
        out[h] = np.median(sliding_window_view(padded, filter_width, axis=-1), axis=-1)
    return out.reshape(x32.shape)


def backtrace(trace: np.ndarray) -> np.ndarray:
    """Follow ``trace`` from the bottom-right corner; returns a (2, K) array of (i, j)."""
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1
    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))
        t = trace[i, j]
        if t == 0:
            i -= 1
            j -= 1
        elif t == 1:
            i -= 1
        elif t == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")
    return np.array(result)[::-1, :].T


def dtw(x: np.ndarray) -> np.ndarray:
    """Dynamic time warping path through cost matrix ``x`` (N, M), as upstream ``dtw_cpu``.

    Cells on one anti-diagonal ``d = i + j`` depend only on diagonals ``d - 1`` and ``d - 2``,
    so each diagonal is computed in one vectorised step with upstream's tie-break.
    """
    n, m = x.shape
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    trace = -np.ones((n + 1, m + 1), dtype=np.float32)
    cost[0, 0] = 0
    flat_cost, flat_trace, stride = cost.reshape(-1), trace.reshape(-1), m + 1
    for d in range(2, n + m + 1):
        i = np.arange(max(1, d - m), min(n, d - 1) + 1)
        j = d - i
        k = i * stride + j
        c0 = flat_cost[k - stride - 1]  # cost[i - 1, j - 1]
        c1 = flat_cost[k - stride]  # cost[i - 1, j]
        c2 = flat_cost[k - 1]  # cost[i, j - 1]
        pick0 = (c0 < c1) & (c0 < c2)
        pick1 = ~pick0 & (c1 < c0) & (c1 < c2)
        flat_cost[k] = x[i - 1, j - 1] + np.where(pick0, c0, np.where(pick1, c1, c2))
        flat_trace[k] = np.where(pick0, 0, np.where(pick1, 1, 2))
    return backtrace(trace)
