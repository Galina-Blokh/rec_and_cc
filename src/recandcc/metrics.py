"""Per-session pipeline metrics (stored in ``SessionInfo.metrics``) and their log summary."""

from __future__ import annotations

import logging
import statistics

from .models import SessionInfo

log = logging.getLogger("recandcc.metrics")


def rounded(x: float, digits: int = 3) -> float:
    return round(float(x), digits)


def latency_stats(latencies: list[float]) -> dict:
    """Median / max / count of a list of seconds ({} when empty)."""
    if not latencies:
        return {}
    return {
        "latency_median_s": rounded(statistics.median(latencies)),
        "latency_max_s": rounded(max(latencies)),
        "phrases": len(latencies),
    }


def summary_line(metrics: dict) -> str:
    """``stage.key=value`` pairs in one line (nested dicts flattened)."""
    parts: list[str] = []

    def walk(prefix: str, value) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            parts.append(f"{prefix}={value}")

    walk("", metrics)
    return " ".join(parts) or "(none)"


def log_session_metrics(session: SessionInfo) -> None:
    log.info("session %s metrics: %s", session.id, summary_line(session.metrics))
