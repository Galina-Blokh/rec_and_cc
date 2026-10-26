"""Audio activity detection backends. Platform modules are imported lazily."""

from __future__ import annotations

import sys

from .base import ActivityMonitor

__all__ = ["ActivityMonitor", "create_monitor"]


def create_monitor() -> ActivityMonitor:
    if sys.platform == "darwin":
        from .mac import MacActivityMonitor

        return MacActivityMonitor()
    if sys.platform == "win32":
        from .win import WinActivityMonitor

        return WinActivityMonitor()
    raise NotImplementedError(f"Audio activity detection is not supported on {sys.platform}")
