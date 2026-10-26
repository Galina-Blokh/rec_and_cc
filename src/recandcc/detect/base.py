"""Audio activity detection interface implemented per platform."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import AudioProcess


class ActivityMonitor(ABC):
    @abstractmethod
    def snapshot(self) -> list[AudioProcess]:
        """Processes currently doing audio input and/or output. Must be cheap (called ~1/s)."""

    def close(self) -> None:  # noqa: B027 — optional hook
        """Release resources (helper processes, COM objects)."""
