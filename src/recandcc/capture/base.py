"""Audio capture interface implemented per platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from ..models import AppTarget

# Receives float32 mono SAMPLE_RATE frames. Called on the capture thread:
# implementations of the callback must return quickly (just enqueue the data).
FrameCallback = Callable[[np.ndarray], None]


class AudioSource(ABC):
    name: str = "source"
    # What a system-audio source records: "all" (everything) or "app" (one app's processes).
    scope: str = "all"

    @abstractmethod
    def start(self, on_frames: FrameCallback) -> None:
        """Begin capturing; returns immediately. Frames arrive on a background thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device. Idempotent."""

    def confirmed_scope(self, timeout: float) -> str:
        """``scope`` once the source knows it for sure (a per-app tap may fall back to "all"
        after ``start``); "pending" if not within ``timeout`` seconds. Not on audio threads."""
        return self.scope

    def add_target(self, target: AppTarget) -> bool:
        """Also record ``target`` (per-app system audio only). False if not supported."""
        return False
