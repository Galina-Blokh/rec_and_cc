"""Audio capture interface implemented per platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

# Receives float32 mono SAMPLE_RATE frames. Called on the capture thread:
# implementations of the callback must return quickly (just enqueue the data).
FrameCallback = Callable[[np.ndarray], None]


class AudioSource(ABC):
    name: str = "source"

    @abstractmethod
    def start(self, on_frames: FrameCallback) -> None:
        """Begin capturing; returns immediately. Frames arrive on a background thread."""

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and release the device. Idempotent."""
