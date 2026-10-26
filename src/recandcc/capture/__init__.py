"""Audio capture backends. Use the factories; platform modules are imported lazily."""

from __future__ import annotations

import sys

from .base import AudioSource, FrameCallback

__all__ = ["AudioSource", "FrameCallback", "create_mic_source", "create_system_source"]


def create_mic_source() -> AudioSource:
    from .mic import MicSource

    return MicSource()


def create_system_source() -> AudioSource:
    if sys.platform == "darwin":
        from .system_mac import MacSystemAudioSource

        return MacSystemAudioSource()
    if sys.platform == "win32":
        from .system_win import WinLoopbackSource

        return WinLoopbackSource()
    raise NotImplementedError(f"System audio capture is not supported on {sys.platform}")
