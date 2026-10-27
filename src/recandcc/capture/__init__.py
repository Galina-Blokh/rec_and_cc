"""Audio capture backends. Use the factories; platform modules are imported lazily."""

from __future__ import annotations

import logging
import sys

from ..models import AppTarget
from .base import AudioSource, FrameCallback

log = logging.getLogger(__name__)

__all__ = ["AudioSource", "FrameCallback", "create_mic_source", "create_system_source"]


def create_mic_source() -> AudioSource:
    from .mic import MicSource

    return MicSource()


def create_system_source(target: AppTarget | None = None) -> AudioSource:
    """System audio; with ``target``, only that app's audio where supported (macOS)."""
    if sys.platform == "darwin":
        from .system_mac import MacSystemAudioSource

        return MacSystemAudioSource(target=target)
    if sys.platform == "win32":
        from .system_win import WinLoopbackSource

        if target is not None:
            log.info("per-app system audio is macOS only; recording all system audio")

        return WinLoopbackSource()
    raise NotImplementedError(f"System audio capture is not supported on {sys.platform}")
