"""Sensitivity: map the two 0-100 settings to effective trigger thresholds (pure functions).

*Outside* (``detection.outside_sensitivity``) is the room around the laptop, heard by the
microphone (room listening). *Inside* (``detection.inside_sensitivity``) is audio the laptop
plays or captures in an app (calls, videos). Higher = less needed to start a recording.
Unset (None) leaves the configured thresholds exactly as they are, with no noise-floor gate.
"""

from __future__ import annotations

from .config import DetectionConfig

# Menu presets: label -> (outside, inside). "Off" = both unset (the plain thresholds).
PRESETS: dict[str, tuple[int | None, int | None]] = {
    "Off": (None, None),
    "Quiet room": (75, 50),
    "Normal": (50, 50),
    "Noisy place": (25, 50),
}
MIN_MARGIN_DB, MAX_MARGIN_DB = 3.0, 15.0  # room level above its noise floor, at 100 / 0


def _clamp(s: float) -> float:
    return min(100.0, max(0.0, float(s)))


def scale(s: float | None) -> float:
    """Threshold multiplier: 4x at 0, 1x at 50 (= the configured value), 0.25x at 100."""
    return 1.0 if s is None else 4.0 ** ((50.0 - _clamp(s)) / 50.0)


def effective_min_active_s(det: DetectionConfig) -> float:
    """How long an app must do audio before it triggers (inside sensitivity)."""
    return det.min_active_s * scale(det.inside_sensitivity)


def room_trigger(det: DetectionConfig) -> tuple[float, float, float | None]:
    """(speech_s, window_s, margin_db) for the room monitor (outside sensitivity).

    speech_s never exceeds the window; margin_db is how far above the room's adaptive noise
    floor a block must be to count as speech (None = no level gate)."""
    s = det.outside_sensitivity
    window = float(det.room_window_s)
    if s is None:
        return float(det.room_speech_s), window, None
    speech = min(det.room_speech_s * scale(s), window)
    margin = MAX_MARGIN_DB - (MAX_MARGIN_DB - MIN_MARGIN_DB) * _clamp(s) / 100.0
    return speech, window, margin


def current_preset(det: DetectionConfig) -> str | None:
    """The preset matching the config, or None for custom values."""
    now = (det.outside_sensitivity, det.inside_sensitivity)
    return next((name for name, vals in PRESETS.items() if vals == now), None)


def apply_preset(det: DetectionConfig, name: str) -> None:
    det.outside_sensitivity, det.inside_sensitivity = PRESETS[name]
