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
MIN_MARGIN_DB, MAX_MARGIN_DB = 9.0, 12.0  # step loudness above the noise floor, at 100 / 0
# Speech needed is at most this share of the window (the window widens instead): even a busy
# conversation leaves pauses, so every setting stays reachable.
MAX_COVERAGE = 0.5


def valid(s: object) -> float | None:
    """A sensitivity as a float in 0-100, or None if unset or not a number."""
    if s is None or isinstance(s, bool) or not isinstance(s, int | float) or s != s:
        return None
    return min(100.0, max(0.0, float(s)))


def scale(s: float | None) -> float:
    """Threshold multiplier: 4x at 0, 1x at 50 (= the configured value), 0.25x at 100."""
    v = valid(s)
    return 1.0 if v is None else 4.0 ** ((50.0 - v) / 50.0)


def effective_min_active_s(det: DetectionConfig) -> float:
    """How long an app must do audio before it triggers (inside sensitivity)."""
    return det.min_active_s * scale(det.inside_sensitivity)


def room_trigger(det: DetectionConfig) -> tuple[float, float, float | None]:
    """(speech_s, window_s, margin_db) for the room monitor (outside sensitivity).

    The window widens so speech_s is at most MAX_COVERAGE of it; margin_db is how far above
    the room's noise floor a step must be to count (None = no level gate)."""
    s = valid(det.outside_sensitivity)
    if s is None:  # as configured; speech longer than the window is clamped to it
        window = float(det.room_window_s)
        return min(float(det.room_speech_s), window), window, None
    speech = det.room_speech_s * scale(s)
    window = max(float(det.room_window_s), speech / MAX_COVERAGE)
    margin = MAX_MARGIN_DB - (MAX_MARGIN_DB - MIN_MARGIN_DB) * s / 100.0
    return speech, window, margin


def current_preset(det: DetectionConfig) -> str | None:
    """The preset matching the config, or None for custom values."""
    now = (det.outside_sensitivity, det.inside_sensitivity)
    return next((name for name, vals in PRESETS.items() if vals == now), None)


def apply_preset(det: DetectionConfig, name: str) -> None:
    det.outside_sensitivity, det.inside_sensitivity = PRESETS[name]
