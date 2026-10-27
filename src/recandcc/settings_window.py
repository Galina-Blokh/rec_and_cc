"""The native settings window: ``rcc-audiotap settings`` (macOS, SwiftUI).

Python stays the single source of truth for config: it writes a JSON snapshot of the current
values, the helper shows the window and prints one JSON line when it closes, and the values
come back through ``config.validate_settings`` / ``save_settings``. The helper never reads
or writes config.toml (so it needs no TOML parser and cannot bypass validation).

Protocol (stdout, one line): ``{"settings": "saved", "values": {...}}`` or
``{"settings": "cancelled"}``. An old helper has no ``settings`` subcommand: it exits with
status 2 and no JSON, which ``run_settings`` reports as ``SettingsUnavailable``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# The window waits for the owner: generous, but never hangs forever.
SETTINGS_TIMEOUT_S = 15 * 60.0


class SettingsUnavailable(RuntimeError):
    """No usable native settings window (not macOS, helper missing or too old)."""


def snapshot(cfg: Config) -> dict:
    """The values the window shows. Settings only, no transcript or audio content."""
    tr = cfg.transcription
    return {
        "mode": str(cfg.detection.mode),
        "output_dir": str(cfg.output_dir),
        "languages": list(tr.languages),
        "keep_audio": tr.keep_audio,
        "live_captions": cfg.live_captions,
        "speaker_names": tr.speaker_names,
        "hotkey": cfg.hotkey,
    }


def run_settings(cfg: Config, timeout: float = SETTINGS_TIMEOUT_S) -> dict | None:
    """Show the window and wait for it to close (blocks: call off the UI thread).

    Returns the raw values the user saved, or None if they cancelled. Raises
    SettingsUnavailable when the helper cannot show the window."""
    if sys.platform != "darwin":
        raise SettingsUnavailable("the native settings window is macOS-only")
    from .native import HelperNotFoundError, find_helper

    try:
        helper = find_helper()
    except HelperNotFoundError as e:
        raise SettingsUnavailable(str(e)) from e
    with tempfile.TemporaryDirectory(prefix="rcc-settings-") as tmp:
        snap = Path(tmp) / "settings.json"
        snap.write_text(json.dumps(snapshot(cfg)), encoding="utf-8")
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv: our own helper, no shell
                [str(helper), "settings", "--config", str(snap)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise SettingsUnavailable(f"settings window failed: {e}") from e
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    try:
        reply = json.loads(lines[-1])
        state = reply["settings"]
    except (IndexError, ValueError, KeyError, TypeError) as e:
        log.info("rcc-audiotap settings gave no answer (exit %s)", result.returncode)
        raise SettingsUnavailable("the helper has no settings window (rebuild it)") from e
    if state == "saved" and isinstance(reply.get("values"), dict):
        return reply["values"]
    return None
