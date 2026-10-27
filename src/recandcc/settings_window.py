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
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)

# The window waits for the owner: generous, but never hangs forever.
SETTINGS_TIMEOUT_S = 15 * 60.0


_proc_lock = threading.Lock()
_proc: subprocess.Popen | None = None  # the open window's helper, so quitting can close it
_closing = threading.Event()  # set when the app quits: a window starting now is killed


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


def raise_settings() -> None:
    """Bring the open settings window to the front (SIGUSR1: the helper handles it)."""
    with _proc_lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGUSR1)


def close_settings() -> None:
    """Close the settings window if one is open (the app is quitting)."""
    _closing.set()
    with _proc_lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        proc.terminate()  # never waits here: this runs on the UI thread during Quit
        threading.Timer(2.0, proc.kill).start()  # in case it ignores SIGTERM


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
    if _closing.is_set():
        return None
    shown = snapshot(cfg)  # what the window starts with: the baseline for "changed"
    with tempfile.TemporaryDirectory(prefix="rcc-settings-") as tmp:
        snap = Path(tmp) / "settings.json"
        snap.write_text(json.dumps(shown), encoding="utf-8")
        global _proc
        try:
            proc = subprocess.Popen(  # noqa: S603 - fixed argv: our own helper, no shell
                [str(helper), "settings", "--config", str(snap)],
                stdin=subprocess.PIPE,  # held open: EOF (we died) closes the window
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # nothing worth keeping, and never a full pipe
                text=True,
            )
        except OSError as e:
            raise SettingsUnavailable(f"settings window failed: {e}") from e
        with _proc_lock:
            _proc = proc
            if _closing.is_set():
                proc.kill()  # quit raced with the start: don't leave a Dock icon behind
        timer = threading.Timer(timeout, proc.kill)
        timer.start()
        try:
            stdout = proc.stdout.read() if proc.stdout else ""  # until the helper exits
            proc.wait()
            timed_out = not timer.is_alive() and proc.returncode == -signal.SIGKILL
        finally:
            timer.cancel()
            with _proc_lock:
                _proc = None
            if proc.stdin is not None:
                proc.stdin.close()
        if timed_out:
            log.warning("settings window timed out; treated as cancelled")
            return None
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    try:
        reply = json.loads(lines[-1])
        state = reply["settings"]
    except (IndexError, ValueError, KeyError, TypeError) as e:
        if proc.returncode == 0 or proc.returncode < 0:
            return None  # closed by SIGTERM (app quitting) or killed: same as cancelled
        log.info("rcc-audiotap settings gave no answer (exit %s)", proc.returncode)
        raise SettingsUnavailable("the helper has no settings window (rebuild it)") from e
    if state == "saved" and isinstance(reply.get("values"), dict):
        # only what the user changed: edits made to config.toml meanwhile are not reverted
        return {k: v for k, v in reply["values"].items() if shown.get(k) != v}
    return None
