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

import contextlib
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
_ready = threading.Event()  # the open window printed "ready": its signal handlers are in
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


def _changed(key: str, before, after) -> bool:
    """Did the user change this value? The window returns languages in its own fixed order
    (en, ru, he first), so the set is compared: a reorder is not an edit, and the user's own
    spelling ("iw") and order in config.toml are left alone."""
    if key == "languages" and isinstance(before, list) and isinstance(after, list):
        try:
            return set(before) != set(after)
        except TypeError:
            return True
    return before != after


def _kill(proc: subprocess.Popen) -> None:
    with contextlib.suppress(OSError):
        proc.kill()


def raise_settings() -> None:
    """Bring the open settings window to the front (SIGUSR1: the helper handles it)."""
    with _proc_lock:
        proc = _proc
    if proc is not None and _ready.is_set() and proc.poll() is None:
        with contextlib.suppress(OSError):  # it may have just exited
            proc.send_signal(signal.SIGUSR1)


def close_settings() -> None:
    """Close the settings window if one is open (the app is quitting)."""
    _closing.set()
    with _proc_lock:
        proc = _proc
    if proc is not None and proc.poll() is None:
        with contextlib.suppress(OSError):  # it may have just exited
            proc.terminate()  # never waits here: this runs on the UI thread during Quit
        killer = threading.Timer(2.0, _kill, args=(proc,))  # in case it ignores SIGTERM
        killer.daemon = True
        killer.start()


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
                _kill(proc)  # quit raced with the start: don't leave a Dock icon behind
        _ready.clear()
        timer = threading.Timer(timeout, _kill, args=(proc,))
        timer.daemon = True
        timer.start()
        out: list[str] = []
        try:
            for line in proc.stdout or ():  # until the helper exits
                out.append(line)
                if '"ready"' in line:
                    _ready.set()
            stdout = "".join(out)
            proc.wait()
            timed_out = not timer.is_alive() and proc.returncode == -signal.SIGKILL
        finally:
            timer.cancel()
            _ready.clear()
            with _proc_lock:
                _proc = None
            if proc.stdin is not None:
                proc.stdin.close()
        if timed_out:
            log.warning("settings window timed out; treated as cancelled")
            return None
    lines = [ln for ln in stdout.splitlines() if ln.strip() and '"ready"' not in ln]
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
        return {k: v for k, v in reply["values"].items() if _changed(k, shown.get(k), v)}
    return None
