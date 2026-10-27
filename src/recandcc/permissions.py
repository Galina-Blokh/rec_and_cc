"""Privacy permissions (microphone, system audio recording): query, request, open Settings.

macOS: everything goes through the `rcc-audiotap` helper, so when the app runs from
RecAndCC.app the permissions are attributed to "RecAndCC" (not Terminal/python).

Status values: "granted" | "denied" | "undetermined" | "restricted" (microphone only) |
"unknown" (could not be determined).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

log = logging.getLogger(__name__)

MICROPHONE = "microphone"
SYSTEM_AUDIO = "system_audio"
KINDS = (MICROPHONE, SYSTEM_AUDIO)

STATUS_TIMEOUT_S = 10.0
REQUEST_TIMEOUT_S = 60.0  # how long the user gets to answer the system prompt


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown permission kind {kind!r}, expected one of {KINDS}")


def _run_helper(args: list[str], timeout: float) -> dict:
    from .native import find_helper

    result = subprocess.run(  # noqa: S603 - fixed argv: our own bundled helper, no shell
        [str(find_helper()), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stderr.strip():
        log.debug("rcc-audiotap %s: %s", args[0], result.stderr.strip())
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            f"rcc-audiotap {' '.join(args)} printed nothing (exit {result.returncode})"
        )
    return json.loads(lines[-1])


def status() -> dict[str, str]:
    """Current status of every permission, e.g. {"microphone": "granted", ...}."""
    if sys.platform == "win32":
        # Windows loopback capture needs no permission; mic privacy state isn't easily queryable.
        return {MICROPHONE: "unknown", SYSTEM_AUDIO: "granted"}
    if sys.platform != "darwin":
        return {MICROPHONE: "unknown", SYSTEM_AUDIO: "unknown"}
    try:
        data = _run_helper(["permissions"], STATUS_TIMEOUT_S)
        return {kind: str(data.get(kind, "unknown")) for kind in KINDS}
    except Exception:
        log.exception("could not query permissions")
        return {MICROPHONE: "unknown", SYSTEM_AUDIO: "unknown"}


def request(kind: str, timeout: float = REQUEST_TIMEOUT_S) -> str:
    """Show the system permission prompt (if not decided yet) and return the resulting status.

    Blocks until the user answers or `timeout` expires - call it off the UI thread.
    Once a permission was denied macOS will not prompt again; use open_settings() then.
    """
    _check_kind(kind)
    if sys.platform != "darwin":
        return status()[kind]
    try:
        data = _run_helper(["request", kind, "--timeout", str(timeout)], timeout + 15)
        return str(data.get(kind, "unknown"))
    except Exception:
        log.exception("permission request for %s failed", kind)
        return "unknown"


def open_settings(kind: str) -> None:
    """Open the System Settings / Windows Settings page where the permission can be changed."""
    _check_kind(kind)
    if sys.platform == "win32":
        os.startfile("ms-settings:privacy-microphone")  # noqa: S606, S607 - constant URI
        return
    if sys.platform != "darwin":
        return
    try:
        _run_helper(["open-settings", kind], STATUS_TIMEOUT_S)
    except Exception:
        log.exception("could not open settings for %s", kind)
        anchor = "Privacy_Microphone" if kind == MICROPHONE else "Privacy_AudioCapture"
        subprocess.run(  # noqa: S603 - fixed argv (absolute path), constant URL, no shell
            ["/usr/bin/open", f"x-apple.systempreferences:com.apple.preference.security?{anchor}"],
            check=False,
        )
