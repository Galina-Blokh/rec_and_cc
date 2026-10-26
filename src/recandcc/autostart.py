"""Launch-at-login: macOS LaunchAgent / Windows HKCU Run key.

macOS: if RecAndCC.app exists (we're running inside it — ``RCC_APP_BUNDLE`` — or it's
installed in ~/Applications or /Applications) the agent launches the *app* via
``/usr/bin/open -a``, so TCC permissions (microphone, system audio) belong to RecAndCC.app.
Otherwise it falls back to ``<current python> -m recandcc run``.

The plist only takes effect at the next login (we deliberately don't ``launchctl load`` it,
which would start a second instance right now).

Windows (UNTESTED): ``HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\RecAndCC``.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path

from .config import LOG_DIR

LABEL = "com.gblokh.recandcc"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
WIN_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
WIN_VALUE = "RecAndCC"
APP_NAME = "RecAndCC.app"


def find_app_bundle() -> Path | None:
    env = os.environ.get("RCC_APP_BUNDLE")
    candidates = [Path(env)] if env else []
    candidates += [Path.home() / "Applications" / APP_NAME, Path("/Applications") / APP_NAME]
    for c in candidates:
        if c.suffix == ".app" and c.is_dir():
            return c
    return None


def mac_program_arguments() -> list[str]:
    app = find_app_bundle()
    if app is not None:
        return ["/usr/bin/open", "-a", str(app)]
    return [sys.executable, "-m", "recandcc", "run"]


def _mac_plist() -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "Label": LABEL,
        "ProgramArguments": mac_program_arguments(),
        "RunAtLoad": True,
        "KeepAlive": False,
        "ProcessType": "Interactive",
        "StandardOutPath": str(LOG_DIR / "launchd.out.log"),
        "StandardErrorPath": str(LOG_DIR / "launchd.err.log"),
        "WorkingDirectory": str(Path.home()),
    }


def _win_command() -> str:
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")  # no console window
    return f'"{pyw if pyw.exists() else exe}" -m recandcc run'


def enable() -> None:
    if sys.platform == "darwin":
        PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLIST_PATH, "wb") as f:
            plistlib.dump(_mac_plist(), f)
    elif sys.platform == "win32":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, WIN_VALUE, 0, winreg.REG_SZ, _win_command())
    else:
        raise NotImplementedError(sys.platform)


def disable() -> None:
    if sys.platform == "darwin":
        PLIST_PATH.unlink(missing_ok=True)
    elif sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE
            ) as k:
                winreg.DeleteValue(k, WIN_VALUE)
        except FileNotFoundError:
            pass
    else:
        raise NotImplementedError(sys.platform)


def is_enabled() -> bool:
    if sys.platform == "darwin":
        return PLIST_PATH.exists()
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY) as k:
                winreg.QueryValueEx(k, WIN_VALUE)
            return True
        except FileNotFoundError:
            return False
    return False


def describe() -> str:
    if sys.platform == "darwin":
        where = str(PLIST_PATH)
        cmd = " ".join(mac_program_arguments())
    elif sys.platform == "win32":
        where, cmd = rf"HKCU\{WIN_RUN_KEY}\{WIN_VALUE}", _win_command()
    else:
        return "unsupported platform"
    return f"{'enabled' if is_enabled() else 'disabled'} ({where}; command: {cmd})"
