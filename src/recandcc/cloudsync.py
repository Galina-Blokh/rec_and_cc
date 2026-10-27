"""Is a folder synced to a cloud drive? (iCloud Drive, OneDrive, Dropbox, Google Drive.)

Used to warn when ``output_dir`` would upload recordings and transcripts off this device.
Purely local and cheap: path checks plus a few ``stat`` calls, no subprocess, no network.
The result is a hint for a warning, never a reason to refuse a folder.

- macOS: anything under ``~/Library/Mobile Documents`` (iCloud Drive) or
  ``~/Library/CloudStorage`` (File Provider: OneDrive, Dropbox, Google Drive, ...), and
  ``~/Desktop`` / ``~/Documents`` when iCloud "Desktop & Documents Folders" is on. That
  setting shows up as ``Desktop`` / ``Documents`` folders at the top of iCloud Drive
  (``~/Library/Mobile Documents/com~apple~CloudDocs``).
- Windows: anything under the folders in ``%OneDrive%``, ``%OneDriveConsumer%`` and
  ``%OneDriveCommercial%`` (compared case-insensitively).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path, PurePath, PureWindowsPath

ICLOUD_DRIVE = "iCloud Drive"
ONEDRIVE_ENV = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
_PROVIDERS = {  # ~/Library/CloudStorage/<prefix>-<account>
    "OneDrive": "OneDrive",
    "Dropbox": "Dropbox",
    "GoogleDrive": "Google Drive",
    "Box": "Box",
}


def _under(path: PurePath, root: PurePath) -> PurePath | None:
    """``path`` relative to ``root`` if it is inside it, else None."""
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError):
        return path.expanduser().absolute()


def mac_provider(path: Path, home: Path | None = None) -> str | None:
    home = _resolve(home or Path.home())
    path = _resolve(Path(path))
    lib = home / "Library"
    if _under(path, lib / "Mobile Documents") is not None:
        return ICLOUD_DRIVE
    rel = _under(path, lib / "CloudStorage")
    if rel is not None and rel.parts:
        name = rel.parts[0]
        for prefix, label in _PROVIDERS.items():
            if name.startswith(prefix):
                return label
        return name.split("-", 1)[0] or "a cloud drive"
    rel = _under(path, home)
    if rel is not None and rel.parts and rel.parts[0] in ("Desktop", "Documents"):
        cloud_docs = lib / "Mobile Documents" / "com~apple~CloudDocs" / rel.parts[0]
        try:
            if cloud_docs.is_dir():
                return f"{ICLOUD_DRIVE} (Desktop & Documents)"
        except OSError:
            pass
    return None


def windows_provider(path: str | os.PathLike, env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env

    def norm(p: str | os.PathLike) -> PureWindowsPath:
        text = os.path.realpath(p) if sys.platform == "win32" else str(p)
        return PureWindowsPath(text.lower())

    target = norm(path)
    for key in ONEDRIVE_ENV:
        root = env.get(key) or env.get(key.upper())
        if root and _under(target, norm(root)) is not None:
            return "OneDrive"
    return None


def cloud_sync_provider(path: str | os.PathLike, platform: str | None = None) -> str | None:
    """Name of the cloud service syncing ``path`` ("iCloud Drive", "OneDrive", ...) or None."""
    platform = platform or sys.platform
    try:
        if platform == "darwin":
            return mac_provider(Path(path))
        if platform == "win32":
            return windows_provider(path)
    except Exception:  # a warning helper must never break startup
        return None
    return None


def warning_text(path: str | os.PathLike, provider: str) -> str:
    return (
        f"The transcripts folder {path} is synced to {provider}: recordings and transcripts "
        "may be uploaded. Set output_dir to a local folder to keep them on this device."
    )
