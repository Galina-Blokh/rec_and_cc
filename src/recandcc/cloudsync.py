"""Is a folder synced to a cloud drive? (iCloud Drive, OneDrive, Dropbox, Google Drive.)

Used to warn when ``output_dir`` would upload recordings and transcripts off this device.
Purely local and cheap: path-prefix checks (paths are not resolved, only a symlink at
``output_dir`` itself is followed one level) plus reading Finder's preferences file; no
subprocess, no network, and no ``stat`` into a cloud mount (those can stall). The result is a
hint for a warning, never a reason to refuse a folder, and not exhaustive.

- macOS: anything under ``~/Library/Mobile Documents`` (iCloud Drive),
  ``~/Library/CloudStorage`` (File Provider: OneDrive, Dropbox, Google Drive, Box, ...),
  ``~/Dropbox``, ``~/Google Drive`` or ``/Volumes/GoogleDrive``; and ``~/Desktop`` /
  ``~/Documents`` when iCloud "Desktop & Documents Folders" is on, read from Finder's
  ``FXICloudDriveDesktop`` / ``FXICloudDriveDocuments`` preferences.
- Windows: anything under ``%OneDrive%``, ``%OneDriveConsumer%``, ``%OneDriveCommercial%``
  or ``%USERPROFILE%\\Dropbox`` (compared case-insensitively).
"""

from __future__ import annotations

import logging
import ntpath
import os
import plistlib
import sys
from collections.abc import Mapping
from pathlib import Path, PurePath, PureWindowsPath

log = logging.getLogger(__name__)

ICLOUD_DRIVE = "iCloud Drive"
ONEDRIVE_ENV = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")
_PROVIDERS = {  # ~/Library/CloudStorage/<prefix>-<account>
    "OneDrive": "OneDrive",
    "Dropbox": "Dropbox",
    "GoogleDrive": "Google Drive",
    "Box": "Box",
}
_MAC_FOLDERS = {  # legacy (non-File-Provider) sync folders
    "Dropbox": "Dropbox",
    "Google Drive": "Google Drive",
    "/Volumes/GoogleDrive": "Google Drive",
}


def _under(path: PurePath, root: PurePath) -> PurePath | None:
    """``path`` relative to ``root`` if it is inside it, else None."""
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _absolute(path: str | os.PathLike) -> Path:
    """Absolute, normalised, but not resolved: never follows links into a cloud mount."""
    return Path(os.path.normpath(Path(path).expanduser().absolute()))


_prefs_cache: dict[Path, tuple[tuple[int, int], set[str]]] = {}


def icloud_desktop_documents(home: Path) -> set[str]:
    """Which of "Desktop" / "Documents" iCloud syncs, from Finder's preferences file
    (``FXICloudDrive*`` keys, stored as bool or 0/1). Re-parsed only when the file changes."""
    prefs = home / "Library" / "Preferences" / "com.apple.finder.plist"
    try:
        st = prefs.stat()
        key = (st.st_mtime_ns, st.st_size)
        cached = _prefs_cache.get(prefs)
        if cached is not None and cached[0] == key:
            return cached[1]
        data = plistlib.loads(prefs.read_bytes())
    except Exception:  # missing, unreadable or not a plist: assume not synced
        return set()
    keys = {"Desktop": "FXICloudDriveDesktop", "Documents": "FXICloudDriveDocuments"}
    found = {name for name, k in keys.items() if data.get(k) in (True, 1)}
    _prefs_cache[prefs] = (key, found)
    return found


def _mac_classify(path: Path, home: Path) -> str | None:
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
    for root, label in _MAC_FOLDERS.items():
        if _under(path, home / root) is not None:  # an absolute root replaces home
            return label
    rel = _under(path, home)
    if rel is not None and rel.parts and rel.parts[0] in icloud_desktop_documents(home):
        return f"{ICLOUD_DRIVE} (Desktop & Documents)"
    return None


def mac_provider(path: str | os.PathLike, home: Path | None = None) -> str | None:
    home = _absolute(home or Path.home())
    path = _absolute(path)
    found = _mac_classify(path, home)
    if found is None and os.path.islink(path):  # e.g. ~/Transcribes -> iCloud Drive
        found = _mac_classify(_absolute(path.parent / os.readlink(path)), home)
    return found


def windows_provider(path: str | os.PathLike, env: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if env is None else env

    def norm(p: str | os.PathLike) -> PureWindowsPath:  # no realpath: OneDrive placeholders
        return PureWindowsPath(ntpath.normpath(str(p)).lower())

    target = norm(path)
    for key in ONEDRIVE_ENV:
        root = env.get(key) or env.get(key.upper())
        if root and _under(target, norm(root)) is not None:
            return "OneDrive"
    profile = env.get("USERPROFILE")
    if profile and _under(target, norm(PureWindowsPath(profile) / "Dropbox")) is not None:
        return "Dropbox"
    return None


def cloud_sync_provider(path: str | os.PathLike, platform: str | None = None) -> str | None:
    """Name of the cloud service syncing ``path`` ("iCloud Drive", "OneDrive", ...) or None."""
    platform = platform or sys.platform
    try:
        if platform == "darwin":
            return mac_provider(Path(path))
        if platform == "win32":
            return windows_provider(path)
    except Exception:  # a best-effort warning must never break startup or the refresh
        log.debug("cloud-sync check failed for %s", path, exc_info=True)
        return None
    return None


def warning_text(path: str | os.PathLike, provider: str) -> str:
    return (
        f"The transcripts folder {path} is synced to {provider}: recordings and transcripts "
        "may be uploaded. Set output_dir to a local folder to keep them on this device."
    )
