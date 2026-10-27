"""Is a folder synced to a cloud drive? (iCloud Drive, OneDrive, Dropbox, Google Drive.)

Used to warn when ``output_dir`` would upload recordings and transcripts off this device.
Purely local and cheap: path-prefix checks plus reading Finder's preferences file. Paths are
not resolved: symlinks in ``output_dir`` are followed by reading the link text, and nothing
inside a known cloud root is ever stat-ed (that can stall). No subprocess, no network. The
result is a hint for a warning, never a reason to refuse a folder, and not exhaustive.

- macOS: anything under ``~/Library/Mobile Documents`` (iCloud Drive),
  ``~/Library/CloudStorage`` (File Provider: OneDrive, Dropbox, Google Drive, Box, ...),
  ``~/Dropbox``, ``~/Google Drive``, ``~/OneDrive*``, ``~/Box``, ``~/Box Sync`` or
  ``/Volumes/GoogleDrive`` (case-insensitive); and ``~/Desktop`` / ``~/Documents`` when
  iCloud "Desktop & Documents Folders" is on, read from Finder's ``FXICloudDriveDesktop`` /
  ``FXICloudDriveDocuments`` preferences.
- Windows: anything under ``%OneDrive%``, ``%OneDriveConsumer%``, ``%OneDriveCommercial%``
  or ``%USERPROFILE%\\Dropbox``, and ``%USERPROFILE%\\Documents`` / ``Desktop`` when OneDrive
  folder backup has moved them (compared case-insensitively).
"""

from __future__ import annotations

import logging
import ntpath
import os
import plistlib
import sys
import time
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
_MAC_HOME_FOLDERS = {  # legacy (non-File-Provider) sync folders in ~, compared casefolded
    "dropbox": "Dropbox",
    "google drive": "Google Drive",
    "box": "Box",
    "box sync": "Box",
}
_MAC_VOLUMES = {"/volumes/googledrive": "Google Drive"}
_MAX_LINKS = 8  # symlinks followed per path (stops cycles)


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
    """Pure prefix checks. macOS volumes are usually case-insensitive, so compare casefolded."""
    cf = PurePath(str(path).casefold())
    h = PurePath(str(home).casefold())
    if _under(cf, h / "library" / "mobile documents") is not None:
        return ICLOUD_DRIVE
    rel = _under(cf, h / "library" / "cloudstorage")
    if rel is not None and rel.parts:
        for prefix, label in _PROVIDERS.items():
            if rel.parts[0].startswith(prefix.casefold()):
                return label
        return path.parts[len(cf.parts) - len(rel.parts)].split("-", 1)[0] or "a cloud drive"
    for root, label in _MAC_VOLUMES.items():
        if _under(cf, PurePath(root)) is not None:
            return label
    rel = _under(cf, h)
    if rel is None or not rel.parts:
        return None
    top = rel.parts[0]
    if top in _MAC_HOME_FOLDERS:
        return _MAC_HOME_FOLDERS[top]
    if top.startswith("onedrive"):  # ~/OneDrive, ~/OneDrive - Acme
        return "OneDrive"
    if top in {n.casefold() for n in icloud_desktop_documents(home)}:
        return f"{ICLOUD_DRIVE} (Desktop & Documents)"
    return None


def _is_link(path: Path) -> bool:
    """Symlink, or (Windows) directory junction. One lstat, nothing inside the target."""
    return os.path.islink(path) or (hasattr(path, "is_junction") and path.is_junction())


def _follow_one_link(path: Path) -> Path | None:
    """``path`` with its first linked component replaced by the link's target, or None.
    Reads link text only (lstat + readlink); never stats inside the target."""
    for anc in [*reversed(path.parents), path]:
        if anc.parent != anc and _is_link(anc):
            text = os.readlink(anc)
            if text.startswith("\\\\?\\"):  # Windows junction targets: \\?\C:\...
                text = text[4:]
            return _absolute(anc.parent / text) / path.relative_to(anc)
    return None


def _following_links(path: str | os.PathLike, classify) -> str | None:
    """``classify(path)``; if nothing matches, follow links in the path (``~/Transcribes`` ->
    iCloud Drive, or a linked parent) one at a time, re-checking after each. A path already
    inside a known cloud root is classified before anything in it is touched."""
    path = _absolute(path)
    for _ in range(_MAX_LINKS):
        found = classify(path)
        if found is not None:
            return found
        nxt = _follow_one_link(path)
        if nxt is None:
            return None
        path = nxt
    return None


def mac_provider(path: str | os.PathLike, home: Path | None = None) -> str | None:
    home = _absolute(home or Path.home())
    return _following_links(path, lambda p: _mac_classify(p, home))


_KNOWN_TTL_S = 300.0
_known_cache: tuple[float, list[str]] | None = None


def _known_folders() -> list[str]:
    """Where Windows really keeps Documents / Desktop (moved into OneDrive by folder backup).
    Re-read every few minutes, so turning folder backup on while running is noticed."""
    global _known_cache
    if sys.platform != "win32":
        return []
    now = time.monotonic()
    if _known_cache is None or now - _known_cache[0] > _KNOWN_TTL_S:
        import platformdirs

        _known_cache = (now, [platformdirs.user_documents_dir(), platformdirs.user_desktop_dir()])
    return _known_cache[1]


def _win_classify(path: str | os.PathLike, env: Mapping[str, str], known: list[str]) -> str | None:
    def norm(p: str | os.PathLike) -> PureWindowsPath:  # no realpath: OneDrive placeholders
        return PureWindowsPath(ntpath.normpath(ntpath.abspath(str(p))).lower())

    target = norm(path)
    roots = [norm(r) for k in ONEDRIVE_ENV if (r := env.get(k) or env.get(k.upper()))]
    if any(_under(target, r) is not None for r in roots):
        return "OneDrive"
    profile = env.get("USERPROFILE")
    if not profile:
        return None
    if _under(target, norm(PureWindowsPath(profile) / "Dropbox")) is not None:
        return "Dropbox"
    # Folder backup moves Documents / Desktop into OneDrive and may leave a junction behind
    # at %USERPROFILE%\Documents: a path through it is synced too.
    for folder in known:
        moved = norm(folder)
        stub = norm(PureWindowsPath(profile) / moved.name)
        if any(_under(moved, r) is not None for r in roots) and _under(target, stub) is not None:
            return "OneDrive"
    return None


def windows_provider(
    path: str | os.PathLike,
    env: Mapping[str, str] | None = None,
    known: list[str] | None = None,
    follow_links: bool = True,
) -> str | None:
    env = os.environ if env is None else env
    known = _known_folders() if known is None else known
    found = _win_classify(path, env, known)
    if found is None and follow_links:  # e.g. C:\Transcribes is a junction into OneDrive
        found = _following_links(path, lambda p: _win_classify(p, env, known))
    return found


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
