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
import posixpath
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


# ------------------------------------------------------------------ network volumes
# A network share is as much "off this device" as a cloud drive: the same warning UX.
_NETWORK_FS = {  # filesystem type -> label
    "smbfs": "SMB",
    "afpfs": "AFP",
    "nfs": "NFS",
    "webdav": "WebDAV",
    "cifs": "SMB",
    "smb3": "SMB",
    "smb2": "SMB",
    "sshfs": "SSHFS",
    "fuse.sshfs": "SSHFS",
    "nfs4": "NFS",
    "9p": "9P",
}
_MNT_LOCAL = 0x1000  # macOS statfs f_flags
_DRIVE_REMOTE = 4  # GetDriveTypeW


def _existing_ancestor(path: Path) -> Path:
    """Closest existing parent: output_dir may not have been created yet."""
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def _mac_statfs(path: Path) -> tuple[str, int]:
    """``(f_fstypename, f_flags)`` of the volume holding ``path``, via libc ``statfs``."""
    import ctypes
    import ctypes.util

    class StatFS(ctypes.Structure):  # struct statfs, 64-bit inodes (sys/mount.h)
        _fields_ = [
            ("f_bsize", ctypes.c_uint32),
            ("f_iosize", ctypes.c_int32),
            ("f_blocks", ctypes.c_uint64),
            ("f_bfree", ctypes.c_uint64),
            ("f_bavail", ctypes.c_uint64),
            ("f_files", ctypes.c_uint64),
            ("f_ffree", ctypes.c_uint64),
            ("f_fsid", ctypes.c_int32 * 2),
            ("f_owner", ctypes.c_uint32),
            ("f_type", ctypes.c_uint32),
            ("f_flags", ctypes.c_uint32),
            ("f_fssubtype", ctypes.c_uint32),
            ("f_fstypename", ctypes.c_char * 16),
            ("f_mntonname", ctypes.c_char * 1024),
            ("f_mntfromname", ctypes.c_char * 1024),
            ("f_reserved", ctypes.c_uint32 * 8),
        ]

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.dylib", use_errno=True)
    fn = getattr(libc, "statfs$INODE64", None) or libc.statfs
    buf = StatFS()
    if fn(os.fsencode(_existing_ancestor(path)), ctypes.byref(buf)) != 0:
        raise OSError(ctypes.get_errno(), "statfs failed")
    return buf.f_fstypename.decode("ascii", "replace"), buf.f_flags


def mac_network_volume(path: str | os.PathLike, statfs=None) -> str | None:
    fstype, flags = (statfs or _mac_statfs)(_absolute(path))
    fstype = fstype.lower()
    if fstype in _NETWORK_FS:
        return _NETWORK_FS[fstype]
    if not flags & _MNT_LOCAL and fstype not in ("devfs", "autofs"):
        return "a network volume"  # not local and not a known pseudo fs
    return None


def windows_network_volume(path: str | os.PathLike, drive_type=None) -> str | None:
    """Judge the path as written first (string checks and GetDriveTypeW only: nothing here
    contacts a share, so a hostile UNC path never starts an SMB session); only a local-looking
    path is resolved, to catch junctions and links that lead to a share."""
    raw = str(path)
    found = _win_network(ntpath.normpath(raw), drive_type)
    if found is None:
        real = os.path.realpath(raw)
        if real != raw:
            found = _win_network(ntpath.normpath(real), drive_type)
    return found


def _win_network(text: str, drive_type) -> str | None:
    if text.lower().startswith("\\\\?\\unc\\"):
        return "a network share"
    if text.startswith(("\\\\?\\", "\\\\.\\")):  # \\?\Z:\... -> Z:\...
        text = text[4:]
    elif text.startswith("\\\\"):
        return "a network share"  # UNC path
    drive = ntpath.splitdrive(ntpath.abspath(text))[0]
    if len(drive) != 2:
        return None
    if drive_type is None:
        import ctypes

        drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
    return "a network drive" if drive_type(drive + "\\") == _DRIVE_REMOTE else None


def linux_network_volume(path: str | os.PathLike, mounts: str | None = None) -> str | None:
    """Longest matching mount point in /proc/mounts decides."""
    if mounts is None:
        mounts = Path("/proc/mounts").read_text(encoding="utf-8", errors="replace")
    table = []
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            table.append((parts[1].replace("\\040", " "), parts[2]))

    def classify(text: str) -> str | None:
        target = posixpath.normpath(text.replace("\\", "/"))
        best, best_type = "", ""
        for mnt, fstype in table:
            if (target == mnt or target.startswith(mnt.rstrip("/") + "/")) and len(mnt) > len(best):
                best, best_type = mnt, fstype
        return _NETWORK_FS.get(best_type)

    raw = str(path)  # as written first; resolve links only when it looks local
    found = classify(raw)
    if found is None:
        real = os.path.realpath(raw)
        if real != raw:
            found = classify(real)
    return found


def network_volume(path: str | os.PathLike, platform: str | None = None) -> str | None:
    """Kind of network share holding ``path`` ("SMB", "NFS", "a network drive", ...) or None."""
    platform = platform or sys.platform
    try:
        if platform == "darwin":
            return mac_network_volume(path)
        if platform == "win32":
            return windows_network_volume(path)
        if platform.startswith("linux"):
            return linux_network_volume(path)
    except Exception:  # best effort, like the cloud check
        log.debug("network-volume check failed", exc_info=True)
    return None


def network_warning_text(path: str | os.PathLike, kind: str) -> str:
    where = kind if kind.startswith("a ") else f"{kind} storage"
    return (
        f"The transcripts folder {path} is on {where}: recordings and transcripts leave this "
        "device over the network. Set output_dir to a local folder to keep them here."
    )
