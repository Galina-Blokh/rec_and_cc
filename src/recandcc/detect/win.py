"""Windows audio activity detection. UNTESTED (written without access to Windows).

Output: pycaw ``AudioUtilities.GetAllSessions()`` enumerates WASAPI render sessions of the
default endpoint; a session whose ``State == AudioSessionStateActive (1)`` is playing.

Input (best effort): Windows has no public per-process "is capturing" API. We read the
privacy "CapabilityAccessManager" consent store, which Windows updates whenever an app
opens/closes the microphone::

    HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\microphone
        <PackageFamilyName>\\   (packaged / Store apps, e.g. MSTeams_8wekyb3d8bbwe)
        NonPackaged\\C:#Program Files#Zoom#bin#Zoom.exe\\   (classic apps, '\\' replaced by '#')
            LastUsedTimeStart  QWORD FILETIME
            LastUsedTimeStop   QWORD FILETIME, 0 while the app is using the mic

So ``LastUsedTimeStop == 0 and LastUsedTimeStart != 0`` means "mic in use right now".
Caveats: browsers report as the browser exe (good enough for us), stale entries can remain
0 after a crash, and our own python.exe shows up while we record (filtered out below).

COM must be initialised on each thread that calls into pycaw; we do it lazily per thread.
"""

from __future__ import annotations

# The `pyright: ignore`s below mark Windows-only APIs: comtypes/pycaw are installed, and winreg
# is typed, only on Windows, while type checking runs on Linux and macOS.
import logging
import ntpath
import os
import re
import sys
import threading
import time

from ..models import AudioProcess
from .base import ActivityMonitor

log = logging.getLogger(__name__)

_MIC_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
)
_STATE_ACTIVE = 1
_TABLE_TTL_S = 10.0  # re-walk the process list at most this often while the mic set is stable
_HELPER_WORDS = ("autostart", "update", "crash", "helper")


def _alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


class WinActivityMonitor(ActivityMonitor):
    def __init__(self) -> None:
        self._tls = threading.local()
        self._own_exe = os.path.normcase(os.path.abspath(sys.executable))
        self._table: tuple | None = None  # (mic key, time, by_family, by_name)
        self._chosen: dict[str, str] = {}  # package family -> exe reported last time

    def _ensure_com(self) -> None:
        if getattr(self._tls, "com", False):
            return
        import comtypes  # pyright: ignore[reportMissingImports]

        comtypes.CoInitialize()
        self._tls.com = True

    # ---- output -------------------------------------------------------------
    def _output_sessions(self) -> dict[str, AudioProcess]:
        from pycaw.pycaw import AudioUtilities  # pyright: ignore[reportMissingImports]

        out: dict[str, AudioProcess] = {}
        for s in AudioUtilities.GetAllSessions():
            try:
                proc = s.Process
                if proc is None:  # system sounds session
                    continue
                if s.State != _STATE_ACTIVE:
                    continue
                exe = proc.name()
                if proc.pid == os.getpid():
                    continue
                key = exe.lower()
                out[key] = AudioProcess(
                    pid=proc.pid,
                    bundle_id=exe.lower(),
                    name=exe,
                    input_active=False,
                    output_active=True,
                )
            except Exception:  # process vanished etc.
                log.debug("skipping audio session", exc_info=True)
        return out

    # ---- input --------------------------------------------------------------
    def _mic_users(self) -> list[tuple[str, str]]:
        """[(exe_name_lower, full_path_or_package)] of apps currently using the mic."""
        import winreg

        users: list[tuple[str, str]] = []

        def in_use(key) -> bool:
            try:
                stop, _ = winreg.QueryValueEx(key, "LastUsedTimeStop")  # pyright: ignore[reportAttributeAccessIssue]
                start, _ = winreg.QueryValueEx(key, "LastUsedTimeStart")  # pyright: ignore[reportAttributeAccessIssue]
            except OSError:
                return False
            return stop == 0 and start != 0

        def walk(path: str, packaged: bool):
            try:
                root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path)  # pyright: ignore[reportAttributeAccessIssue]
            except OSError:
                return
            with root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)  # pyright: ignore[reportAttributeAccessIssue]
                    except OSError:
                        break
                    i += 1
                    if sub == "NonPackaged":
                        continue
                    try:
                        with winreg.OpenKey(root, sub) as k:  # pyright: ignore[reportAttributeAccessIssue]
                            if not in_use(k):
                                continue
                    except OSError:
                        continue
                    if packaged:
                        users.append((sub.split("_")[0].lower(), sub))
                    else:
                        full = sub.replace("#", "\\")
                        users.append((ntpath.basename(full).lower(), full))

        walk(_MIC_KEY, packaged=True)
        walk(_MIC_KEY + r"\NonPackaged", packaged=False)
        return users

    def _process_table(
        self, key: frozenset[str]
    ) -> tuple[dict[str, list[tuple[str, int]]], dict[str, int]]:
        """One psutil walk -> (package family -> [(exe_lower, pid)], exe_lower -> pid).

        The consent store names packaged apps by family (``MSTeams_8wekyb3d8bbwe``) while
        WASAPI sessions and the app lists use exe names (``ms-teams.exe``). Packaged apps run
        from ``...\\WindowsApps\\<Name>_<Version>_<Arch>_<ResourceId>_<PublisherId>\\``, and
        the family name is ``<Name>_<PublisherId>``. Cached until the set of mic users
        (``key``) changes or ``_TABLE_TTL_S`` passes, so polling does not walk every second.
        """
        now = time.monotonic()
        if self._table and self._table[0] == key and now - self._table[1] < _TABLE_TTL_S:
            return self._table[2], self._table[3]
        by_family: dict[str, list[tuple[str, int]]] = {}
        by_name: dict[str, int] = {}
        try:
            import psutil  # pyright: ignore[reportMissingModuleSource] - pycaw dependency, Windows only

            # ad_value=None: protected processes yield None, not an AccessDenied instance
            for p in psutil.process_iter(["name", "exe"], ad_value=None):
                name, exe = p.info.get("name"), p.info.get("exe")
                name = name.lower() if isinstance(name, str) else ""
                if name:
                    by_name.setdefault(name, p.pid)
                if not isinstance(exe, str):
                    continue
                parts = exe.lower().split("\\")
                if "windowsapps" not in parts[:-2]:
                    continue
                fields = parts[parts.index("windowsapps") + 1].split("_")
                family = f"{fields[0]}_{fields[-1]}"
                by_family.setdefault(family, []).append((name or parts[-1], p.pid))
        except Exception:
            log.debug("process lookup failed", exc_info=True)
            return by_family, by_name  # partial: use it this poll, but don't cache it
        self._table = (key, now, by_family, by_name)
        return by_family, by_name

    def _choose(self, family: str, cands: list[tuple[str, int]], procs: dict) -> tuple[str, int]:
        """Pick one process per package, deterministically and stable across polls: a key
        that flips between polls looks like a new app to the engine and ends the session."""
        cands = sorted(cands)
        app = _alnum(family.split("_")[0])
        pick = (
            # Sticky first: a helper briefly playing audio must not take over the key.
            next((c for c in cands if c[0] == self._chosen.get(family)), None)
            or next((c for c in cands if c[0] in procs), None)  # has the WASAPI session
            or next((c for c in cands if _alnum(c[0].rsplit(".", 1)[0]) == app), None)
            or next((c for c in cands if not any(w in c[0] for w in _HELPER_WORDS)), None)
            or cands[0]
        )
        self._chosen[family] = pick[0]
        return pick

    def snapshot(self) -> list[AudioProcess]:
        self._ensure_com()
        try:
            procs = self._output_sessions()
        except Exception:
            log.exception("pycaw session enumeration failed")
            procs = {}
        try:
            mic = self._mic_users()
        except Exception:
            log.exception("reading microphone consent store failed")
            mic = []
        key = frozenset(full.lower() for _, full in mic)
        by_family, by_name = self._process_table(key) if mic else ({}, {})
        for exe, full in mic:
            if os.path.normcase(full) == self._own_exe:
                continue
            if "\\" not in full and (cands := by_family.get(full.lower())):
                exe, pid = self._choose(full.lower(), cands, procs)
            else:
                pid = by_name.get(exe, 0)
            existing = procs.get(exe)
            if existing:
                procs[exe] = AudioProcess(
                    existing.pid,
                    existing.bundle_id,
                    existing.name,
                    input_active=True,
                    output_active=existing.output_active,
                )
            else:
                procs[exe] = AudioProcess(
                    pid=pid,
                    bundle_id=exe,
                    name=exe,
                    input_active=True,
                    output_active=False,
                )
        self._chosen = {f: e for f, e in self._chosen.items() if f in key}  # off the mic
        return list(procs.values())

    def close(self) -> None:
        if getattr(self._tls, "com", False):
            try:
                import comtypes  # pyright: ignore[reportMissingImports]

                comtypes.CoUninitialize()
            except Exception:
                log.debug("CoUninitialize failed", exc_info=True)
            self._tls.com = False
