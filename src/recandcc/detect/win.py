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

import logging
import ntpath
import os
import sys
import threading

from ..models import AudioProcess
from .base import ActivityMonitor

log = logging.getLogger(__name__)

_MIC_KEY = (
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone"
)
_STATE_ACTIVE = 1


class WinActivityMonitor(ActivityMonitor):
    def __init__(self) -> None:
        self._tls = threading.local()
        self._own_exe = os.path.normcase(os.path.abspath(sys.executable))

    def _ensure_com(self) -> None:
        if getattr(self._tls, "com", False):
            return
        import comtypes

        comtypes.CoInitialize()
        self._tls.com = True

    # ---- output -------------------------------------------------------------
    def _output_sessions(self) -> dict[str, AudioProcess]:
        from pycaw.pycaw import AudioUtilities

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
                stop, _ = winreg.QueryValueEx(key, "LastUsedTimeStop")
                start, _ = winreg.QueryValueEx(key, "LastUsedTimeStart")
            except OSError:
                return False
            return stop == 0 and start != 0

        def walk(path: str, packaged: bool):
            try:
                root = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path)
            except OSError:
                return
            with root:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(root, i)
                    except OSError:
                        break
                    i += 1
                    if sub == "NonPackaged":
                        continue
                    try:
                        with winreg.OpenKey(root, sub) as k:
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

    def _pid_for_exe(self, exe_lower: str) -> int:
        try:
            import psutil  # pycaw dependency

            for p in psutil.process_iter(["name"]):
                if (p.info.get("name") or "").lower() == exe_lower:
                    return p.pid
        except Exception:
            log.debug("process lookup for %s failed", exe_lower, exc_info=True)
        return 0

    def _packaged_exes(self, families: set[str]) -> dict[str, list[tuple[str, int]]]:
        """Package family (lowercase) -> [(exe_name_lower, pid)] of its running processes.

        The consent store names packaged apps by family (``MSTeams_8wekyb3d8bbwe``) while
        WASAPI sessions and the app lists use exe names (``ms-teams.exe``). Packaged apps run
        from ``...\\WindowsApps\\<Name>_<Version>_<Arch>_<ResourceId>_<PublisherId>\\``, and
        the family name is ``<Name>_<PublisherId>``.
        """
        found: dict[str, list[tuple[str, int]]] = {}
        try:
            import psutil

            for p in psutil.process_iter(["name", "exe"]):
                parts = (p.info.get("exe") or "").split("\\")
                low = [x.lower() for x in parts]
                if "windowsapps" not in low[:-2]:
                    continue
                fields = low[low.index("windowsapps") + 1].split("_")
                family = f"{fields[0]}_{fields[-1]}"
                if family in families:
                    exe = (p.info.get("name") or parts[-1]).lower()
                    found.setdefault(family, []).append((exe, p.pid))
        except Exception:
            log.debug("packaged app lookup failed", exc_info=True)
        return found

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
        families = {full.lower() for _, full in mic if "\\" not in full}
        packaged = self._packaged_exes(families) if families else {}
        for exe, full in mic:
            if os.path.normcase(full) == self._own_exe:
                continue
            pid = 0
            if cands := packaged.get(full.lower()):
                exe, pid = next((c for c in cands if c[0] in procs), cands[0])
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
                    pid=pid or self._pid_for_exe(exe),
                    bundle_id=exe,
                    name=exe,
                    input_active=True,
                    output_active=False,
                )
        return list(procs.values())

    def close(self) -> None:
        if getattr(self._tls, "com", False):
            try:
                import comtypes

                comtypes.CoUninitialize()
            except Exception:
                log.debug("CoUninitialize failed", exc_info=True)
            self._tls.com = False
