"""Shared data types. Every module talks to the others through these."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

# All captured audio is normalised to this format before it leaves a capture backend:
# float32, mono, 16 kHz (what Whisper expects).
SAMPLE_RATE = 16_000

# Whisper languages written right to left (Hebrew, Yiddish, Arabic, Persian, Urdu, Pashto,
# Sindhi). Captions and transcripts render their segments right-aligned / with bidi marks.
RTL_LANGUAGES = frozenset({"he", "yi", "ar", "fa", "ur", "ps", "sd"})

log = logging.getLogger(__name__)

# SessionInfo.save: how often / how long to retry replacing a session.json someone has open.
_REPLACE_TRIES, _REPLACE_WAIT_S = 5, 0.05

# Guards SessionInfo.metrics updates against a concurrent save (several pipeline threads).
_METRICS_LOCK = threading.RLock()
# Serialises session.json writes in this process, so a title change (set_title) can't land
# between a save's read of the on-disk title and its replace.
_SAVE_LOCK = threading.Lock()


class Mode(StrEnum):
    OFF = "off"  # never record
    ASK = "ask"  # offer to record when audio is detected
    AUTO = "auto"  # record automatically when audio is detected


class Track(StrEnum):
    ME = "me"  # microphone
    OTHERS = "others"  # system / loopback audio


class TriggerKind(StrEnum):
    CALL = "call"
    MEDIA = "media"
    MANUAL = "manual"
    ROOM = "room"  # people talking near the laptop, no app involved (room listening)


@dataclass(frozen=True)
class AudioProcess:
    """A process currently doing audio I/O, as reported by an ActivityMonitor."""

    pid: int
    bundle_id: str  # macOS bundle id ("us.zoom.xos") or Windows exe name ("zoom.exe")
    name: str  # human readable name ("zoom.us")
    input_active: bool  # process is capturing from a microphone
    output_active: bool  # process is playing audio


@dataclass(frozen=True)
class AppTarget:
    """The app a session records, for a per-app system-audio tap: its processes' pids plus
    patterns (case-insensitive substrings of bundle id / name) that also catch helper
    processes it starts later, like the patterns detection groups an app's processes by."""

    match: tuple[str, ...] = ()
    pids: tuple[int, ...] = ()


@dataclass
class Segment:
    start: float  # seconds from session start
    end: float
    text: str
    speaker: str  # "Me", "Others" (later: "Speaker 1", ...)
    language: str  # ISO 639-1 code, e.g. "en", "ru", "he"
    overlap: bool = False  # diarization: two other speakers at once (json only, when True)
    # (start, end, text) per word, when the backend gave word timestamps (faster-whisper, only
    # for "Others" while diarizing). In memory only: used to split at a speaker change, never
    # written to transcript.json.
    words: list[tuple[float, float, str]] = field(default_factory=list, compare=False, repr=False)


def is_rtl(language: str, text: str = "") -> bool:
    """True if a segment reads right to left: by its language, or (no language) by the first
    strong-direction character of ``text``."""
    if language:
        return language in RTL_LANGUAGES
    for ch in text:
        kind = unicodedata.bidirectional(ch)
        if kind in ("R", "AL"):
            return True
        if kind == "L":
            return False
    return False


@dataclass
class SessionInfo:
    """One recording. Persisted as <directory>/session.json."""

    id: str
    directory: Path
    started_at: datetime
    trigger_kind: TriggerKind
    trigger_app: str | None = None
    ended_at: datetime | None = None
    # Track -> WAV filename relative to `directory` (16 kHz mono).
    tracks: dict[Track, str] = field(default_factory=dict)
    transcribed: bool = False
    # Pipeline measurements, by stage ("recording", "live", "transcription", ...): plain
    # JSON values. Absent in session.json files written before it existed.
    metrics: dict[str, dict] = field(default_factory=dict)
    # Started by room listening, even if a call upgraded it later: everyone in the room is on
    # the mic track. Absent (False) in older session.json files.
    started_as_room: bool = False
    # Display title the user gave it in Meeting notes (None: never renamed, "": reset).
    # Changed only by set_title; save() keeps whatever title the file already has.
    title: str | None = None
    # The voices store's forget generation when recording began: if "Forget all voices" ran
    # since, no voice data or call-app names are kept for it (None: older files, unknown).
    voices_generation: int | None = None

    FILENAME = "session.json"
    # Present in a folder the user discarded: never transcribe or list it (discard_session).
    DISCARD_MARKER = ".discarded"

    def update_metrics(self, stage: str, values: dict) -> None:
        """Merge `values` into ``metrics[stage]`` (thread-safe; not saved until ``save``)."""
        with _METRICS_LOCK:
            self.metrics[stage] = {**self.metrics.get(stage, {}), **values}

    def save(self) -> None:
        with _METRICS_LOCK:
            data = asdict(self)
        with _SAVE_LOCK:
            # The title on disk wins: a copy loaded before (or after) a rename never undoes it.
            target = self.directory / self.FILENAME
            for attempt in range(_REPLACE_TRIES):
                try:
                    data["title"] = json.loads(target.read_text("utf-8")).get("title")
                    break
                except FileNotFoundError:
                    break  # a new session: its own title
                except (OSError, ValueError, AttributeError):  # locked / half-written (Windows)
                    if attempt + 1 < _REPLACE_TRIES:
                        time.sleep(_REPLACE_WAIT_S)
            self._write(data)

    @classmethod
    def set_title(cls, directory: Path, title: str) -> None:
        """Change only the ``title`` key of a session.json ("" resets it)."""
        d = Path(directory)
        with _SAVE_LOCK:
            data = json.loads((d / cls.FILENAME).read_text(encoding="utf-8"))
            data["title"] = title
            _replace_json(d / cls.FILENAME, json.dumps(data, indent=2, ensure_ascii=False))

    def _write(self, data: dict) -> None:
        data["directory"] = str(self.directory)
        data["started_at"] = self.started_at.isoformat()
        data["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        data["tracks"] = {str(k): v for k, v in self.tracks.items()}
        _replace_json(
            self.directory / self.FILENAME, json.dumps(data, indent=2, ensure_ascii=False)
        )

    @classmethod
    def load(cls, directory: Path) -> SessionInfo:
        data = json.loads((Path(directory) / cls.FILENAME).read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            directory=Path(directory),
            started_at=datetime.fromisoformat(data["started_at"]),
            trigger_kind=TriggerKind(data["trigger_kind"]),
            trigger_app=data.get("trigger_app"),
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            tracks={Track(k): v for k, v in data.get("tracks", {}).items()},
            transcribed=data.get("transcribed", False),
            metrics=data.get("metrics") or {},
            started_as_room=data.get("started_as_room", False),
            title=t if isinstance(t := data.get("title"), str) else None,
            voices_generation=g if type(g := data.get("voices_generation")) is int else None,
        )


def _replace_json(target: Path, text: str) -> None:
    """Write ``target`` atomically, so a crash mid-write can't leave it truncated; the temp
    name is unique per process and thread (the app and the CLI may both save)."""
    tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        for attempt in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, target)
                return
            except PermissionError:  # Windows: someone (antivirus, a reader) has it open
                if attempt + 1 < _REPLACE_TRIES:
                    time.sleep(_REPLACE_WAIT_S)
        log.warning("%s stays in use; writing it in place", target)
        target.write_text(text, encoding="utf-8")
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


RECORDING_LOCK = ".recording.lock"  # in a session folder another process records into


def pid_alive(pid: int) -> bool:
    """Whether process ``pid`` exists (never signals it: ``os.kill`` would end it on
    Windows)."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED: it exists
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # someone else's process
        return True
    except OSError:
        return False
    return True


def hold_recording_lock(directory: Path) -> Path:
    """Mark ``directory`` as being recorded by this process (``recandcc record``): startup
    recovery leaves it alone while this pid is alive. Remove the returned file when done."""
    lock = Path(directory) / RECORDING_LOCK
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def boot_time() -> float | None:
    """When the system last booted (epoch seconds), or None if unknown. Counts time asleep,
    unlike ``time.monotonic`` on macOS."""
    try:
        if sys.platform == "win32":
            import ctypes

            k32 = ctypes.WinDLL("kernel32")
            k32.GetTickCount64.restype = ctypes.c_uint64  # ms since boot, sleep included
            return time.time() - k32.GetTickCount64() / 1000
        if sys.platform == "darwin":
            import ctypes

            class Timeval(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int32)]

            tv, size = Timeval(), ctypes.c_size_t(ctypes.sizeof(Timeval))
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.sysctlbyname(b"kern.boottime", ctypes.byref(tv), ctypes.byref(size), None, 0):
                return None
            return tv.tv_sec + tv.tv_usec / 1e6
        with open("/proc/stat", encoding="ascii") as f:  # Linux
            for line in f:
                if line.startswith("btime "):
                    return float(line.split()[1])
    except Exception:  # noqa: BLE001 - unknown: locks are judged by their pid alone
        return None
    return None


def recording_lock_held(directory: Path) -> bool:
    """True if a live process holds ``directory``'s recording lock (an unreadable lock counts
    as held). A lock whose process is gone, or written before the system last booted (its pid
    may now be another process), is deleted."""
    lock = Path(directory) / RECORDING_LOCK
    try:
        text = lock.read_text(encoding="utf-8").strip()
        written = lock.stat().st_mtime
    except FileNotFoundError:
        return False
    except OSError:
        return True
    try:
        pid = int(text)
    except ValueError:
        pid = 0  # cut short while being written: its writer died with it
    booted = boot_time()
    if (booted is None or written >= booted) and pid_alive(pid):
        return True
    with contextlib.suppress(OSError):
        lock.unlink(missing_ok=True)
    log.info("removed stale recording lock in %s", directory)
    return False


def mark_discarded(directory: Path) -> None:
    """Mark a session folder as discarded, so it is never transcribed or listed again: create
    ``SessionInfo.DISCARD_MARKER``, or if that fails rename session.json to it, or as a last
    resort delete session.json. Never raises."""
    d = Path(directory)
    marker, meta = d / SessionInfo.DISCARD_MARKER, d / SessionInfo.FILENAME
    try:
        marker.touch()
        return
    except OSError:
        pass
    try:
        os.replace(meta, marker)
    except FileNotFoundError:
        pass
    except OSError:
        with contextlib.suppress(OSError):
            meta.unlink(missing_ok=True)


def discard_session(directory: Path) -> bool:
    """Delete a session folder the user discarded; True once it is gone.

    The folder is marked (``mark_discarded``) before any audio is touched and the marker is
    deleted last. So if a file can't be deleted (e.g. Windows still locks a WAV) the leftover
    is never transcribed and is deleted again at the next start. Files that vanish meanwhile
    are fine. May take a while (big WAVs): call it off latency-sensitive threads."""
    d = Path(directory)
    marker, meta = d / SessionInfo.DISCARD_MARKER, d / SessionInfo.FILENAME
    failed: list[str] = []
    for _ in range(2):  # a second pass catches files written meanwhile (e.g. a racing purge)
        mark_discarded(d)  # every pass runs marked
        failed = []
        try:
            entries = list(d.iterdir())
        except FileNotFoundError:
            return True
        except OSError as e:
            failed.append(str(e))
            break
        # Audio and transcripts first, then session.json, the marker last.
        for p in sorted(entries, key=lambda p: (p == marker, p == meta)):
            if p == marker and failed:
                break
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except OSError as e:
                failed.append(f"{p.name}: {e.strerror or e}")
        if not failed:
            try:
                d.rmdir()
                return True
            except FileNotFoundError:
                return True
            except OSError as e:
                failed.append(str(e))
    with contextlib.suppress(OSError):
        if not d.exists():
            return True
    log.warning("could not delete discarded session %s: %s", d, "; ".join(failed))
    return False
