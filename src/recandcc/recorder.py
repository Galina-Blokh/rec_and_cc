"""Session recorder: capture sources -> queues -> WAV writer threads.

Thread layout per track::

    AudioSource worker thread --on_frames()--> queue.Queue --> writer thread (soundfile)

``on_frames`` only measures the level (cheap numpy RMS) and enqueues a copy, so capture
never blocks on disk I/O. Frame listeners (``add_listener``, used for live transcription) get
the same copy on the capture thread and must only enqueue it. ``stop()`` stops the sources,
sends a sentinel to each writer, joins them, and persists session.json.
"""

from __future__ import annotations

import errno
import logging
import ntpath
import os
import queue
import re
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from functools import cache
from pathlib import Path

import numpy as np
import soundfile as sf

from .capture.base import AudioSource
from .config import Config
from .models import SAMPLE_RATE, AppTarget, SessionInfo, Track, TriggerKind

log = logging.getLogger(__name__)

PRIVATE_DIR_MODE = 0o700  # owner-only on POSIX; Windows gets an owner-only DACL instead
_IS_WINDOWS = sys.platform == "win32"  # module flag so tests can fake Windows safely
_ERROR_ALREADY_EXISTS = 183
_FILE_PERSISTENT_ACLS = 0x8  # GetVolumeInformationW file-system flag
_warned: set[str] = set()  # "<reason>:<volume>" keys already logged
_noticed: set[str] = set()  # "<reason>:<volume>" keys already shown to the user
_keys_lock = threading.Lock()  # engine and UI threads both create folders
PRIVACY_NOTICE = (
    "Recordings may be readable by other users",
    "RecAndCC couldn't make its recording folder private (no owner-only access list on this "
    "drive). Keep output_dir on a local NTFS drive; see the log for details.",
)


def _not_private(key: str, msg: str, *args: object) -> str:
    """Log a folder-not-private warning once per ``key`` and return ``key``."""
    with _keys_lock:
        first = key not in _warned
        _warned.add(key)
    if first:
        log.warning(msg, *args)
    return key


def _win_long_path(path: Path | str) -> str:
    """``\\\\?\\`` form of an absolute Windows path, so folders past MAX_PATH stay private."""
    s = ntpath.normpath(str(path))
    if s.startswith("\\\\?\\") or not ntpath.isabs(s):
        return s
    if s.startswith("\\\\"):  # \\host\share\... -> \\?\UNC\host\share\...
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s if s[1:3] == ":\\" else s


def _volume_key(path: Path) -> tuple[str, bool]:
    """``(volume key, persists ACLs)``; can't tell -> ``(drive, False)``: assume the worst."""
    try:
        return _volume_acl_support(path)
    except Exception as e:
        log.debug("could not check ACL support for %s: %s", path, e)
        return path.anchor or str(path), False


@cache
def _current_user_sid() -> str:
    """SID of the user running the app, e.g. ``S-1-5-21-...-1001`` (Windows only)."""
    if sys.platform != "win32":  # also narrows the Win32-only ctypes names for pyright
        raise OSError("Windows only")
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.LocalFree.argtypes = [ctypes.c_void_p]
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p]
    adv.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    token = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise ctypes.WinError(ctypes.get_last_error())  # TOKEN_QUERY ^
    try:
        size = wintypes.DWORD()
        adv.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))  # 1 = TokenUser
        buf = ctypes.create_string_buffer(size.value)
        if not adv.GetTokenInformation(token, 1, buf, size, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        psid = ctypes.cast(buf, ctypes.POINTER(ctypes.c_void_p))[0]  # TOKEN_USER.User.Sid
        text = wintypes.LPWSTR()
        if not adv.ConvertSidToStringSidW(psid, ctypes.byref(text)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return str(text.value)
        finally:
            k32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        k32.CloseHandle(token)


def owner_only_sddl(user_sid: str) -> str:
    """DACL with full control for ``user_sid`` and SYSTEM only.

    ``P`` (protected) blocks inheritance from the parent folder (e.g. ``Users: Read`` on a
    shared drive); ``OICI`` passes the same two entries to files and subfolders inside.
    """
    return f"D:P(A;OICI;FA;;;{user_sid})(A;OICI;FA;;;SY)"


def _exists_error(err: int, path: Path) -> FileExistsError | None:
    """``FileExistsError`` if ``CreateDirectoryW`` failed with ERROR_ALREADY_EXISTS (183)."""
    if err == _ERROR_ALREADY_EXISTS:
        return FileExistsError(errno.EEXIST, "folder already exists", str(path))
    return None


def _win_mkdir_owner_only(path: Path) -> None:
    """Create one folder with an owner-only DACL in a single call (no permissive window)."""
    sddl = owner_only_sddl(_current_user_sid())
    if sys.platform != "win32":  # also narrows the Win32-only ctypes names for pyright
        raise OSError("Windows only")
    import ctypes
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    k32.LocalFree.argtypes = [ctypes.c_void_p]
    adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    sd = ctypes.c_void_p()
    if not adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl,
        1,
        ctypes.byref(sd),
        None,  # 1 = SDDL_REVISION_1
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sa = SecurityAttributes(ctypes.sizeof(SecurityAttributes), sd, False)
        if not k32.CreateDirectoryW(_win_long_path(path), ctypes.byref(sa)):
            err = ctypes.get_last_error()
            raise _exists_error(err, path) or ctypes.WinError(err)
    finally:
        k32.LocalFree(sd)


def _volume_acl_support(path: Path) -> tuple[str, bool]:
    """``(volume key, persists ACLs)`` for an existing ``path``; key = root + serial number.

    FAT/exFAT and some shares accept ``CreateDirectoryW`` with a security descriptor but
    silently drop it; their volumes lack ``FILE_PERSISTENT_ACLS``. Two in-process API calls,
    made every time (no cache): a USB stick swapped under the same drive letter has another
    serial number, so it gets its own key.
    """
    root, serial, flags = _volume_info(_win_long_path(path))  # \\?\: works past MAX_PATH
    return f"{root} ({serial:08X})", bool(flags & _FILE_PERSISTENT_ACLS)


def _short_root(root: str) -> str:
    r"""``\\?\C:\`` -> ``C:\``, ``\\?\UNC\h\s\`` -> ``\\h\s\``; GUID roots stay."""
    if root.startswith("\\\\?\\UNC\\"):
        return "\\\\" + root[8:]
    if root.startswith("\\\\?\\") and root[5:6] == ":":
        return root[4:]
    return root


def _volume_info(long_path: str) -> tuple[str, int, int]:
    """``(volume root, serial number, file-system flags)`` via the Win32 volume API."""
    if sys.platform != "win32":  # also narrows the Win32-only ctypes names for pyright
        raise OSError("Windows only")
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetVolumePathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    k32.GetVolumeInformationW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    buf = ctypes.create_unicode_buffer(1024)
    if not k32.GetVolumePathNameW(long_path, buf, len(buf)):
        raise ctypes.WinError(ctypes.get_last_error())
    root = _short_root(buf.value)
    serial, flags = wintypes.DWORD(), wintypes.DWORD()
    if not k32.GetVolumeInformationW(
        root, None, 0, ctypes.byref(serial), None, ctypes.byref(flags), None, 0
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return root, serial.value, flags.value


def _mkdir_private(path: Path) -> str | None:
    """Create one folder (parent must exist), private to the current user from the start.

    POSIX: ``0700`` (chmod-ed too: ``mkdir(mode=...)`` is filtered by the umask). Windows: an
    owner-only DACL set at creation. If Windows refuses the DACL (e.g. some network shares),
    this falls back to a plain ``mkdir(mode=0o700)``; if the volume doesn't keep ACLs (FAT/
    exFAT, or it can't tell), the folder is not private. Either way recording goes on, a
    warning is logged once per (reason, volume), and that key is returned (None = private).
    Raises ``FileExistsError`` if something is already at ``path``.
    """
    if not _IS_WINDOWS:
        path.mkdir(mode=PRIVATE_DIR_MODE)
        path.chmod(PRIVATE_DIR_MODE)
        return None
    try:
        _win_mkdir_owner_only(path)
    except FileExistsError:
        raise  # created concurrently / something in the way: not a privacy problem
    except Exception as e:  # keep recording, just warn
        # FileExistsError here reports nothing; \\?\ so long paths don't abort the start
        os.mkdir(_win_long_path(path) if sys.platform == "win32" else path, PRIVATE_DIR_MODE)
        return _not_private(
            f"acl-fallback:{_volume_key(path)[0]}",
            "could not create %s with an owner-only ACL (%s); recordings may be readable "
            "by other users",
            path,
            e,
        )
    volume, persists = _volume_key(path)
    if persists:
        return None
    return _not_private(
        f"no-acl:{volume}",
        "%s may not keep file permissions (e.g. FAT/exFAT): recordings in %s may be "
        "readable by other users",
        volume,
        path,
    )


def make_private_dir(path: Path) -> str | None:
    """Create ``path`` (and missing parents) with owner-only permissions, if it doesn't exist.

    Each folder this creates is made private (``_mkdir_private``); existing folders are left
    untouched. Returns a not-private key if a folder it created isn't private, else None.
    """
    if path.is_dir():
        return None
    if path.parent == path:  # a missing drive / UNC share root: nothing left to create
        raise FileNotFoundError(f"{path} does not exist")
    issue = make_private_dir(path.parent) if not path.parent.is_dir() else None
    try:
        return _mkdir_private(path) or issue
    except FileExistsError:
        if path.is_dir():
            return issue  # created concurrently
        raise  # a file (or dangling symlink) is in the way


# Sessions with less audio than this (longest track) are deleted instead of transcribed.
MIN_SESSION_SECONDS = 3.0

# Peak below this (~ -100 dBFS) counts as "digital silence" (no permission / dead device).
DIGITAL_SILENCE = 1e-5

TRACK_FILES = {Track.OTHERS: "others.wav", Track.ME: "me.wav"}

# Call apps whose audio a system daemon plays, not the app: a per-app tap on the app alone
# would record silence. Sessions they start use the global tap; added to a running per-app
# tap (a call upgrading a media session), the daemon is added too. Pattern -> daemon name.
DAEMON_PLAYED_APPS = {"com.apple.facetime": "avconferenced"}

SourceFactory = Callable[[], AudioSource]
# Called with no argument (all system audio) or with an AppTarget (only that app's audio).
SystemSourceFactory = Callable[..., AudioSource]
# listener(track, start_frame, frames): ``start_frame`` is the sample index of frames[0] in the
# track's timeline (= in its WAV), so a mic track that joined mid-session starts at its padding.
FrameListener = Callable[[Track, int, np.ndarray], None]
# Captured audio kept per track to replay to a listener added after the session started.
REPLAY_SECONDS = 10.0


def slugify(text: str | None, default: str = "session") -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:40] or default


def dbfs(frames: np.ndarray) -> float:
    if frames.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frames, dtype=np.float64))))
    return 20.0 * np.log10(rms) if rms > 1e-6 else -120.0


def should_record_mic(policy: str, kind: TriggerKind) -> bool:
    policy = (policy or "calls").lower()
    if policy == "always":
        return True
    if policy == "never":
        return False
    return kind in (TriggerKind.CALL, TriggerKind.MANUAL)


class _TrackWriter:
    def __init__(self, track: Track, path: Path, source: AudioSource, recorder: Recorder):
        self.track = track
        self.path = path
        self.source = source
        self.recorder = recorder
        self.q: queue.Queue[np.ndarray | None] = queue.Queue()
        self.frames_written = 0
        self.frames_received = 0  # updated on the capture thread; ahead of frames_written
        self.backlog_max = 0  # most samples ever waiting for the writer (disk falling behind)
        self._padding = 0  # leading zeros (mid-session join): not counted as backlog
        self.level_dbfs = -120.0
        self.peak = 0.0  # max |sample| over the session (0.0 => digital silence throughout)
        self.error: BaseException | None = None
        self._tap_lock = threading.Lock()  # frames_received + listeners + replay buffer
        self._listeners: tuple[FrameListener, ...] = recorder._listeners
        self._recent: deque[tuple[int, np.ndarray]] = deque()
        self._recent_frames = 0
        self._file = sf.SoundFile(
            str(path), mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16", format="WAV"
        )
        self.thread = threading.Thread(target=self._run, name=f"writer-{track}", daemon=True)
        self.thread.start()

    def on_frames(self, frames: np.ndarray) -> None:  # capture thread: keep it cheap
        frames = np.asarray(frames, dtype=np.float32).reshape(-1)
        level = dbfs(frames)
        self.level_dbfs = level
        if frames.size:
            self.peak = max(self.peak, float(np.max(np.abs(frames))))
        self.recorder._note_level(level)
        block = frames.copy()
        with self._tap_lock:
            start = self.frames_received
            self.frames_received += block.size
            self._recent.append((start, block))
            self._recent_frames += block.size
            while self._recent_frames - self._recent[0][1].size >= REPLAY_SECONDS * SAMPLE_RATE:
                self._recent_frames -= self._recent.popleft()[1].size
            for cb in self._listeners:
                self._call(cb, start, block)
        self.q.put(block)
        written = self.frames_written  # an int written by the writer thread: atomic read
        if written >= self._padding:
            self.backlog_max = max(self.backlog_max, start + block.size - written)

    def _call(self, cb: FrameListener, start: int, block: np.ndarray) -> None:
        try:
            cb(self.track, start, block)
        except Exception:  # a listener must never break capture
            log.exception("frame listener failed (%s)", self.track)

    def add_listener(self, cb: FrameListener) -> None:
        """Replay the recent audio to ``cb``, then deliver new frames to it."""
        with self._tap_lock:
            for start, block in self._recent:
                self._call(cb, start, block)
            self._listeners = (*self._listeners, cb)

    def remove_listener(self, cb: FrameListener) -> None:
        with self._tap_lock:
            self._listeners = tuple(c for c in self._listeners if c is not cb)

    def pad_silence(self, frames: int) -> None:
        """Queue ``frames`` zeros ahead of any captured audio (track joining mid-session).
        Listeners are not sent the zeros: the first captured block's ``start_frame`` already
        includes the padding."""
        with self._tap_lock:
            self.frames_received += frames
        self._padding += frames
        for i in range(0, frames, SAMPLE_RATE):
            self.q.put(np.zeros(min(SAMPLE_RATE, frames - i), dtype=np.float32))

    def _run(self) -> None:
        try:
            while True:
                block = self.q.get()
                if block is None:
                    break
                self._file.write(np.clip(block, -1.0, 1.0))
                self.frames_written += block.size
        except BaseException as e:  # disk full etc. — drain so capture isn't stuck
            self.error = e
            log.exception("writer %s failed", self.track)
            while self.q.get() is not None:
                pass
        finally:
            self._file.close()

    def finish(self) -> None:
        self.q.put(None)
        self.thread.join()


class Recorder:
    """Records one session at a time. Thread-safe; ``start``/``stop`` may block briefly."""

    def __init__(
        self,
        cfg: Config,
        mic_factory: SourceFactory | None = None,
        system_factory: SystemSourceFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        min_session_s: float = MIN_SESSION_SECONDS,
    ):
        if mic_factory is None or system_factory is None:
            from . import capture

            mic_factory = mic_factory or capture.create_mic_source
            system_factory = system_factory or capture.create_system_source
        self.cfg = cfg
        self.mic_factory = mic_factory
        self.system_factory = system_factory
        self.clock = clock
        self.min_session_s = min_session_s
        self._op_lock = threading.Lock()  # serialises start/stop
        self._lvl_lock = threading.Lock()  # guards _last_sound
        self._session: SessionInfo | None = None
        self._privacy_issue: str | None = None  # last start's not-private key (privacy_notice)
        self._writers: list[_TrackWriter] = []
        self._started_mono = 0.0
        self._last_sound = 0.0
        # Peak |sample| per track of the most recently stopped session (incl. discarded).
        self.last_peaks: dict[Track, float] = {}
        self._listeners: tuple[FrameListener, ...] = ()

    # ---- frame tap ----------------------------------------------------------
    def add_listener(self, cb: FrameListener) -> None:
        """Receive every track's frames as they are captured: ``cb(track, start_frame,
        frames)`` on the capture thread (keep it cheap: enqueue only; don't mutate
        ``frames``). Added during a session, it first gets the last ``REPLAY_SECONDS`` of
        audio, so nothing since the start is missed. Tracks added later (mic joining a call)
        are included. Listeners are per session: ``stop()`` removes them all."""
        with self._op_lock:
            self._listeners = (*self._listeners, cb)
            for w in self._writers:
                w.add_listener(cb)

    def remove_listener(self, cb: FrameListener) -> None:
        with self._op_lock:
            self._listeners = tuple(c for c in self._listeners if c is not cb)
            for w in self._writers:
                w.remove_listener(cb)

    # ---- levels -------------------------------------------------------------
    def _note_level(self, level: float) -> None:
        if level > self.cfg.detection.silence_dbfs:
            now = self.clock()
            with self._lvl_lock:
                self._last_sound = now

    def seconds_since_sound(self) -> float:
        """Seconds since any track was above ``silence_dbfs`` (since start if never)."""
        with self._lvl_lock:
            last = self._last_sound
        return max(0.0, self.clock() - last) if self.is_recording else 0.0

    def privacy_notice(self) -> tuple[str, str] | None:
        """``(title, message)`` if the last successful start created a folder that isn't
        private and no notice was delivered yet for its (reason, volume) in this process.
        Call ``privacy_noticed()`` once the notice was shown."""
        key = self._privacy_issue
        with _keys_lock:
            return None if key is None or key in _noticed else PRIVACY_NOTICE

    def privacy_noticed(self) -> None:
        """The notice from ``privacy_notice()`` reached the user: don't repeat it."""
        key, self._privacy_issue = self._privacy_issue, None
        if key is not None:
            with _keys_lock:
                _noticed.add(key)

    def silent_tracks(self) -> list[Track]:
        """Tracks of the last stopped session that were digital silence the whole time
        (typically a missing microphone / system-audio permission)."""
        return [t for t, p in self.last_peaks.items() if p <= DIGITAL_SILENCE]

    def levels(self) -> dict[Track, float]:
        return {w.track: w.level_dbfs for w in list(self._writers)}

    @property
    def is_recording(self) -> bool:
        return self._session is not None

    @property
    def session(self) -> SessionInfo | None:
        return self._session

    def elapsed(self) -> float:
        return self.clock() - self._started_mono if self.is_recording else 0.0

    # ---- lifecycle ----------------------------------------------------------
    def _make_dir(
        self, now: datetime, app: str | None, kind: TriggerKind
    ) -> tuple[str, Path, str | None]:
        day = self.cfg.output_dir / now.strftime("%Y-%m-%d")
        slug = slugify(app, default=str(kind))
        base = f"{now.strftime('%H%M%S')}_{slug}"
        d = day / base
        n = 2
        while d.exists():
            d = day / f"{base}-{n}"
            n += 1
        # Recordings and transcripts are private: owner-only folders (0700 / owner-only DACL)
        # at every level the recorder creates. An existing output folder keeps its permissions.
        issues = [make_private_dir(p) for p in (self.cfg.output_dir, day)]
        issues.append(_mkdir_private(d))
        issue = next((i for i in issues if i), None)
        return f"{now.strftime('%Y-%m-%d')}_{d.name}", d, issue

    def _per_app(self, target: AppTarget | None) -> AppTarget | None:
        """``target`` if the Others track should record only that app, else None."""
        if target is None or not (target.match or target.pids):
            return None
        scope = str(self.cfg.detection.system_audio_scope or "all").lower()
        return target if scope == "app" else None

    @staticmethod
    def _daemons(target: AppTarget) -> list[str]:
        return [d for p, d in DAEMON_PLAYED_APPS.items() for m in target.match if p in m.lower()]

    def _system_source(self, target: AppTarget | None) -> AudioSource:
        if target is None:
            return self.system_factory()
        try:
            return self.system_factory(target)
        except Exception:
            log.exception("per-app system audio unavailable; recording all system audio")
            return self.system_factory()

    def start(
        self,
        kind: TriggerKind,
        app: str | None = None,
        record_mic: bool | None = None,
        target: AppTarget | None = None,
    ) -> SessionInfo:
        """Start a session. ``record_mic`` overrides the config policy when not None.
        ``target``: the triggering app, recorded alone with ``system_audio_scope = "app"``."""
        with self._op_lock:
            if self._session is not None:
                raise RuntimeError("already recording")
            now = datetime.now().astimezone()
            self._privacy_issue = None
            sid, directory, privacy_issue = self._make_dir(now, app, kind)
            session = SessionInfo(
                id=sid, directory=directory, started_at=now, trigger_kind=kind, trigger_app=app
            )
            want_mic = (
                should_record_mic(self.cfg.detection.record_mic, kind)
                if record_mic is None
                else record_mic
            )
            per_app = self._per_app(target)
            if per_app is not None and self._daemons(per_app):
                log.info("system audio: %s plays calls via a daemon; recording everything", app)
                per_app = None
            if per_app is not None:
                log.info(
                    "system audio: asking for only %s", ", ".join(per_app.match) or per_app.pids
                )
            plan: list[tuple[Track, SourceFactory]] = [
                (Track.OTHERS, lambda: self._system_source(per_app))
            ]
            if want_mic:
                plan.append((Track.ME, self.mic_factory))

            writers: list[_TrackWriter] = []
            for track, factory in plan:
                w = self._open_track(session, track, factory)
                if w is not None:
                    writers.append(w)

            if not writers:
                shutil.rmtree(directory, ignore_errors=True)
                raise RuntimeError("no audio source could be started")
            self._privacy_issue = privacy_issue  # reported by privacy_notice()
            self._started_mono = self.clock()
            with self._lvl_lock:
                self._last_sound = self._started_mono
            self._writers = writers
            session.save()
            self._session = session
            log.info(
                "recording started: %s (tracks: %s)",
                directory,
                ", ".join(str(t) for t in session.tracks),
            )
            return session

    def _open_track(
        self, session: SessionInfo, track: Track, factory: SourceFactory, pad_frames: int = 0
    ) -> _TrackWriter | None:
        """Create and start one track; None (logged) if its source can't be started."""
        path = session.directory / TRACK_FILES[track]
        try:
            src = factory()
            w = _TrackWriter(track, path, src, self)
        except Exception:
            log.exception("could not create %s source", track)
            return None
        if pad_frames:
            w.pad_silence(pad_frames)
        try:
            src.start(w.on_frames)
        except Exception:
            log.exception("could not start %s source", track)
            w.finish()
            path.unlink(missing_ok=True)
            return None
        session.tracks[track] = TRACK_FILES[track]
        return w

    def upgrade_to_call(self, app: str | None, target: AppTarget | None = None) -> bool:
        """Turn a running media session into a call session (a call started while e.g. a
        video was playing): adds the mic track if the policy wants it for calls. The mic
        track starts with silence up to "now" so it stays aligned with the other tracks.
        A per-app Others track is extended to the call app ``target`` (both apps recorded)."""
        with self._op_lock:
            session = self._session
            if session is None or session.trigger_kind != TriggerKind.MEDIA:
                return False
            session.trigger_kind = TriggerKind.CALL
            session.trigger_app = app or session.trigger_app
            if (per_app := self._per_app(target)) is not None:
                if daemons := self._daemons(per_app):
                    per_app = AppTarget((*per_app.match, *daemons), per_app.pids)
                for w in self._writers:
                    if w.track == Track.OTHERS and not w.source.add_target(per_app):
                        log.info("Others track: could not add %s to the per-app tap", app)
            have_mic = any(w.track == Track.ME for w in self._writers)
            if not have_mic and should_record_mic(self.cfg.detection.record_mic, TriggerKind.CALL):
                pad = max((w.frames_received for w in self._writers), default=0)
                w = self._open_track(session, Track.ME, self.mic_factory, pad_frames=pad)
                if w is not None:
                    self._writers = [*self._writers, w]
            session.save()
            log.info(
                "session upgraded to call (%s); tracks: %s",
                app,
                ", ".join(str(t) for t in session.tracks),
            )
            return True

    def stop(self) -> SessionInfo | None:
        """Stop, flush and close. Returns the session, or None if nothing was recording
        or the session was shorter than ``min_session_s`` (its directory is deleted)."""
        with self._op_lock:
            session, writers = self._session, self._writers
            if session is None:
                return None
            for w in writers:
                try:
                    w.source.stop()
                except Exception:
                    log.exception("stopping %s source failed", w.track)
            for w in writers:
                w.finish()
            self._session, self._writers, self._listeners = None, [], ()
            self.last_peaks = {w.track: w.peak for w in writers}
            session.ended_at = datetime.now().astimezone()

            longest = max((w.frames_written for w in writers), default=0) / SAMPLE_RATE
            if longest < self.min_session_s:
                log.info("discarding short session %s (%.1fs)", session.directory, longest)
                shutil.rmtree(session.directory, ignore_errors=True)
                return None
            for w in writers:  # drop empty tracks (e.g. a source that never delivered)
                if w.frames_written == 0:
                    w.path.unlink(missing_ok=True)
                    session.tracks.pop(w.track, None)
            session.update_metrics(
                "recording",
                {
                    "audio_s": round(longest, 1),
                    # Effective: "all" also when a per-app tap fell back to the global one.
                    "system_audio_scope": next(
                        (w.source.scope for w in writers if w.track == Track.OTHERS), None
                    ),
                    "writer_backlog_max_s": {
                        str(w.track): round(w.backlog_max / SAMPLE_RATE, 3) for w in writers
                    },
                },
            )
            session.save()
            log.info("recording stopped: %s (%.1fs)", session.directory, longest)
            return session
