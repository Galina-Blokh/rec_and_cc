"""Session recorder: capture sources -> queues -> WAV writer threads.

Thread layout per track::

    AudioSource worker thread --on_frames()--> queue.Queue --> writer thread (soundfile)

``on_frames`` only measures the level (cheap numpy RMS) and enqueues a copy, so capture
never blocks on disk I/O. Frame listeners (``add_listener``, used for live transcription) get
the same copy on the capture thread and must only enqueue it. ``stop()`` stops the sources,
sends a sentinel to each writer, joins them, and persists session.json.
"""

from __future__ import annotations

import logging
import queue
import re
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from .capture.base import AudioSource
from .config import Config
from .models import SAMPLE_RATE, SessionInfo, Track, TriggerKind

log = logging.getLogger(__name__)

PRIVATE_DIR_MODE = 0o700  # owner-only (POSIX; ignored on Windows, where ACLs apply)


def make_private_dir(path: Path) -> None:
    """Create ``path`` (and missing parents) with owner-only permissions, if it doesn't exist.

    ``mkdir(mode=...)`` is filtered by the umask and only applies to the last component, so
    each folder this creates is chmod-ed explicitly. Existing folders are left untouched.
    """
    if path.is_dir():
        return
    if path.parent == path:  # a missing drive / UNC share root: nothing left to create
        raise FileNotFoundError(f"{path} does not exist")
    if not path.parent.is_dir():
        make_private_dir(path.parent)
    try:
        path.mkdir(mode=PRIVATE_DIR_MODE)
    except FileExistsError:
        if path.is_dir():
            return  # created concurrently
        raise  # a file (or dangling symlink) is in the way
    path.chmod(PRIVATE_DIR_MODE)


# Sessions with less audio than this (longest track) are deleted instead of transcribed.
MIN_SESSION_SECONDS = 3.0

# Peak below this (~ -100 dBFS) counts as "digital silence" (no permission / dead device).
DIGITAL_SILENCE = 1e-5

TRACK_FILES = {Track.OTHERS: "others.wav", Track.ME: "me.wav"}

SourceFactory = Callable[[], AudioSource]
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
    return kind in (TriggerKind.CALL, TriggerKind.MANUAL, TriggerKind.ROOM)


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
        system_factory: SourceFactory | None = None,
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
    def _make_dir(self, now: datetime, app: str | None, kind: TriggerKind) -> tuple[str, Path]:
        day = self.cfg.output_dir / now.strftime("%Y-%m-%d")
        slug = slugify(app, default=str(kind))
        base = f"{now.strftime('%H%M%S')}_{slug}"
        d = day / base
        n = 2
        while d.exists():
            d = day / f"{base}-{n}"
            n += 1
        # Recordings and transcripts are private: owner-only folders (0700) at every level the
        # recorder creates. An output folder that already exists keeps its permissions.
        for p in (self.cfg.output_dir, day):
            make_private_dir(p)
        d.mkdir(mode=PRIVATE_DIR_MODE)
        return f"{now.strftime('%Y-%m-%d')}_{d.name}", d

    def start(
        self,
        kind: TriggerKind,
        app: str | None = None,
        record_mic: bool | None = None,
        mic_source: SourceFactory | None = None,
    ) -> SessionInfo:
        """Start a session. ``record_mic`` overrides the config policy when not None.
        ``mic_source`` replaces the mic factory (room listening: a source that replays its
        pre-roll first); it opens first and the other tracks start padded to its length."""
        with self._op_lock:
            if self._session is not None:
                raise RuntimeError("already recording")
            now = datetime.now().astimezone()
            sid, directory = self._make_dir(now, app, kind)
            session = SessionInfo(
                id=sid, directory=directory, started_at=now, trigger_kind=kind, trigger_app=app
            )
            want_mic = (
                should_record_mic(self.cfg.detection.record_mic, kind)
                if record_mic is None
                else record_mic
            )
            plan = [(Track.OTHERS, self.system_factory)]
            if want_mic:
                plan.append((Track.ME, mic_source or self.mic_factory))
            if mic_source is not None:
                plan.reverse()

            writers: list[_TrackWriter] = []
            for track, factory in plan:
                pad = max((w.frames_received for w in writers), default=0) if mic_source else 0
                w = self._open_track(session, track, factory, pad_frames=pad)
                if w is not None:
                    writers.append(w)

            if not writers:
                shutil.rmtree(directory, ignore_errors=True)
                raise RuntimeError("no audio source could be started")
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

    def upgrade_to_call(self, app: str | None) -> bool:
        """Turn a running media or room session into a call session (a call started while e.g.
        a video was playing): adds the mic track if the policy wants it for calls. The mic
        track starts with silence up to "now" so it stays aligned with the other tracks."""
        with self._op_lock:
            session = self._session
            if session is None or session.trigger_kind not in (TriggerKind.MEDIA, TriggerKind.ROOM):
                return False
            session.trigger_kind = TriggerKind.CALL
            session.trigger_app = app or session.trigger_app
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
                    "writer_backlog_max_s": {
                        str(w.track): round(w.backlog_max / SAMPLE_RATE, 3) for w in writers
                    },
                },
            )
            session.save()
            log.info("recording stopped: %s (%.1fs)", session.directory, longest)
            return session
