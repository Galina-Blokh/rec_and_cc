"""A few seconds of one speaker's voice, cut from the session's own audio (US-26).

Used by the macOS "Name speakers…" dialog. The clip is cut with the stdlib ``wave`` module
into a private temp folder, played by ``afplay`` on a worker thread, and deleted after
playback (the folder on exit). Nothing about the content is logged.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import SessionInfo, Track

log = logging.getLogger(__name__)

MIN_S = 2.0  # the first segment at least this long is used, else the longest one
MAX_S = 6.0  # a clip never runs past this, nor past its segment
PLAY_TIMEOUT_S = MAX_S + 10.0
POLL_S = 0.25  # how often a playing sample re-checks that nothing records


@dataclass(frozen=True)
class Clip:
    wav: Path
    start: float
    end: float


def pick_span(spans: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    """(start, end) of a sample: the first span of at least ``MIN_S`` (else the longest),
    clamped to ``MAX_S`` and to that span. None if there is no usable span."""
    spans = [(float(a), float(b)) for a, b in spans if b > a >= 0]
    if not spans:
        return None
    start, end = next((s for s in spans if s[1] - s[0] >= MIN_S), None) or max(
        spans, key=lambda s: s[1] - s[0]
    )
    return start, min(end, start + MAX_S)


def _track_wav(session: SessionInfo, track: Track) -> Path | None:
    """The track's WAV if it is still a plain file inside the session folder."""
    name = str(session.tracks.get(track) or "")
    if not name or Path(name).name != name or "\\" in name or not name.endswith(".wav"):
        return None
    path = session.directory / name
    return path if path.is_file() and not path.is_symlink() else None


def _read(wav: Path, start: float, end: float) -> tuple[wave._wave_params, bytes]:
    with wave.open(str(wav), "rb") as w:
        params = w.getparams()
        first = int(start * params.framerate)
        if first >= params.nframes:
            raise ValueError("clip past the end of the audio")
        w.setpos(first)
        return params, w.readframes(int((end - start) * params.framerate))


def _loudness(wav: Path, start: float, end: float) -> float:
    params, data = _read(wav, start, end)
    if params.sampwidth != 2 or not data:
        return 0.0
    return float(np.sqrt(np.mean(np.frombuffer(data, "<i2").astype(np.float64) ** 2)))


def find_clip(directory: Path, label: str) -> Clip | None:
    """Where `label`'s sample is in the session in `directory`, or None if its audio is
    gone (retention purged it, the file is missing or unreadable). Never raises."""
    try:
        from ..transcribe.output import JSON_NAME

        directory = Path(directory)
        data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
        mine = [s for s in data.get("segments", []) if s["speaker"] == label]
        # two voices at once make a poor sample: overlaps only if nothing else is left
        clean = [s for s in mine if not s.get("overlap")] or mine
        span = pick_span([(s["start"], s["end"]) for s in clean])
        if span is None:
            return None
        from ..transcribe.engine import diarized_tracks  # lazy: on the dialog worker

        session = SessionInfo.load(directory)
        tracks = diarized_tracks(session)
        # Others whenever it was diarized: a room session a call upgraded diarizes both
        # tracks, but nothing records which track a label came from, so a label is never
        # played from the user's own mic unless the mic is the only diarized track.
        track = Track.OTHERS if Track.OTHERS in tracks else Track.ME
        wav = _track_wav(session, track)
        if wav is None:
            return None
        if len(tracks) > 1 and _loudness(wav, *span) == 0.0:
            return None  # the call's padding before the upgrade: a room speaker, not on Others
        _read(wav, *span)  # readable and long enough
        return Clip(wav, *span)
    except Exception as e:  # noqa: BLE001 - no sample is the fallback, never a crash
        log.debug("no speaker sample: %s", type(e).__name__)
        return None


def cut_clip(clip: Clip, out: Path) -> Path:
    """Write the clip's frames as a WAV with the source's format."""
    params, data = _read(clip.wav, clip.start, clip.end)
    with wave.open(str(out), "wb") as w:
        w.setparams(params)
        w.writeframes(data)
    return out


AFPLAY = "/usr/bin/afplay"  # an absolute path: never a PATH lookup
PREFIX = "rcc-sample-"


def sweep_stale(root: Path | None = None) -> None:
    """Remove ``rcc-sample-*`` folders the user owns, left in the temp dir by a crash."""
    try:
        uid = os.getuid() if hasattr(os, "getuid") else None
        for d in Path(root or tempfile.gettempdir()).glob(PREFIX + "*"):
            if d.is_dir() and not d.is_symlink() and (uid is None or d.stat().st_uid == uid):
                shutil.rmtree(d, ignore_errors=True)
    except OSError as e:
        log.debug("could not sweep old speaker samples: %s", type(e).__name__)


class SamplePlayer:
    """Plays clips one at a time on a worker thread; ``play`` returns at once.

    ``popen`` is injectable for tests; ``allowed()`` is checked before playing (False while
    a session records: system capture would record the sample). Starting a sample stops
    the one playing; each clip is deleted after playback; ``close`` kills the player and
    removes the private temp folder (also at exit). Old folders are swept at start."""

    def __init__(
        self,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        command: Sequence[str] = (AFPLAY,),
        allowed: Callable[[], bool] = lambda: True,
        root: Path | None = None,
    ) -> None:
        self._popen = popen
        self._command = list(command)
        self._allowed = allowed
        self._root = root
        self._lock = threading.Lock()  # covers stop-old, cut and register-new
        self._dir: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._count = 0
        self._closed = False
        sweep_stale(root)
        atexit.register(self.close)

    def play(self, clip: Clip) -> threading.Thread | None:
        """Start playing `clip` on a worker thread; None if not allowed now."""
        if self._closed or not self._allowed():
            log.debug("speaker sample refused: busy or closed")
            return None
        t = threading.Thread(target=self._run, args=(clip,), daemon=True, name="speaker-sample")
        t.start()
        return t

    def _run(self, clip: Clip) -> None:
        path: Path | None = None
        try:
            with self._lock:
                self._kill()  # one sample at a time: the newest wins
                if self._closed or not self._allowed():
                    return
                if self._dir is None:  # mkdtemp: owner-only (0700)
                    self._dir = Path(tempfile.mkdtemp(prefix=PREFIX, dir=self._root))
                self._count += 1
                path = cut_clip(clip, self._dir / f"sample-{self._count}.wav")
                self._kill()  # still under the lock: no other afplay can overlap this one
                proc = self._proc = self._popen(
                    [*self._command, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._watch(proc)
            log.debug("played a speaker sample of %.1f s", clip.end - clip.start)
        except Exception as e:  # noqa: BLE001 - a missing sample is never fatal
            log.debug(
                "speaker sample of %.1f s failed: %s", clip.end - clip.start, type(e).__name__
            )
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _watch(self, proc: subprocess.Popen) -> None:
        """Wait for the player; kill it once a session starts recording (it would record
        the sample) or after ``PLAY_TIMEOUT_S``."""
        deadline = time.monotonic() + PLAY_TIMEOUT_S
        while True:
            try:
                proc.wait(timeout=POLL_S)
                return
            except subprocess.TimeoutExpired:
                if not self._allowed() or time.monotonic() > deadline:
                    with contextlib.suppress(OSError):
                        proc.kill()
                    proc.wait()
                    return

    def stop(self) -> None:
        """Kill the sample playing now, if any; later plays still work. Never blocks (the
        menu calls it on the main thread when a session starts)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(OSError):
                proc.kill()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._kill()
            folder, self._dir = self._dir, None
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
