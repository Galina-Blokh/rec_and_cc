"""A few seconds of one speaker's voice, cut from the session's own audio (US-26).

Used by the macOS "Name speakers…" dialog. The clip is cut with the stdlib ``wave`` module
into a private temp folder, played by ``afplay`` on a worker thread, and deleted after
playback (the folder on exit). Nothing about the content is logged.
"""

from __future__ import annotations

import atexit
import json
import logging
import shutil
import subprocess
import tempfile
import threading
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..models import SessionInfo, Track, TriggerKind

log = logging.getLogger(__name__)

MIN_S = 2.0  # the first segment at least this long is used, else the longest one
MAX_S = 6.0  # a clip never runs past this, nor past its segment
PLAY_TIMEOUT_S = MAX_S + 10.0


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
        span = pick_span(
            [(s["start"], s["end"]) for s in data.get("segments", []) if s["speaker"] == label]
        )
        if span is None:
            return None
        session = SessionInfo.load(directory)
        # the diarized tracks: the mic in a room session, both if a call upgraded it
        if session.trigger_kind == TriggerKind.ROOM:
            order = (Track.ME,)
        elif session.started_as_room:
            order = (Track.ME, Track.OTHERS)
        else:
            order = (Track.OTHERS,)
        wavs = [p for t in order if (p := _track_wav(session, t))]
        if not wavs:
            return None
        # both tracks: the speaker is on the one that is louder during the clip
        wav = max(wavs, key=lambda p: _loudness(p, *span)) if len(wavs) > 1 else wavs[0]
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


class SamplePlayer:
    """Plays clips one at a time on a worker thread; ``play`` returns at once.

    ``popen`` is injectable for tests. Each clip is deleted after playback; ``close``
    stops playback and removes the private temp folder (also run at exit)."""

    def __init__(
        self,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        command: Sequence[str] = ("afplay",),
    ) -> None:
        self._popen = popen
        self._command = list(command)
        self._lock = threading.Lock()
        self._dir: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._count = 0
        atexit.register(self.close)

    def play(self, clip: Clip) -> threading.Thread:
        t = threading.Thread(target=self._run, args=(clip,), daemon=True, name="speaker-sample")
        t.start()
        return t

    def _run(self, clip: Clip) -> None:
        path: Path | None = None
        try:
            with self._lock:
                if self._dir is None:  # mkdtemp: owner-only (0700)
                    self._dir = Path(tempfile.mkdtemp(prefix="rcc-sample-"))
                self._count += 1
                path = self._dir / f"sample-{self._count}.wav"
                self._stop()  # one sample at a time
            cut_clip(clip, path)
            with self._lock:
                proc = self._proc = self._popen(
                    [*self._command, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            try:
                proc.wait(timeout=PLAY_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.debug("played a speaker sample of %.1f s", clip.end - clip.start)
        except Exception as e:  # noqa: BLE001 - a missing sample is never fatal
            log.debug(
                "speaker sample of %.1f s failed: %s", clip.end - clip.start, type(e).__name__
            )
        finally:
            if path is not None:
                path.unlink(missing_ok=True)

    def _stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None

    def close(self) -> None:
        with self._lock:
            self._stop()
            folder, self._dir = self._dir, None
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
