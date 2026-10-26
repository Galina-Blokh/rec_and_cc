"""macOS system-audio capture via the rcc-audiotap helper (Core Audio process tap)."""

from __future__ import annotations

import logging
import os
import subprocess
import threading

import numpy as np

from ..native import find_helper
from .base import AudioSource, FrameCallback

log = logging.getLogger(__name__)

# ~100 ms of float32 mono 16 kHz audio.
BLOCK_SAMPLES = 1600
BLOCK_BYTES = BLOCK_SAMPLES * 4


class MacSystemAudioSource(AudioSource):
    """Everything the Mac plays (except this Python process), as float32 mono 16 kHz."""

    name = "system"

    def __init__(self, exclude_pids: list[int] | None = None) -> None:
        self._exclude_pids = list(exclude_pids or [])
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, on_frames: FrameCallback) -> None:
        with self._lock:
            if self._proc is not None:
                raise RuntimeError("MacSystemAudioSource already started")
            cmd = [str(find_helper()), "capture", "--exclude-pid", str(os.getpid())]
            for pid in self._exclude_pids:
                cmd += ["--exclude-pid", str(pid)]
            log.debug("starting %s", cmd)
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv: our own bundled helper, no shell
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._reader = threading.Thread(
                target=self._read_audio,
                args=(self._proc, on_frames),
                name="mac-system-audio",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=_pump_stderr, args=(self._proc,), name="mac-system-audio-stderr", daemon=True
            )
            self._reader.start()
            self._stderr_thread.start()

    def stop(self) -> None:
        with self._lock:
            proc, reader, err = self._proc, self._reader, self._stderr_thread
            self._proc = self._reader = self._stderr_thread = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()  # SIGTERM: helper tears down the tap/aggregate device and exits
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.warning("rcc-audiotap did not exit after SIGTERM, killing it")
                proc.kill()
                proc.wait(timeout=3)
        for thread in (reader, err):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3)
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()

    @staticmethod
    def _read_audio(proc: subprocess.Popen[bytes], on_frames: FrameCallback) -> None:
        assert proc.stdout is not None  # noqa: S101 - type narrowing: the pipe was requested in Popen
        pending = b""
        while True:
            try:
                chunk = proc.stdout.read(BLOCK_BYTES - len(pending))
            except (OSError, ValueError):
                break
            if not chunk:
                break
            pending += chunk
            if len(pending) < BLOCK_BYTES:
                continue
            frames = np.frombuffer(pending, dtype="<f4").astype(np.float32, copy=False)
            pending = b""
            try:
                on_frames(frames)
            except Exception:
                log.exception("on_frames callback failed")
        usable = len(pending) - len(pending) % 4
        if usable:
            on_frames(np.frombuffer(pending[:usable], dtype="<f4").astype(np.float32, copy=False))
        code = proc.poll()
        if code not in (None, 0, -15):
            log.error("rcc-audiotap capture exited with code %s", code)


def _pump_stderr(proc: subprocess.Popen[bytes]) -> None:
    """Forward helper diagnostics to logging."""
    assert proc.stderr is not None  # noqa: S101 - type narrowing: the pipe was requested in Popen
    for raw in iter(proc.stderr.readline, b""):
        line = raw.decode(errors="replace").rstrip()
        if line:
            log.info("%s", line)
