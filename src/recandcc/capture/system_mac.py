"""macOS system-audio capture via the rcc-audiotap helper (Core Audio process tap)."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

from ..models import AppTarget
from ..native import find_helper
from .base import AudioSource, FrameCallback

log = logging.getLogger(__name__)

# ~100 ms of float32 mono 16 kHz audio.
BLOCK_SAMPLES = 1600
BLOCK_BYTES = BLOCK_SAMPLES * 4


def _target_args(target: AppTarget) -> list[str]:
    args: list[str] = []
    for pid in target.pids:
        args += ["--pid", str(int(pid))]
    for text in target.match:
        if text := " ".join(text.split()):  # one line per stdin command
            args += ["--match", text]
    return args


_SUPPORTS: dict[tuple[Path, float], bool] = {}


def _supports_targets(helper: Path, mtime: float) -> bool:
    """Whether this helper build knows ``capture --pid/--match`` (asked once per build).

    Only a real answer is remembered: a failed or timed-out probe (e.g. under load) returns
    False for this session but is asked again next time, so one hiccup can't pin a helper
    build to the global tap for the rest of the run."""
    if (key := (helper, mtime)) in _SUPPORTS:
        return _SUPPORTS[key]
    try:
        out = subprocess.run(  # noqa: S603 - fixed argv: our own bundled helper, no shell
            [str(helper), "help"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return False
    _SUPPORTS[key] = ok = "--match" in out.stdout
    return ok


class MacSystemAudioSource(AudioSource):
    """Everything the Mac plays (except this Python process), or with ``target`` only that
    app's processes (per-app tap; the helper falls back to everything if it can't), as
    float32 mono 16 kHz."""

    name = "system"

    def __init__(
        self, exclude_pids: list[int] | None = None, target: AppTarget | None = None
    ) -> None:
        self._exclude_pids = list(exclude_pids or [])
        self._target_args = _target_args(target) if target is not None else []
        # "app" until the helper reports otherwise ("scope: all" on stderr = it fell back).
        self.scope = "app" if self._target_args else "all"
        self._proc: subprocess.Popen[bytes] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, on_frames: FrameCallback) -> None:
        with self._lock:
            if self._proc is not None:
                raise RuntimeError("MacSystemAudioSource already started")
            helper = find_helper()
            if self._target_args and not _supports_targets(helper, helper.stat().st_mtime):
                log.warning(
                    "%s predates per-app capture (rebuild it with native/macos/build.sh); "
                    "recording all system audio",
                    helper,
                )
                self._target_args, self.scope = [], "all"
            cmd = [str(helper), "capture", "--exclude-pid", str(os.getpid())]
            for pid in self._exclude_pids:
                cmd += ["--exclude-pid", str(pid)]
            cmd += self._target_args
            log.debug("starting %s", cmd)
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv: our own bundled helper, no shell
                cmd,
                # Per-app tap: stdin carries "pid N" / "match TEXT" lines (add_target).
                stdin=subprocess.PIPE if self._target_args else subprocess.DEVNULL,
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
                target=self._pump_stderr,
                args=(self._proc,),
                name="mac-system-audio-stderr",
                daemon=True,
            )
            self._reader.start()
            self._stderr_thread.start()

    def add_target(self, target: AppTarget) -> bool:
        """Extend a running per-app tap to ``target`` too (a call started during media)."""
        args = _target_args(target)
        with self._lock:
            proc = self._proc
            if proc is None or proc.stdin is None or self.scope != "app":
                return False
            lines = "".join(f"{args[i][2:]} {args[i + 1]}\n" for i in range(0, len(args), 2))
            try:
                proc.stdin.write(lines.encode())
                proc.stdin.flush()
            except (OSError, ValueError):
                return False
        return True

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
        for pipe in (proc.stdin, proc.stdout):
            if pipe:
                pipe.close()
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

    def _pump_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        """Forward helper diagnostics to logging; notice a per-app -> global fallback."""
        assert proc.stderr is not None  # noqa: S101 - type narrowing: the pipe was requested in Popen
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if line.endswith("scope: all") and self.scope == "app":
                self.scope = "all"
                log.warning("per-app system audio unavailable; recording all system audio")
            elif line:
                log.info("%s", line)
