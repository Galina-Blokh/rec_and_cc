"""macOS audio activity detection via `rcc-audiotap monitor` (Core Audio process objects)."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time

from ..models import AudioProcess
from ..native import find_helper
from .base import ActivityMonitor

log = logging.getLogger(__name__)

STALE_AFTER_S = 5.0
MAX_BACKOFF_S = 30.0


class MacActivityMonitor(ActivityMonitor):
    """Keeps an `rcc-audiotap monitor` helper running and caches its latest report."""

    def __init__(self, interval: float = 1.0) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._latest: list[AudioProcess] = []
        self._latest_at = 0.0  # time.monotonic() of the last line
        self._proc: subprocess.Popen[str] | None = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="mac-activity-monitor", daemon=True)
        self._thread.start()

    def snapshot(self) -> list[AudioProcess]:
        with self._lock:
            if time.monotonic() - self._latest_at > STALE_AFTER_S:
                return []
            return list(self._latest)

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    # -- helper management --------------------------------------------------------------

    def _run(self) -> None:
        backoff = 1.0
        while not self._closed.is_set():
            started = time.monotonic()
            try:
                self._run_once()
            except Exception:
                log.exception("rcc-audiotap monitor failed")
            with self._lock:
                self._latest, self._latest_at = [], 0.0
            if self._closed.is_set():
                break
            # Reset the backoff if the helper ran fine for a while.
            if time.monotonic() - started > 60:
                backoff = 1.0
            log.warning("rcc-audiotap monitor exited; restarting in %.0f s", backoff)
            self._closed.wait(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF_S)

    def _run_once(self) -> None:
        cmd = [str(find_helper()), "monitor", "--interval", str(self._interval)]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        with self._lock:
            self._proc = proc
        if self._closed.is_set():  # close() raced with the start
            proc.kill()
        threading.Thread(target=_pump_stderr, args=(proc,), daemon=True).start()
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                parsed = _parse(line)
            except (ValueError, KeyError, TypeError) as exc:
                log.debug("bad monitor line %r: %s", line, exc)
                continue
            with self._lock:
                self._latest, self._latest_at = parsed, time.monotonic()
        proc.wait()
        with self._lock:
            self._proc = None


def _parse(line: str) -> list[AudioProcess]:
    return [
        AudioProcess(
            pid=int(item["pid"]),
            bundle_id=str(item.get("bundle_id") or ""),
            name=str(item.get("name") or ""),
            input_active=bool(item.get("input")),
            output_active=bool(item.get("output")),
        )
        for item in json.loads(line)
    ]


def _pump_stderr(proc: subprocess.Popen[str]) -> None:
    assert proc.stderr is not None
    for line in proc.stderr:
        if line.strip():
            log.info("%s", line.rstrip())
