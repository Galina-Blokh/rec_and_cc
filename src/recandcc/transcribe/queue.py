"""Background queue so the recorder never waits for transcription."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from ..models import Segment, SessionInfo
from .engine import Transcriber

log = logging.getLogger("recandcc.transcribe")

DoneCallback = Callable[[SessionInfo, list[Segment] | Exception], None]
# A job that produces the final transcript some other way (live transcription's finish()).
Finalizer = Callable[[], list[Segment]]
_STOP = object()


class TranscriptionQueue:
    """Processes submitted sessions one at a time, in order, on a single dispatcher thread.

    Errors are logged and passed to `on_done` (as the exception); the thread never dies.
    A session submitted with `finalize` (live transcription already did the work) runs that
    instead of a full transcription; if it fails, the session is transcribed from its WAVs.
    """

    def __init__(self, transcriber: Transcriber, on_done: DoneCallback | None = None) -> None:
        self.transcriber = transcriber
        self.on_done = on_done
        self._q: queue.Queue = queue.Queue()
        self._busy = 0
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name="transcription-queue", daemon=True)
        self._thread.start()

    def submit(self, session: SessionInfo, finalize: Finalizer | None = None) -> None:
        if self._closed:
            raise RuntimeError("TranscriptionQueue is shut down")
        with self._lock:
            self._busy += 1
        self._q.put((session, finalize))
        log.info("queued session %s (%d pending)", session.id, self.pending())

    def pending(self) -> int:
        """Sessions submitted and not finished yet (including the one in progress)."""
        with self._lock:
            return self._busy

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting work. wait=True finishes queued sessions first; wait=False drops
        sessions that haven't started (the current one still completes in the background)."""
        self._closed = True
        if not wait:
            while True:
                try:
                    item = self._q.get_nowait()
                except queue.Empty:
                    break
                if item is not _STOP:
                    with self._lock:
                        self._busy -= 1
        self._q.put(_STOP)
        if wait:
            self._thread.join()

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            session, finalize = item
            result: list[Segment] | Exception
            try:
                result = self._process(session, finalize)
            except Exception as e:  # noqa: BLE001 - must never kill the thread
                log.exception("transcription of session %s failed", session.id)
                result = e
            finally:
                with self._lock:
                    self._busy -= 1
            if self.on_done:
                try:
                    self.on_done(session, result)
                except Exception:  # noqa: BLE001
                    log.exception("on_done callback failed")

    def _process(self, session: SessionInfo, finalize: Finalizer | None) -> list[Segment]:
        if finalize is not None:
            try:
                return finalize()
            except Exception:  # noqa: BLE001
                log.exception(
                    "live transcription of session %s failed; transcribing it from the audio",
                    session.id,
                )
        return self.transcriber.transcribe_session(session)
