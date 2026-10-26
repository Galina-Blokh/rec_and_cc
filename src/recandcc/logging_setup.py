"""Logging: rotating file in LOG_DIR plus stderr, and crash reporting.

``install_crash_handlers`` routes uncaught exceptions (main thread: ``sys.excepthook``, other
threads: ``threading.excepthook``) into the log with their full traceback and enables
``faulthandler`` (segfaults, aborts in native code) into ``<log dir>/faults.log``. A thread
crash is also reported to the user through ``set_crash_notifier`` (the UI Controller posts it
as a ``notice`` event, so the toast is shown by the main thread), at most once per thread name
per ``NOTICE_INTERVAL_S``.
"""

from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import LOG_DIR

LOG_FILE = LOG_DIR / "recandcc.log"
FAULTS_FILE = LOG_DIR / "faults.log"
_FMT = "%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"
NOTICE_INTERVAL_S = 60.0

log = logging.getLogger("recandcc.crash")

Notifier = Callable[[str, str], None]  # (title, message)
_notifier: Notifier | None = None
_last_notice: dict[str, float] = {}
_notice_lock = threading.Lock()
_faults_file = None  # kept open for faulthandler
_installed = False


def setup_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    root = logging.getLogger()
    if getattr(root, "_recandcc_configured", False):
        return
    root.setLevel(level)
    handlers: list[logging.Handler] = []
    if to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
        )
    if sys.stderr is not None:  # pythonw.exe has no stderr
        handlers.append(logging.StreamHandler(sys.stderr))
    for h in handlers:
        h.setFormatter(logging.Formatter(_FMT))
        root.addHandler(h)
    for noisy in ("faster_whisper", "httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    root._recandcc_configured = True  # type: ignore[attr-defined]
    install_crash_handlers(FAULTS_FILE)


def set_crash_notifier(notifier: Notifier | None) -> None:
    """Who tells the user about a crashed thread (any thread may call it; must not block)."""
    global _notifier
    _notifier = notifier


def install_crash_handlers(faults_file: Path | None = FAULTS_FILE) -> None:
    """Log uncaught exceptions of every thread; native crashes go to `faults_file`."""
    global _installed, _faults_file
    if not _installed:
        sys.excepthook = _excepthook
        threading.excepthook = _thread_excepthook
        _installed = True
    if faults_file is not None and _faults_file is None:
        try:
            Path(faults_file).parent.mkdir(parents=True, exist_ok=True)
            _faults_file = open(faults_file, "a", encoding="utf-8")  # noqa: SIM115 - kept open
            faulthandler.enable(file=_faults_file, all_threads=True)
        except Exception:  # noqa: BLE001 - never fail startup over crash reporting
            log.exception("could not enable faulthandler (%s)", faults_file)


def _excepthook(exc_type, exc, tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    log.critical("uncaught exception", exc_info=(exc_type, exc, tb))


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    if issubclass(args.exc_type, SystemExit):  # like the default hook
        return
    name = args.thread.name if args.thread is not None else "unknown thread"
    log.error(
        "uncaught exception in thread %s",
        name,
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )
    notify_thread_crash(name, args.exc_type.__name__)


def notify_thread_crash(name: str, error: str, now: float | None = None) -> bool:
    """Tell the user (rate-limited per thread name). True if a notice was posted."""
    notifier = _notifier
    if notifier is None:
        return False
    now = time.monotonic() if now is None else now
    with _notice_lock:
        last = _last_notice.get(name)
        if last is not None and now - last < NOTICE_INTERVAL_S:
            return False
        _last_notice[name] = now
    try:
        notifier("Internal error", f"{name} crashed ({error}). Details: menu > Reveal logs")
    except Exception:  # noqa: BLE001 - the hook itself must never raise
        log.exception("crash notice failed")
        return False
    return True
