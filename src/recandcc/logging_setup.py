"""Logging: rotating file in LOG_DIR plus stderr."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from .config import LOG_DIR

LOG_FILE = LOG_DIR / "recandcc.log"
_FMT = "%(asctime)s %(levelname)-7s %(threadName)s %(name)s: %(message)s"


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
