"""Opt-in semantic search: a small multilingual sentence-embedding model run through ONNX Runtime.

``paraphrase-multilingual-MiniLM-L12-v2`` (50+ languages incl. en and ru, Apache-2.0, 384-d)
in its int8 ONNX export from ``Xenova/paraphrase-multilingual-MiniLM-L12-v2``. Both files are
pinned to an exact repo commit and a sha256, and downloaded once (only when
``[search] semantic = true``) into the platform cache dir; afterwards nothing here touches
the network. ONNX Runtime and ``tokenizers`` are already installed as faster-whisper
dependencies, so no new dependency is needed.

``Embedder.embed`` returns L2-normalised float32 rows (cosine = dot product), or None when the
model isn't available (offline and not yet downloaded; retried at most every ``RETRY_AFTER_S``).
It blocks: call it from a worker thread, never from the UI or an audio thread.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import platformdirs

from .config import APP_NAME

log = logging.getLogger("recandcc.semantic")

MODELS_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
REVISION = "2c4055b12046f11709e9df2c122e59ffbdc2f900"
_BASE = f"https://huggingface.co/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/{REVISION}/"
MAX_TOKENS = 128
BATCH = 32
DIM = 384
RETRY_AFTER_S = 600.0  # after a failed download, don't try again for this long
DOWNLOAD_TIMEOUT_S = 30.0
LOAD_ATTEMPTS = 3  # a fresh file may be held for a moment (antivirus on Windows)
LOAD_RETRY_PAUSE_S = 0.5
LOAD_RETRY_AFTER_S = 60.0  # after a load that kept failing softly, don't try again for this long
DOWNLOAD_LOCK_WAIT_S = 600.0  # wait this long for another process's download of the same file
DOWNLOAD_LOCK_POLL_S = 0.2
REOPEN_AFTER_PAUSE_S = 5.0  # a pause this long may outlive the server's idle timeout: reopen
_SOFT_MARKERS = ("being used by another process", "sharing violation", "access is denied")
# Windows messages are localised; the codes are not: ERROR_ACCESS_DENIED 5, SHARING_VIOLATION 32,
# LOCK_VIOLATION 33 (as "os error 32", "[WinError 32]", "error 33", "0x80070020" ...).
_SOFT_CODES = re.compile(
    r"(?:os error|winerror|error)\s*(?:code)?\s*[:=]?\s*(?:5|32|33)\b|0x8007(?:0005|0020|0021)\b",
    re.IGNORECASE,
)
_SOFT_WINERRORS = (5, 32, 33)


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


MODEL = ModelFile(
    name=f"minilm-l12-multilingual-{REVISION[:8]}-int8.onnx",
    url=_BASE + "onnx/model_quantized.onnx",
    sha256="66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc",
    size=118_308_126,
)
TOKENIZER = ModelFile(
    name=f"minilm-l12-multilingual-{REVISION[:8]}-tokenizer.json",
    url=_BASE + "tokenizer.json",
    sha256="b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441",
    size=17_082_913,
)


class Embedder(Protocol):
    """Optional extras: ``model_id`` (vectors of another model are re-embedded),
    ``ready()`` (True once ``embed`` won't block on a model load) and ``prepare(wait)``
    (download and load the model from the embed worker; ``wait`` pauses while the app is busy
    and returns False to give up). ``embed`` itself never starts a download."""

    def embed(self, texts: list[str]) -> np.ndarray | None: ...

    # Optional: ``embed_query(texts)``: like ``embed`` but returns None instead of waiting
    # when the embedder is busy (a search must not queue behind the indexer).


_download_lock = threading.Lock()
_last_failure = 0.0
_verified: set[Path] = set()  # model files whose sha256 was checked in this process


def _is_valid(path: Path, model: ModelFile) -> bool:
    """On disk with the right size and, checked once per process, the right sha256.
    A corrupt file is deleted so it downloads again."""
    if not path.is_file() or path.stat().st_size != model.size:
        return False
    if path in _verified:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != model.sha256:
        log.warning("search model %s is corrupt; removing it", model.name)
        path.unlink(missing_ok=True)
        return False
    _verified.add(path)
    return True


def _is_contention(e: OSError) -> bool:
    """True if a non-blocking lock failed because another process holds it."""
    if sys.platform == "win32":  # msvcrt.locking(LK_NBLCK): EACCES, or EDEADLOCK after retries
        return e.errno in (errno.EACCES, getattr(errno, "EDEADLOCK", errno.EACCES))
    return e.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK)


class _FileLock:
    """Exclusive advisory lock on a file, shared between processes (the app and a CLI search):
    fcntl on POSIX, msvcrt on Windows (as the single-instance lock in ``cli.py``)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f: Any = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = open(self.path, "a+")  # noqa: SIM115 - held open: it is the lock
        try:
            if sys.platform == "win32":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            f.close()
            if not _is_contention(e):
                raise  # not "held by someone else" (bad folder, permissions): fail fast
            return False
        self._f = f
        return True

    def release(self) -> None:
        f, self._f = self._f, None
        if f is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            # POSIX: closing the descriptor drops the flock
        except OSError:
            pass
        finally:
            f.close()


class _Abandoned(Exception):
    """The caller's ``wait`` said stop (a recording started and the setting was switched off)."""


def model_path(
    model: ModelFile,
    download: bool = True,
    wait: Callable[[], bool] | None = None,
    lock_wait_s: float | None = None,
) -> Path | None:
    """Local path of `model`, downloading (and verifying) it once if needed. None if it is not
    on disk and can't be downloaded now. A model on disk is used without any network access.
    ``wait`` is called before every block of a download: it blocks while the app is busy
    (a recording) and returns False to abandon the download (the ``.part`` file is deleted).
    ``lock_wait_s`` bounds the wait for another process's download (``DOWNLOAD_LOCK_WAIT_S``)."""
    global _last_failure
    path = MODELS_DIR / model.name
    if _is_valid(path, model):
        return path
    if not download:
        return None
    with _download_lock:
        if _is_valid(path, model):
            return path
        if _last_failure and time.monotonic() - _last_failure < RETRY_AFTER_S:
            return None
        lock = _FileLock(MODELS_DIR / ".download.lock")
        try:
            wait_s = DOWNLOAD_LOCK_WAIT_S if lock_wait_s is None else lock_wait_s
            deadline = time.monotonic() + wait_s
            while not lock.try_acquire():
                # another process is downloading: use its result when it is done
                if _is_valid(path, model):
                    return path
                if time.monotonic() >= deadline:
                    return None  # not our failure: no backoff
                time.sleep(DOWNLOAD_LOCK_POLL_S)
            if _is_valid(path, model):  # the other process finished while we waited
                return path
            _download(model, path, wait)
        except _Abandoned:
            return None  # not a failure: no backoff
        except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
            _last_failure = time.monotonic()
            log.warning("semantic search unavailable: can't download %s (%s)", model.name, e)
            return None
        finally:
            lock.release()
    return path


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse a redirect to anything but https (the checksum would catch tampering, but the
    request itself must not leave over plain http)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        if not newurl.lower().startswith("https://"):
            raise urllib.error.URLError(f"refusing redirect to non-https URL {newurl!r}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_HttpsOnlyRedirect)


def _open(url: str, offset: int) -> Any:
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"}) if offset else url  # noqa: S310
    return _opener().open(req, timeout=DOWNLOAD_TIMEOUT_S)  # https only (checked by the caller)


def _resumes_at(r: Any, offset: int) -> bool:
    """True if the server answered the Range request with 206 starting at `offset`."""
    if getattr(r, "status", None) != 206:
        return False
    headers = getattr(r, "headers", None)
    return headers is not None and str(headers.get("Content-Range", "")).startswith(
        f"bytes {offset}-"
    )


def _download(model: ModelFile, path: Path, wait: Callable[[], bool] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part.{os.getpid()}")  # per process, under the file lock
    log.info("downloading search model %s (%.0f MB, once)", model.name, model.size / 1e6)
    if not model.url.startswith("https://"):
        raise ValueError(f"refusing non-https model URL {model.url!r}")
    digest = hashlib.sha256()
    written = 0
    try:
        r = _open(model.url, 0)
        try:
            with tmp.open("wb") as f:
                while block := r.read(1 << 20):
                    paused = False
                    if wait is not None:
                        t0 = time.monotonic()
                        if not wait():
                            raise _Abandoned
                        paused = time.monotonic() - t0 >= REOPEN_AFTER_PAUSE_S
                    written += len(block)
                    if written > model.size:  # a bad or hostile server must not fill the disk
                        raise ValueError(f"download larger than the expected {model.size} bytes")
                    digest.update(block)
                    f.write(block)
                    if paused and written < model.size:
                        # The connection sat idle and the server may have dropped it: resume
                        # on a fresh one from the bytes on disk (the final sha256 still decides).
                        # Not when every byte is here: ``Range: bytes=<size>-`` is a 416.
                        r.close()
                        r = _open(model.url, written)
                        if not _resumes_at(r, written):  # no range support: start clean
                            r.close()
                            f.seek(0)
                            f.truncate()
                            digest, written = hashlib.sha256(), 0
                            r = _open(model.url, 0)
        finally:
            r.close()
        if written != model.size:
            raise ValueError(f"size mismatch ({written} bytes)")
        if digest.hexdigest() != model.sha256:
            raise ValueError(f"sha256 mismatch ({digest.hexdigest()})")
        tmp.replace(path)
        _verified.add(path)  # just checked while downloading
    finally:
        tmp.unlink(missing_ok=True)


def _is_soft(e: Exception) -> bool:
    """A failure that says nothing about the files: locked, permissions, sharing."""
    if isinstance(e, OSError) or getattr(e, "winerror", None) in _SOFT_WINERRORS:
        return True
    msg = str(e)
    return _SOFT_CODES.search(msg) is not None or any(m in msg.lower() for m in _SOFT_MARKERS)


class OnnxEmbedder:
    """The sentence model on the CPU. Loaded on first use; thread-safe.

    Locks, in the only order they are ever nested: ``_load_lock`` (one load or download at a
    time; held across the slow work, taken by ``prepare`` and, non-blocking, by ``embed``)
    -> ``_compute`` (the tokenizer and the ONNX session) -> ``_lock`` (state: ``_loaded``,
    ``_gen``, ``_broken``; held for a few assignments, never across a call). ``close`` takes
    only ``_lock``, so it never waits for a load, a download or an embed: it bumps ``_gen``,
    and a load that started before it drops its result instead of publishing it."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._compute = threading.Lock()
        self._load_lock = threading.Lock()
        self._loaded: tuple[Any, Any] | None = None
        self._gen = 0  # bumped by close(): cancels loads and downloads in flight
        self._reload_ok = True  # False after close(): only prepare() loads again
        self._broken = False  # verified files the runtime can't load: don't retry per search
        self._retry_at = 0.0  # soft load failure (locked file ...): no new attempt before this

    model_id = MODEL.name
    downloads = True  # False: use model files on disk only (a CLI search while the app runs)
    lock_wait_s: float | None = None  # wait for another process's download (None: default)

    def ready(self) -> bool:
        """True once the model is loaded: ``embed`` then doesn't wait for anything."""
        return self._loaded is not None

    def close(self) -> None:
        """Drop the session and tokenizer (~120 MB), cancel a load or download in flight and
        forget past failures; the next ``prepare`` / ``embed`` starts afresh. Never blocks."""
        global _last_failure
        _last_failure = 0.0  # the download backoff too: switching off and on retries at once
        with self._lock:
            self._gen += 1
            self._reload_ok = False  # a straggling embed() must not reload ~120 MB
            self._loaded = None
            self._broken = False
            self._retry_at = 0.0

    def prepare(self, wait: Callable[[], bool] | None = None) -> bool:
        """Download (if needed), verify and load the model; True once ready. Run from the embed
        worker. ``wait`` is polled between the steps and while downloading: it blocks while the
        app is busy and returns False to give up (a switched-off setting)."""
        if self._loaded is not None:
            return True
        with self._load_lock:
            if self._loaded is not None:
                return True
            if self._blocked():
                return False
            with self._lock:
                gen = self._gen
                self._reload_ok = True  # an explicit request (the embed worker) loads again
            paths: list[Path] = []
            for m in (MODEL, TOKENIZER):
                if self._cancelled(gen) or (wait is not None and not wait()):
                    return False
                p = model_path(
                    m,
                    download=self.downloads,
                    wait=lambda: self._cancelled(gen) is False and _ok(wait),
                    lock_wait_s=self.lock_wait_s,
                )
                if p is None:
                    return False
                paths.append(p)
            if self._cancelled(gen) or (wait is not None and not wait()):
                return False
            return self._build(paths[0], paths[1], gen, wait)

    def _cancelled(self, gen: int) -> bool:
        return self._gen != gen

    def _blocked(self) -> bool:
        with self._lock:
            return self._broken or time.monotonic() < self._retry_at

    def _load_cached(self) -> tuple[Any, Any] | None:
        """Load from files already on disk (never downloads, never waits for another load)."""
        if not self._load_lock.acquire(blocking=False):
            return None
        try:
            if self._loaded is None and self._reload_ok and not self._blocked():
                gen = self._gen
                m, t = model_path(MODEL, download=False), model_path(TOKENIZER, download=False)
                if m is not None and t is not None:
                    self._build(m, t, gen)
            return self._loaded
        finally:
            self._load_lock.release()

    def _build(self, m: Path, t: Path, gen: int, wait: Callable[[], bool] | None = None) -> bool:
        """Open the tokenizer and session. ``wait`` (the worker's) is polled between the steps
        and the retries; False cancels without publishing."""
        global _last_failure
        import onnxruntime as ort  # lazy: only when semantic search is on
        from tokenizers import Tokenizer

        for attempt in range(LOAD_ATTEMPTS):
            try:
                if self._cancelled(gen) or not _ok(wait):
                    return False
                tok = Tokenizer.from_file(str(t))
                tok.enable_truncation(MAX_TOKENS)
                tok.enable_padding()
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2  # background work: leave cores to transcription
                if self._cancelled(gen) or not _ok(wait):
                    return False
                # CPU only: never CoreML or the GPU, which would run outside MLX_LOCK
                session = ort.InferenceSession(str(m), opts, providers=["CPUExecutionProvider"])
                loaded = (session, tok)
            except Exception as e:  # noqa: BLE001 - unreadable model files
                if _is_soft(e):
                    if attempt + 1 < LOAD_ATTEMPTS and not self._cancelled(gen):
                        time.sleep(LOAD_RETRY_PAUSE_S)
                        if not _ok(wait):
                            return False
                        continue
                    log.warning("search model can't be opened now (%s); retrying later", e)
                    with self._lock:
                        if self._gen == gen:  # not after a close(): it forgot failures
                            self._retry_at = time.monotonic() + LOAD_RETRY_AFTER_S
                    return False
                log.warning("search model failed to load (%s)", e)
                if not self._cancelled(gen):
                    _last_failure = time.monotonic()  # the download backoff applies
                for f in (m, t):
                    if f in _verified:
                        with self._lock:
                            if self._gen == gen:
                                self._broken = True  # intact files: the runtime is the problem
                    else:
                        # safety net for files put on disk outside this process (model_path
                        # verifies everything it returns): a later attempt downloads it again
                        f.unlink(missing_ok=True)
                return False
            with self._lock:
                if self._gen != gen:
                    return False  # closed meanwhile: drop it
                self._loaded = loaded
            return True
        return False

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """Embeddings, or None if the model isn't loaded and can't be loaded from disk now."""
        loaded = self._loaded or self._load_cached()
        if loaded is None:
            return None
        with self._compute:
            return self._run(loaded, texts)

    def embed_query(self, texts: list[str]) -> np.ndarray | None:
        """Like ``embed`` for a loaded model, but None at once if the indexer holds the
        compute lock (a search then shows full-text results instead of queueing behind it)."""
        loaded = self._loaded
        if loaded is None or not self._compute.acquire(blocking=False):
            return None
        try:
            return self._run(loaded, texts)
        finally:
            self._compute.release()

    @staticmethod
    def _run(loaded: tuple[Any, Any], texts: list[str]) -> np.ndarray:
        session, tok = loaded
        out = []
        for i in range(0, len(texts), BATCH):
            enc = tok.encode_batch(texts[i : i + BATCH])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            hidden = session.run(None, feed)[0]
            pooled = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-9))
        if not out:
            return np.zeros((0, DIM), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)


_default: OnnxEmbedder | None = None
_default_lock = threading.Lock()


def _ok(wait: Callable[[], bool] | None) -> bool:
    return wait is None or wait()


def default_embedder() -> OnnxEmbedder:
    global _default
    with _default_lock:
        if _default is None:
            _default = OnnxEmbedder()
        return _default


def current_default_embedder() -> OnnxEmbedder | None:
    """The process-wide embedder if it was ever created (never creates it)."""
    with _default_lock:
        return _default


def close_default_embedder() -> None:
    """Unload the process-wide model, if it was ever created (semantic search turned off)."""
    e = current_default_embedder()
    if e is not None:
        e.close()
