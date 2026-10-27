"""Full-text search over the transcripts: an incremental SQLite index of transcript.json.

The index lives in the platform app-data dir (never inside ``output_dir``, so cloud sync of
the transcripts folder never picks it up) and is owner-only. ``sync`` reindexes a session
whose transcript.json changed (mtime, size) and drops sessions that were deleted or discarded;
``index_session`` updates one session after its transcript is final. With FTS5 the text is
tokenised by ``unicode61 remove_diacritics 2`` (Russian and other scripts work) and hits are
ranked by BM25; an SQLite built without FTS5 falls back to a (slower) ``LIKE`` scan. ё is
indexed and searched as е. Deletes are secure (``secure_delete`` and FTS5 ``secure-delete``),
so the text of a dropped session doesn't linger in the file.

With an ``embedder`` (opt-in, ``[search] semantic``) every segment also gets a sentence
embedding in ``seg_vec``, computed outside the write lock right after the FTS update, and
``search`` fuses the BM25 and the cosine rankings by reciprocal rank. Without one nothing
is embedded, stored or fused, and results are exactly the full-text ones; vectors left from
an earlier opt-in are deleted at the next ``sync``.

Repeated queries score an in-process cache of the vector matrix (built lazily by the first
query once the embed worker is idle; at most ``CACHE_MAX_ROWS`` rows, about 1.5 KB of float32
per 384-d segment plus ~0.3-0.5 KB of per-row metadata and text). Every committed change of
``seg_vec`` writes a fresh random token into ``vec_version`` in the same transaction (an
explicit update plus triggers on ``seg_vec``, so even an older build of the app or CLI that
doesn't know the token invalidates it) and, in this process, drops the cache; a query reuses
the cache only while the token matches, so writes of another process (the CLI) are seen too.
Without a fresh cache a query streams the vectors from SQLite; an index that can't be cached
(over the cap, a foreign model's vectors) is remembered per token, so each query streams once
instead of attempting a build first.

Every call opens its own connection, so any thread may use an index; writes are serialised
by a lock (and by SQLite's file lock across processes, e.g. the app and the CLI). Transcript
text is never logged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import platformdirs

from .config import APP_NAME
from .models import SessionInfo
from .semantic import Embedder

log = logging.getLogger(__name__)

INDEX_PATH = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / "search.sqlite3"
TRANSCRIPT = "transcript.json"
SCHEMA_VERSION = 3
_YO = str.maketrans("ёЁ", "еЕ")
_MARKS = "\x01\x02"  # highlight marks, shown as [ ]
SEMANTIC_MIN_SIM = 0.45  # cosine below this is not a semantic match (0.3 matched unrelated lines)
RRF_K = 60  # reciprocal rank fusion constant
POOL = 4  # each ranking contributes up to POOL * limit candidates to the fusion
SCAN_CHUNK = 2048  # vectors scored (or cached) per fetch
# Vector cache cap. Per row: 1,536 B of 384-d float32 plus ~300-500 B of Python metadata (the
# 7-tuple, a float, the list slot and the segment text; folder/speaker strings are pooled), so
# ~190 MB at the cap for an always-resident menu-bar app. Above it queries stream (as before).
CACHE_MAX_ROWS = 100_000
CACHE_IDLE_S = 15 * 60  # an unused cache is released after this long (rebuilt by the next query)
EMBED_CHUNK = 64  # segments embedded between two checks of ``busy``
BUSY_POLL_S = 0.5
NO_VECTORS = np.zeros((0, 0), dtype=np.float32)  # a session without segments
_TOKEN = re.compile(r"\w+", re.UNICODE)
_LIKE_SQL = (
    "SELECT p.folder, s.session_id, s.started_at, p.start, p.speaker, p.language, p.text "
    "FROM seg_plain p JOIN sessions s ON s.folder = p.folder WHERE {where} "
    "ORDER BY s.started_at DESC, p.start LIMIT ?"
)


@dataclass(frozen=True)
class Hit:
    folder: Path  # the session folder
    session_id: str
    started_at: str  # ISO timestamp of the session start
    start: float  # seconds from the session start
    speaker: str
    language: str
    snippet: str  # matched text, the match marked with [ ]
    # identity of the segment text (not shown, not compared): two segments of one speaker may
    # share a start time and must not be merged when the rankings are fused
    seg: str = field(default="", compare=False, repr=False)

    @property
    def transcript(self) -> Path:
        return self.folder / "transcript.md"


@dataclass(frozen=True, eq=False)
class _VecCache:
    """A snapshot of the stored vectors of one model, consistent with ``token``."""

    token: str
    model: str
    matrix: np.ndarray  # (rows, dim) float32, L2-normalised as stored, in rowid order
    # per row: (folder, session_id, started_at, start, speaker, language, text), kept with the
    # vector so a hit is never looked up again by rowid (no reuse race)
    meta: list[tuple[str, str, str, float, str, str, str]]


def has_fts5() -> bool:
    try:
        with contextlib.closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.Error:
        return False


class TranscriptIndex:
    """``fts=None`` uses FTS5 when this SQLite has it; ``False`` forces the LIKE fallback."""

    def __init__(
        self,
        path: Path | None = None,
        fts: bool | None = None,
        embedder: Embedder | None = None,
        busy: Callable[[], bool] | None = None,
    ) -> None:
        self.embedder = embedder  # swapped by default_index when the setting changes
        self.busy = busy  # True while recording: the embedding worker then pauses
        self.busy_poll_s = BUSY_POLL_S
        self._embed_guard = threading.Lock()  # guards the two fields below (never held long)
        self._embed_thread: threading.Thread | None = None  # the one background embedding worker
        self._embed_dirty = False  # work arrived while the worker was running
        self.path = Path(path) if path is not None else INDEX_PATH
        self.fts = has_fts5() if fts is None else fts
        self._lock = threading.Lock()  # serialises writes
        self._init_lock = threading.Lock()  # first connection: files + schema
        self._ready = False
        self._fts_secure = True  # FTS5 'secure-delete' (SQLite >= 3.42) is on
        self._vec_dirty = False  # a vector change awaits its commit (only under ``_lock``)
        self._cache: _VecCache | None = None
        self._cache_used = 0.0  # ``_now()`` of the last hit or install
        self._cache_gen = 0  # bumped after each committed vector change in this process
        self._cache_lock = threading.Lock()  # guards the two above; held only for assignments
        self._cache_build = threading.Lock()  # one cache builder at a time (never waited for)
        # (token, model, dim, cap) for which a build can't succeed (over the cap, foreign
        # vectors): skipped until the token changes. Guarded by ``_cache_lock``.
        self._no_cache: tuple[str, str, int, int] | None = None

    # ------------------------------------------------------------------ storage
    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            with self._init_lock:
                if not self._ready:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with contextlib.suppress(OSError):  # owner-only (no-op for Windows ACLs)
                        os.chmod(self.path.parent, 0o700)
                        self.path.touch(mode=0o600, exist_ok=True)
                        os.chmod(self.path, 0o600)
                    with contextlib.closing(self._open()) as db:
                        self._create(db)
                    self._ready = True
        return self._open()

    def _owner_only(self) -> None:
        """Re-apply 0600 after a VACUUM rebuilt the file (it may come back with the umask)."""
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    def _open(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        # Deleted rows are overwritten, so a dropped session's text can't be recovered.
        db.execute("PRAGMA secure_delete = ON")
        return db

    def _create(self, db: sqlite3.Connection) -> None:
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            tables = ("sessions", "seg_fts", "seg_plain", "seg_vec", "sem_sessions", "vec_version")
            for table in tables:
                db.execute(f"DROP TABLE IF EXISTS {table}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.execute(
            "CREATE TABLE IF NOT EXISTS sessions (folder TEXT PRIMARY KEY, session_id TEXT, "
            "started_at TEXT, stamp TEXT, fts INTEGER)"
        )
        if self.fts:
            # text: normalised (ё -> е) and indexed; orig: the text when it differs.
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS seg_fts USING fts5(text, orig UNINDEXED, "
                "folder UNINDEXED, start UNINDEXED, speaker UNINDEXED, language UNINDEXED, "
                "tokenize = 'unicode61 remove_diacritics 2')"
            )
            try:  # also purge deleted terms from the FTS index itself
                db.execute("INSERT INTO seg_fts(seg_fts, rank) VALUES ('secure-delete', 1)")
            except sqlite3.OperationalError:
                self._fts_secure = False  # older SQLite: optimize + VACUUM after drops
        else:
            db.execute(
                "CREATE TABLE IF NOT EXISTS seg_plain (folder TEXT, start REAL, speaker TEXT, "
                "language TEXT, text TEXT, norm TEXT)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS seg_plain_folder ON seg_plain(folder)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS seg_vec (folder TEXT, start REAL, speaker TEXT, "
            "language TEXT, text TEXT, vec BLOB)"
        )
        db.execute("CREATE INDEX IF NOT EXISTS seg_vec_folder ON seg_vec(folder)")
        db.execute(
            "CREATE TABLE IF NOT EXISTS sem_sessions "
            "(folder TEXT PRIMARY KEY, stamp TEXT, model TEXT)"
        )
        # A random token, replaced by every vector change: a recreated table never repeats one.
        db.execute(
            "CREATE TABLE IF NOT EXISTS vec_version (id INTEGER PRIMARY KEY CHECK (id = 0), "
            "token TEXT)"
        )
        db.execute("INSERT OR IGNORE INTO vec_version VALUES (0, ?)", (secrets.token_hex(8),))
        # Also bump it from SQLite itself, so a writer that doesn't know the token (an older
        # app or CLI on the same file, same schema version) still invalidates every cache.
        for event in ("INSERT", "DELETE", "UPDATE"):
            db.execute(  # only the constant event names are interpolated
                f"CREATE TRIGGER IF NOT EXISTS seg_vec_{event.lower()} AFTER {event} ON seg_vec "  # noqa: S608
                "BEGIN UPDATE vec_version SET token = lower(hex(randomblob(8))) WHERE id = 0; END"
            )
        db.commit()

    @property
    def _table(self) -> str:
        return "seg_fts" if self.fts else "seg_plain"

    def _bump(self, db: sqlite3.Connection) -> None:
        """Vectors change in this (uncommitted) transaction: new token, cache dropped at commit."""
        db.execute("UPDATE vec_version SET token = ? WHERE id = 0", (secrets.token_hex(8),))
        self._vec_dirty = True

    def _commit(self, db: sqlite3.Connection) -> None:
        """Commit (under ``_lock``); after a vector change drop the cache only now, so a cache
        built from the old rows meanwhile is refused at install (generation changed)."""
        db.commit()
        if self._vec_dirty:
            self._vec_dirty = False
            self._drop_cache()

    def release_idle_cache(self, now: float | None = None) -> bool:
        """Free the vector cache if no query used it for ``CACHE_IDLE_S``: the app is always
        on, and a rarely used cache would hold up to ~190 MB for nothing. The generation bump
        makes a build in flight discard its result; the next query rebuilds lazily."""
        now = _now() if now is None else now
        with self._cache_lock:
            if self._cache is None or now - self._cache_used < CACHE_IDLE_S:
                return False
            self._cache = None
            self._cache_gen += 1
        log.debug("search vector cache released (idle)")
        return True

    def _drop_cache(self) -> None:
        with self._cache_lock:
            self._cache_gen += 1
            self._cache = None
            self._no_cache = None

    def _purge(self, db: sqlite3.Connection) -> None:
        """After deletes, when FTS5 can't secure-delete: merge the index and rewrite the file
        so no deleted text survives in old segments or free pages."""
        if self.fts and not self._fts_secure:
            db.execute("INSERT INTO seg_fts(seg_fts) VALUES ('optimize')")
            db.commit()
            db.execute("VACUUM")
            self._owner_only()

    # ------------------------------------------------------------------ updates
    def index_session(self, directory: Path) -> bool:
        """(Re)index one session folder, or drop it when it has no transcript or was
        discarded. True if it is indexed now."""
        d = Path(directory)
        f = d / TRANSCRIPT
        parsed = None
        if not (d / SessionInfo.DISCARD_MARKER).exists() and f.exists():
            parsed = (_stamp(f), _read(d))  # file I/O outside the write lock
        with self._lock, contextlib.closing(self._connect()) as db:
            removed = self._drop(db, str(d))
            if parsed is not None:
                self._write(db, d, *parsed)
            self._commit(db)
            if removed:
                self._purge(db)
        self._kick_embed()
        return parsed is not None and parsed[1] is not None

    def sync(self, output_dir: Path) -> int:
        """Bring the index in line with <output_dir>/<day>/<session>/transcript.json;
        returns the number of sessions (re)indexed."""
        found: dict[str, str] = {}
        for f in Path(output_dir).glob(f"*/*/{TRANSCRIPT}"):
            try:
                if not (f.parent / SessionInfo.DISCARD_MARKER).exists():
                    found[str(f.parent)] = _stamp(f)
            except OSError:
                continue
        with contextlib.closing(self._connect()) as db:
            known = {
                folder: (stamp, fts)
                for folder, stamp, fts in db.execute("SELECT folder, stamp, fts FROM sessions")
            }
        todo = [f for f, stamp in found.items() if known.get(f) != (stamp, int(self.fts))]
        # Read and parse outside the write lock, so a big first sync doesn't block indexing.
        parsed = {f: _read(Path(f)) for f in todo}
        with self._lock, contextlib.closing(self._connect()) as db:
            removed = 0
            for folder in known.keys() - found.keys():
                removed += self._drop(db, folder)
            for folder in todo:
                removed += self._drop(db, folder)
                self._write(db, Path(folder), found[folder], parsed[folder])
            self._commit(db)
            if removed:
                self._purge(db)
        self._kick_embed()
        changed = sum(p is not None for p in parsed.values())
        log.info("search index synced: %d sessions, %d updated", len(found), changed)
        return changed

    def _drop(self, db: sqlite3.Connection, folder: str) -> bool:
        """Remove a session; True if it had segments (so there is text to purge)."""
        db.execute("DELETE FROM sessions WHERE folder = ?", (folder,))
        vec = db.execute("DELETE FROM seg_vec WHERE folder = ?", (folder,))
        sem = db.execute("DELETE FROM sem_sessions WHERE folder = ?", (folder,))
        if vec.rowcount > 0 or sem.rowcount > 0:
            self._bump(db)
        table = self._table  # a constant, never user input
        cur = db.execute(f"DELETE FROM {table} WHERE folder = ?", (folder,))  # noqa: S608
        return cur.rowcount > 0 or vec.rowcount > 0

    def _write(self, db: sqlite3.Connection, d: Path, stamp: str, data: dict | None) -> None:
        """Insert a parsed transcript (``_read``). An unreadable one (None) is recorded with
        its stamp and no segments: no stale hits, and no re-read until it changes."""
        folder = str(d)
        data = data or {"segments": []}
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (folder, data.get("session_id", d.name), data.get("started_at", ""), stamp, self.fts),
        )
        segs = data["segments"]
        if self.fts:
            rows = [(_yo(t), t if _yo(t) != t else None, folder, *rest) for t, *rest in segs]
            db.executemany("INSERT INTO seg_fts VALUES (?, ?, ?, ?, ?, ?)", rows)
        else:
            rows = [(folder, *rest, t, _fold(t)) for t, *rest in segs]
            db.executemany("INSERT INTO seg_plain VALUES (?, ?, ?, ?, ?, ?)", rows)

    # ------------------------------------------------------------------ embeddings
    def set_embedder(self, embedder: Embedder | None) -> None:
        """Switch semantic search on or off (None). Vectors of another model are re-embedded;
        a running worker notices the swap before it stores anything. Off releases the cache."""
        self.embedder = embedder
        if embedder is None:
            self._drop_cache()

    def _wait_idle(self, embedder: Embedder) -> bool:
        """Pause while ``busy`` (a recording or a prompt); False if the embedder was swapped."""
        while self.busy is not None and self.busy():
            if self.embedder is not embedder:
                return False
            time.sleep(self.busy_poll_s)
        return self.embedder is embedder

    def _model_id(self) -> str:
        e = self.embedder
        return str(getattr(e, "model_id", type(e).__name__))

    def _kick_embed(self) -> None:
        """Start the background embedding worker unless one is running (then just tell it
        there is more to do). Never blocks: ``sync`` and ``search`` don't wait for models."""
        if self.embedder is None:
            self._forget_vectors()
            return
        with self._embed_guard:
            self._embed_dirty = True
            if self._embed_thread is not None:
                return
            # Daemon: being killed at interpreter exit loses nothing. SQLite writes are atomic
            # and a leftover model ``.part`` file is overwritten by the next download.
            t = threading.Thread(target=self._embed_worker, name="search-embed", daemon=True)
            self._embed_thread = t
            try:
                t.start()  # inside the guard: nobody can join an unstarted thread
            except RuntimeError:
                self._embed_thread = None
                log.warning("could not start the embedding worker")

    def _embed_worker(self) -> None:
        while True:
            with self._embed_guard:
                if not self._embed_dirty:
                    self._embed_thread = None
                    return
                self._embed_dirty = False
            self._embed_pending()  # never raises

    def wait_embedded(self, timeout: float = 30.0) -> bool:
        """Block until the embedding worker is idle (tests, CLI); False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._embed_guard:
                if self._embed_thread is None:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)  # polling: a thread can't be joined before it has started

    def _forget_vectors(self) -> None:
        """Semantic search is off: delete every vector (the full-text index stays)."""
        try:
            with self._lock, contextlib.closing(self._connect()) as db:
                if self.embedder is not None:
                    return  # switched on again meanwhile
                had = db.execute("SELECT 1 FROM seg_vec LIMIT 1").fetchone()
                had = had or db.execute("SELECT 1 FROM sem_sessions LIMIT 1").fetchone()
                if had:
                    db.execute("DELETE FROM seg_vec")
                    db.execute("DELETE FROM sem_sessions")
                    self._bump(db)
                    self._commit(db)  # secure_delete overwrites freed pages: no VACUUM in the lock
        except Exception as e:  # noqa: BLE001 - semantic search is optional
            log.error("dropping search vectors failed: %s", type(e).__name__)

    def _embed_pending(self) -> None:
        """Embed the sessions whose vectors are missing or stale (other transcript stamp or
        other model); never raises. Runs only on the worker thread. Embedding is outside the
        write lock; a session changed (or the embedder swapped) meanwhile is skipped."""
        try:
            embedder = self.embedder
            if embedder is None:
                return
            model = self._model_id()
            with contextlib.closing(self._connect()) as db:
                todo = db.execute(
                    "SELECT s.folder, s.stamp FROM sessions s LEFT JOIN sem_sessions v "
                    "ON v.folder = s.folder WHERE v.stamp IS NULL OR v.stamp != s.stamp "
                    "OR v.model IS NOT ?",
                    (model,),
                ).fetchall()
            ready = getattr(embedder, "ready", None)
            if (todo or (ready is not None and not ready())) and not self._wait_idle(embedder):
                return
            if ready is not None and not ready():
                # Load (download first, if needed) here, on the worker: it starts only while
                # idle and pauses or gives up when a recording starts or the setting is
                # switched off. ``embed`` on the search path never downloads.
                prepare = getattr(embedder, "prepare", None)
                if prepare is not None:
                    if not prepare(lambda: self._wait_idle(embedder)):
                        return  # unavailable now: retry at the next sync
                elif not todo:
                    embedder.embed([])
            if not todo:
                return
            for folder, stamp in todo:
                if not self._wait_idle(embedder):
                    return
                data = _read(Path(folder))
                segs = data["segments"] if data else []
                parts: list[np.ndarray] = []
                for k in range(0, len(segs), EMBED_CHUNK):
                    if not self._wait_idle(embedder):
                        return
                    part = embedder.embed([_yo(s[0]) for s in segs[k : k + EMBED_CHUNK]])
                    if part is None:
                        return  # model unavailable now: retry at the next sync
                    parts.append(part)
                vecs = np.concatenate(parts) if parts else NO_VECTORS
                with self._lock, contextlib.closing(self._connect()) as db:
                    if self.embedder is not embedder:
                        return  # switched off or swapped: don't store
                    cur = db.execute("SELECT stamp FROM sessions WHERE folder = ?", (folder,))
                    if (row := cur.fetchone()) is None or row[0] != stamp:
                        continue
                    db.execute("DELETE FROM seg_vec WHERE folder = ?", (folder,))
                    rows = [
                        (folder, start, speaker, lang, text, vecs[i].astype(np.float32).tobytes())
                        for i, (text, start, speaker, lang) in enumerate(segs)
                    ]
                    db.executemany("INSERT INTO seg_vec VALUES (?, ?, ?, ?, ?, ?)", rows)
                    db.execute(
                        "INSERT OR REPLACE INTO sem_sessions VALUES (?, ?, ?)",
                        (folder, stamp, model),
                    )
                    self._bump(db)
                    self._commit(db)
        except Exception as e:  # noqa: BLE001 - semantic search is optional
            log.error("semantic indexing failed: %s", type(e).__name__)

    def _semantic(self, query: str, limit: int) -> list[Hit]:
        """Segments closest in meaning to ``query`` (cosine), best first, from the vectors
        stored so far; [] if the model isn't loaded yet (the worker is loading it)."""
        embedder = self.embedder
        if embedder is None:
            return []
        ready = getattr(embedder, "ready", None)
        if ready is not None and not ready():
            self._kick_embed()  # never wait for a download or a model load here
            return []
        embed_query = getattr(embedder, "embed_query", None) or embedder.embed
        q = embed_query([_yo(query)])  # None: the indexer holds the model; full-text only
        if q is None or not len(q):
            return []
        model, dim, q0 = self._model_id(), q.shape[1], q[0]
        cache = self._fresh_cache(model, dim)
        if cache is not None:
            return _score_cached(cache, q0, limit)
        top_ids = np.zeros(0, dtype=np.int64)
        top_sims = np.zeros(0, dtype=np.float32)
        last = 0
        blobs: dict[int, bytes] = {}  # the scored vector of each survivor (rowid reuse check)
        while True:
            with contextlib.closing(self._connect()) as db:  # not held while scoring
                rows = db.execute(
                    "SELECT v.rowid, v.vec FROM seg_vec v JOIN sem_sessions m "
                    "ON m.folder = v.folder WHERE m.model = ? AND v.rowid > ? "
                    "ORDER BY v.rowid LIMIT ?",
                    (model, last, SCAN_CHUNK),
                ).fetchall()
            if not rows:
                break
            last = rows[-1][0]
            matrix = np.frombuffer(b"".join(r[1] for r in rows), dtype=np.float32)
            if matrix.size != len(rows) * dim:  # embedder swapped meanwhile
                return []
            sims = matrix.reshape(len(rows), -1) @ q0
            keep = sims >= SEMANTIC_MIN_SIM
            top_ids = np.concatenate(
                [top_ids, np.array([r[0] for r in rows], dtype=np.int64)[keep]]
            )
            blobs.update((r[0], r[1]) for r, k in zip(rows, keep, strict=True) if k)
            top_sims = np.concatenate([top_sims, sims[keep]])
            if len(top_ids) > limit:  # running top-k
                part = np.argpartition(-top_sims, limit - 1)[:limit]
                top_ids, top_sims = top_ids[part], top_sims[part]
                blobs = {int(i): blobs[int(i)] for i in top_ids}
            if len(rows) < SCAN_CHUNK:
                break
        order = np.lexsort((top_ids, -top_sims))  # best first; ties by insertion order
        ids = [int(top_ids[i]) for i in order[:limit]]
        if not ids:
            return []
        found: dict[int, tuple] = {}
        with contextlib.closing(self._connect()) as db:
            for k in range(0, len(ids), 500):
                chunk = ids[k : k + 500]
                marks = ",".join("?" * len(chunk))
                sql = (  # only placeholders are interpolated
                    "SELECT v.rowid, v.folder, s.session_id, s.started_at, v.start, v.speaker, "  # noqa: S608
                    "v.language, v.text, v.vec FROM seg_vec v "
                    "JOIN sessions s ON s.folder = v.folder "
                    f"WHERE v.rowid IN ({marks})"
                )
                for r in db.execute(sql, chunk):
                    if r[8] == blobs.get(r[0]):  # a reused rowid holds another segment now
                        found[r[0]] = r
        return [
            Hit(
                Path(r[1]), r[2], r[3], float(r[4]), r[5], r[6], _plain_snippet(r[7]), _seg_id(r[7])
            )
            for i in ids
            if (r := found.get(i)) is not None
        ]

    def _vec_token(self) -> str | None:
        with contextlib.closing(self._connect()) as db:
            row = db.execute("SELECT token FROM vec_version").fetchone()
        return row[0] if row else None

    def _fresh_cache(self, model: str, dim: int) -> _VecCache | None:
        """The cached vectors if still current, else a new cache; None (stream instead) while
        the embed worker runs (it would invalidate it at once), while another query builds
        one, above ``CACHE_MAX_ROWS``, or if the vectors changed during the build. A build that
        can't succeed for this token (over the cap, foreign vectors) isn't retried until the
        token changes, so such an index costs one streaming scan per query, not a build too."""
        with self._cache_lock:
            cache, gen, no_cache = self._cache, self._cache_gen, self._no_cache
        token = self._vec_token()
        if cache is not None and cache.token == token and cache.model == model:
            with self._cache_lock:
                self._cache_used = _now()
            return cache if cache.matrix.shape[1] == dim else None
        if cache is not None:  # stale (e.g. another process wrote): release it now
            with self._cache_lock:
                if self._cache is cache:
                    self._cache = None
        with self._embed_guard:
            if self._embed_thread is not None:
                return None
        if token is None:
            return None
        key = (token, model, dim, CACHE_MAX_ROWS)
        if no_cache == key or not self._cache_build.acquire(blocking=False):
            return None
        try:
            built = self._build_cache(token, model, dim)
        finally:
            self._cache_build.release()
        if built is None:
            if self._vec_token() == token:  # not a concurrent write: retrying won't help
                with self._cache_lock:
                    if self._cache_gen == gen:
                        self._no_cache = key
            return None
        with self._cache_lock:
            if self._cache_gen != gen:
                return None  # changed in this process meanwhile: never keep old rows
            self._cache = built
            self._cache_used = _now()
        log.debug("search vector cache: %d rows", len(built.meta))
        return built

    def _build_cache(self, token: str, model: str, dim: int) -> _VecCache | None:
        """Read the vectors in chunks (no long read lock, so writers aren't held up) and keep
        them only if the token is unchanged afterwards: no change committed in between. A
        cheap count first keeps an over-cap index from reading anything; the matrix is filled
        in place (no second copy at the end)."""
        with contextlib.closing(self._connect()) as db:
            total = db.execute(  # this model's rows only, as the chunked read below selects
                "SELECT count(*) FROM seg_vec v JOIN sem_sessions m ON m.folder = v.folder "
                "JOIN sessions s ON s.folder = v.folder WHERE m.model = ?",
                (model,),
            ).fetchone()[0]
        if total > CACHE_MAX_ROWS:
            return None
        matrix = np.empty((total, dim), dtype=np.float32)
        meta: list[tuple[str, str, str, float, str, str, str]] = []
        pool: dict[str, str] = {}  # one string object per folder, speaker, ...
        last = 0
        while True:
            with contextlib.closing(self._connect()) as db:
                rows = db.execute(
                    "SELECT v.rowid, v.vec, v.folder, s.session_id, s.started_at, v.start, "
                    "v.speaker, v.language, v.text FROM seg_vec v "
                    "JOIN sem_sessions m ON m.folder = v.folder "
                    "JOIN sessions s ON s.folder = v.folder "
                    "WHERE m.model = ? AND v.rowid > ? ORDER BY v.rowid LIMIT ?",
                    (model, last, SCAN_CHUNK),
                ).fetchall()
            if len(meta) + len(rows) > total:  # rows added since the count: token changed
                return None
            if rows:
                last = rows[-1][0]
                blob = b"".join(r[1] for r in rows)
                if len(blob) != len(rows) * dim * 4:  # another model's vectors
                    return None
                n = len(meta)
                matrix[n : n + len(rows)] = np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)
                p = pool.setdefault
                meta.extend(
                    (p(r[2], r[2]), p(r[3], r[3]), p(r[4], r[4]), float(r[5]))
                    + (p(r[6], r[6]), p(r[7], r[7]), r[8])
                    for r in rows
                )
            if len(rows) < SCAN_CHUNK:
                break
        if self._vec_token() != token:
            return None
        if len(meta) < total:  # rows vanished between the count and the reads
            matrix = matrix[: len(meta)].copy()
        return _VecCache(token, model, matrix, meta)

    # ------------------------------------------------------------------ queries
    def search(self, query: str, limit: int = 20) -> list[Hit]:
        """Full-text matches (``_lexical``); with an embedder, fused by reciprocal rank with
        the segments closest in meaning. A failing embedder falls back to the full-text hits."""
        if self.embedder is None or limit <= 0 or not _TOKEN.search(query):
            return self._lexical(query, limit)
        lexical = self._lexical(query, limit * POOL)
        if self.busy is not None and self.busy():  # a call is live: no ONNX, no O(n) scan
            return lexical[:limit]
        try:
            semantic = self._semantic(query, limit * POOL)
        except Exception as e:  # noqa: BLE001 - semantic search is optional
            log.error("semantic search failed: %s", type(e).__name__)
            return lexical[:limit]
        return fuse(lexical, semantic)[:limit]

    def _lexical(self, query: str, limit: int = 20) -> list[Hit]:
        """Segments containing every word of ``query`` (FTS5: prefix match, BM25-ranked;
        fallback: substring match, newest sessions first). ё and е match each other."""
        words = _TOKEN.findall(_yo(query))
        if not words or limit <= 0:
            return []
        with contextlib.closing(self._connect()) as db:
            if self.fts:
                match = " ".join('"' + w.replace('"', "") + '"*' for w in words)
                rows = db.execute(
                    "SELECT f.folder, s.session_id, s.started_at, f.start, f.speaker, "
                    "f.language, highlight(seg_fts, 0, char(1), char(2)), f.orig, f.text "
                    "FROM seg_fts f "
                    "JOIN sessions s ON s.folder = f.folder WHERE seg_fts MATCH ? "
                    "ORDER BY bm25(seg_fts), s.started_at DESC, f.start LIMIT ?",
                    (match, limit),
                ).fetchall()
                rows = [(*r[:6], _snippet(_restore(r[6], r[7])), _seg_id(r[8])) for r in rows]
            else:
                folded = [_fold(w) for w in words]
                where = " AND ".join(["p.norm LIKE ? ESCAPE '\\'"] * len(folded))
                sql = _LIKE_SQL.format(where=where)  # only constants and placeholders
                params = (*(f"%{_escape_like(w)}%" for w in folded), limit)
                rows = db.execute(sql, params).fetchall()
                rows = [(*r[:6], _snippet(_mark(r[6], folded[0])), _seg_id(r[6])) for r in rows]
        return [Hit(Path(r[0]), r[1], r[2], float(r[3]), r[4], r[5], r[6], r[7]) for r in rows]


def _now() -> float:
    return time.monotonic()


def _score_cached(cache: _VecCache, q0: np.ndarray, limit: int) -> list[Hit]:
    """The streaming scan's ranking over the cached matrix: best first, ties by rowid."""
    sims = cache.matrix @ q0
    ids = np.flatnonzero(sims >= SEMANTIC_MIN_SIM)
    if len(ids) > limit:
        ids = ids[np.argpartition(-sims[ids], limit - 1)[:limit]]
    hits = []
    for i in ids[np.lexsort((ids, -sims[ids]))][:limit]:
        folder, session_id, started_at, start, speaker, language, text = cache.meta[i]
        snippet = _plain_snippet(text)
        hits.append(
            Hit(
                Path(folder),
                session_id,
                started_at,
                start,
                speaker,
                language,
                snippet,
                _seg_id(text),
            )
        )
    return hits


def _seg_id(text: str) -> str:
    """Identity of a segment's text; ё/е-insensitive, as the full-text index stores it."""
    return hashlib.blake2b(_yo(text).encode(), digest_size=8).hexdigest()


def _key(h: Hit) -> tuple[Path, float, str, str]:
    # Me and Others can speak at the same timestamp, and so can two segments of one speaker
    return (h.folder, h.start, h.speaker, h.seg)


def fuse(lexical: Sequence[Hit], semantic: Sequence[Hit]) -> list[Hit]:
    """Hits found by both rankings first (by reciprocal rank), then the full-text-only ones,
    then the meaning-only ones: a semantic hit never outranks an exact word match."""
    lkeys, skeys = {_key(h) for h in lexical}, {_key(h) for h in semantic}
    fused = rrf([lexical, semantic])
    tier = [
        (0 if _key(h) in lkeys and _key(h) in skeys else 1 if _key(h) in lkeys else 2)
        for h in fused
    ]
    return [h for _, h in sorted(zip(tier, fused, strict=True), key=lambda p: p[0])]


def rrf(rankings: Sequence[Sequence[Hit]]) -> list[Hit]:
    """Reciprocal rank fusion of rankings of hits (one segment = folder + start + speaker).
    The first ranking's copy of a hit is kept (the full-text one has the highlighted snippet)."""
    score: dict[tuple[Path, float, str, str], float] = {}
    first: dict[tuple[Path, float, str, str], Hit] = {}
    for ranking in rankings:
        for rank, h in enumerate(ranking):
            key = _key(h)
            score[key] = score.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            first.setdefault(key, h)
    return [first[k] for k in sorted(score, key=lambda k: -score[k])]


def _plain_snippet(text: str, width: int = 120) -> str:
    return text if len(text) <= width else text[:width] + "…"


def _read(d: Path) -> dict | None:
    """A session's transcript.json as {session_id, started_at, segments: [(text, start,
    speaker, language)]}, or None (logged) if it can't be read."""
    try:
        data = json.loads((d / TRANSCRIPT).read_text(encoding="utf-8"))
        segs = [
            (str(s["text"]), float(s["start"]), str(s["speaker"]), str(s["language"]))
            for s in data.get("segments", [])
        ]
        return {
            "session_id": str(data.get("session_id", d.name)),
            "started_at": str(data.get("started_at", "")),
            "segments": segs,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        log.warning("search: cannot index %s: %s", d, type(e).__name__)
        return None


def _stamp(f: Path) -> str:
    """Change detection: mtime and size (a coarse mtime alone can miss a quick rewrite)."""
    st = f.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


def _yo(s: str) -> str:
    """ё -> е, Ё -> Е: Russian text often omits the diaeresis (a 1:1 character mapping)."""
    return s.translate(_YO)


def _fold(s: str) -> str:
    """Case- and ё-insensitive form for the LIKE fallback (``lower``: same length, mostly)."""
    return _yo(s).lower()


def _restore(marked: str, orig: str | None) -> str:
    """Put the original characters (ё) back into FTS5's highlighted, normalised text."""
    if not orig:
        return marked
    chars = iter(orig)
    return "".join(c if c in _MARKS else next(chars, c) for c in marked)


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _mark(text: str, word: str) -> str:
    """``text`` with the first ``word`` (fallback search) wrapped in the highlight marks."""
    norm = _fold(text)
    i = norm.find(word)
    if i < 0 or len(norm) != len(text):
        return text
    j = i + len(word)
    return f"{text[:i]}\x01{text[i:j]}\x02{text[j:]}"


def _snippet(marked: str, width: int = 60) -> str:
    """About ``width`` characters either side of the first match, the matches in [ ]."""
    i = max(0, marked.find("\x01"))
    j = marked.find("\x02", i)
    lo, hi = max(0, i - width), min(len(marked), (j if j >= 0 else i) + 1 + width)
    text = marked[lo:hi].replace("\x01", "[").replace("\x02", "]")
    return ("…" if lo else "") + text + ("…" if hi < len(marked) else "")


_default: TranscriptIndex | None = None
_default_lock = threading.Lock()


def default_index(
    semantic: bool = False, busy: Callable[[], bool] | None = None
) -> TranscriptIndex:
    """The process-wide index at ``INDEX_PATH`` (created on first use). ``semantic`` is
    ``[search] semantic``; a change of it takes effect on the next call. ``busy`` (when given)
    pauses background embedding while it is true (a recording). Turning semantic off also
    unloads the model."""
    global _default
    dropped = None
    with _default_lock:
        if _default is None or _default.path != INDEX_PATH:
            _default = TranscriptIndex()
        # one index (one lock, one worker) per path: only its embedder follows the setting
        if busy is not None:
            _default.busy = busy
        if not semantic:
            if _default.embedder is not None:
                dropped = _default.embedder
                _default.set_embedder(None)
        elif _default.embedder is None:
            from .semantic import default_embedder

            _default.set_embedder(default_embedder())
        index = _default
    if dropped is not None:
        # Close only if no concurrent default_index(True) re-attached the very object we
        # dropped (closing it would disable the model that index now uses). The re-check and
        # the close share one lock hold, so no re-attach slips between them; close() takes only
        # the embedder's own state lock for a few assignments, so it cannot wedge a search.
        with _default_lock:
            if _default is None or _default.embedder is not dropped:
                close = getattr(dropped, "close", None)
                if close is not None:
                    close()
    return index


def search_transcripts(
    output_dir: Path,
    query: str,
    limit: int = 20,
    semantic: bool = False,
    wait_s: float = 0.0,
    notify: Callable[[str], None] | None = None,
    busy: Callable[[], bool] | None = None,
) -> list[Hit]:
    """Sync the default index with ``output_dir`` (incremental: only changed sessions are
    read), then search it. Blocking: call it off the UI thread. Embedding runs in the
    background and is not waited for (meaning matches cover what is embedded so far), unless
    ``wait_s`` > 0: a one-shot process (the CLI) loads the model (the one-time download, with
    a ``notify`` notice), indexes, and waits up to ``wait_s`` seconds for it, because it would
    otherwise exit before the daemon worker produced anything."""
    index = default_index(semantic, busy)
    if semantic and wait_s > 0 and index.embedder is not None:
        prepare = getattr(index.embedder, "prepare", None)
        ready = getattr(index.embedder, "ready", None)
        if prepare is not None and ready is not None and not ready():
            if notify:
                notify("Downloading the semantic search model (once)...")
            prepare()  # no `wait`: a CLI search has no recording to yield to
    index.sync(output_dir)
    if semantic and wait_s > 0 and not index.wait_embedded(wait_s) and notify:
        notify("still indexing; showing what's ready")
    return index.search(query, limit)
