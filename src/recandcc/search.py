"""Full-text search over the transcripts: an incremental SQLite index of transcript.json.

The index lives in the platform app-data dir (never inside ``output_dir``, so cloud sync of
the transcripts folder never picks it up) and is owner-only. ``sync`` reindexes a session
whose transcript.json changed (mtime) and drops sessions that were deleted or discarded;
``index_session`` updates one session after its transcript is final. With FTS5 the text is
tokenised by ``unicode61 remove_diacritics 2`` (Russian and other scripts work) and hits are
ranked by BM25; an SQLite built without FTS5 falls back to a (slower) ``LIKE`` scan.

Every call opens its own connection, so any thread may use an index; writes are serialised
by a lock (and by SQLite's file lock across processes, e.g. the app and the CLI). Transcript
text is never logged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import platformdirs

from .config import APP_NAME
from .models import SessionInfo

log = logging.getLogger(__name__)

INDEX_PATH = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / "search.sqlite3"
TRANSCRIPT = "transcript.json"
SCHEMA_VERSION = 1
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

    @property
    def transcript(self) -> Path:
        return self.folder / "transcript.md"


def has_fts5() -> bool:
    try:
        with contextlib.closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.Error:
        return False


class TranscriptIndex:
    """``fts=None`` uses FTS5 when this SQLite has it; ``False`` forces the LIKE fallback."""

    def __init__(self, path: Path | None = None, fts: bool | None = None) -> None:
        self.path = Path(path) if path is not None else INDEX_PATH
        self.fts = has_fts5() if fts is None else fts
        self._lock = threading.Lock()
        self._ready = False

    # ------------------------------------------------------------------ storage
    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):  # owner-only (a no-op for ACLs on Windows)
                os.chmod(self.path.parent, 0o700)
                self.path.touch(mode=0o600, exist_ok=True)
                os.chmod(self.path, 0o600)
        db = sqlite3.connect(self.path, timeout=10)
        if not self._ready:
            self._create(db)
            self._ready = True
        return db

    def _create(self, db: sqlite3.Connection) -> None:
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            for table in ("sessions", "seg_fts", "seg_plain"):
                db.execute(f"DROP TABLE IF EXISTS {table}")
            db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        db.execute(
            "CREATE TABLE IF NOT EXISTS sessions (folder TEXT PRIMARY KEY, session_id TEXT, "
            "started_at TEXT, mtime INTEGER, fts INTEGER)"
        )
        if self.fts:
            db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS seg_fts USING fts5(text, folder UNINDEXED, "
                "start UNINDEXED, speaker UNINDEXED, language UNINDEXED, "
                "tokenize = 'unicode61 remove_diacritics 2')"
            )
        else:
            db.execute(
                "CREATE TABLE IF NOT EXISTS seg_plain (folder TEXT, start REAL, speaker TEXT, "
                "language TEXT, text TEXT, norm TEXT)"
            )
            db.execute("CREATE INDEX IF NOT EXISTS seg_plain_folder ON seg_plain(folder)")
        db.commit()

    @property
    def _table(self) -> str:
        return "seg_fts" if self.fts else "seg_plain"

    # ------------------------------------------------------------------ updates
    def index_session(self, directory: Path) -> bool:
        """(Re)index one session folder, or drop it when it has no transcript or was
        discarded. True if it is indexed now."""
        d = Path(directory)
        f = d / TRANSCRIPT
        with self._lock, contextlib.closing(self._connect()) as db:
            if (d / SessionInfo.DISCARD_MARKER).exists() or not f.exists():
                self._drop(db, str(d))
                db.commit()
                return False
            ok = self._index(db, d, f.stat().st_mtime_ns)
            db.commit()
            return ok

    def sync(self, output_dir: Path) -> int:
        """Bring the index in line with <output_dir>/<day>/<session>/transcript.json;
        returns the number of sessions (re)indexed."""
        found: dict[str, int] = {}
        for f in Path(output_dir).glob(f"*/*/{TRANSCRIPT}"):
            try:
                if not (f.parent / SessionInfo.DISCARD_MARKER).exists():
                    found[str(f.parent)] = f.stat().st_mtime_ns
            except OSError:
                continue
        changed = 0
        with self._lock, contextlib.closing(self._connect()) as db:
            known = {
                folder: (mtime, fts)
                for folder, mtime, fts in db.execute("SELECT folder, mtime, fts FROM sessions")
            }
            for folder in known.keys() - found.keys():
                self._drop(db, folder)
            for folder, mtime in found.items():
                if known.get(folder) != (mtime, int(self.fts)):
                    changed += self._index(db, Path(folder), mtime)
            db.commit()
        log.info("search index synced: %d sessions, %d updated", len(found), changed)
        return changed

    def _drop(self, db: sqlite3.Connection, folder: str) -> None:
        db.execute("DELETE FROM sessions WHERE folder = ?", (folder,))
        table = self._table  # a constant, never user input
        db.execute(f"DELETE FROM {table} WHERE folder = ?", (folder,))  # noqa: S608

    def _index(self, db: sqlite3.Connection, d: Path, mtime: int) -> bool:
        folder = str(d)
        try:
            data = json.loads((d / TRANSCRIPT).read_text(encoding="utf-8"))
            segs = [
                (folder, float(s["start"]), str(s["speaker"]), str(s["language"]), str(s["text"]))
                for s in data.get("segments", [])
            ]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            log.warning("search: cannot index %s: %s", d, type(e).__name__)
            return False
        self._drop(db, folder)
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (folder, data.get("session_id", d.name), data.get("started_at", ""), mtime, self.fts),
        )
        if self.fts:
            db.executemany(
                "INSERT INTO seg_fts VALUES (?, ?, ?, ?, ?)", [(s[4], *s[:4]) for s in segs]
            )
        else:
            db.executemany(
                "INSERT INTO seg_plain VALUES (?, ?, ?, ?, ?, ?)",
                [(*s, s[4].casefold()) for s in segs],
            )
        return True

    # ------------------------------------------------------------------ queries
    def search(self, query: str, limit: int = 20) -> list[Hit]:
        """Segments containing every word of ``query`` (FTS5: prefix match, BM25-ranked;
        fallback: substring match, newest sessions first)."""
        words = _TOKEN.findall(query)
        if not words or limit <= 0:
            return []
        with contextlib.closing(self._connect()) as db:
            if self.fts:
                match = " ".join('"' + w.replace('"', "") + '"*' for w in words)
                rows = db.execute(
                    "SELECT f.folder, s.session_id, s.started_at, f.start, f.speaker, "
                    "f.language, snippet(seg_fts, 0, '[', ']', '…', 16) FROM seg_fts f "
                    "JOIN sessions s ON s.folder = f.folder WHERE seg_fts MATCH ? "
                    "ORDER BY bm25(seg_fts), s.started_at DESC, f.start LIMIT ?",
                    (match, limit),
                ).fetchall()
            else:
                folded = [w.casefold() for w in words]
                where = " AND ".join(["p.norm LIKE ? ESCAPE '\\'"] * len(folded))
                sql = _LIKE_SQL.format(where=where)  # only constants and placeholders
                params = (*(f"%{_escape_like(w)}%" for w in folded), limit)
                rows = db.execute(sql, params).fetchall()
                rows = [(*r[:6], _mark(r[6], folded[0])) for r in rows]
        return [Hit(Path(r[0]), r[1], r[2], float(r[3]), r[4], r[5], r[6]) for r in rows]


def _escape_like(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _mark(text: str, word: str, width: int = 60) -> str:
    """Snippet of ``text`` around the first ``word`` (fallback search), marked like FTS5's."""
    i = text.casefold().find(word)
    if i < 0:
        return text[: 2 * width]
    j = i + len(word)
    lo, hi = max(0, i - width), min(len(text), j + width)
    pre, post = ("…" if lo else ""), ("…" if hi < len(text) else "")
    return f"{pre}{text[lo:i]}[{text[i:j]}]{text[j:hi]}{post}"


_default: TranscriptIndex | None = None
_default_lock = threading.Lock()


def default_index() -> TranscriptIndex:
    """The process-wide index at ``INDEX_PATH`` (created on first use)."""
    global _default
    with _default_lock:
        if _default is None or _default.path != INDEX_PATH:
            _default = TranscriptIndex()
        return _default


def search_transcripts(output_dir: Path, query: str, limit: int = 20) -> list[Hit]:
    """Sync the default index with ``output_dir`` (incremental: only changed sessions are
    read), then search it. Blocking: call it off the UI thread."""
    index = default_index()
    index.sync(output_dir)
    return index.search(query, limit)
