"""Full-text search over the transcripts: an incremental SQLite index of transcript.json.

The index lives in the platform app-data dir (never inside ``output_dir``, so cloud sync of
the transcripts folder never picks it up) and is owner-only. ``sync`` reindexes a session
whose transcript.json changed (mtime, size) and drops sessions that were deleted or discarded;
``index_session`` updates one session after its transcript is final. With FTS5 the text is
tokenised by ``unicode61 remove_diacritics 2`` (Russian and other scripts work) and hits are
ranked by BM25; an SQLite built without FTS5 falls back to a (slower) ``LIKE`` scan. ё is
indexed and searched as е. Deletes are secure (``secure_delete`` and FTS5 ``secure-delete``),
so the text of a dropped session doesn't linger in the file.

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
SCHEMA_VERSION = 2
_YO = str.maketrans("ёЁ", "еЕ")
_MARKS = "\x01\x02"  # highlight marks, shown as [ ]
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
        self._lock = threading.Lock()  # serialises writes
        self._init_lock = threading.Lock()  # first connection: files + schema
        self._ready = False
        self._fts_secure = True  # FTS5 'secure-delete' (SQLite >= 3.42) is on

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

    def _open(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        # Deleted rows are overwritten, so a dropped session's text can't be recovered.
        db.execute("PRAGMA secure_delete = ON")
        return db

    def _create(self, db: sqlite3.Connection) -> None:
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            for table in ("sessions", "seg_fts", "seg_plain"):
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
        db.commit()

    @property
    def _table(self) -> str:
        return "seg_fts" if self.fts else "seg_plain"

    def _purge(self, db: sqlite3.Connection) -> None:
        """After deletes, when FTS5 can't secure-delete: merge the index and rewrite the file
        so no deleted text survives in old segments or free pages."""
        if self.fts and not self._fts_secure:
            db.execute("INSERT INTO seg_fts(seg_fts) VALUES ('optimize')")
            db.commit()
            db.execute("VACUUM")

    # ------------------------------------------------------------------ updates
    def index_session(self, directory: Path) -> bool:
        """(Re)index one session folder, or drop it when it has no transcript or was
        discarded. True if it is indexed now."""
        d = Path(directory)
        f = d / TRANSCRIPT
        with self._lock, contextlib.closing(self._connect()) as db:
            if (d / SessionInfo.DISCARD_MARKER).exists() or not f.exists():
                self._drop(db, str(d))
                ok = False
            else:
                ok = self._index(db, d, _stamp(f))
            db.commit()
            self._purge(db)
            return ok

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
        changed = dropped = 0
        with self._lock, contextlib.closing(self._connect()) as db:
            known = {
                folder: (stamp, fts)
                for folder, stamp, fts in db.execute("SELECT folder, stamp, fts FROM sessions")
            }
            for folder in known.keys() - found.keys():
                self._drop(db, folder)
                dropped += 1
            for folder, stamp in found.items():
                if known.get(folder) != (stamp, int(self.fts)):
                    changed += self._index(db, Path(folder), stamp)
            db.commit()
            if changed or dropped:
                self._purge(db)
        log.info("search index synced: %d sessions, %d updated", len(found), changed)
        return changed

    def _drop(self, db: sqlite3.Connection, folder: str) -> None:
        db.execute("DELETE FROM sessions WHERE folder = ?", (folder,))
        table = self._table  # a constant, never user input
        db.execute(f"DELETE FROM {table} WHERE folder = ?", (folder,))  # noqa: S608

    def _index(self, db: sqlite3.Connection, d: Path, stamp: str) -> bool:
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
            (folder, data.get("session_id", d.name), data.get("started_at", ""), stamp, self.fts),
        )
        if self.fts:
            rows = []
            for s in segs:
                norm = _yo(s[4])
                rows.append((norm, s[4] if norm != s[4] else None, *s[:4]))
            db.executemany("INSERT INTO seg_fts VALUES (?, ?, ?, ?, ?, ?)", rows)
        else:
            db.executemany(
                "INSERT INTO seg_plain VALUES (?, ?, ?, ?, ?, ?)",
                [(*s, _fold(s[4])) for s in segs],
            )
        return True

    # ------------------------------------------------------------------ queries
    def search(self, query: str, limit: int = 20) -> list[Hit]:
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
                    "f.language, highlight(seg_fts, 0, char(1), char(2)), f.orig FROM seg_fts f "
                    "JOIN sessions s ON s.folder = f.folder WHERE seg_fts MATCH ? "
                    "ORDER BY bm25(seg_fts), s.started_at DESC, f.start LIMIT ?",
                    (match, limit),
                ).fetchall()
                rows = [(*r[:6], _snippet(_restore(r[6], r[7]))) for r in rows]
            else:
                folded = [_fold(w) for w in words]
                where = " AND ".join(["p.norm LIKE ? ESCAPE '\\'"] * len(folded))
                sql = _LIKE_SQL.format(where=where)  # only constants and placeholders
                params = (*(f"%{_escape_like(w)}%" for w in folded), limit)
                rows = db.execute(sql, params).fetchall()
                rows = [(*r[:6], _snippet(_mark(r[6], folded[0]))) for r in rows]
        return [Hit(Path(r[0]), r[1], r[2], float(r[3]), r[4], r[5], r[6]) for r in rows]


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
