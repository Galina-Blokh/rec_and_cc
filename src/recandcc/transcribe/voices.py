"""Voice profiles: speaker names remembered across sessions (opt-in, issue #1).

Voice data is biometric, so nothing here is stored unless
``transcription.speaker_names = true``, and all of it lives in the owner-only ``VOICES_DIR``
(app data, never inside ``output_dir``, which may be synced to a cloud drive):

- ``<VOICES_DIR>/<hash>.json``: one profile per name: the sum of the voice centroids it was
  named from (so a wrong one can be taken out again), their count and the embedding model.
- ``<VOICES_DIR>/sessions/<hash>.json``: each diarized session's speaker centroids (label ->
  unit vector) and its folder, so a speaker can be named after the fact without re-running
  diarization. ``prune_session_voices`` deletes them once that folder is gone.

``match_profiles`` names a new session's speakers: a cluster centroid takes a profile's name
if their cosine similarity is >= ``MATCH_SIMILARITY`` and beats the second-best profile by
``MATCH_MARGIN``; each name is used at most once per session. ``Me`` is never relabelled.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import platformdirs

from ..config import APP_NAME
from ..models import Segment, SessionInfo
from .diarize import EMBEDDING_MODEL
from .merge import ME, OTHERS
from .output import JSON_NAME, write_outputs

log = logging.getLogger("recandcc.transcribe.voices")

VOICES_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / "voices"
MODEL = EMBEDDING_MODEL.name  # centroids of another embedding model are not comparable
# Cosine similarity of a session's speaker centroid to a profile. Deliberately strict (the
# clustering itself merges windows at >= 0.5): a wrong name is worse than "Speaker N", and a
# match also needs a clear margin over the next profile. To be tuned on real speech (#13).
MATCH_SIMILARITY = 0.6
MATCH_MARGIN = 0.1
REPLACE_TRIES = 5
NUMBERED = re.compile(r"speaker\s*\d+")  # labels the diarizer hands out itself


@dataclass(frozen=True)
class Profile:
    name: str
    centroid: np.ndarray  # unit vector
    count: int  # how many sessions' speakers it is the mean of
    updated_at: str = ""
    total: np.ndarray | None = None  # sum of those centroids, so one can be taken out again


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def _key(text: str) -> str:
    return _norm(text).casefold()


def check_name(name: str) -> str:
    """The normalised name, or ValueError if it can't be a speaker name."""
    name = _norm(name)
    if not name or any(unicodedata.category(ch).startswith("C") for ch in name):
        raise ValueError("a name must be non-empty text without control characters or newlines")
    if "**" in name or _key(name) in {_key(ME), _key(OTHERS)} or NUMBERED.fullmatch(_key(name)):
        raise ValueError(f"{name!r} is reserved (Me, Others, Speaker N) or breaks the markdown")
    return name


# Every change of the voice store holds LOCK (re-entrant: name_speaker calls the others), so
# "Forget all voices" can't interleave with a naming or a diarization. `delete_all_profiles`
# also bumps the generation: a write that began before it (`since=`) is dropped.
LOCK = threading.RLock()
# Held by name_speaker for its whole run, before LOCK; "forget all" never takes it, so a slow
# transcript folder (cloud sync) can't stall a forget or the diarizer's voice writes.
NAMING_LOCK = threading.Lock()
_forgets = 0


def forget_generation() -> int:
    """Changes every time all voices are deleted; see ``write_session_voices(since=)``."""
    return _forgets


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with LOCK:
            return fn(*args, **kwargs)

    return wrapper


def _dir(voices_dir: Path | None) -> Path:
    return Path(voices_dir) if voices_dir is not None else VOICES_DIR


def _hashed(folder: Path, text: str) -> Path:
    return folder / f"{hashlib.sha256(text.encode()).hexdigest()[:16]}.json"  # any text is safe


def _write_private(path: Path, data: dict) -> None:
    """Atomic write of an owner-only (0600) JSON file in an owner-only (0700) folder."""
    nested = path.parent.name == "sessions"
    for d in [path.parent.parent, path.parent] if nested else [path.parent]:
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            d.chmod(0o700)  # mkdir's mode is filtered by the umask
    # unique per process and thread (the app and the CLI may both write), like SessionInfo
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    _retry(lambda: tmp.unlink(missing_ok=True))  # a stale one may have other permissions
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())  # the new file is complete before it replaces the old one
        _retry(lambda: os.replace(tmp, path))
    finally:
        tmp.unlink(missing_ok=True)


def _retry(op) -> None:
    """Run `op`, retrying a PermissionError (Windows: antivirus or a reader holds the file)."""
    for attempt in range(REPLACE_TRIES):
        try:
            op()
            return
        except PermissionError:
            if attempt + 1 == REPLACE_TRIES:
                raise
            time.sleep(0.05 * (attempt + 1))


def _vector(c) -> list[float]:
    return [round(float(x), 6) for x in _unit(c)]


# --------------------------------------------------------------------------- profiles store
def load_profiles(voices_dir: Path | None = None) -> list[Profile]:
    """Profiles of the current embedding model, sorted by name; unreadable files are skipped
    (logged by file name)."""
    d = _dir(voices_dir)
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("model") == MODEL:
                count = int(data["count"])
                if "sum" in data:
                    total = np.asarray(data["sum"], dtype=np.float64)
                else:  # written before sums were stored: its mean stands for every session
                    total = _unit(data["centroid"]) * count
                out.append(Profile(data["name"], _unit(total), count, data["updated"], total))
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("skipping unreadable voice profile %s", f.name)
    return sorted(out, key=lambda p: _key(p.name))


@_locked
def save_profile(name: str, centroid, voices_dir: Path | None = None) -> Profile:
    """Fold `centroid` into the profile of `name` (created if new): the mean of all of them."""
    profile = _fold(name, centroid, +1, voices_dir)
    if profile is None:  # unreachable: adding never empties a profile
        raise RuntimeError("voice profile vanished")
    return profile


def _fold(name: str, centroid, sign: int, voices_dir: Path | None) -> Profile | None:
    """Add (+1) or take out (-1) one centroid; deletes the profile when nothing is left."""
    name = check_name(name)
    path = _hashed(_dir(voices_dir), _key(name))
    old = next((p for p in load_profiles(voices_dir) if _key(p.name) == _key(name)), None)
    c = _unit(centroid)
    total, count = np.zeros_like(c), 0
    if old is not None and old.total is not None:
        if old.total.shape != c.shape:  # another kind of vector: never mix or reset them
            raise ValueError(f"the voice doesn't fit the profile of {name!r}")
        total, count = old.total, old.count
    elif sign < 0:
        return None  # nothing to take out of
    total, count = total + sign * c, count + sign
    if count <= 0:
        path.unlink(missing_ok=True)
        return None
    now = datetime.now().isoformat(timespec="seconds")
    data = {"name": name, "model": MODEL, "count": count, "updated": now}
    _write_private(path, {**data, "sum": [round(float(x), 6) for x in total]})
    return Profile(name, _unit(total), count, now, total)


@_locked
def delete_profile(name: str, voices_dir: Path | None = None) -> bool:
    """Delete the profile of `name` and every session's centroid labelled `name` or folded
    into it, so the voice can't come back from a session."""
    path = _hashed(_dir(voices_dir), _key(name))
    existed = path.exists()
    path.unlink(missing_ok=True)
    for f in _session_files(voices_dir):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            speakers = data.get("speakers", {})
            keep = {
                lbl: v
                for lbl, v in speakers.items()
                if _key(name) not in (_key(lbl), _key(v.get("profile") or ""))
            }
            if len(keep) != len(speakers):
                existed = True
                _write_private(f, {**data, "speakers": keep})
        except (OSError, ValueError, AttributeError):
            log.warning("could not clean session voices %s", f.name)
    return existed


@_locked
def delete_all_profiles(voices_dir: Path | None = None) -> int:
    """Every profile, every session's speaker voices and any temp file left by a crash
    (they hold voice data too); returns the number of profiles."""
    d = _dir(voices_dir)
    profiles = list(d.glob("*.json")) if d.is_dir() else []
    stale = [*d.glob(".*.tmp"), *d.glob("sessions/.*.tmp")] if d.is_dir() else []
    global _forgets
    _forgets += 1
    for f in [*profiles, *_session_files(voices_dir), *stale]:
        f.unlink(missing_ok=True)
    return len(profiles)


# --------------------------------------------------------------------------- matching
def match_profiles(
    centroids: Mapping[int, np.ndarray],
    profiles: list[Profile],
    min_similarity: float = MATCH_SIMILARITY,
    margin: float = MATCH_MARGIN,
) -> dict[int, str]:
    """Speaker id -> profile name for the clusters that match one profile clearly."""
    candidates = []
    for k, c in centroids.items():
        c = _unit(c)
        sims = sorted(
            ((float(c @ p.centroid), p.name) for p in profiles if p.centroid.shape == c.shape),
            reverse=True,
        )
        if not sims:
            continue
        second = sims[1][0] if len(sims) > 1 else -1.0
        if sims[0][0] >= min_similarity and sims[0][0] - second >= margin:
            candidates.append((sims[0][0], k, sims[0][1]))
    names: dict[int, str] = {}
    for _, k, name in sorted(candidates, reverse=True):  # the closest cluster gets the name
        if _key(name) not in {_key(n) for n in names.values()}:
            names[k] = name
    return names


# --------------------------------------------------------------------------- sessions
def _session_path(session_id: str, voices_dir: Path | None) -> Path:
    return _hashed(_dir(voices_dir) / "sessions", session_id)


def _session_files(voices_dir: Path | None) -> list[Path]:
    d = _dir(voices_dir) / "sessions"
    return sorted(d.glob("*.json")) if d.is_dir() else []


def _deleted(directory: Path) -> bool:
    """True only if `directory` is surely deleted: its parent (the day folder) or, when that
    is gone too, the parent's parent (output_dir) can be listed and lacks it. An unmounted
    drive, an offline share or a permission error is never taken for a deletion."""
    for parent, child in ((directory.parent, directory.name), (directory.parent.parent, None)):
        try:
            names = os.listdir(parent)
        except OSError:
            continue
        return (child or directory.parent.name) not in names
    return False


@_locked
def prune_session_voices(voices_dir: Path | None = None) -> int:
    """Delete the voices of sessions whose folder was deleted; returns how many. Files that
    can't be read or name no folder (written before folders were stored) are kept."""
    n = 0
    for f in _session_files(voices_dir):
        try:
            directory = json.loads(f.read_text(encoding="utf-8")).get("directory")
            if isinstance(directory, str) and directory and _deleted(Path(directory)):
                f.unlink()
                n += 1
        except (OSError, ValueError, AttributeError):
            log.warning("could not check session voices %s", f.name)
    if n:
        log.info("deleted the voices of %d deleted session(s)", n)
    return n


@_locked
def write_session_voices(
    session: SessionInfo,
    speakers: Mapping[str, np.ndarray],
    folded: Mapping[str, str] | None = None,
    voices_dir: Path | None = None,
    since: int | None = None,
) -> None:
    """Save a session's label -> centroid. `folded`: label -> the profile it was already
    added to (so naming it again doesn't count the same session twice). `since`: the
    ``forget_generation()`` when the work began; if all voices were deleted since, nothing
    is written."""
    if since is not None and since != _forgets:
        return
    folded = folded or {}
    entries = {
        label: {"centroid": _vector(c), "profile": folded.get(label)}
        for label, c in speakers.items()
    }
    data = {
        "version": 1,
        "session_id": session.id,
        "directory": str(Path(session.directory).resolve()),  # to prune deleted sessions
        "model": MODEL,
        "speakers": entries,
    }
    _write_private(_session_path(session.id, voices_dir), data)


def read_session_voices(session_id: str, voices_dir: Path | None = None) -> dict[str, dict]:
    """label -> {"centroid": unit vector, "profile": name | None}; {} if none are stored."""
    path = _session_path(session_id, voices_dir)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model") != MODEL:
        return {}
    return {
        label: {"centroid": _unit(v["centroid"]), "profile": v.get("profile")}
        for label, v in data.get("speakers", {}).items()
    }


@_locked
def delete_session_voices(session_id: str, voices_dir: Path | None = None) -> None:
    _session_path(session_id, voices_dir).unlink(missing_ok=True)


def session_speakers(directory: Path, lines: int = 2) -> list[tuple[str, list[str]]]:
    """Each label except Me with its first `lines` texts, in order of first appearance."""
    data = json.loads((Path(directory) / JSON_NAME).read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for s in data.get("segments", []):
        if s["speaker"] != ME and len(texts := out.setdefault(s["speaker"], [])) < lines:
            texts.append(s["text"].strip())
    return list(out.items())


def name_speaker(directory: Path, label: str, name: str, voices_dir: Path | None = None) -> Profile:
    """Save `label`'s voice of the session in `directory` as `name` and rewrite its transcript
    files with the name. Labels and names compare case-insensitively. Raises ValueError when
    that is not possible."""
    with NAMING_LOCK:  # one renaming at a time: each rewrites the whole transcript
        return _name_speaker(Path(directory), label, check_name(name), voices_dir)


def _name_speaker(directory: Path, label: str, name: str, voices_dir: Path | None) -> Profile:
    session = SessionInfo.load(directory)
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    found = next((s.speaker for s in segments if _key(s.speaker) == _key(label)), None)
    if found is None:
        raise ValueError("no such speaker in this transcript; see `recandcc speakers`")
    label = found
    if label == ME:
        raise ValueError(f'"{ME}" is your microphone track and is never renamed')
    if any(_key(s.speaker) == _key(name) and s.speaker != label for s in segments):
        raise ValueError(f"{name!r} is already another speaker of this session")
    with LOCK:  # the voice store only; the transcript is rewritten outside it
        stored = read_session_voices(session.id, voices_dir)
        folded_as = {k: _key(v["profile"] or "") for k, v in stored.items()}
        # `name`: its voice was renamed already, but rewriting the transcript failed last time
        found = [k for k in stored if _key(k) == _key(label)] or [
            k for k in stored if _key(k) == _key(name) == folded_as[k]
        ]
        if not found:
            raise ValueError(
                f"no voice data for {label!r} (speaker_names was off when this session was "
                "transcribed, or its voices were deleted)"
            )
        entry, previous = stored[found[0]], folded_as[found[0]]
        known = {_key(p.name): p for p in load_profiles(voices_dir)}
        voices = {name if k == found[0] else k: v for k, v in stored.items()}
        # Mark this session as folded into `name` first: a failure below can then leave the
        # profile one session short, but never count this session twice.
        write_session_voices(
            session,
            {lbl: v["centroid"] for lbl, v in voices.items()},
            {lbl: v["profile"] for lbl, v in voices.items() if v["profile"]} | {name: name},
            voices_dir,
        )
        profile = known.get(_key(name))
        if previous != _key(name) or profile is None:
            if previous and previous != _key(name) and previous in known:
                _fold(known[previous].name, entry["centroid"], -1, voices_dir)  # Anna -> Bob
            profile = save_profile(name, entry["centroid"], voices_dir)
    if label != name:
        segments = [replace(s, speaker=name) if s.speaker == label else s for s in segments]
        write_outputs(session, segments, data["duration"], data["model"])
    log.info("named a speaker of session %s (profile of %d sessions)", session.id, profile.count)
    return profile


__all__ = [
    "LOCK",
    "MATCH_MARGIN",
    "NAMING_LOCK",
    "MATCH_SIMILARITY",
    "VOICES_DIR",
    "Profile",
    "check_name",
    "delete_all_profiles",
    "delete_profile",
    "delete_session_voices",
    "load_profiles",
    "match_profiles",
    "name_speaker",
    "prune_session_voices",
    "read_session_voices",
    "save_profile",
    "session_speakers",
    "write_session_voices",
]
