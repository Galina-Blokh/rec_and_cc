"""Voice profiles: speaker names remembered across sessions (opt-in, issue #1).

Voice data is biometric, so nothing here is stored unless
``transcription.speaker_names = true``, and all of it lives in the owner-only ``VOICES_DIR``
(app data, never inside ``output_dir``, which may be synced to a cloud drive):

- ``<VOICES_DIR>/<hash>.json``: one profile per name, the running mean of the voice centroids
  it was named from, their count and the embedding model.
- ``<VOICES_DIR>/sessions/<hash>.json``: each diarized session's speaker centroids (label ->
  unit vector), so a speaker can be named after the fact without re-running diarization.

``match_profiles`` names a new session's speakers: a cluster centroid takes a profile's name
if their cosine similarity is >= ``MATCH_SIMILARITY`` and beats the second-best profile by
``MATCH_MARGIN``; each name is used at most once per session. ``Me`` is never relabelled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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
NUMBERED = re.compile(r"speaker\s*\d+")  # labels the diarizer hands out itself


@dataclass(frozen=True)
class Profile:
    name: str
    centroid: np.ndarray  # unit vector
    count: int  # how many sessions' speakers it is the mean of
    updated_at: str = ""


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
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.unlink(missing_ok=True)  # a stale one from a crash may have other permissions
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(path)


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
                c = _unit(data["centroid"])
                out.append(Profile(data["name"], c, int(data["count"]), data["updated"]))
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("skipping unreadable voice profile %s", f.name)
    return sorted(out, key=lambda p: _key(p.name))


def save_profile(name: str, centroid, voices_dir: Path | None = None) -> Profile:
    """Create the profile of `name`, or fold `centroid` into it (running mean)."""
    name = check_name(name)
    old = next((p for p in load_profiles(voices_dir) if _key(p.name) == _key(name)), None)
    c, count = _unit(centroid), 1
    if old is not None and old.centroid.shape == c.shape:
        c, count = _unit(old.centroid * old.count + c), old.count + 1
    profile = Profile(name, c, count, datetime.now().isoformat(timespec="seconds"))
    data = {"name": name, "model": MODEL, "count": count, "updated": profile.updated_at}
    _write_private(_hashed(_dir(voices_dir), _key(name)), {**data, "centroid": _vector(c)})
    return profile


def delete_profile(name: str, voices_dir: Path | None = None) -> bool:
    path = _hashed(_dir(voices_dir), _key(name))
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def delete_all_profiles(voices_dir: Path | None = None) -> int:
    """Every profile and every session's speaker voices; returns the number of profiles."""
    d = _dir(voices_dir)
    profiles = list(d.glob("*.json")) if d.is_dir() else []
    for f in [*profiles, *(d / "sessions").glob("*.json")]:
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


def write_session_voices(
    session_id: str,
    speakers: Mapping[str, np.ndarray],
    folded: Mapping[str, str] | None = None,
    voices_dir: Path | None = None,
) -> None:
    """Save a session's label -> centroid. `folded`: label -> the profile it was already
    added to (so naming it again doesn't count the same session twice)."""
    folded = folded or {}
    entries = {
        label: {"centroid": _vector(c), "profile": folded.get(label)}
        for label, c in speakers.items()
    }
    data = {"version": 1, "session_id": session_id, "model": MODEL, "speakers": entries}
    _write_private(_session_path(session_id, voices_dir), data)


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
    directory, name = Path(directory), check_name(name)
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
    stored = read_session_voices(session.id, voices_dir)
    entry = next((v for lbl, v in stored.items() if _key(lbl) == _key(label)), None)
    if entry is None:
        raise ValueError(
            f"no voice data for {label!r} (speaker_names was off when this session was "
            "transcribed, or its voices were deleted)"
        )
    known = {_key(p.name): p for p in load_profiles(voices_dir)}
    if entry["profile"] and _key(entry["profile"]) == _key(name) and _key(name) in known:
        profile = known[_key(name)]  # this session is already part of that profile
    else:
        profile = save_profile(name, entry["centroid"], voices_dir)
    if label != name:
        segments = [replace(s, speaker=name) if s.speaker == label else s for s in segments]
        write_outputs(session, segments, data["duration"], data["model"])
    voices = {name if _key(lbl) == _key(label) else lbl: v for lbl, v in stored.items()}
    write_session_voices(
        session.id,
        {lbl: v["centroid"] for lbl, v in voices.items()},
        {lbl: v["profile"] for lbl, v in voices.items() if v["profile"]} | {name: name},
        voices_dir,
    )
    log.info("named a speaker of session %s (profile of %d sessions)", session.id, profile.count)
    return profile


__all__ = [
    "MATCH_MARGIN",
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
    "read_session_voices",
    "save_profile",
    "session_speakers",
    "write_session_voices",
]
