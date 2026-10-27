"""Voice profiles: speaker names remembered across sessions (opt-in, issue #1).

Voice profiles are biometric data, so nothing here is stored unless
``transcription.speaker_names = true``:

- A final transcript with diarized speakers gets ``speakers.json`` beside it: each label's
  voice centroid (unit vector), so a speaker can be named later without re-running
  diarization.
- ``name_speaker`` saves (or updates, as a running mean) the profile of a name in
  ``VOICES_DIR`` (owner-only app-data folder, never inside ``output_dir``; one small JSON file
  per name) and rewrites that session's transcript.md/.srt/.json labels.
- ``match_profiles`` names a new session's speakers: a cluster centroid takes a profile's name
  if their cosine similarity is >= ``MATCH_SIMILARITY`` and beats the second-best profile by
  ``MATCH_MARGIN``; each name is used at most once per session. ``Me`` is never relabelled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import platformdirs

from ..config import APP_NAME
from ..models import Segment, SessionInfo
from .merge import ME, OTHERS
from .output import JSON_NAME, atomic_write, write_outputs

log = logging.getLogger("recandcc.transcribe.voices")

VOICES_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False)) / "voices"
SPEAKERS_NAME = "speakers.json"  # per-session centroids, beside transcript.json
# Cosine similarity of a session's speaker centroid to a profile. Deliberately strict (the
# clustering itself merges windows at >= 0.5): a wrong name is worse than "Speaker N", and a
# match also needs a clear margin over the next profile. To be tuned on real speech (#13).
MATCH_SIMILARITY = 0.6
MATCH_MARGIN = 0.1
RESERVED = {ME.casefold(), OTHERS.casefold()}


@dataclass(frozen=True)
class Profile:
    name: str
    centroid: np.ndarray  # unit vector
    count: int  # how many named speakers it is the mean of
    updated_at: str = ""


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _dir(voices_dir: Path | None) -> Path:
    return Path(voices_dir) if voices_dir is not None else VOICES_DIR


def _profile_path(name: str, voices_dir: Path | None) -> Path:
    key = hashlib.sha256(name.strip().casefold().encode()).hexdigest()[:16]
    return _dir(voices_dir) / f"{key}.json"  # any name (Cyrillic, "/") gives a safe filename


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


# --------------------------------------------------------------------------- profiles store
def load_profiles(voices_dir: Path | None = None) -> list[Profile]:
    """All profiles, sorted by name; unreadable files are skipped (logged by file name)."""
    d = _dir(voices_dir)
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append(
                Profile(data["name"], _unit(data["centroid"]), int(data["count"]), data["updated"])
            )
        except (OSError, ValueError, KeyError, TypeError):
            log.warning("skipping unreadable voice profile %s", f.name)
    return sorted(out, key=lambda p: p.name.casefold())


def save_profile(name: str, centroid, voices_dir: Path | None = None) -> Profile:
    """Create the profile of `name`, or fold `centroid` into it (running mean)."""
    name = name.strip()
    path = _profile_path(name, voices_dir)
    old = next((p for p in load_profiles(voices_dir) if p.name.casefold() == name.casefold()), None)
    c, count = _unit(centroid), 1
    if old is not None and old.centroid.shape == c.shape:
        c, count = _unit(old.centroid * old.count + c), old.count + 1
    profile = Profile(name, c, count, datetime.now().isoformat(timespec="seconds"))
    _private_dir(path.parent)
    data = {"name": name, "count": count, "updated": profile.updated_at}
    data["centroid"] = [round(float(x), 6) for x in c]
    fd = os.open(path.with_name(f".{path.name}.tmp"), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    path.with_name(f".{path.name}.tmp").replace(path)
    return profile


def delete_profile(name: str, voices_dir: Path | None = None) -> bool:
    path = _profile_path(name, voices_dir)
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def delete_all_profiles(voices_dir: Path | None = None) -> int:
    d = _dir(voices_dir)
    files = list(d.glob("*.json")) if d.is_dir() else []
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)


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
        if name not in names.values():
            names[k] = name
    return names


# --------------------------------------------------------------------------- sessions
def write_session_speakers(directory: Path, speakers: Mapping[str, np.ndarray]) -> None:
    """speakers.json: label -> voice centroid of each diarized speaker of the session."""
    data = {
        "version": 1,
        "speakers": {
            label: {"centroid": [round(float(x), 6) for x in _unit(c)]}
            for label, c in speakers.items()
        },
    }
    atomic_write(Path(directory) / SPEAKERS_NAME, json.dumps(data, ensure_ascii=False))


def read_session_speakers(directory: Path) -> dict[str, np.ndarray]:
    path = Path(directory) / SPEAKERS_NAME
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {label: _unit(v["centroid"]) for label, v in data.get("speakers", {}).items()}


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
    files with the name. Raises ValueError when that is not possible."""
    directory, name = Path(directory), name.strip()
    if label == ME:
        raise ValueError(f'"{ME}" is your microphone track and is never renamed')
    if not name or name.casefold() in RESERVED:
        raise ValueError(f"choose a name other than {ME!r} / {OTHERS!r}")
    centroids = read_session_speakers(directory)
    if label not in centroids:
        raise ValueError(
            f"no voice data for {label!r} in this session (speaker_names was off, or the "
            "session was not diarized); see `recandcc speakers`"
        )
    data = json.loads((directory / JSON_NAME).read_text(encoding="utf-8"))
    segments = [Segment(**s) for s in data["segments"]]
    if name != label and any(s.speaker == name for s in segments):
        raise ValueError(f"{name!r} is already a speaker of this session")
    profile = save_profile(name, centroids[label], voices_dir)
    if name != label:
        segments = [replace(s, speaker=name) if s.speaker == label else s for s in segments]
        write_outputs(SessionInfo.load(directory), segments, data["duration"], data["model"])
        renamed = {name if lbl == label else lbl: c for lbl, c in centroids.items()}
        write_session_speakers(directory, renamed)
    log.info("named a speaker of session %s (profile seen %d times)", directory.name, profile.count)
    return profile


__all__ = [
    "MATCH_MARGIN",
    "MATCH_SIMILARITY",
    "SPEAKERS_NAME",
    "VOICES_DIR",
    "Profile",
    "delete_all_profiles",
    "delete_profile",
    "load_profiles",
    "match_profiles",
    "name_speaker",
    "read_session_speakers",
    "save_profile",
    "session_speakers",
    "write_session_speakers",
]
