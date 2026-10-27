"""Offline speaker diarization of the "Others" track: which of the other participants spoke when.

Pipeline (CPU only; ~2 s for 13 min of audio on an M5 Max, see docs/ARCHITECTURE.md)::

    others.wav -> Silero VAD speech regions (the VAD already used for chunking)
               -> regions cut into ~2 s windows
               -> one speaker embedding per window: 3D-Speaker CAM++ (zh+en "advanced",
                  28 MB ONNX) through sherpa-onnx, on a small thread pool (ONNX Runtime
                  releases the GIL, so windows are embedded in parallel)
               -> agglomerative clustering (average linkage, cosine distance) of the windows
                  >= 1 s into k speakers, k picked by silhouette (optional min/max hints);
                  shorter windows and tiny clusters join the nearest speaker
               -> turns (start, end, speaker id)

``assign_speakers`` then gives every "Others" transcript segment the speaker it overlaps
most, numbered "Speaker 1", "Speaker 2", ... by first appearance. With a single speaker the
label stays "Others". With ``speaker_names`` on, speakers that match a voice profile
(voices.py) are named instead of numbered.

The embedding model is downloaded once (sha256-verified) into the platform cache dir
(``MODELS_DIR``); after that nothing here touches the network. If the download fails (offline)
diarization is skipped and retried at most every ``RETRY_AFTER_S``.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import logging
import os
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import platformdirs

from ..config import APP_NAME
from ..models import SAMPLE_RATE, Segment
from .merge import OTHERS

log = logging.getLogger("recandcc.transcribe.diarize")

MODELS_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


# 3D-Speaker CAM++ trained on a large zh+en corpus ("common_advanced"), Apache-2.0. Chosen over
# the VoxCeleb CAM++/ERes2Net, WeSpeaker ResNet34 and NeMo TitaNet models: best clustering on
# en/ru test voices, insensitive to the threshold, and the fastest of them.
EMBEDDING_MODEL = ModelFile(
    name="3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
    ),
    sha256="aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2",
    size=28_281_164,
)

# --- tuning (measured on synthetic en/ru multi-voice tracks, see the slow test) -------------
WINDOW_S = 2.0  # speech regions are cut into windows of about this length
MIN_UNIT_S = 1.0  # shorter windows are too noisy to define a speaker; they join the nearest
CLUSTER_THRESHOLD = 0.5  # cosine distance: clusters closer than this are the same speaker
MIN_SPEAKER_S = 2.0  # a "speaker" with less clustered speech than this joins the nearest one
MAX_CLUSTER_UNITS = 3000  # beyond this, cluster an even subsample (memory: n^2 floats)
# Adaptive speaker count (ADR 0013): with at least MIN_ADAPTIVE_UNITS windows to cluster, the
# number of speakers is the k <= MAX_AUTO_SPEAKERS (or the max_speakers hint) with the best
# silhouette; one speaker if no split scores MIN_SPLIT_SCORE. Fewer windows: CLUSTER_THRESHOLD.
MIN_ADAPTIVE_UNITS = 10
MAX_AUTO_SPEAKERS = 20
MIN_SPLIT_SCORE = 0.3
RETRY_AFTER_S = 600.0  # after a failed download, don't try again for this long
DOWNLOAD_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Turn:
    """A stretch of one speaker's speech (seconds from session start)."""

    start: float
    end: float
    speaker: int


class Turns(list):
    """The turns of one diarization (a ``list[Turn]``) plus each speaker's voice centroid (the
    unit mean of its windows' embeddings), which voices.py matches against voice profiles."""

    def __init__(self, turns=(), centroids: dict[int, np.ndarray] | None = None) -> None:
        super().__init__(turns)
        self.centroids = centroids or {}


# --------------------------------------------------------------------------- model files
_download_lock = threading.Lock()
_last_failure = 0.0


def model_path(model: ModelFile = EMBEDDING_MODEL, download: bool = True) -> Path | None:
    """Local path of `model`, downloading (and verifying) it once if needed. None if it is not
    on disk and can't be downloaded now. A model on disk is used without any network access."""
    global _last_failure
    path = MODELS_DIR / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        return None
    with _download_lock:
        if path.is_file() and path.stat().st_size == model.size:
            return path
        if _last_failure and time.monotonic() - _last_failure < RETRY_AFTER_S:
            return None
        try:
            _download(model, path)
        except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
            _last_failure = time.monotonic()
            log.warning("speaker diarization unavailable: can't download %s (%s)", model.name, e)
            return None
    return path


def _download(model: ModelFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part")
    log.info("downloading speaker model %s (%.0f MB, once)", model.name, model.size / 1e6)
    t0 = time.monotonic()
    if not model.url.startswith("https://"):
        raise ValueError(f"refusing non-https model URL {model.url!r}")
    digest = hashlib.sha256()
    try:
        with (
            urllib.request.urlopen(model.url, timeout=DOWNLOAD_TIMEOUT_S) as r,  # noqa: S310 - https only, checked above
            tmp.open("wb") as f,
        ):
            while block := r.read(1 << 20):
                digest.update(block)
                f.write(block)
        if digest.hexdigest() != model.sha256:
            raise ValueError(f"sha256 mismatch ({digest.hexdigest()})")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("downloaded %s in %.1fs", path, time.monotonic() - t0)


# --------------------------------------------------------------------------- pure logic
def speech_windows(
    regions: Sequence[tuple[float, float]], window_s: float = WINDOW_S
) -> list[tuple[float, float]]:
    """Cut each region into equal windows of about `window_s` (a short region stays whole)."""
    out = []
    for s, e in regions:
        if e <= s:
            continue
        n = max(1, int(round((e - s) / window_s)))
        step = (e - s) / n
        out.extend((s + i * step, s + (i + 1) * step) for i in range(n))
    return out


def cluster_embeddings(
    embeddings: np.ndarray,
    durations: np.ndarray,
    threshold: float = CLUSTER_THRESHOLD,
    min_unit_s: float = MIN_UNIT_S,
    min_speaker_s: float = MIN_SPEAKER_S,
    max_units: int = MAX_CLUSTER_UNITS,
    min_speakers: int = 0,
    max_speakers: int = 0,
) -> np.ndarray:
    """Speaker id (0..k-1) per embedding row.

    Rows with duration >= `min_unit_s` are clustered (average linkage on cosine similarity)
    into k speakers, k chosen by ``_pick_k`` (silhouette, or the distance `threshold` for few
    rows) within the optional `min_speakers` / `max_speakers` hints (0 = no hint); clusters
    with less than `min_speaker_s` of speech are dissolved (keeping at least `min_speakers`);
    then every row (short ones too) takes its nearest cluster centroid.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    x = np.array(embeddings, dtype=np.float32)
    finite = np.isfinite(x).all(axis=1)  # a NaN/inf row would break the linkage
    x[~finite] = 0.0  # not clustered; takes speaker 0
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    durations = np.asarray(durations, dtype=np.float64)
    units = np.flatnonzero((durations >= min_unit_s) & finite)
    if len(units) == 0:
        units = np.flatnonzero(finite)
    if len(units) == 0:
        return np.zeros(n, dtype=int)
    if len(units) > max_units:
        units = units[np.linspace(0, len(units) - 1, max_units).astype(int)]
    min_speakers, max_speakers = max(0, min_speakers), max(0, max_speakers)
    cap = max(MAX_AUTO_SPEAKERS, min_speakers, max_speakers)  # scores for the free estimate too
    merges, scores = _dendrogram(x[units], cap)
    k = _pick_k(len(units), merges, scores, threshold, min_speakers, max_speakers)
    groups = [units[g] for g in _cut(len(units), merges, k)]
    groups.sort(key=lambda g: -durations[g].sum())
    big = sum(durations[g].sum() >= min_speaker_s for g in groups)
    groups = groups[: max(1, big, min(min_speakers, len(groups)))]
    centroids = np.stack([x[g].mean(axis=0) for g in groups])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
    return np.argmax(x @ centroids.T, axis=1)


def _pick_k(
    n: int,
    merges: list[tuple[int, int, float]],
    scores: dict[int, float],
    threshold: float,
    min_speakers: int,
    max_speakers: int,
) -> int:
    """Number of speakers among `n` clustered rows. With at least ``MIN_ADAPTIVE_UNITS`` rows:
    the k >= 2 with the best silhouette, or 1 if no split reaches ``MIN_SPLIT_SCORE``. With
    fewer: the cut at the distance `threshold`. Either way within the hints."""
    hi = min(max_speakers or n, n)
    lo = min(max(1, min_speakers), hi)
    if n < MIN_ADAPTIVE_UNITS:
        free = n - sum(1 for *_, sim in merges if sim >= 1.0 - threshold)
        k = min(max(free, lo), hi)
    else:
        free, k = _best_k(scores, 1, n), _best_k(scores, lo, hi)
    if k != free:
        log.info(
            "speaker count hint (min %d, max %d): %d speakers instead of %d estimated",
            min_speakers,
            max_speakers,
            k,
            free,
        )
    return k


def _best_k(scores: dict[int, float], lo: int, hi: int) -> int:
    cand = {k: s for k, s in scores.items() if lo <= k <= hi}
    if not cand:
        return lo
    best = max(cand, key=lambda k: (cand[k], -k))
    return 1 if lo == 1 and cand[best] < MIN_SPLIT_SCORE else best


def _cut(n: int, merges: list[tuple[int, int, float]], k: int) -> list[np.ndarray]:
    """The k clusters left after the first ``n - k`` merges, as index arrays."""
    parent = np.arange(n)
    for i, j, _ in merges[: n - k]:
        parent[j] = i
    for _ in range(n):  # flatten: follow links until every row points at its cluster's root
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            break
        parent = nxt
    return [np.flatnonzero(parent == r) for r in np.unique(parent)]


def speaker_centroids(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Speaker id -> unit mean of its (normalised) embedding rows."""
    x = np.asarray(embeddings, dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    out = {}
    for k in np.unique(labels).tolist():
        c = x[labels == k].mean(axis=0)
        out[int(k)] = c / max(float(np.linalg.norm(c)), 1e-9)
    return out


def _dendrogram(x: np.ndarray, max_k: int) -> tuple[list[tuple[int, int, float]], dict[int, float]]:
    """Average-linkage agglomerative clustering of unit vectors (cosine similarity), run to a
    single cluster. Returns the merges ``(kept, merged, similarity)`` in order (similarity
    non-increasing) and, for every 2 <= k <= `max_k` clusters on the way, a size-weighted
    cluster silhouette: per cluster ``(b - a) / max(a, b)`` with ``a`` its mean within-cluster
    cosine distance and ``b`` the average-linkage distance to its nearest other cluster
    (0 for a single row). O(k^2) per score, from the linkage matrix it maintains anyway."""
    n = len(x)
    merges: list[tuple[int, int, float]] = []
    scores: dict[int, float] = {}
    if n < 2:
        return merges, scores
    sim = x @ x.T  # float32: 36 MB for MAX_CLUSTER_UNITS
    np.fill_diagonal(sim, -np.inf)
    size = np.ones(n, dtype=np.float64)
    within = np.zeros(n, dtype=np.float64)  # sum of within-cluster pair similarities
    alive = np.ones(n, dtype=bool)
    best = sim.argmax(axis=1)
    best_val = sim[np.arange(n), best]

    def score() -> float:
        a_idx = np.flatnonzero(alive)
        b = 1.0 - sim[np.ix_(a_idx, a_idx)].max(axis=1)
        sz = size[a_idx]
        pairs = sz * (sz - 1) / 2
        a = 1.0 - within[a_idx] / np.maximum(pairs, 1)
        s = np.where(sz > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-9), 0.0)
        return float((s * sz).sum() / sz.sum())

    for k in range(n, 1, -1):  # k clusters before this merge
        if 2 <= k <= max_k:
            scores[k] = score()
        i = int(np.argmax(best_val))
        j = int(best[i])
        if not np.isfinite(best_val[i]):  # unreachable for finite embeddings
            break
        merges.append((i, j, float(best_val[i])))
        # Lance-Williams update for average linkage: merge j into i.
        within[i] += within[j] + float(sim[i, j]) * size[i] * size[j]
        row = (sim[i] * size[i] + sim[j] * size[j]) / (size[i] + size[j])
        sim[i], sim[:, i] = row, row
        sim[i, i] = -np.inf
        sim[j], sim[:, j] = -np.inf, -np.inf
        size[i] += size[j]
        alive[j] = False
        best_val[j] = -np.inf
        # rows whose best partner was i or j must be recomputed; others may now prefer i
        stale = np.flatnonzero((best == i) | (best == j))
        for r in [i, *stale.tolist()]:
            if alive[r]:
                best[r] = int(np.argmax(sim[r]))
                best_val[r] = sim[r, best[r]]
        better = sim[:, i] > best_val
        best[better], best_val[better] = i, sim[better, i]
    return merges, scores


def assign_speakers(
    segments: Sequence[Segment],
    turns: Sequence[Turn],
    speaker: str = OTHERS,
    names: dict[int, str] | None = None,
) -> list[Segment]:
    """Relabel the `speaker` ("Others") segments with the diarized speaker they overlap most
    (the nearest turn if none overlaps), numbered "Speaker 1", "Speaker 2", ... by first
    appearance. If they all map to one speaker (or there are no turns), nothing changes.
    `names`: speaker id -> name (matched voice profiles), used instead of a number."""
    return label_speakers(segments, turns, speaker, names)[0]


def label_speakers(
    segments: Sequence[Segment],
    turns: Sequence[Turn],
    speaker: str = OTHERS,
    names: dict[int, str] | None = None,
) -> tuple[list[Segment], dict[int, str]]:
    """``assign_speakers`` plus the label given to each speaker id that got segments (a lone
    unnamed speaker keeps `speaker`)."""
    segments = list(segments)
    if not turns:
        return segments, {}
    names = names or {}
    turns = sorted(turns, key=lambda t: (t.start, t.end))
    starts = [t.start for t in turns]
    longest = max(t.end - t.start for t in turns)
    picks: dict[int, int] = {}
    for idx, seg in enumerate(segments):
        if seg.speaker == speaker:
            picks[idx] = _best_speaker(seg, turns, starts, longest)
    ids = set(picks.values())
    if not ids:
        return segments, {}
    if len(ids) == 1 and not ids & names.keys():
        return segments, {ids.pop(): speaker}
    taken = {n.casefold() for n in names.values()}
    numbers = (f"Speaker {n}" for n in itertools.count(1))
    free = (label for label in numbers if label.casefold() not in taken)  # no duplicate labels
    labels: dict[int, str] = {}
    for idx in sorted(picks, key=lambda i: (segments[i].start, segments[i].end, i)):
        k = picks[idx]
        if k not in labels:
            labels[k] = names[k] if k in names else next(free)
    out = [
        replace(seg, speaker=labels[picks[i]]) if i in picks else seg
        for i, seg in enumerate(segments)
    ]
    return out, labels


def _best_speaker(seg: Segment, turns: Sequence[Turn], starts: list[float], longest: float) -> int:
    lo = bisect.bisect_left(starts, seg.start - longest)  # earlier turns end before seg
    hi = bisect.bisect_left(starts, seg.end)  # turns[hi:] start at/after the segment's end
    overlap: dict[int, float] = defaultdict(float)
    for t in turns[lo:hi]:
        o = min(seg.end, t.end) - max(seg.start, t.start)
        if o > 0:
            overlap[t.speaker] += o
    if overlap:
        return max(overlap, key=lambda k: (overlap[k], -k))
    # no overlap: the turn nearest in time
    near = turns[max(0, lo - 1) : hi + 1]
    return min(near, key=lambda t: max(t.start - seg.end, seg.start - t.end)).speaker


# --------------------------------------------------------------------------- runtime
def default_threads(cores: int | None = None) -> int:
    cores = cores or os.cpu_count() or 4
    return max(1, min(6, cores // 2))


VadFn = Callable[[np.ndarray], list[tuple[float, float]]]


def _silero(audio: np.ndarray) -> list[tuple[float, float]]:
    from .chunking import find_speech

    return find_speech(audio)


class Diarizer:
    """Loads the embedding model lazily; runs one diarization at a time on its own thread
    (``submit``), embedding windows on `threads` worker threads. Thread-safe; reusable after
    ``close``."""

    def __init__(
        self, threads: int = 0, vad: VadFn = _silero, min_speakers: int = 0, max_speakers: int = 0
    ) -> None:
        self.threads = threads or default_threads()
        self.vad = vad
        self.min_speakers, self.max_speakers = min_speakers, max_speakers  # 0 = no hint
        self._lock = threading.Lock()
        self._extractor = None
        self._pool: ThreadPoolExecutor | None = None  # embedding workers
        self._jobs: ThreadPoolExecutor | None = None  # one diarization at a time
        self.load_s: float | None = None  # last model load time, until a job reports it

    def _load(self) -> tuple[Any, ThreadPoolExecutor] | None:
        with self._lock:
            if self._extractor is None:
                path = model_path()
                if path is None:
                    return None
                import sherpa_onnx

                t0 = time.monotonic()
                cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(path), num_threads=1)
                self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
                self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="diarize-emb")
                self.load_s = time.monotonic() - t0
                log.info(
                    "loaded speaker model %s in %.2fs (%d threads)",
                    path.name,
                    self.load_s,
                    self.threads,
                )
            if self._pool is None:  # unreachable: set and cleared together with the extractor
                return None
            return self._extractor, self._pool

    def diarize(self, audio: np.ndarray) -> list[Turn] | None:
        """Speaker turns of a 16 kHz mono track; None if diarization is unavailable."""
        t0 = time.monotonic()
        loaded = self._load()
        if loaded is None:
            return None
        extractor, pool = loaded
        regions = self.vad(audio)
        t_vad = time.monotonic() - t0
        windows = speech_windows(regions)
        if not windows:
            return []

        def embed(window: tuple[float, float]) -> np.ndarray:
            stream = extractor.create_stream()
            s, e = window
            stream.accept_waveform(SAMPLE_RATE, audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)])
            stream.input_finished()
            return np.asarray(extractor.compute(stream), dtype=np.float32)

        embeddings = np.stack(list(pool.map(embed, windows)))
        labels = cluster_embeddings(
            embeddings,
            np.array([e - s for s, e in windows]),
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        turns = Turns(centroids=speaker_centroids(embeddings, labels))
        turns += [
            Turn(round(s, 3), round(e, 3), int(k))
            for (s, e), k in zip(windows, labels, strict=True)
        ]
        log.info(
            "diarized %.0fs of audio: %d windows -> %d speakers in %.2fs (vad %.2fs)",
            len(audio) / SAMPLE_RATE,
            len(windows),
            len(set(labels.tolist())),
            time.monotonic() - t0,
            t_vad,
        )
        return turns

    def diarize_file(self, path: Path) -> list[Turn] | None:
        from .chunking import load_track

        return self.diarize(load_track(Path(path)))

    def submit(self, path: Path, timing: dict | None = None) -> Future:
        """Diarize the WAV at `path` in the background. The future's result is the turns, or
        None when unavailable or failed; it never raises (errors are logged). When it ran,
        `timing` gets ``elapsed_s`` (and ``model_load_s`` if the model was loaded for it)
        before the result is set."""
        with self._lock:
            if self._jobs is None:
                self._jobs = ThreadPoolExecutor(1, thread_name_prefix="diarize")
            return self._jobs.submit(self._safe, Path(path), timing)

    def _safe(self, path: Path, timing: dict | None = None) -> list[Turn] | None:
        try:
            if not path.exists():
                log.info("no audio to diarize (%s was deleted)", path.name)
                return None
            t0 = time.monotonic()
            turns = self.diarize_file(path)
            if timing is not None and turns is not None:
                timing["elapsed_s"] = round(time.monotonic() - t0, 2)
                load, self.load_s = self.load_s, None
                if load is not None:
                    timing["model_load_s"] = round(load, 2)
            return turns
        except Exception as e:  # noqa: BLE001 - diarization is optional
            log.warning("speaker diarization of %s failed: %r", path, e)
            return None

    def close(self, wait: bool = True) -> None:
        """Release the model and threads. `wait`: let a running diarization (and the
        transcript rewrite that follows it) finish first; else drop it."""
        with self._lock:
            jobs, pool = self._jobs, self._pool
            self._jobs = self._pool = self._extractor = None
        if jobs is not None:
            jobs.shutdown(wait=wait, cancel_futures=not wait)
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=not wait)


__all__ = [
    "EMBEDDING_MODEL",
    "MODELS_DIR",
    "Diarizer",
    "Turn",
    "Turns",
    "assign_speakers",
    "cluster_embeddings",
    "label_speakers",
    "model_path",
    "speaker_centroids",
    "speech_windows",
]
