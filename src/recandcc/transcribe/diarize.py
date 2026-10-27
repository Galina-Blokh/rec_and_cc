"""Offline speaker diarization of the "Others" track: which of the other participants spoke when.

Pipeline (CPU only; ~2 s for 13 min of audio on an M5 Max, see docs/ARCHITECTURE.md)::

    others.wav -> Silero VAD speech regions (the VAD already used for chunking)
               -> (optional, ADR 0014) pyannote segmentation-3.0 (6 MB ONNX) on 10 s chunks:
                  speaker-change points cut the regions, overlapped speech is marked
               -> regions (or their pieces) cut into ~2 s windows
               -> one speaker embedding per window: 3D-Speaker CAM++ (zh+en "advanced",
                  28 MB ONNX) through sherpa-onnx, on a small thread pool (ONNX Runtime
                  releases the GIL, so windows are embedded in parallel)
               -> agglomerative clustering (average linkage, cosine distance) of the windows
                  >= 1 s into k speakers, k picked by silhouette (optional min/max hints);
                  shorter windows, overlapped windows and tiny clusters join the nearest speaker
               -> turns (start, end, speaker id)

``assign_speakers`` then gives every "Others" transcript segment the speaker it overlaps
most, numbered "Speaker 1", "Speaker 2", ... by first appearance. With a single speaker the
label stays "Others". Segments with overlapped speech get ``overlap=True``. With
``speaker_names`` on, speakers that match a voice profile (voices.py) are named instead of
numbered.

The models are downloaded once (sha256-verified) into the platform cache dir
(``MODELS_DIR``); after that nothing here touches the network. If the download fails (offline)
diarization is skipped and retried at most every ``RETRY_AFTER_S``.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import logging
import os
import shutil
import tarfile
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
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
    sha256: str  # of the download (the archive if `member` is set)
    size: int  # of the file on disk
    member: str = ""  # file to extract from a .tar.bz2 download


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

# pyannote segmentation-3.0 (MIT) as converted to ONNX for sherpa-onnx: 10 s chunk -> per
# ~17 ms frame one of 7 powerset classes (silence, 3 local speakers, their 3 pairs).
SEGMENTATION_MODEL = ModelFile(
    name="pyannote-segmentation-3.0.onnx",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
        "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    ),
    sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    size=5_992_913,
    member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
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
# Segmentation (ADR 0014): chunks every SEG_STEP_S; each chunk is trusted for the middle
# SEG_STEP_S of it (its edges see too little context). Class runs shorter than MIN_RUN_S are
# jitter; a cut never leaves a piece shorter than MIN_PIECE_S; a segment is marked overlapped
# with at least OVERLAP_MARK_S of overlapped speech.
SEG_STEP_S = 5.0
SEG_BATCH = 16
MIN_RUN_S = 0.25
MIN_PIECE_S = 0.4
OVERLAP_MARK_S = 0.5
RETRY_AFTER_S = 600.0  # after a failed download, don't try again for this long
DOWNLOAD_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class Turn:
    """A stretch of one speaker's speech (seconds from session start)."""

    start: float
    end: float
    speaker: int
    overlap: bool = False  # two speakers at once; `speaker` is the dominant one


class Turns(list):
    """The turns of one diarization (a ``list[Turn]``) plus each speaker's voice centroid (the
    unit mean of its windows' embeddings), which voices.py matches against voice profiles."""

    def __init__(self, turns=(), centroids: dict[int, np.ndarray] | None = None) -> None:
        super().__init__(turns)
        self.centroids = centroids or {}


# --------------------------------------------------------------------------- model files
_download_lock = threading.Lock()
_last_failure: dict[str, float] = {}  # model name -> time of its last failed download


def model_path(model: ModelFile = EMBEDDING_MODEL, download: bool = True) -> Path | None:
    """Local path of `model`, downloading (and verifying) it once if needed. None if it is not
    on disk and can't be downloaded now. A model on disk is used without any network access."""
    path = MODELS_DIR / model.name
    if path.is_file() and path.stat().st_size == model.size:
        return path
    if not download:
        return None
    with _download_lock:
        if path.is_file() and path.stat().st_size == model.size:
            return path
        failed = _last_failure.get(model.name)
        if failed and time.monotonic() - failed < RETRY_AFTER_S:
            return None
        try:
            _download(model, path)
        except Exception as e:  # noqa: BLE001 - offline, proxy, disk full, bad checksum ...
            _last_failure[model.name] = time.monotonic()
            log.warning("can't download speaker model %s (%s)", model.name, e)
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
        if model.member:
            _extract(tmp, model, path)
        else:
            tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    log.info("downloaded %s in %.1fs", path, time.monotonic() - t0)


def _extract(archive: Path, model: ModelFile, path: Path) -> None:
    """Copy `model.member` out of the verified `archive` to `path` (only that file)."""
    out = path.with_name(f".{path.name}.x")
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            src = tar.extractfile(tar.getmember(model.member))  # KeyError if missing
            if src is None:
                raise ValueError(f"{model.member} is not a file")
            with src, out.open("wb") as f:
                shutil.copyfileobj(src, f)
        if out.stat().st_size != model.size:
            raise ValueError(f"{model.member}: unexpected size {out.stat().st_size}")
        out.replace(path)
    finally:
        out.unlink(missing_ok=True)


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


def chunk_events(
    classes: np.ndarray, times: np.ndarray, lo: float, hi: float, n_local: int = 3
) -> tuple[list[float], list[tuple[float, float]]]:
    """Speaker-change points and overlapped stretches in [lo, hi) of one segmentation chunk.

    `classes`: powerset class per frame (0 silence, 1..n_local one local speaker, above that
    two at once); `times`: frame centres in seconds. Runs shorter than ``MIN_RUN_S`` and silence
    are skipped; a change is where the next kept run has another class (between the two runs).
    """
    if len(classes) == 0:
        return [], []
    frame_s = float(times[1] - times[0]) if len(times) > 1 else 0.0
    edges = np.flatnonzero(np.diff(classes)) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(classes)]
    min_frames = MIN_RUN_S / frame_s if frame_s else 0
    runs = [
        (int(classes[a]), float(times[a]) - frame_s / 2, float(times[b - 1]) + frame_s / 2)
        for a, b in zip(starts, ends, strict=True)
        if classes[a] != 0 and b - a >= min_frames
    ]
    changes = [
        t
        for (c0, _, e0), (c1, s1, _) in itertools.pairwise(runs)
        if c0 != c1 and lo <= (t := (e0 + s1) / 2) < hi
    ]
    overlaps = [
        (max(s, lo), min(e, hi)) for c, s, e in runs if c > n_local and min(e, hi) > max(s, lo)
    ]
    return changes, overlaps


def cut_regions(
    regions: Sequence[tuple[float, float]],
    changes: Sequence[float],
    min_piece_s: float = MIN_PIECE_S,
) -> list[tuple[float, float]]:
    """Cut each region at the change points inside it; a cut that would leave a piece shorter
    than `min_piece_s` is skipped."""
    changes = sorted(changes)
    out = []
    for s, e in regions:
        a = s
        for t in changes[bisect.bisect_right(changes, s) : bisect.bisect_left(changes, e)]:
            if t - a >= min_piece_s and e - t >= min_piece_s:
                out.append((a, t))
                a = t
        out.append((a, e))
    return out


def overlap_flags(
    windows: Sequence[tuple[float, float]], overlaps: Sequence[tuple[float, float]]
) -> list[bool]:
    """Per window: is at least half of it overlapped speech?"""
    return [_covered(s, e, overlaps) >= (e - s) / 2 > 0 for s, e in windows]


def _covered(s: float, e: float, spans: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, min(e, b) - max(s, a)) for a, b in spans)


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
    reserved: Collection[str] = (),
) -> tuple[list[Segment], dict[int, str]]:
    """``assign_speakers`` plus the label given to each speaker id that got segments (a lone
    unnamed speaker keeps `speaker`). Segments with at least ``OVERLAP_MARK_S`` of overlapped
    turns get ``overlap=True`` (their label is still the speaker they overlap most).
    `reserved`: labels already given on another track, skipped when numbering so two tracks
    never share a "Speaker N"."""
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
    spans = [(t.start, t.end) for t in turns if t.overlap]
    for i in picks if spans else ():  # turns don't overlap each other: no double counting
        if _covered(segments[i].start, segments[i].end, spans) >= OVERLAP_MARK_S:
            segments[i] = replace(segments[i], overlap=True)
    if len(ids) == 1 and not ids & names.keys():
        return segments, {ids.pop(): speaker}
    taken = {n.casefold() for n in (*names.values(), *reserved)}
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


class Segmenter:
    """pyannote segmentation-3.0 over the speech of a track: speaker-change points and
    overlapped stretches (seconds). ONNX Runtime on `threads` intra-op threads."""

    def __init__(self, path: Path, threads: int) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.window = int(meta["window_size"])  # samples per chunk (10 s)
        self.shift = int(meta["receptive_field_shift"])  # samples per output frame
        self.offset = int(meta["receptive_field_size"]) / 2  # first frame's centre
        self.n_local = int(meta["num_speakers"])

    def __call__(
        self, audio: np.ndarray, regions: Sequence[tuple[float, float]]
    ) -> tuple[list[float], list[tuple[float, float]]]:
        win, step = self.window, int(SEG_STEP_S * SAMPLE_RATE)
        margin = (win - step) // 2  # each chunk is trusted for its middle `step` samples
        spans = [(int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)) for s, e in regions]
        last = max(0, -(-(len(audio) - win) // step)) * step  # chunk reaching the end
        starts = [  # chunks containing speech
            c for c in range(0, last + 1, step) if any(s < c + win and e > c for s, e in spans)
        ]
        changes: list[float] = []
        overlaps: list[tuple[float, float]] = []
        for b in range(0, len(starts), SEG_BATCH):
            batch = starts[b : b + SEG_BATCH]
            x = np.zeros((len(batch), 1, win), dtype=np.float32)
            for row, c in enumerate(batch):
                piece = audio[c : c + win]
                x[row, 0, : len(piece)] = piece
            y = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: x})[0])
            for c, scores in zip(batch, y, strict=True):
                times = (c + self.offset + self.shift * np.arange(len(scores))) / SAMPLE_RATE
                lo = (c + margin) / SAMPLE_RATE if c else -np.inf
                hi = (c + margin + step) / SAMPLE_RATE if c < last else np.inf
                ch, ov = chunk_events(np.argmax(scores, axis=-1), times, lo, hi, self.n_local)
                changes += ch
                overlaps += ov
        return changes, overlaps


class Diarizer:
    """Loads the models lazily; runs one diarization at a time on its own thread
    (``submit``), embedding windows on `threads` worker threads. Thread-safe; reusable after
    ``close``. `segmentation`: cut windows at speaker changes and mark overlapped speech
    (``Segmenter``); without its model, fixed ~2 s windows as before."""

    def __init__(
        self,
        threads: int = 0,
        vad: VadFn = _silero,
        min_speakers: int = 0,
        max_speakers: int = 0,
        segmentation: bool = False,
    ) -> None:
        self.threads = threads or default_threads()
        self.vad = vad
        self.min_speakers, self.max_speakers = min_speakers, max_speakers  # 0 = no hint
        self.segmentation = segmentation
        self._lock = threading.Lock()
        self._extractor = None
        self._segmenter: Segmenter | None = None
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

    def _load_segmenter(self) -> Segmenter | None:
        with self._lock:
            if not self.segmentation or self._segmenter is not None:
                return self._segmenter if self.segmentation else None
        path = model_path(SEGMENTATION_MODEL)  # may download: outside the lock
        if path is None:
            log.info("no segmentation model: speaker turns from fixed ~2 s windows")
            return None
        t0 = time.monotonic()
        try:
            segmenter = Segmenter(path, self.threads)
        except Exception as e:  # noqa: BLE001 - corrupt file, missing metadata, ORT error
            self._segmentation_failed("load", e)
            return None
        with self._lock:
            if self._segmenter is None:
                self._segmenter = segmenter
                self.load_s = (self.load_s or 0.0) + time.monotonic() - t0
            return self._segmenter

    def _segment(
        self, audio: np.ndarray, regions: list[tuple[float, float]]
    ) -> tuple[list[float], list[tuple[float, float]]]:
        segmenter = self._load_segmenter() if regions else None
        if segmenter is None:
            return [], []
        try:
            return segmenter(audio, regions)
        except Exception as e:  # noqa: BLE001 - optional: fixed windows still work
            self._segmentation_failed("run", e)
            return [], []

    def _segmentation_failed(self, what: str, e: Exception) -> None:
        """A broken segmentation model must not cost the speaker labels: fixed windows from now
        on for this Diarizer."""
        log.warning(
            "segmentation model failed to %s (%s); speaker turns from fixed ~2 s windows",
            what,
            type(e).__name__,
        )
        with self._lock:
            self.segmentation, self._segmenter = False, None

    def diarize(self, audio: np.ndarray) -> list[Turn] | None:
        """Speaker turns of a 16 kHz mono track; None if diarization is unavailable."""
        t0 = time.monotonic()
        loaded = self._load()
        if loaded is None:
            return None
        extractor, pool = loaded
        regions = self.vad(audio)
        t_vad = time.monotonic() - t0
        changes, overlaps = self._segment(audio, regions)
        t_seg = time.monotonic() - t0 - t_vad
        windows = speech_windows(cut_regions(regions, changes))
        if not windows:
            return []
        flags = overlap_flags(windows, overlaps)

        def embed(window: tuple[float, float]) -> np.ndarray:
            stream = extractor.create_stream()
            s, e = window
            stream.accept_waveform(SAMPLE_RATE, audio[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)])
            stream.input_finished()
            return np.asarray(extractor.compute(stream), dtype=np.float32)

        embeddings = np.stack(list(pool.map(embed, windows)))
        mixed = np.array(flags, dtype=bool)
        # overlapped windows mix two voices: they don't shape the clusters (duration 0 = not a
        # clustering unit) or the voice centroids, and take the nearest (dominant) speaker
        labels = cluster_embeddings(
            embeddings,
            np.where(mixed, 0.0, [e - s for s, e in windows]),
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )
        clean = ~mixed if not mixed.all() else mixed
        turns = Turns(centroids=speaker_centroids(embeddings[clean], labels[clean]))
        turns += [
            Turn(round(s, 3), round(e, 3), int(k), overlap)
            for (s, e), k, overlap in zip(windows, labels, flags, strict=True)
        ]
        log.info(
            "diarized %.0fs of audio: %d windows (%d overlapped) -> %d speakers in %.2fs "
            "(vad %.2fs, segmentation %.2fs)",
            len(audio) / SAMPLE_RATE,
            len(windows),
            int(mixed.sum()),
            len(set(labels.tolist())),
            time.monotonic() - t0,
            t_vad,
            t_seg,
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
            self._jobs = self._pool = self._extractor = self._segmenter = None
        if jobs is not None:
            jobs.shutdown(wait=wait, cancel_futures=not wait)
        if pool is not None:
            pool.shutdown(wait=wait, cancel_futures=not wait)


__all__ = [
    "EMBEDDING_MODEL",
    "MODELS_DIR",
    "SEGMENTATION_MODEL",
    "Diarizer",
    "Segmenter",
    "Turn",
    "Turns",
    "assign_speakers",
    "chunk_events",
    "cluster_embeddings",
    "cut_regions",
    "label_speakers",
    "model_path",
    "overlap_flags",
    "speaker_centroids",
    "speech_windows",
]
