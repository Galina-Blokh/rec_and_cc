"""Speaker diarization: label assignment, clustering, model download and both final paths
(batch + live) with fake diarization results. No model needed.

Stories: US-12, US-14.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pytest
import soundfile as sf
from test_live import FakeTranscriber, Feeder, burst, make_session, silence
from test_transcribe import make_session as make_batch_session

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track
from recandcc.transcribe import diarize
from recandcc.transcribe.diarize import (
    Diarizer,
    ModelFile,
    Segmenter,
    Turn,
    assign_speakers,
    chunk_events,
    cluster_embeddings,
    cut_regions,
    label_speakers,
    overlap_flags,
    speech_windows,
)
from recandcc.transcribe.diarize import model_path as real_model_path
from recandcc.transcribe.engine import Transcriber
from recandcc.transcribe.live import LiveTranscriber


def seg(start, end, speaker="Others", text="x"):
    return Segment(start, end, text, speaker, "en")


def speakers(segments):
    return [s.speaker for s in segments]


# --------------------------------------------------------------------------- assign_speakers
def test_speakers_numbered_by_first_appearance():
    segs = [seg(0, 2), seg(3, 5), seg(6, 8), seg(9, 11)]
    turns = [Turn(0, 2.5, 7), Turn(2.5, 5.5, 2), Turn(5.5, 8.5, 7), Turn(8.5, 12, 4)]
    assert speakers(assign_speakers(segs, turns)) == [
        "Speaker 1",
        "Speaker 2",
        "Speaker 1",
        "Speaker 3",
    ]


def test_me_is_never_relabelled_and_order_is_kept():
    segs = [seg(0, 2, "Me"), seg(1, 3), seg(4, 6, "Me"), seg(5, 7)]
    turns = [Turn(0, 3.5, 0), Turn(4.5, 8, 1)]
    out = assign_speakers(segs, turns)
    assert speakers(out) == ["Me", "Speaker 1", "Me", "Speaker 2"]
    assert [(s.start, s.text) for s in out] == [(s.start, s.text) for s in segs]
    assert speakers(segs) == ["Me", "Others", "Me", "Others"]  # input untouched


def test_single_speaker_keeps_others():
    segs = [seg(0, 2), seg(3, 5), seg(1, 2, "Me")]
    turns = [Turn(0, 2, 3), Turn(3, 5, 3)]
    assert speakers(assign_speakers(segs, turns)) == ["Others", "Others", "Me"]


def test_no_turns_keeps_others():
    segs = [seg(0, 2), seg(3, 5)]
    assert assign_speakers(segs, []) == segs


def test_segment_spanning_a_change_takes_the_majority():
    segs = [seg(0, 10), seg(11, 12)]
    turns = [Turn(0, 3, 0), Turn(3, 10, 1), Turn(10.5, 12, 0)]
    # 0-10 overlaps speaker 0 for 3 s and speaker 1 for 7 s -> speaker 1 (first seen)
    assert speakers(assign_speakers(segs, turns)) == ["Speaker 1", "Speaker 2"]


def test_segment_without_overlap_takes_the_nearest_turn():
    segs = [seg(0, 2), seg(2.6, 3.0), seg(10, 12)]
    turns = [Turn(0, 2, 0), Turn(3.2, 5, 1), Turn(10, 12, 1)]
    # 2.6-3.0: 0.6 s after speaker 0, 0.2 s before speaker 1
    assert speakers(assign_speakers(segs, turns)) == ["Speaker 1", "Speaker 2", "Speaker 2"]
    far = [seg(20, 21), seg(0, 1)]
    assert speakers(assign_speakers(far, turns)) == ["Speaker 2", "Speaker 1"]


def test_overlapping_and_unsorted_turns():
    segs = [seg(0, 4), seg(5, 9)]
    turns = [Turn(5, 9, 1), Turn(0, 9, 0), Turn(0, 4, 1)]
    # 0-4: speaker 0 for 4 s, speaker 1 for 4 s -> tie goes to the lower id (0)
    # 5-9: both 4 s again -> 0  => one speaker, stays Others
    assert speakers(assign_speakers(segs, turns)) == ["Others", "Others"]


# --------------------------------------------------------------------------- clustering
def _voices(rng, n_voices, per_voice, dim=64, noise=0.35):
    centres = rng.normal(size=(n_voices, dim))
    labels = np.repeat(np.arange(n_voices), per_voice)
    rng.shuffle(labels)
    emb = centres[labels] + noise * rng.normal(size=(len(labels), dim))
    return emb, labels


def _same_partition(a, b):
    pairs = {(int(x), int(y)) for x, y in zip(a, b, strict=True)}
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


@pytest.mark.parametrize("n_voices", [1, 2, 3, 5])
def test_cluster_embeddings_recovers_voices(n_voices):
    rng = np.random.default_rng(n_voices)
    emb, truth = _voices(rng, n_voices, 20)
    got = cluster_embeddings(emb, np.full(len(emb), 2.0))
    assert _same_partition(got, truth)


def test_short_units_join_the_nearest_speaker_and_tiny_clusters_dissolve():
    rng = np.random.default_rng(0)
    emb, truth = _voices(rng, 2, 10)
    dur = np.full(len(emb), 2.0)
    # a short unit of a third voice can't found a speaker of its own
    odd = rng.normal(size=(1, emb.shape[1])) * 3
    got = cluster_embeddings(np.vstack([emb, odd]), np.append(dur, 0.5))
    assert len(set(got.tolist())) == 2 and _same_partition(got[:-1], truth)
    # nor can a long but lone (1.5 s < MIN_SPEAKER_S) one
    got = cluster_embeddings(np.vstack([emb, odd]), np.append(dur, 1.5))
    assert len(set(got.tolist())) == 2


def test_cluster_embeddings_edge_cases():
    assert cluster_embeddings(np.zeros((0, 8)), np.zeros(0)).tolist() == []
    one = cluster_embeddings(np.ones((1, 8)), np.array([0.4]))  # only short units
    assert one.tolist() == [0]


def test_cluster_subsample_for_huge_inputs():
    rng = np.random.default_rng(3)
    emb, truth = _voices(rng, 3, 100)
    got = cluster_embeddings(emb, np.full(len(emb), 2.0), max_units=60)
    assert _same_partition(got, truth)


# --------------------------------------------------------------------------- adaptive count
def _k(emb, **hints):
    return len(set(cluster_embeddings(emb, np.full(len(emb), 2.0), **hints).tolist()))


@pytest.mark.parametrize("n_voices", [2, 4, 8, 12])
def test_adaptive_count_recovers_many_voices(n_voices):
    rng = np.random.default_rng(10 + n_voices)
    emb, truth = _voices(rng, n_voices, 20, dim=192, noise=0.8)
    assert _same_partition(cluster_embeddings(emb, np.full(len(emb), 2.0)), truth)


def test_one_noisy_speaker_stays_one():
    # within-speaker cosine ~0.5: the fixed threshold alone split this into 4-27 "speakers"
    for s in range(5):
        emb, _ = _voices(np.random.default_rng(s), 1, 60, noise=1.0)
        assert _k(emb) == 1


def _close_pair(rng, cos, per=30, dim=64):
    base, other = rng.normal(size=(2, dim))
    base /= np.linalg.norm(base)
    other -= other @ base * base
    other /= np.linalg.norm(other)
    centres = 8 * np.stack([base, cos * base + np.sqrt(1 - cos**2) * other])
    labels = np.repeat([0, 1], per)
    return centres[labels] + 0.35 * rng.normal(size=(len(labels), dim)), labels


def test_close_but_distinct_voices_separate():
    # windows of the two voices are ~0.55 cosine apart: within the fixed threshold (merged)
    for s in range(5):
        emb, truth = _close_pair(np.random.default_rng(s), cos=0.65)
        assert _same_partition(cluster_embeddings(emb, np.full(len(emb), 2.0)), truth)


def test_speaker_hints_are_respected():
    emb, truth = _voices(np.random.default_rng(7), 6, 15)
    assert _k(emb) == 6
    assert _k(emb, max_speakers=3) == 3
    assert _k(emb, min_speakers=8) >= 8
    assert _same_partition(cluster_embeddings(emb, np.full(len(emb), 2.0), min_speakers=6), truth)
    one, _ = _voices(np.random.default_rng(8), 1, 40)
    assert _k(one, min_speakers=2) == 2
    assert _k(one, min_speakers=3, max_speakers=2) == 2  # max wins
    assert _k(emb[:4], max_speakers=1) == 1  # few windows: threshold cut, still clamped


def test_hint_override_is_logged_with_counts_only(caplog):
    emb, _ = _voices(np.random.default_rng(7), 6, 15)
    with caplog.at_level("INFO", logger="recandcc.transcribe.diarize"):
        _k(emb)
        assert not caplog.records
        _k(emb, max_speakers=3)
    assert caplog.messages == [
        "speaker count hint (min 0, max 3): 3 speakers instead of 6 estimated"
    ]


def test_non_finite_embeddings_do_not_break_clustering():
    emb, truth = _voices(np.random.default_rng(12), 3, 15)
    emb[[0, 5]] = np.nan
    emb[9, 3] = np.inf
    got = cluster_embeddings(emb, np.full(len(emb), 2.0), max_speakers=3)
    ok = np.isfinite(emb).all(axis=1)
    assert len(set(got.tolist())) <= 3 and _same_partition(got[ok], truth[ok])
    assert cluster_embeddings(np.full((3, 8), np.nan), np.full(3, 2.0)).tolist() == [0, 0, 0]


def test_few_windows_keep_the_threshold_behaviour():
    emb, truth = _voices(np.random.default_rng(9), 2, 4)  # 8 < MIN_ADAPTIVE_UNITS
    assert _same_partition(cluster_embeddings(emb, np.full(len(emb), 2.0)), truth)
    assert _k(emb, threshold=1.5) == 1  # the threshold still decides


def test_adaptive_clustering_of_an_hour_takes_seconds():
    # an hour of speech is ~1800 two-second windows; MAX_CLUSTER_UNITS bounds bigger inputs
    emb, truth = _voices(np.random.default_rng(11), 12, 500, dim=192, noise=0.8)
    t0 = time.perf_counter()
    got = cluster_embeddings(emb, np.full(len(emb), 2.0))
    assert time.perf_counter() - t0 < 5.0  # ~0.15 s on an M5 Max
    assert _same_partition(got, truth)


def test_speech_windows():
    assert speech_windows([(0, 1.2), (2, 6), (7, 7)]) == [(0, 1.2), (2, 4), (4, 6)]
    w = speech_windows([(10, 15)])
    assert len(w) == 2 and w[0] == (10, 12.5) and w[-1][1] == 15


# --------------------------------------------------------------------------- segmentation
def test_chunk_events_finds_changes_and_overlaps():
    # 0.1 s frames: A 1 s, jitter of C (0.1 s), A, silence, B, A+B overlap, B, short C
    classes = np.array([1] * 10 + [3] + [1] * 5 + [0] * 3 + [2] * 10 + [4] * 6 + [2] * 4 + [3])
    times = 0.05 + 0.1 * np.arange(len(classes))
    changes, overlaps = chunk_events(classes, times, -np.inf, np.inf)
    assert changes == pytest.approx([1.75, 2.9, 3.5])  # A|B mid-pause, B|AB, AB|B; no C
    assert overlaps == [pytest.approx((2.9, 3.5))]
    # only [lo, hi) of the chunk is used
    changes, overlaps = chunk_events(classes, times, 2.0, 3.2)
    assert changes == pytest.approx([2.9]) and overlaps == [pytest.approx((2.9, 3.2))]
    assert chunk_events(np.zeros(0, int), np.zeros(0), 0, 1) == ([], [])


def test_cut_regions_at_changes_but_not_into_slivers():
    regions = [(0.0, 4.0), (5.0, 9.0)]
    assert cut_regions(regions, [1.5, 4.5, 8.8, 6.0, 6.2]) == [
        (0.0, 1.5),
        (1.5, 4.0),
        (5.0, 6.0),  # 6.2 would leave a 0.2 s piece, 8.8 one at the end
        (6.0, 9.0),
    ]
    assert cut_regions(regions, []) == regions


def test_overlap_flags_need_half_the_window():
    flags = overlap_flags([(0, 2), (2, 4), (4, 4)], [(1.2, 2.5), (3.5, 3.6)])
    assert flags == [False, False, False]
    assert overlap_flags([(0, 2), (2, 3)], [(0.5, 1.0), (1.0, 1.6), (2.0, 3.0)]) == [True, True]


def test_segmenter_chunks_count_every_event_once():
    """Chunks every 5 s overlap; each instant is read from one chunk only."""
    change, ov = 7.45, (12.3, 13.3)  # next to the 7.5 s / 12.5 s hand-over points

    class FakeSession:
        batches = []

        def get_inputs(self):
            return [type("I", (), {"name": "x"})]

        def run(self, _out, feeds):
            x = feeds["x"]
            self.batches.append(len(x))
            t = (x[:, 0, :1] + seg.offset + seg.shift * np.arange(589)) / SAMPLE_RATE  # (N, 589)
            cls = np.where(t < change, 1, 2) + 2 * ((t >= ov[0]) & (t < ov[1]))
            return [np.eye(7)[cls]]

    seg = Segmenter.__new__(Segmenter)
    seg.session, seg.window, seg.shift, seg.offset, seg.n_local = FakeSession(), 160000, 270, 495, 3
    audio = np.arange(60 * SAMPLE_RATE, dtype=np.float32)  # sample value = its index
    changes, overlaps = seg(audio, [(0.0, 30.0)])
    assert changes == pytest.approx([change, ov[0], ov[1]], abs=0.03)
    assert sum(e - s for s, e in overlaps) == pytest.approx(1.0, abs=0.03)
    assert FakeSession.batches == [6]  # chunks at 0..25 s; 30..50 s hold no speech


class FakeExtractor:
    """Embedding = direction set by the window's mean sample: 1.0 voice A, 2.0 voice B."""

    def create_stream(self):
        return type("S", (), {"accept_waveform": _accept, "input_finished": lambda self: None})()

    def compute(self, stream):
        theta = (float(np.mean(stream.audio)) - 1.0) * np.pi / 2
        return [np.cos(theta), np.sin(theta), 0.0]


def _accept(stream, sr, audio):
    stream.audio = audio


def _voices_track():
    """A 0-5.3 s, B 5.3-12 s (no pause), A+B 12-13 s (A louder), A 13-16 s."""
    level = np.r_[np.full(53, 1.0), np.full(67, 2.0), np.full(10, 1.3), np.full(30, 1.0)]
    return np.repeat(level, SAMPLE_RATE // 10).astype(np.float32)


def _fake_diarizer(monkeypatch, segmenter=None, **kwargs) -> Diarizer:
    d = Diarizer(threads=1, vad=lambda a: [(0.0, 16.0)], **kwargs)
    pool = ThreadPoolExecutor(1)
    monkeypatch.setattr(d, "_load", lambda: (FakeExtractor(), pool))
    if segmenter is not None:
        monkeypatch.setattr(d, "_load_segmenter", lambda: segmenter)
    return d


def test_windows_follow_segmentation_and_overlap_takes_the_dominant_speaker(monkeypatch):
    d = _fake_diarizer(monkeypatch, lambda audio, regions: ([5.3, 12.0, 13.0], [(12.0, 13.0)]))
    turns = d.diarize(_voices_track())
    a, b = turns[0].speaker, next(t.speaker for t in turns if t.start >= 5.3)
    assert a != b
    assert [(t.start, t.end) for t in turns if t.speaker == b] == pytest.approx(
        [(5.3, 7.533), (7.533, 9.767), (9.767, 12.0)], abs=1e-3
    )
    assert [(t.start, t.end, t.speaker) for t in turns if t.overlap] == [(12.0, 13.0, a)]
    assert np.allclose(turns.centroids[a], [1, 0, 0])  # the mixed window is left out


def test_missing_segmentation_model_falls_back_to_fixed_windows(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(diarize, "model_path", lambda m=None, **k: calls.append(m))
    with caplog.at_level("INFO", logger="recandcc.transcribe.diarize"):
        turns = _fake_diarizer(monkeypatch, segmentation=True).diarize(_voices_track())
    assert [(t.start, t.end) for t in turns] == FIXED
    assert not any(t.overlap for t in turns)
    assert calls == [diarize.SEGMENTATION_MODEL]
    assert "no segmentation model: speaker turns from fixed ~2 s windows" in caplog.messages
    calls.clear()
    _fake_diarizer(monkeypatch).diarize(_voices_track())
    assert calls == []  # diarize_overlap = false (default): never looked up or downloaded
    assert TranscriptionConfig().diarize_overlap is False


FIXED = [(2.0 * i, 2.0 * i + 2) for i in range(8)]


def test_broken_segmentation_model_keeps_speaker_labels(tmp_path, monkeypatch, caplog):
    bad = tmp_path / "seg.onnx"
    bad.write_bytes(b"\0" * diarize.SEGMENTATION_MODEL.size)  # right size, not a model
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: bad)
    d = _fake_diarizer(monkeypatch, segmentation=True)
    with caplog.at_level("WARNING", logger="recandcc.transcribe.diarize"):
        turns = d.diarize(_voices_track())
    assert [(t.start, t.end) for t in turns] == FIXED and len({t.speaker for t in turns}) == 2
    assert d.segmentation is False  # not retried for this Diarizer
    assert caplog.messages[0].startswith("segmentation model failed to load (")

    def broken(audio, regions):
        raise RuntimeError("ORT run failed: secret detail")

    d = _fake_diarizer(monkeypatch, broken, segmentation=True)
    caplog.clear()
    with caplog.at_level("WARNING", logger="recandcc.transcribe.diarize"):
        assert [(t.start, t.end) for t in d.diarize(_voices_track())] == FIXED
    assert d.segmentation is False
    assert caplog.messages == [
        "segmentation model failed to run (RuntimeError); speaker turns from fixed ~2 s windows"
    ]


def test_overlap_marks_only_segments_with_enough_overlapped_speech():
    turns = [Turn(0, 5, 0), Turn(5, 5.3, 1, True), Turn(5.3, 8, 1), Turn(8, 9, 1, True)]
    out, _ = label_speakers([seg(1, 4), seg(4, 6), seg(6, 10), seg(7, 9, "Me")], turns)
    assert [s.overlap for s in out] == [False, False, True, False]  # 0.3 s < OVERLAP_MARK_S
    assert speakers(out) == ["Speaker 1", "Speaker 1", "Speaker 2", "Me"]
    from recandcc.transcribe.output import paragraphs

    assert [p.overlap for p in paragraphs(out)] == [False, True, False]  # carried into paragraphs


# --------------------------------------------------------------------------- model files
def test_segmentation_model_is_pinned_like_the_speaker_model():
    for m in (diarize.EMBEDDING_MODEL, diarize.SEGMENTATION_MODEL):
        assert m.url.startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
        assert len(m.sha256) == 64 and int(m.sha256, 16) and m.size > 1_000_000
    assert diarize.SEGMENTATION_MODEL.member.endswith("/model.onnx")
    assert diarize.SEGMENTATION_MODEL.name != diarize.EMBEDDING_MODEL.name  # same folder


def _archive(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_archived_model_is_verified_then_extracted(model_dir, monkeypatch):
    onnx = b"segmentation" * 100
    data = _archive({"pkg/model.onnx": onnx, "pkg/run.sh": b"rm -rf /"})
    sha = hashlib.sha256(data).hexdigest()
    calls = _urlopen(monkeypatch, data)
    good = ModelFile(
        "seg.onnx", "https://example.invalid/s.tar.bz2", sha, len(onnx), "pkg/model.onnx"
    )
    assert real_model_path(good).read_bytes() == onnx and len(calls) == 1
    assert real_model_path(good) is not None and len(calls) == 1  # on disk now
    assert sorted(p.name for p in model_dir.iterdir()) == ["seg.onnx"]  # nothing else unpacked
    assert real_model_path(ModelFile("x.onnx", good.url, sha, 1, "pkg/missing.onnx")) is None
    assert real_model_path(ModelFile("y.onnx", good.url, "0" * 64, len(onnx), good.member)) is None
    assert sorted(p.name for p in model_dir.iterdir()) == ["seg.onnx"]  # no partial files


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(diarize, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(diarize, "_last_failure", {})
    return tmp_path / "models"


def _fake_model(data: bytes) -> ModelFile:
    return ModelFile(
        "m.onnx", "https://example.invalid/m.onnx", hashlib.sha256(data).hexdigest(), len(data)
    )


def _urlopen(monkeypatch, data: bytes | None):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        if data is None:
            raise OSError("network is unreachable")
        return io.BytesIO(data)

    monkeypatch.setattr(diarize.urllib.request, "urlopen", urlopen)
    return calls


def test_model_on_disk_is_used_without_network(model_dir, monkeypatch):
    data = b"onnx" * 100
    model_dir.mkdir()
    (model_dir / "m.onnx").write_bytes(data)
    calls = _urlopen(monkeypatch, None)
    assert real_model_path(_fake_model(data)) == model_dir / "m.onnx"
    assert calls == []


def test_model_is_downloaded_once_and_verified(model_dir, monkeypatch):
    data = b"weights" * 1000
    calls = _urlopen(monkeypatch, data)
    path = real_model_path(_fake_model(data))
    assert path.read_bytes() == data and len(calls) == 1
    assert real_model_path(_fake_model(data)) == path and len(calls) == 1
    assert [p.name for p in model_dir.iterdir()] == ["m.onnx"]  # no temp file left


def test_bad_checksum_is_rejected(model_dir, monkeypatch):
    data = b"weights" * 1000
    _urlopen(monkeypatch, b"tampered" + data[8:])
    assert real_model_path(_fake_model(data)) is None
    assert list(model_dir.iterdir()) == []


def test_offline_download_fails_gracefully_and_backs_off(model_dir, monkeypatch):
    data = b"weights"
    calls = _urlopen(monkeypatch, None)
    assert real_model_path(_fake_model(data)) is None
    assert real_model_path(_fake_model(data)) is None
    assert len(calls) == 1  # not retried right away
    assert real_model_path(_fake_model(data), download=False) is None


# --------------------------------------------------------------------------- batch final path
def _done(turns) -> Future:
    f = Future()
    f.set_result(turns)
    return f


TURNS = [Turn(0, 5, 3), Turn(5, 10, 8), Turn(10, 15, 3)]
SEGS = [seg(1, 4, text="hello"), seg(2, 3, "Me", "yes"), seg(6, 9, text="hi"), seg(11, 14)]


def _session_with_audio(tmp_path):
    for t in ("me", "others"):
        sf.write(tmp_path / f"{t}.wav", np.zeros(SAMPLE_RATE, np.float32), SAMPLE_RATE)
    return make_batch_session(tmp_path, {Track.ME: "me.wav", Track.OTHERS: "others.wav"})


def test_write_final_applies_ready_labels(tmp_path, monkeypatch):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", keep_audio=False))
    monkeypatch.setattr(t, "start_diarization", lambda _s: _done(TURNS))
    out = t.write_final(s, SEGS, 15.0)
    assert speakers(out) == ["Speaker 1", "Me", "Speaker 2", "Speaker 1"]
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:** hello" in md and "Speaker 2:** hi" in md and "Others" not in md
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [x["speaker"] for x in data["segments"]] == speakers(out)
    assert "Speaker 2: hi" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")
    assert not (tmp_path / "others.wav").exists()  # keep_audio = False


def test_slow_diarization_rewrites_the_transcript_later(tmp_path, monkeypatch):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", keep_audio=False))
    t.diarize_wait_s = 0.05
    pending: Future = Future()
    out = t.write_final(s, SEGS, 15.0, pending)
    md = tmp_path / "transcript.md"
    # written right away, without speaker labels, audio kept for the diarizer
    assert speakers(out) == ["Others", "Me", "Others", "Others"]
    assert "Others:** hello" in md.read_text(encoding="utf-8")
    assert s.transcribed and (tmp_path / "others.wav").exists()
    pending.set_result(TURNS)  # diarization ends -> rewrite + audio deleted
    assert "Speaker 2:** hi" in md.read_text(encoding="utf-8")
    assert "Speaker 1" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")
    assert not (tmp_path / "others.wav").exists()


def test_retention_waits_for_a_slow_diarization(tmp_path):
    from recandcc.retention import purge_old_audio

    d = tmp_path / "2026-10-01" / "s"
    d.mkdir(parents=True)
    s = _session_with_audio(d)
    t = Transcriber(TranscriptionConfig(model="fake"))  # keep_audio: retention deletes it
    t.diarize_wait_s = 0.01
    pending: Future = Future()
    t.write_final(s, SEGS, 15.0, pending)
    assert s.transcribed and s.ended_at is not None
    later = s.ended_at.astimezone() + timedelta(days=2)
    assert purge_old_audio(tmp_path, 1, now=later) == []  # the diarizer still needs the WAVs
    assert (d / "others.wav").exists()
    pending.set_result(TURNS)  # labels written: the audio may go now
    assert not (d / SessionInfo.LABELS_PENDING_MARKER).exists()
    assert len(purge_old_audio(tmp_path, 1, now=later)) == 2
    assert not (d / "others.wav").exists()


def test_labels_final_even_if_the_audio_cannot_be_deleted(tmp_path, monkeypatch):
    """A WAV still open elsewhere (Windows: PermissionError) must not leave the marker
    behind or keep after-labels callbacks waiting."""
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", keep_audio=False))
    t.diarize_wait_s = 0.01
    pending: Future = Future()
    t.write_final(s, SEGS, 15.0, pending)
    done = []
    t.after_labels(s.directory, lambda: done.append(1))

    def locked(session):
        raise PermissionError("in use")

    monkeypatch.setattr(t, "_drop_audio", locked)
    pending.set_result(TURNS)
    assert not (tmp_path / SessionInfo.LABELS_PENDING_MARKER).exists()
    assert done == [1]


def test_after_labels_waits_for_a_slow_diarization(tmp_path):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    t.diarize_wait_s = 0.01
    pending: Future = Future()
    t.write_final(s, SEGS, 15.0, pending)
    seen = []
    data = tmp_path / "transcript.json"
    t.after_labels(s.directory, lambda: seen.append(json.loads(data.read_text("utf-8"))))
    assert seen == []  # still "Others": wait for the labels
    pending.set_result(TURNS)
    assert [x["speaker"] for x in seen[0]["segments"]][0] == "Speaker 1"
    t.after_labels(s.directory, lambda: seen.append("now"))  # labels final: runs right away
    assert seen[1:] == ["now"]


def test_a_diarization_that_never_ends_stops_holding_its_waiters(tmp_path):
    from recandcc.transcribe import engine

    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    t.diarize_wait_s, now = 0.01, [0.0]
    t.clock = lambda: now[0]
    t.write_final(s, SEGS, 15.0, Future())  # never completes
    seen = []
    t.after_labels(s.directory, lambda: seen.append(1))
    assert t.labels_pending(s.directory) and seen == []
    now[0] = engine.LABEL_WAIT_MAX_S + 1  # dropped: the summary marker re-queues it
    assert not t.labels_pending(s.directory) and t._label_waiters == {}


def test_after_labels_runs_now_without_a_pending_diarization(tmp_path, monkeypatch):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    monkeypatch.setattr(t, "start_diarization", lambda _s: _done(TURNS))
    t.write_final(s, SEGS, 15.0)
    seen = []
    t.after_labels(s.directory, lambda: seen.append(1))
    assert seen == [1]


def test_slow_diarization_without_result_still_deletes_audio(tmp_path):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", keep_audio=False))
    t.diarize_wait_s = 0.01
    pending: Future = Future()
    t.write_final(s, SEGS, 15.0, pending)
    before = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    pending.set_result(None)  # unavailable
    assert (tmp_path / "transcript.md").read_text(encoding="utf-8") == before
    assert not (tmp_path / "me.wav").exists()


def test_json_marks_overlap_only_where_found(tmp_path, monkeypatch):
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    files = ("transcript.md", "transcript.srt")
    monkeypatch.setattr(t, "start_diarization", lambda _s: _done(TURNS))
    t.write_final(s, SEGS, 15.0)
    before = [(tmp_path / f).read_text(encoding="utf-8") for f in files]
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert not any("overlap" in x for x in data["segments"])
    mixed = [Turn(0, 5, 3), Turn(5, 7, 8), Turn(7, 8, 8, overlap=True), Turn(8, 15, 3)]
    monkeypatch.setattr(t, "start_diarization", lambda _s: _done(mixed))
    t.write_final(s, SEGS, 15.0)
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert [x.get("overlap") for x in data["segments"]] == [None, None, True, None]
    assert [Segment(**x).overlap for x in data["segments"]] == [False, False, True, False]
    assert [(tmp_path / f).read_text(encoding="utf-8") for f in files] == before  # unchanged


def test_start_diarization_respects_config_and_missing_audio(tmp_path):
    s = _session_with_audio(tmp_path)
    off = Transcriber(TranscriptionConfig(model="fake", diarize=False))
    assert off.start_diarization(s) is None
    on = Transcriber(TranscriptionConfig(model="fake"))
    (tmp_path / "others.wav").unlink()  # keep_audio = False / retention
    assert on.start_diarization(s) is None
    me_only = make_batch_session(tmp_path, {Track.ME: "me.wav"})
    assert on.start_diarization(me_only) is None


def test_no_model_keeps_others(tmp_path):
    """Offline / download failed (conftest: no model): the transcript keeps "Others"."""
    s = _session_with_audio(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake"))
    fut = t.start_diarization(s)
    assert fut is not None and fut.result(timeout=10) is None
    assert speakers(t.write_final(s, SEGS, 15.0, fut)) == speakers(SEGS)
    t.close()


# --------------------------------------------------------------------------- live final path
def test_live_finish_labels_final_transcript_but_not_captions(tmp_path):
    tr = FakeTranscriber()
    session = make_session(tmp_path)
    started = threading.Event()

    def start(sess):
        assert sess is session
        started.set()
        return _done([Turn(0, 2.5, 5), Turn(2.5, 10, 1)])

    tr.start_diarization = start
    emitted = []
    live = LiveTranscriber(tr, session, emitted.extend, step_s=0.02)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(1), silence(1.5), burst(1.5)]))
    session.ended_at = session.started_at + timedelta(seconds=6)
    segs = live.finish(session, timeout=10)
    assert started.is_set()
    assert speakers(segs) == ["Speaker 1", "Speaker 2"]
    assert speakers(emitted) == ["Others", "Others"]  # live captions unchanged
    md = (session.directory / "transcript.md").read_text(encoding="utf-8")
    assert "Speaker 1:**" in md and "Speaker 2:**" in md


def test_shared_extractor_does_not_hold_its_lock_during_a_download(monkeypatch):
    """A slow model_path (download) doesn't block another caller; one extractor is made."""
    import sys
    import threading
    import types

    made = []

    class Ex:
        def __init__(self, cfg):
            made.append(self)

    fake = types.SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=lambda **kw: kw, SpeakerEmbeddingExtractor=Ex
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    gate, entered = threading.Event(), threading.Event()

    def slow_path(*a, download=True, **k):
        if download:
            entered.set()
            gate.wait(5)
            return "m.onnx"
        return None

    monkeypatch.setattr(diarize, "model_path", slow_path)
    diarize.reset_shared_extractor()
    try:
        results = []
        t = threading.Thread(target=lambda: results.append(diarize.shared_extractor(True)))
        t.start()
        assert entered.wait(5)
        t0 = time.monotonic()
        assert diarize.shared_extractor(download=False) is None  # not blocked by the download
        assert time.monotonic() - t0 < 1.0
        gate.set()
        t.join(5)
        assert results[0] is diarize.shared_extractor(True) and len(made) == 1
    finally:
        gate.set()
        diarize.reset_shared_extractor()


def test_live_transcription_never_requests_word_timestamps(tmp_path):
    """Words cost live latency: a live session never asks for them, even with the opt-in on.
    Stories: US-12."""
    asked: list[dict] = []

    class Spy(FakeTranscriber):
        def transcribe_audio(self, audio, offset, speaker, language, **kw):
            asked.append(kw)
            return super().transcribe_audio(audio, offset, speaker, language, **kw)

    tr = Spy(TranscriptionConfig(model="fake", split_on_speaker_change=True))
    session = make_session(tmp_path)
    tr.start_diarization = lambda sess: _done([Turn(0, 2.5, 5), Turn(2.5, 10, 1)])
    live = LiveTranscriber(tr, session, lambda _s: None, step_s=0.02)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(1), silence(1.5), burst(1.5)]))
    session.ended_at = session.started_at + timedelta(seconds=6)
    live.finish(session, timeout=10)
    assert asked  # the live path did transcribe
    assert not any(kw.get("words") for kw in asked)


def test_per_track_hints_replace_the_diarizers(monkeypatch):
    d = _fake_diarizer(monkeypatch, min_speakers=3)  # two voices, forced into three
    assert len({t.speaker for t in d.diarize(_voices_track()) or []}) == 3
    assert len({t.speaker for t in d.diarize(_voices_track(), (0, 0)) or []}) == 2
