"""Live "Speaker N" labels for Others captions (fake embedder, no models).

Stories: US-33.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
from test_live import FakeTranscriber, Feeder, burst, make_session, manual, silence

from recandcc.models import SAMPLE_RATE, Track
from recandcc.transcribe import diarize
from recandcc.transcribe.live_speakers import SpeakerClusterer, SpeakerLabeler

SR = SAMPLE_RATE


def vec(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)


def test_clusterer_labels_are_stable_and_by_first_appearance():
    c = SpeakerClusterer(threshold=0.7)
    assert [c.assign(vec(a)) for a in (0, 90, 10, 95, 5)] == [1, 2, 1, 2, 1]


def test_clusterer_new_speaker_past_threshold_and_running_mean():
    c = SpeakerClusterer(threshold=0.9)
    assert c.assign(vec(0)) == 1
    assert c.assign(vec(40)) == 2  # cos 40 deg = 0.77 < 0.9
    assert c.assign(vec(-20)) == 1  # cos 0.94 to speaker 1; the centroid moves to -10 deg
    assert c.assign(vec(-32)) == 1  # 32 deg from the first voice but 22 from the mean (0.93)
    assert len(c) == 2


def test_clusterer_respects_the_cap_and_rejects_zero_vectors():
    c = SpeakerClusterer(threshold=0.99, max_speakers=2)
    assert [c.assign(vec(a)) for a in (0, 60, 120, 30)] == [1, 2, 2, 1]
    assert len(c) == 2
    with pytest.raises(ValueError):
        c.assign(np.zeros(3))


@pytest.fixture
def emitted():
    return []


def long_audio() -> np.ndarray:
    return np.zeros(2 * SR, np.float32)


def test_labeler_without_the_model_keeps_others(monkeypatch, caplog):
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: None)
    lab = SpeakerLabeler()  # default embedder, model missing
    with caplog.at_level("DEBUG", logger="recandcc.transcribe.live_speakers"):
        embs = [lab.embed(long_audio()) for _ in range(3)]
    assert embs == [None, None, None]
    assert lab.label(None) == "Others"
    assert sum("not on disk" in r.message for r in caplog.records) == 1  # logged once
    assert all(r.levelname == "DEBUG" for r in caplog.records)
    lab.close()


def test_labeler_labels_segments_and_keeps_previous_for_short_ones():
    lab = SpeakerLabeler(embedder=lambda a: vec(0 if a[0] == 0 else 90))
    a = long_audio()
    b = long_audio() + 1
    assert lab.label(lab.embed(a)) == "Speaker 1"
    assert lab.label(lab.embed(b)) == "Speaker 2"
    assert lab.embed(np.zeros(100, np.float32)) is None  # too short
    assert lab.label(None) == "Speaker 2"
    lab.close()


def test_full_queue_drops_instead_of_blocking():
    gate = threading.Event()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(5) and vec(0), queue_size=2)
    futs = [lab.submit(long_audio()) for _ in range(6)]  # 1 running + 2 queued fit
    assert sum(f is None for f in futs) >= 3 and lab.dropped >= 3
    gate.set()
    for f in futs:
        if f is not None:
            f.result(5)
    lab.close()


def test_embedder_failure_gives_none():
    def boom(a):
        raise RuntimeError("x")

    lab = SpeakerLabeler(embedder=boom)
    assert lab.embed(long_audio()) is None
    lab.close()


def test_live_captions_get_speaker_labels_but_stored_segments_keep_others(tmp_path, emitted):
    tr = FakeTranscriber()
    # two phrases from different "voices": embedder keyed on the audio level
    lab = SpeakerLabeler(embedder=lambda a: vec(0 if float(np.abs(a).max()) < 0.4 else 90))
    live = manual(tr, make_session(tmp_path), emitted, labeler=lab)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(2.5, amp=0.3), silence(1.5)]))
    live._step(False)
    feed(np.concatenate([burst(2.5, amp=0.9), silence(1.5)]))
    live._step(False)
    live._step(True)
    assert [s.speaker for s in emitted] == ["Speaker 1", "Speaker 2"]
    assert {s.speaker for s in live.segments()} == {"Others"}  # final transcript untouched
    lab.close()


def test_live_without_labeler_is_unchanged(tmp_path, emitted):
    tr = FakeTranscriber()
    tr.cfg.live_speaker_labels = False
    live = manual(tr, make_session(tmp_path), emitted)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2), silence(1.5)]))
    live._step(True)
    assert [s.speaker for s in emitted] == ["Others"]
