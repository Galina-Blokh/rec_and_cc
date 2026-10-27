"""Live "Speaker N" labels for Others captions (fake embedder, no models).

Stories: US-33.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import numpy as np
import pytest
from test_live import FakeTranscriber, Feeder, burst, make_session, manual, silence

from recandcc.models import SAMPLE_RATE, Track
from recandcc.transcribe import diarize, live_speakers
from recandcc.transcribe.live import EMBED_WAIT_S
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


def embed(lab: SpeakerLabeler, audio: np.ndarray):
    """Embedding of `audio` through the labeler thread (None if not queued)."""
    fut = lab.submit(audio)
    return None if fut is None else fut.result(5)


@pytest.fixture(autouse=True)
def fresh_extractor_cache():
    live_speakers._reset_extractor_cache()
    yield
    live_speakers._reset_extractor_cache()


def test_labeler_without_the_model_keeps_others(monkeypatch, caplog):
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: None)
    lab = SpeakerLabeler()  # default embedder, model missing
    with caplog.at_level("DEBUG", logger="recandcc.transcribe.live_speakers"):
        embs = [embed(lab, long_audio()) for _ in range(3)]
    assert embs == [None, None, None]
    assert lab.label(None) == "Others"
    assert sum("not available" in r.message for r in caplog.records) == 1  # logged once
    assert all(r.levelname == "DEBUG" for r in caplog.records)
    lab.close()


def test_labeler_labels_segments_and_keeps_previous_for_short_ones():
    lab = SpeakerLabeler(embedder=lambda a: vec(0 if a[0] == 0 else 90))
    a = long_audio()
    b = long_audio() + 1
    assert lab.label(embed(lab, a)) == "Speaker 1"
    assert lab.label(embed(lab, b)) == "Speaker 2"
    assert embed(lab, np.zeros(100, np.float32)) is None  # too short
    assert lab.label(None) == "Speaker 2"
    lab.close()


def test_short_segment_inherits_only_after_a_recent_one():
    lab = SpeakerLabeler(embedder=lambda a: vec(0))
    assert lab.label(vec(0), 0.0, 3.0) == "Speaker 1"
    assert lab.label(None, 3.4, 3.9) == "Speaker 1"  # 0.4 s after the previous one
    assert lab.label(None, 9.0, 9.5) == "Others"  # long gap: quick exchange, unknown voice
    assert lab.label(None, 9.6, 12.0) == "Speaker 1"  # long segment without embedding: keeps
    lab.close()


def _fake_sherpa(monkeypatch, fail: list[bool]) -> list[int]:
    """A fake sherpa_onnx whose extractor fails while fail[0]; returns the load-attempt log."""
    loads: list[int] = []

    class Cfg:
        def __init__(self, model, num_threads):
            pass

    class Extractor:
        def __init__(self, cfg):
            loads.append(1)
            if fail[0]:
                raise RuntimeError("bad model")

    mod = types.SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=Cfg, SpeakerEmbeddingExtractor=Extractor
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", mod)
    return loads


def test_extractor_load_failure_is_remembered_until_the_model_file_changes(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"broken")
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: model)
    fail = [True]
    loads = _fake_sherpa(monkeypatch, fail)
    for _ in range(3):
        assert live_speakers._get_extractor() is None
    assert len(loads) == 1  # not retried for every segment
    fail[0] = False
    model.write_bytes(b"verified copy")  # replaced: new mtime
    os.utime(model, (time.time() + 10, time.time() + 10))
    assert live_speakers._get_extractor() is not None
    assert len(loads) == 2


def test_expired_requests_are_skipped_not_embedded():
    clock = [0.0]
    calls = []
    gate = threading.Event()

    def slow(a):
        calls.append(1)
        gate.wait(5)
        return vec(0)

    lab = SpeakerLabeler(embedder=slow, clock=lambda: clock[0])
    first = lab.submit(long_audio(), deadline=10.0)
    stale = lab.submit(long_audio(), deadline=1.0)
    clock[0] = 5.0  # the second request expires while the first runs
    gate.set()
    assert first.result(5) is not None
    assert stale.result(5) is None
    assert len(calls) == 1
    lab.close()


def test_base_exception_does_not_kill_the_thread():
    def boom(a):
        if a[0] == 0:
            raise KeyboardInterrupt
        return vec(0)

    lab = SpeakerLabeler(embedder=boom)
    assert embed(lab, long_audio()) is None
    assert embed(lab, long_audio() + 1) is not None
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
    assert embed(lab, long_audio()) is None
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


def test_slow_embedder_does_not_delay_captions(tmp_path, emitted):
    gate = threading.Event()
    tr = FakeTranscriber()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(10) and vec(0))
    live = manual(tr, make_session(tmp_path), emitted, labeler=lab)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2.5), silence(1.5)]))
    t0 = time.monotonic()
    live._step(True)  # final: waits at most EMBED_WAIT_S for the stuck embedding
    took = time.monotonic() - t0
    gate.set()
    assert took < EMBED_WAIT_S + 1.5
    assert [s.speaker for s in emitted] == ["Others"]  # fallback label
    lab.close()


def test_close_with_a_full_queue_stops_the_thread():
    gate = threading.Event()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(5) and vec(0), queue_size=2)
    futs = [lab.submit(long_audio()) for _ in range(5)]
    assert lab._q.full()
    lab.close()
    gate.set()
    assert lab._thread is not None
    lab._thread.join(3)
    assert not lab._thread.is_alive()
    assert all(f.result(3) is None for f in futs if f is not None)


def test_job_held_then_emitted_with_fallback_after_the_deadline(tmp_path, emitted):
    clock = [100.0]
    gate = threading.Event()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(10) and vec(0), clock=lambda: clock[0])
    live = manual(
        FakeTranscriber(), make_session(tmp_path), emitted, labeler=lab, clock=lambda: clock[0]
    )
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2.5), silence(1.5)]))
    live._step(False)
    deadline = time.monotonic() + 5
    while not live._tracks[Track.OTHERS].jobs[0].future.done():  # transcription finished
        assert time.monotonic() < deadline
        time.sleep(0.01)
    live._step(False)
    assert emitted == []  # held: the embedding is pending and the deadline has not passed
    clock[0] += EMBED_WAIT_S + 0.1
    live._step(False)
    assert [s.speaker for s in emitted] == ["Others"]  # fallback label
    gate.set()
    lab.close()


def test_a_finished_embedding_wakes_the_worker(tmp_path, emitted):
    gate = threading.Event()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(10) and vec(0))
    live = manual(FakeTranscriber(), make_session(tmp_path), emitted, labeler=lab)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2.5), silence(1.5)]))
    live._step(False)
    job = live._tracks[Track.OTHERS].jobs[0]
    job.future.result(5)
    while not job.emb_futs:
        time.sleep(0.01)
    # The transcription future's own callback may set live._wake at any time, so the event
    # alone proves nothing: count wake-ups that happen after the embedding future is done.
    wakes = []
    real_set = live._wake.set
    live._wake.set = lambda: (wakes.append(job.emb_futs[0].done()), real_set())[1]  # type: ignore[method-assign]
    gate.set()
    deadline = time.monotonic() + 5
    while True:
        assert time.monotonic() < deadline
        if any(wakes):
            break
        time.sleep(0.01)
    lab.close()


def test_others_captions_stay_in_order_behind_a_stalled_job(tmp_path, emitted):
    clock = [100.0]
    stall = threading.Event()

    def embedder(a):
        if float(np.abs(a).max()) < 0.4:  # the first phrase is quiet: it stalls
            stall.wait(10)
        return vec(0)

    lab = SpeakerLabeler(embedder=embedder, clock=lambda: clock[0])
    live = manual(
        FakeTranscriber(), make_session(tmp_path), emitted, labeler=lab, clock=lambda: clock[0]
    )
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(2.5, amp=0.3), silence(1.5)]))
    live._step(False)
    feed(np.concatenate([burst(2.5, amp=0.9), silence(1.5)]))
    live._step(False)
    deadline = time.monotonic() + 5
    jobs = live._tracks[Track.OTHERS].jobs
    while not (len(jobs) == 2 and jobs[1].future.done()):  # both transcriptions finished
        assert time.monotonic() < deadline
        time.sleep(0.01)
    live._step(False)
    assert emitted == []  # the second phrase waits behind the first
    clock[0] += EMBED_WAIT_S + 0.1
    live._step(False)
    live._step(True)
    stall.set()
    assert [s.start for s in emitted] == sorted(s.start for s in emitted)
    assert len(emitted) == 2
    lab.close()


def test_me_is_held_while_an_others_job_waits_for_embeddings(tmp_path, emitted):
    clock = [100.0]
    gate = threading.Event()
    lab = SpeakerLabeler(embedder=lambda a: gate.wait(10) and vec(0), clock=lambda: clock[0])
    live = manual(
        FakeTranscriber(),
        make_session(tmp_path, (Track.OTHERS, Track.ME)),
        emitted,
        labeler=lab,
        clock=lambda: clock[0],
        echo_hold_s=0.1,
    )
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2.5), silence(1.5)]))
    Feeder(live, Track.ME)(np.concatenate([silence(2), burst(1.5), silence(1.5)]))
    live._step(False)
    deadline = time.monotonic() + 5
    while not all(j.future.done() for tb in live._tracks.values() for j in tb.jobs):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    live._step(False)  # Me is collected and held; Others waits for its stalled embedding
    assert emitted == []
    clock[0] += 0.2  # past echo_hold_s, before the embedding deadline (EMBED_WAIT_S)
    live._step(False)
    assert emitted == []  # Me must not jump ahead of the Others caption
    clock[0] += EMBED_WAIT_S + 0.1
    live._step(False)
    assert [s.speaker for s in emitted] == ["Others", "Me"]  # start order, each once
    live._step(True)
    assert len(emitted) == 2
    gate.set()
    lab.close()


def test_transcriber_that_never_runs_starts_no_labeler_thread(tmp_path, emitted):
    tr = FakeTranscriber()
    tr.cfg.live_speaker_labels = True
    before = set(threading.enumerate())
    live = manual(tr, make_session(tmp_path), emitted)
    assert live._labeler is not None
    assert live._labeler._thread is None
    assert set(threading.enumerate()) == before
