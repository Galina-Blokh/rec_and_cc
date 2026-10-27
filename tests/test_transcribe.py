"""Unit tests for recandcc.transcribe (no Whisper model needed).

Stories: US-08, US-09, US-10, US-13, US-19.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig, load_config, normalize_languages
from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind, is_rtl
from recandcc.transcribe import Transcriber, TranscriptionQueue, auto_parallelism
from recandcc.transcribe.chunking import find_speech, load_track, merge_labelled, merge_regions
from recandcc.transcribe.merge import (
    dedupe_echo,
    language_confidence,
    merge_tracks,
    pick_language,
    similarity,
    smooth_languages,
)
from recandcc.transcribe.output import (
    PDI,
    RLI,
    RLM,
    fmt_srt_time,
    format_markdown,
    format_srt,
    paragraphs,
)


def seg(start, end, text, speaker="Others", lang="en"):
    return Segment(start, end, text, speaker, lang)


def make_session(tmp_path, tracks=None):
    s = SessionInfo(
        id="s1",
        directory=tmp_path,
        started_at=datetime(2026, 10, 26, 14, 3, 12),
        trigger_kind=TriggerKind.CALL,
        trigger_app="zoom.us",
        ended_at=datetime(2026, 10, 26, 14, 4, 0),
        tracks=tracks or {},
    )
    s.save()
    return s


# -- chunking ---------------------------------------------------------------------------------


def test_merge_regions_joins_short_gaps_and_respects_max():
    regions = [(0.0, 5.0), (5.5, 12.0), (13.0, 25.0), (25.5, 33.0)]
    assert merge_regions(regions) == [(0.0, 25.0), (25.5, 33.0)]


def test_merge_regions_splits_on_long_pause():
    assert merge_regions([(0, 3), (10, 12)]) == [(0, 3), (10, 12)]


def test_merge_regions_splits_overlong_region():
    out = merge_regions([(0.0, 70.0)])
    assert len(out) == 3 and all(e - s <= 30.0 + 1e-9 for s, e in out)
    assert out[0][0] == 0.0 and out[-1][1] == 70.0


def test_merge_regions_empty():
    assert merge_regions([]) == []


def test_silence_has_no_speech():
    assert find_speech(np.zeros(SAMPLE_RATE * 5, dtype=np.float32)) == []
    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 5) * 0.001
    assert find_speech(noise.astype(np.float32)) == []


def test_load_track_downmixes_and_resamples(tmp_path):
    p = tmp_path / "x.wav"
    sf.write(p, np.zeros((8000, 2), dtype=np.float32), 8000, subtype="FLOAT")
    a = load_track(p)
    assert a.dtype == np.float32 and a.ndim == 1 and len(a) == 16000


# -- language ---------------------------------------------------------------------------------


def test_pick_language_restricted():
    probs = [("uk", 0.5), ("ru", 0.3), ("bg", 0.15), ("en", 0.05)]
    assert pick_language(probs, ["en", "ru"]) == "ru"


def test_pick_language_unrestricted():
    assert pick_language([("uk", 0.5), ("ru", 0.3)], []) == "uk"


def test_pick_language_none_allowed_present():
    assert pick_language([("de", 1.0)], ["ru", "en"]) == "ru"


# -- merge & echo -----------------------------------------------------------------------------


def test_similarity_normalises():
    assert similarity("Hello, World!", "hello world") == 1.0
    assert similarity("Привет, мир.", "привет мир") == 1.0


def test_dedupe_echo_drops_overlapping_similar():
    others = [seg(10.0, 14.0, "The quarterly report is ready for review.")]
    me = [
        seg(10.3, 14.2, "the quarterly report is ready for a review", "Me"),  # echo
        seg(20.0, 22.0, "The quarterly report is ready for review.", "Me"),  # no overlap
        seg(11.0, 13.0, "Sounds good to me", "Me"),  # different text
    ]
    kept = dedupe_echo(me, others)
    assert [s.start for s in kept] == [20.0, 11.0]


def test_dedupe_echo_partial_inside_longer_segment():
    others = [seg(0.0, 12.0, "Okay everyone, let's start. The quarterly report is ready now.")]
    me = [seg(5.0, 9.0, "the quarterly report is ready", "Me")]
    assert dedupe_echo(me, others) == []


def test_merge_tracks_orders_by_start():
    me = [seg(5.0, 6.0, "b", "Me"), seg(1.0, 2.0, "a", "Me")]
    others = [seg(3.0, 4.0, "x"), seg(0.0, 0.5, "y")]
    merged = merge_tracks(me, others, dedupe=False)
    assert [s.start for s in merged] == [0.0, 1.0, 3.0, 5.0]


# -- output formatting ------------------------------------------------------------------------


def test_paragraphs_join_same_speaker_short_gap():
    segs = [seg(0, 2, "One."), seg(3, 4, "Two."), seg(7, 8, "Three."), seg(8.5, 9, "Me too.", "Me")]
    ps = paragraphs(segs)
    assert [p.text for p in ps] == ["One. Two.", "Three.", "Me too."]


def test_format_markdown(tmp_path):
    s = make_session(tmp_path)
    segs = [seg(83.2, 85, "Hello there."), seg(86, 87, "Привет.", "Me", "ru")]
    md = format_markdown(s, segs, 3725.0)
    assert "2026-10-26 14:03:12" in md
    assert "zoom.us" in md and "01:02:05" in md and "en, ru" in md
    assert "**[00:01:23] Others:** Hello there." in md
    assert "**[00:01:26] Me:** Привет." in md


def test_format_srt():
    srt = format_srt([seg(1.5, 3.25, "Hi"), seg(3661.001, 3662, "Bye", "Me")])
    assert srt.startswith("1\n00:00:01,500 --> 00:00:03,250\nOthers: Hi\n")
    assert "2\n01:01:01,001 --> 01:01:02,000\nMe: Bye\n" in srt
    assert fmt_srt_time(0) == "00:00:00,000"


# -- Hebrew / right to left ------------------------------------------------------------------


def test_normalize_languages_accepts_hebrew_and_cleans_up():
    assert normalize_languages(["en", "RU", " he "]) == ["en", "ru", "he"]
    assert normalize_languages(["iw", "Hebrew", "he"]) == ["he"]  # aliases, no duplicates
    assert normalize_languages("he") == ["he"]
    assert normalize_languages(["en", "xx"]) == ["en"]  # unknown code dropped
    assert normalize_languages(["xx"]) == ["en", "ru"]  # nothing valid -> defaults
    assert normalize_languages([]) == []  # [] still means any language


def test_load_config_validates_languages(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[transcription]\nlanguages = ["en", "ru", "iw"]\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.transcription.languages == ["en", "ru", "he"]
    cfg.live_captions = False  # e.g. a menu toggle saves the config
    cfg.save(path)
    assert load_config(path).transcription.languages == ["en", "ru", "he"]
    assert '"iw"' in path.read_text(encoding="utf-8")  # the user's spelling is kept
    cfg.transcription.languages = ["he"]  # an actual change is saved as is
    cfg.save(path)
    assert '"iw"' not in path.read_text(encoding="utf-8")
    assert load_config(path).transcription.languages == ["he"]


def test_languages_that_are_not_a_list_fall_back_to_defaults(tmp_path):
    assert normalize_languages(5) == ["en", "ru"]
    assert normalize_languages(None) == []
    path = tmp_path / "config.toml"
    path.write_text("[transcription]\nlanguages = 5\n", encoding="utf-8")
    assert load_config(path).transcription.languages == ["en", "ru"]


def test_language_id_restricted_to_three_languages():
    # Hebrew is often confused with Arabic / Yiddish; only the configured list counts.
    probs = [("ar", 0.5), ("yi", 0.1), ("he", 0.3), ("en", 0.05), ("ru", 0.02)]
    lang, conf = language_confidence(probs, ["en", "ru", "he"])
    assert lang == "he" and conf == pytest.approx(0.3 / 0.37)


def test_is_rtl_by_language_or_text():
    assert is_rtl("he") and not is_rtl("en", "שלום")  # the language wins when known
    assert is_rtl("", "  «שלום» Zoom") and not is_rtl("", "Zoom שלום") and not is_rtl("", "")


def test_markdown_isolates_hebrew_only_and_splits_direction(tmp_path):
    s = make_session(tmp_path)
    segs = [
        seg(1, 2, "Hello.", "Me"),
        seg(2.5, 4, "שלום, אני ב-Zoom.", "Me", "he"),  # same speaker, short gap: new paragraph
        seg(4.5, 5, "Привет.", "Me", "ru"),
    ]
    md = format_markdown(s, segs, 10.0)
    assert "**[00:00:01] Me:** Hello.\n" in md
    assert f"**[00:00:02] Me:** {RLI}שלום, אני ב-Zoom.{PDI}\n" in md
    assert "**[00:00:04] Me:** Привет.\n" in md and "en, he, ru" in md
    assert md.count(RLI) == 1 and RLM not in md


def test_srt_marks_hebrew_lines_with_rlm():
    srt = format_srt([seg(0, 1, "Hi"), seg(1, 2, "שלום Zoom.", "Me", "he")])
    assert "\nOthers: Hi\n" in srt
    assert f"\n{RLM}Me: שלום Zoom.\n" in srt and srt.count(RLM) == 1


# -- orchestration with a fake model ----------------------------------------------------------


class FakeModel:
    """Stands in for WhisperModel. Language by loudness: quiet audio "en", loud audio "ru";
    always ranks "uk" first to check the restriction to cfg.languages."""

    def __init__(self):
        self.calls, self.detects = [], 0
        self.lock = threading.Lock()

    def detect_language(self, audio):
        with self.lock:
            self.detects += 1
        loud = float(np.abs(audio).mean()) > 0.1
        second = ("ru", 0.35) if loud else ("en", 0.35)
        return "uk", 0.6, [("uk", 0.6), second, ("ru" if not loud else "en", 0.05)]

    def transcribe(self, audio, **kw):
        with self.lock:
            self.calls.append(kw)
        dur = len(audio) / SAMPLE_RATE
        text = "слово" if kw["language"] == "ru" else "word"
        return iter([type("S", (), {"start": 0.0, "end": dur, "text": f" {text} "})()]), None


def _write_tracks(tmp_path):
    """Both tracks: quiet 'en' at 1-3 s and 3.5-4.5 s, loud 'ru' at 5-8 s, then 'en' 20-22 s."""
    a = np.zeros(SAMPLE_RATE * 25, np.float32)
    for (s, e), amp in [((1, 3), 0.05), ((3.5, 4.5), 0.05), ((5, 8), 0.5), ((20, 22), 0.05)]:
        a[int(s * SAMPLE_RATE) : int(e * SAMPLE_RATE)] = amp
    for name in ("me.wav", "others.wav"):
        sf.write(tmp_path / name, a, SAMPLE_RATE, subtype="FLOAT")
    return [(1.0, 3.0), (3.5, 4.5), (5.0, 8.0), (20.0, 22.0)]


def _install(t, model, lid):
    from concurrent.futures import ThreadPoolExecutor

    if t._model is None:
        t._model, t._lid_model = model, lid
        t._pool = ThreadPoolExecutor(t.num_workers)
    return model


def _run_fake(tmp_path, monkeypatch, lid):
    import recandcc.transcribe.engine as eng

    regions = _write_tracks(tmp_path)
    monkeypatch.setattr(eng, "find_speech", lambda a: regions)
    s = make_session(tmp_path, {Track.ME: "me.wav", Track.OTHERS: "others.wav"})
    t = Transcriber(TranscriptionConfig(model="fake", workers=2, keep_audio=False))
    model = FakeModel()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: _install(t, model, lid))
    progress = []
    segs = t.transcribe_session(s, progress.append)
    t.close()
    return s, segs, model, progress


def test_transcribe_session_with_language_id_model(tmp_path, monkeypatch):
    lid = FakeModel()
    s, segs, model, progress = _run_fake(tmp_path, monkeypatch, lid)
    # per track: en(1-4.5) | ru(5-8) | en(20-22) -> 3 chunks; language split despite short gaps
    assert lid.detects == 8 and model.detects == 0
    assert len(model.calls) == 6 and all(c["vad_filter"] is False for c in model.calls)
    # Me is identical to Others -> all Me dropped as echo
    assert [(x.speaker, x.start, x.language) for x in segs] == [
        ("Others", 1.0, "en"),
        ("Others", 5.0, "ru"),
        ("Others", 20.0, "en"),
    ]
    assert progress[0] == 0.0 and progress[-1] == 1.0
    assert not (tmp_path / "me.wav").exists()
    assert SessionInfo.load(tmp_path).transcribed
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert data["languages"] == ["en", "ru"] and len(data["segments"]) == 3
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert "**[00:00:01] Others:** word слово" in md  # joined: gap < 2 s
    assert (tmp_path / "transcript.srt").exists()


def test_transcribe_session_main_model_detection(tmp_path, monkeypatch):
    s, segs, model, _ = _run_fake(tmp_path, monkeypatch, None)
    # no LID model: regions merged by gap only -> (1-8) and (20-22) per track, detect per chunk
    assert model.detects == 4 and len(model.calls) == 4
    assert {c["language"] for c in model.calls} <= {"en", "ru"}


class HebrewFakeModel(FakeModel):
    """Quiet audio "en", loud audio "he"; ranks "ar" first (a typical Hebrew confusion)."""

    def detect_language(self, audio):
        with self.lock:
            self.detects += 1
        loud = float(np.abs(audio).mean()) > 0.1
        second = ("he", 0.4) if loud else ("en", 0.4)
        return "ar", 0.55, [("ar", 0.55), second, ("ru", 0.05)]

    def transcribe(self, audio, **kw):
        with self.lock:
            self.calls.append(kw)
        text = "שלום" if kw["language"] == "he" else "word"
        dur = len(audio) / SAMPLE_RATE
        return iter([type("S", (), {"start": 0.0, "end": dur, "text": f" {text} "})()]), None


def test_transcribe_session_with_hebrew(tmp_path, monkeypatch):
    import recandcc.transcribe.engine as eng

    regions = _write_tracks(tmp_path)
    monkeypatch.setattr(eng, "find_speech", lambda a: regions)
    s = make_session(tmp_path, {Track.ME: "me.wav", Track.OTHERS: "others.wav"})
    cfg = TranscriptionConfig(model="fake", workers=2, languages=["en", "ru", "he"])
    t = Transcriber(cfg)
    model, lid = HebrewFakeModel(), HebrewFakeModel()
    monkeypatch.setattr(t, "_ensure_loaded", lambda: _install(t, model, lid))
    segs = t.transcribe_session(s)
    t.close()
    assert [(x.start, x.language) for x in segs] == [(1.0, "en"), (5.0, "he"), (20.0, "en")]
    assert {c["language"] for c in model.calls} == {"en", "he"}
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert data["languages"] == ["en", "he"] and data["segments"][1]["text"] == "שלום"
    md = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert f"**[00:00:05] Others:** {RLI}שלום{PDI}" in md
    assert f"{RLM}Others: שלום" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")


def test_smooth_languages_short_uncertain_inherits():
    regions = [(0, 5), (5.5, 6.0), (10, 15)]
    det = [("ru", 0.95), ("en", 0.55), ("en", 0.9)]
    assert smooth_languages(regions, det) == ["ru", "ru", "en"]


def test_merge_regions_splits_on_language():
    out = merge_labelled([(0, 2), (2.5, 4), (4.5, 6)], ["en", "ru", "ru"])
    assert out == [(0, 2, "en"), (2.5, 6, "ru")]


def test_session_without_tracks_writes_empty_transcript(tmp_path):
    s = make_session(tmp_path)
    t = Transcriber(TranscriptionConfig(model="never-loaded"))
    assert t.transcribe_session(s) == []
    assert "No speech detected" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert t._model is None  # nothing to do -> model never loaded


def test_auto_parallelism():
    assert auto_parallelism(0, 18) == (4, 4)
    assert auto_parallelism(0, 4) == (1, 4)
    assert auto_parallelism(3, 12) == (3, 4)


# -- queue ------------------------------------------------------------------------------------


class StubTranscriber:
    def __init__(self):
        self.order = []

    def transcribe_session(self, session, progress=None):
        if session.id == "bad":
            raise ValueError("boom")
        self.order.append(session.id)
        return [seg(0, 1, session.id)]


def _s(sid, tmp_path):
    return SessionInfo(
        id=sid,
        directory=tmp_path,
        started_at=datetime.now() - timedelta(1),
        trigger_kind=TriggerKind.MANUAL,
    )


def test_queue_order_and_errors(tmp_path):
    stub, done = StubTranscriber(), []
    q = TranscriptionQueue(stub, on_done=lambda s, r: done.append((s.id, r)))
    for sid in ["a", "bad", "b"]:
        q.submit(_s(sid, tmp_path))
    q.shutdown(wait=True)
    assert stub.order == ["a", "b"]
    assert [d[0] for d in done] == ["a", "bad", "b"]
    assert isinstance(done[1][1], ValueError)
    assert q.pending() == 0


def test_queue_finalize_replaces_transcription_and_falls_back_on_failure(tmp_path):
    stub, done = StubTranscriber(), []
    q = TranscriptionQueue(stub, on_done=lambda s, r: done.append((s.id, r)))
    q.submit(_s("live", tmp_path), finalize=lambda: [seg(0, 1, "from live")])

    def broken():
        raise RuntimeError("live failed")

    q.submit(_s("fallback", tmp_path), finalize=broken)
    q.shutdown(wait=True)
    assert stub.order == ["fallback"]  # only the failed live session was transcribed
    assert done[0][1][0].text == "from live"
    assert done[1][1][0].text == "fallback"


def test_queue_survives_a_failing_callback_and_rejects_work_after_shutdown(tmp_path):
    stub, calls = StubTranscriber(), []

    def on_done(s, r):
        calls.append(s.id)
        raise RuntimeError("UI gone")

    q = TranscriptionQueue(stub, on_done=on_done)
    q.submit(_s("a", tmp_path))
    q.submit(_s("b", tmp_path))
    q.shutdown(wait=True)
    assert calls == ["a", "b"]
    with pytest.raises(RuntimeError):
        q.submit(_s("c", tmp_path))


def test_queue_shutdown_without_wait_drops_sessions_not_started(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Slow(StubTranscriber):
        def transcribe_session(self, session, progress=None):
            started.set()
            release.wait(5)
            return super().transcribe_session(session, progress)

    stub = Slow()
    q = TranscriptionQueue(stub)
    for sid in ["a", "b", "c"]:
        q.submit(_s(sid, tmp_path))
    assert started.wait(5)
    q.shutdown(wait=False)
    assert q.pending() == 1  # only the session in progress is left
    release.set()
    q._thread.join(5)
    assert stub.order == ["a"] and q.pending() == 0
    assert not q._thread.is_alive()  # the stop marker reached the dispatcher


def test_queue_submit_racing_shutdown_is_never_stranded(monkeypatch, tmp_path):
    """shutdown() runs exactly while submit() is enqueueing: the session must still run."""
    import queue as queue_mod

    from recandcc.transcribe import queue as tq

    stopper: list[threading.Thread] = []
    progressed = threading.Event()  # shutdown() is waiting for the lock, or queued _STOP
    holder = {}

    class SpyLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            if stopper and threading.current_thread() is stopper[0]:
                progressed.set()
            return self._lock.__enter__()

        def __exit__(self, *exc):
            return self._lock.__exit__(*exc)

    class RacingQueue(queue_mod.Queue):
        def put(self, item, *a, **k):
            if item is tq._STOP:
                progressed.set()
            elif not stopper:  # the submit: let shutdown() run as far as it can first
                stopper.append(threading.Thread(target=holder["q"].shutdown))
                stopper[0].start()
                assert progressed.wait(5)
            super().put(item, *a, **k)

    # The dispatcher must block on the racing queue from the start.
    monkeypatch.setattr(tq.queue, "Queue", RacingQueue)
    stub, done = StubTranscriber(), []
    q = holder["q"] = TranscriptionQueue(stub, on_done=lambda s, r: done.append(s.id))
    q._lock = SpyLock()
    q.submit(_s("a", tmp_path))
    stopper[0].join(5)
    q._thread.join(5)
    assert done == ["a"] and q.pending() == 0
    assert not q._thread.is_alive()


def test_queue_second_shutdown_still_waits_for_the_running_session(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Slow(StubTranscriber):
        def transcribe_session(self, session, progress=None):
            started.set()
            release.wait(5)
            return super().transcribe_session(session, progress)

    stub = Slow()
    q = TranscriptionQueue(stub)
    q.submit(_s("a", tmp_path))
    assert started.wait(5)
    q.shutdown(wait=False)
    threading.Timer(0.2, release.set).start()
    q.shutdown(wait=True)  # joins the dispatcher even though it was already closed
    assert stub.order == ["a"] and not q._thread.is_alive()


def test_queue_repeated_shutdown_without_wait_keeps_the_stop_marker(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Slow(StubTranscriber):
        def transcribe_session(self, session, progress=None):
            started.set()
            release.wait(5)
            return super().transcribe_session(session, progress)

    stub = Slow()
    q = TranscriptionQueue(stub)
    q.submit(_s("a", tmp_path))
    q.submit(_s("b", tmp_path))
    assert started.wait(5)
    q.shutdown(wait=False)  # drops "b", queues the stop marker
    q.shutdown(wait=False)  # drains that marker again: it must be put back
    release.set()
    q._thread.join(5)
    assert stub.order == ["a"] and q.pending() == 0
    assert not q._thread.is_alive()


def test_queue_waits_for_a_writer_still_finishing_the_audio(monkeypatch, tmp_path):
    from recandcc.models import clear_writing, mark_writing
    from recandcc.transcribe import queue as qmod

    monkeypatch.setattr(qmod, "WRITING_WAIT_S", 0.05)
    monkeypatch.setattr(qmod, "WRITING_POLL_S", 0.01)
    (tmp_path / "w").mkdir()
    (tmp_path / "stale").mkdir()
    (tmp_path / "stale" / SessionInfo.WRITING_MARKER).write_text("an earlier run")
    mark_writing(tmp_path / "w")
    stub, done = StubTranscriber(), threading.Event()
    q = TranscriptionQueue(stub, on_done=lambda s, r: s.id == "w" and done.set())
    q.submit(_s("w", tmp_path / "w"))
    q.submit(_s("a", tmp_path))
    q.submit(_s("stale", tmp_path / "stale"))  # its writer is gone: final as it gets
    time.sleep(0.3)
    assert stub.order == ["a", "stale"] and q.pending() == 1  # partial audio: not yet
    clear_writing(tmp_path / "w")  # the disk recovered, the WAV is final
    assert done.wait(5)
    assert stub.order == ["a", "stale", "w"]
    q.shutdown(wait=True)


def test_queue_shutdown_leaves_a_still_writing_session_for_the_next_start(monkeypatch, tmp_path):
    from recandcc.models import mark_writing
    from recandcc.transcribe import queue as qmod

    monkeypatch.setattr(qmod, "WRITING_WAIT_S", 0.05)
    monkeypatch.setattr(qmod, "WRITING_POLL_S", 0.01)
    mark_writing(tmp_path)
    stub = StubTranscriber()
    q = TranscriptionQueue(stub)
    q.submit(_s("w", tmp_path))
    q.shutdown(wait=True)  # returns: the job is dropped, not retried forever
    assert stub.order == [] and q.pending() == 0
