"""LiveTranscriber with a fake model (real Silero VAD on synthetic, speech-like bursts).

Stories: US-08, US-09, US-11.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pytest
import soundfile as sf

from recandcc.config import TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Segment, SessionInfo, Track, TriggerKind
from recandcc.transcribe.engine import Transcriber
from recandcc.transcribe.live import LiveTranscriber, live_vad, quietest_point

SR = SAMPLE_RATE


def burst(seconds: float, freq: float = 220.0, amp: float = 0.5) -> np.ndarray:
    """Amplitude-modulated tone: Silero VAD flags it as speech."""
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t) * (1 + np.sin(2 * np.pi * 4 * t)) / 2).astype(
        np.float32
    )


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


class FakeTranscriber(Transcriber):
    """Real Transcriber minus the models: deterministic text per region.

    Text is phrase(N) with N = round(region offset), so an echo of the same phrase on the
    other track (same time) gets the same text. `sync`: run jobs inline (deterministic tests).
    """

    def __init__(self, cfg: TranscriptionConfig | None = None, sync: bool = True, workers=2):
        super().__init__(
            cfg or TranscriptionConfig(model="fake", workers=workers, live_speaker_labels=False)
        )
        self.sync = sync
        self.calls: list[tuple[str, float, float, str]] = []  # speaker, offset, dur, language
        self.lock = threading.Lock()
        self.fail_at: set[int] = set()  # call indexes that raise
        self.barrier: threading.Barrier | None = None
        self.delay = 0.0
        self.language = ("en", 0.95)

    def _ensure_loaded(self):
        if self._pool is None and not self.sync:
            self._pool = ThreadPoolExecutor(self.num_workers)
        return None

    def submit(self, fn, *args):
        if self.sync:
            f = Future()
            try:
                f.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001
                f.set_exception(e)
            return f
        return super().submit(fn, *args)

    @property
    def has_language_model(self):
        return False

    def detect_language(self, audio, main=False):
        lang = self.language
        return lang(audio) if callable(lang) else lang

    def transcribe_audio(self, audio, offset, speaker, language, *, beam_size=5, max_end=None):
        with self.lock:
            i = len(self.calls)
            self.calls.append((speaker, offset, len(audio) / SR, language))
        if self.barrier is not None:
            self.barrier.wait()
        if self.delay:
            time.sleep(self.delay)
        if i in self.fail_at:
            raise RuntimeError("model hiccup")
        end = min(offset + len(audio) / SR, max_end or 1e9)
        return [Segment(offset, end, phrase(round(offset)), speaker, language)]


def phrase(n: int) -> str:
    """Deterministic, mutually dissimilar text for the region starting near second `n`."""
    rng = np.random.default_rng(n)
    return " ".join("".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz"), 5)) for _ in range(4))


def make_session(tmp_path, tracks=(Track.OTHERS,)) -> SessionInfo:
    d = tmp_path / "sess"
    d.mkdir(exist_ok=True)
    s = SessionInfo(
        id="live1",
        directory=d,
        started_at=datetime.now().astimezone(),
        trigger_kind=TriggerKind.CALL,
        trigger_app="zoom.us",
        tracks={t: f"{t}.wav" for t in tracks},
    )
    for t in tracks:
        sf.write(d / f"{t}.wav", np.zeros(SR, np.float32), SR)
    s.save()
    return s


class Feeder:
    """Feeds a track in 100 ms blocks with correct start frames."""

    def __init__(self, live: LiveTranscriber, track: Track, start: int = 0):
        self.live, self.track, self.pos = live, track, start

    def __call__(self, audio: np.ndarray) -> None:
        for i in range(0, len(audio), SR // 10):
            block = audio[i : i + SR // 10]
            self.live.feed(self.track, self.pos, block)
            self.pos += len(block)


@pytest.fixture
def emitted():
    return []


def manual(tr, session, emitted, **kw) -> LiveTranscriber:
    kw.setdefault("md_interval_s", 0.0)
    return LiveTranscriber(tr, session, on_segments=emitted.extend, start=False, **kw)


# --------------------------------------------------------------------------- helpers
def test_vad_flags_bursts_and_quietest_point():
    regions = live_vad(np.concatenate([silence(1), burst(1.5), silence(1)]))
    assert len(regions) == 1
    s, e = regions[0]
    assert 0.6 < s < 1.1 and 2.4 < e < 3.0
    a = np.concatenate([burst(1), silence(0.3) + 0.001, burst(1)])
    assert 1.0 <= quietest_point(a, 0.5, 2.0) <= 1.3


# --------------------------------------------------------------------------- committing
def test_region_committed_only_after_trailing_silence(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(2)]))
    live._step(False)
    assert tr.calls == [] and emitted == []
    feed(silence(0.3))  # VAD needs 0.5 s to end the region, plus the commit margin
    live._step(False)
    assert tr.calls == []
    feed(silence(0.8))
    live._step(False)
    assert len(tr.calls) == 1
    speaker, offset, dur, lang = tr.calls[0]
    assert speaker == "Others" and lang == "en"
    assert 0.6 < offset < 1.1 and 2.0 < dur < 2.9  # padded region around 1.0-3.0 s
    assert [s.text for s in emitted] == [phrase(1)]
    # nothing more to commit
    feed(silence(2))
    live._step(False)
    assert len(tr.calls) == 1


def test_short_blips_ignored(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(0.1), silence(2)]))
    live._step(False)
    assert tr.calls == []


def test_long_speech_is_force_cut(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path), emitted, max_region_s=6.0, min_cut_s=4.0)
    feed = Feeder(live, Track.OTHERS)
    # 9 s of non-stop speech with a dip (quieter, still speech) at 5 s
    talk = np.concatenate([burst(5), burst(0.3, amp=0.05), burst(3.7)])
    for i in range(0, len(talk), SR // 2):
        feed(talk[i : i + SR // 2])
        live._step(False)
    assert len(tr.calls) == 1
    _, offset, dur, _ = tr.calls[0]
    assert offset < 0.1 and 4.0 <= dur <= 6.0
    assert 4.9 <= offset + dur <= 5.4  # cut in the dip
    tb = live._tracks[Track.OTHERS]
    assert tb.base / SR == pytest.approx(offset + dur, abs=0.01)
    feed(silence(1.5))
    live._step(False)
    assert len(tr.calls) == 2 and tr.calls[1][1] == pytest.approx(offset + dur, abs=0.05)


def test_mic_padding_gives_absolute_times(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path, (Track.OTHERS, Track.ME)), emitted)
    Feeder(live, Track.OTHERS)(silence(13))
    # the mic joined at 10 s: its first frame is sample 10*SR of the session timeline
    Feeder(live, Track.ME, start=10 * SR)(np.concatenate([burst(1), silence(2)]))
    live._step(False)
    me = [s for s in emitted if s.speaker == "Me"]
    assert len(me) == 1 and 9.7 < me[0].start <= 10.1
    assert me[0].text == phrase(10)


def test_timeline_gap_flushes_and_continues(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(1)]))  # speech running when the gap starts
    feed.pos += 20 * SR
    feed(np.concatenate([burst(1), silence(2)]))
    live._step(False)
    assert [round(c[1]) for c in tr.calls] == [1, 22]


def test_memory_tail_trimmed(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path), emitted, max_region_s=5.0, min_cut_s=3.0)
    feed = Feeder(live, Track.OTHERS)
    biggest = 0
    for i in range(40):  # 2 minutes: speech, silence, and long monologues
        feed(burst(8) if i % 5 == 0 else np.concatenate([burst(1), silence(2)]))
        live._step(False)
        biggest = max(biggest, live._tracks[Track.OTHERS].size)
    assert biggest / SR < 5.0 + 8.0 + 1.0  # never more than one feed + one region
    assert live._tracks[Track.OTHERS].size / SR <= 1.5
    assert len(live._tracks[Track.OTHERS].blocks) <= 1


# --------------------------------------------------------------------------- results
def test_language_inherited_by_short_uncertain_region(tmp_path, emitted):
    tr = FakeTranscriber()
    tr.language = lambda a: ("ru", 0.95) if len(a) > 2.5 * SR else ("en", 0.5)
    live = manual(tr, make_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(2.5), silence(1.2), burst(1), silence(1.5), burst(1)]))
    feed(silence(5))
    feed(burst(1))
    feed(silence(1.5))
    live._step(False)
    assert [c[3] for c in tr.calls] == ["ru", "ru", "ru", "en"]  # last one: too far away


def test_failed_region_is_skipped(tmp_path, emitted):
    tr = FakeTranscriber()
    tr.fail_at = {0}
    live = manual(tr, make_session(tmp_path), emitted)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([burst(1), silence(1.5), burst(1), silence(1.5)]))
    live._step(False)
    assert len(tr.calls) == 2 and len(emitted) == 1


def test_echo_not_shown_twice(tmp_path, emitted):
    tr = FakeTranscriber()
    live = manual(tr, make_session(tmp_path, (Track.OTHERS, Track.ME)), emitted)
    audio = np.concatenate([silence(1), burst(1.5), silence(1.5)])
    Feeder(live, Track.OTHERS)(audio)
    Feeder(live, Track.ME)(0.3 * audio)  # speaker echo picked up by the mic
    live._step(False)
    assert [(s.speaker, s.text) for s in emitted] == [("Others", phrase(1))]
    Feeder(live, Track.ME, start=len(audio))(np.concatenate([burst(1), silence(1.5)]))
    Feeder(live, Track.OTHERS, start=len(audio))(silence(3))
    live._step(False)
    assert [s.speaker for s in emitted] == ["Others", "Me"]


def test_markdown_rewritten_mid_session(tmp_path, emitted):
    tr = FakeTranscriber()
    session = make_session(tmp_path)
    live = manual(tr, session, emitted, md_interval_s=3.0)
    clock = [100.0]
    live.clock = lambda: clock[0]
    feed = Feeder(live, Track.OTHERS)
    md = session.directory / "transcript.md"
    feed(np.concatenate([silence(1), burst(1), silence(1.5)]))
    live._step(False)
    text = md.read_text(encoding="utf-8")
    assert phrase(1) in text and "recording in progress" in text
    feed(np.concatenate([burst(1), silence(1.5)]))
    clock[0] += 1.0
    live._step(False)
    assert phrase(3) not in md.read_text(encoding="utf-8")  # debounced
    clock[0] += 3.0
    live._step(False)
    assert phrase(3) in md.read_text(encoding="utf-8")
    assert not SessionInfo.load(session.directory).transcribed
    assert not list(session.directory.glob(".*.tmp"))


# --------------------------------------------------------------------------- lifecycle
def test_finish_flushes_tail_and_writes_outputs(tmp_path, emitted):
    tr = FakeTranscriber(TranscriptionConfig(model="fake", workers=2, keep_audio=False))
    session = make_session(tmp_path)
    ended = []
    live = LiveTranscriber(tr, session, emitted.extend, lambda: ended.append(1), step_s=0.02)
    feed = Feeder(live, Track.OTHERS)
    feed(np.concatenate([silence(1), burst(1), silence(1.5), burst(2)]))  # still talking
    session.ended_at = session.started_at + timedelta(seconds=10)
    segs = live.finish(session, timeout=10)
    assert len(segs) == 2 and segs[0].text == phrase(1) and 3.0 < segs[1].start < 4.0
    assert emitted == segs  # the tail is also shown live
    assert ended == [1]
    for name in ("transcript.md", "transcript.json", "transcript.srt"):
        assert (session.directory / name).exists()
    md = (session.directory / "transcript.md").read_text(encoding="utf-8")
    assert "recording in progress" not in md and "00:00:10" in md
    loaded = SessionInfo.load(session.directory)
    assert loaded.transcribed
    assert not (session.directory / "others.wav").exists()  # keep_audio = False
    assert live.finish(timeout=1) == segs  # idempotent


def test_cancel_writes_nothing(tmp_path, emitted):
    tr = FakeTranscriber()
    session = make_session(tmp_path)
    ended = []
    live = LiveTranscriber(tr, session, emitted.extend, lambda: ended.append(1), step_s=0.02)
    Feeder(live, Track.OTHERS)(np.concatenate([silence(1), burst(2)]))
    live.cancel()
    with pytest.raises(RuntimeError):
        live.finish(timeout=5)
    assert ended == []
    assert sorted(p.name for p in session.directory.iterdir()) == ["others.wav", "session.json"]
    assert not SessionInfo.load(session.directory).transcribed


def test_both_tracks_transcribed_concurrently(tmp_path, emitted):
    tr = FakeTranscriber(sync=False, workers=2)
    tr.barrier = threading.Barrier(2, timeout=5)  # both regions must be in flight at once
    session = make_session(tmp_path, (Track.OTHERS, Track.ME))
    live = LiveTranscriber(tr, session, emitted.extend, step_s=0.02, start=False)
    audio = np.concatenate([silence(1), burst(1.5), silence(1.5)])
    Feeder(live, Track.OTHERS)(audio)
    Feeder(live, Track.ME)(np.concatenate([silence(1), burst(1.5, amp=0.4), silence(1.5)]))
    tr.cfg.dedupe_echo = False
    live._thread.start()
    segs = live.finish(session, timeout=10)
    tr.close()
    assert sorted(s.speaker for s in segs) == ["Me", "Others"]
    assert not tr.barrier.broken


def test_model_failure_reported_by_finish(tmp_path, emitted):
    tr = FakeTranscriber()

    def boom():
        raise RuntimeError("no model")

    tr.ensure_loaded = boom
    live = LiveTranscriber(tr, make_session(tmp_path), emitted.extend, step_s=0.02)
    Feeder(live, Track.OTHERS)(burst(1))
    with pytest.raises(RuntimeError, match="no model"):
        live.finish(timeout=5)
