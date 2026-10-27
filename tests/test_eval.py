"""scripts/eval.py: metrics on hand-made references, and a run with a fake pipeline.

Stories: none.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from recandcc.models import SAMPLE_RATE, Segment
from recandcc.transcribe.diarize import Turn

_spec = importlib.util.spec_from_file_location(
    "rcc_eval", Path(__file__).resolve().parents[1] / "scripts" / "eval.py"
)
ev = importlib.util.module_from_spec(_spec)
sys.modules["rcc_eval"] = ev  # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(ev)


def seg(start, end, text, lang="en", speaker="Others"):
    return Segment(start=start, end=end, text=text, speaker=speaker, language=lang)


def utt(start, end, text, lang="en", speaker="A"):
    return ev.Utterance(start, end, speaker, lang, text)


def test_normalize_case_punctuation_and_yo():
    assert ev.normalize("Hello, World! It's 5 o'clock.") == [
        "hello",
        "world",
        "it's",
        "5",
        "o'clock",
    ]
    assert ev.normalize("Ещё — раз…") == ["еще", "раз"]


def test_normalize_curly_apostrophes_and_stress_marks():
    assert ev.normalize("Don\u2019t") == ev.normalize("don't") == ["don't"]
    assert ev.normalize("мо\u0301локо йод") == ["молоко", "йод"]  # й keeps its breve
    assert ev.normalize("Café") == ["cafe"]


@pytest.mark.parametrize(
    ("ref", "hyp", "edits"),
    [
        ("a b c", "a b c", 0),
        ("a b c", "a x c", 1),
        ("a b c", "a c", 1),
        ("a b", "a b c d", 2),
        ("a b", "", 2),
    ],
)
def test_word_edits(ref, hyp, edits):
    assert ev.word_edits(ref.split(), hyp.split()) == edits


def test_text_scores_per_language_and_lid():
    utts = [utt(0, 2, "hello there world"), utt(3, 5, "привет как дела", "ru")]
    segs = [
        seg(0.1, 1.9, "Hello there, word."),  # 1 substitution
        seg(3.0, 4.0, "привет как", "en"),  # tagged wrong, 1 deletion
        seg(4.0, 4.5, "дела", "ru"),  # shorter than the "en" part: LID still wrong
    ]
    s = ev.text_scores(utts, segs)
    assert s["en"] == {"words": 3, "edits": 1, "utts": 1, "lid_ok": 1}
    assert s["ru"] == {"words": 3, "edits": 0, "utts": 1, "lid_ok": 0}


def test_unmatched_segment_goes_to_nearest_utterance_without_language_vote():
    utts = [utt(0, 1, "one"), utt(10, 11, "two")]
    (w0, v0), (w1, v1) = ev.assign_words(utts, [seg(8, 9, "two", "ru")])
    assert w0 == [] and w1 == ["two"] and v1 == {}


def test_unmatched_segment_goes_to_closest_interval_not_midpoint():
    utts = [utt(0, 20, "long one"), utt(30, 31, "short")]
    (w0, _), (w1, _) = ev.assign_words(utts, [seg(21, 22, "extra")])
    assert w0 == ["extra"] and w1 == []  # 1 s after the long one, 8 s before the short one


def test_segment_spanning_utterances_is_split():
    utts = [utt(0, 1, "hello"), utt(1, 2, "world")]
    s = ev.text_scores(utts, [seg(0, 2, "Hello world")])
    assert s["en"] == {"words": 2, "edits": 0, "utts": 2, "lid_ok": 2}


def test_segment_over_overlapping_utterances_splits_words_and_votes():
    utts = [utt(0, 2, "one two", "en", "A"), utt(1, 3, "три четыре", "ru", "B")]
    (w0, v0), (w1, v1) = ev.assign_words(utts, [seg(0, 3, "one two три четыре", "en")])
    assert w0 == ["one", "two"] and w1 == ["три", "четыре"]  # equal overlap: half each
    assert v0 == {"en": 2.0} and v1 == {"en": 2.0}


def test_language_vote_weighs_overlap_not_segment_length():
    utts = [utt(0, 10, "a b c d e f g h i j", "en"), utt(10, 12, "к л", "ru")]
    segs = [seg(0, 10.5, "a b c d e f g h i j", "en"), seg(10.5, 12, "к л", "ru")]
    s = ev.text_scores(utts, segs)  # the long en segment overlaps the ru utterance by 0.5 s only
    assert s["ru"]["lid_ok"] == 1 and s["en"]["lid_ok"] == 1


def test_missing_hypothesis_counts_all_words_and_no_lid():
    s = ev.text_scores([utt(0, 1, "a b c")], [])
    assert s["en"] == {"words": 3, "edits": 3, "utts": 1, "lid_ok": 0}


def test_der_perfect_with_renamed_speakers():
    ref = [(0, 2, "A"), (2, 4, "B")]
    assert ev.der(ref, [(0, 2, "1"), (2, 4, "0")])["der"] == 0.0


def test_der_miss_false_alarm_confusion():
    ref = [(0, 2, "A"), (2, 4, "B")]
    hyp = [(0, 1, "x"), (2, 5, "x")]  # A misses 1 s; B -> x confused (2 s); 1 s false alarm
    d = ev.der(ref, hyp)
    assert d["miss_s"] == pytest.approx(1.0)
    assert d["false_alarm_s"] == pytest.approx(1.0)
    assert d["confusion_s"] == pytest.approx(1.0)  # x maps to B (2 s); A's 1 s is confused
    assert d["der"] == pytest.approx(3.0 / 4.0)


def test_der_many_speakers_is_exact():
    ref = [(i, i + 1, f"R{i}") for i in range(7)]
    hyp = [(i, i + 1, f"H{i}") for i in range(7)] + [(7 + i, 7.5 + i, f"X{i}") for i in range(13)]
    d = ev.der(ref, hyp)  # 7 x 20: permutations would be ~390 million, the DP is instant
    assert d["confusion_s"] == 0.0 and d["false_alarm_s"] == pytest.approx(6.5)


def test_best_mapping_beats_greedy():
    # greedy takes 5 (row 0 -> col 0) and then 0; the optimum pairs 4 + 4
    assert ev._best_mapping([[5, 4], [4, 0]]) == 8
    assert ev._best_mapping([[1, 2, 3]]) == 3 and ev._best_mapping([]) == 0


def test_der_overlapping_reference_speakers():
    ref = [(0, 2, "A"), (1, 3, "B")]  # 1 s of crosstalk: 4 s of reference speech
    perfect = ev.der(ref, [(0, 2, "x"), (1, 3, "y")])
    assert perfect["speech_s"] == pytest.approx(4.0) and perfect["der"] == 0.0
    one_voice = ev.der(ref, [(0, 3, "x")])  # crosstalk heard as one speaker: 1 s missed
    assert one_voice["miss_s"] == pytest.approx(1.0) and one_voice["false_alarm_s"] == 0.0
    assert one_voice["confusion_s"] == pytest.approx(1.0)  # x maps to A or B, not both


def test_best_mapping_greedy_above_twelve_speakers():
    m = [[10 if i == j else 1 for j in range(14)] for i in range(13)]
    assert ev._best_mapping(m) == 130  # diagonal; greedy finds it, the DP would be 2^13 x 14


def test_der_no_reference():
    assert ev.der([], [])["der"] == 0.0


class FakeTranscriber:
    """Like the real one with keep_audio = false: drops the WAV after transcribing."""

    def __init__(self, segs):
        self.segs, self.sessions = segs, []

    def transcribe_session(self, session):
        wav = session.directory / "others.wav"
        assert wav.exists()
        self.sessions.append(session)
        session.update_metrics("transcription", {"audio_s": 5.0, "elapsed_s": 0.5})
        wav.unlink()
        return self.segs


class FakeDiarizer:
    def diarize_file(self, path):
        assert path.exists()  # runs before the transcriber can drop the audio
        return [Turn(0.0, 2.0, 0), Turn(3.0, 5.0, 1)]


def test_manifest_run_and_summary(tmp_path):
    sf.write(tmp_path / "c1.wav", np.zeros(5 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    manifest = {
        "clips": [
            {
                "id": "c1",
                "audio": "c1.wav",
                "utterances": [
                    {"start": 0, "end": 2, "speaker": "A", "language": "en", "text": "hello there"},
                    {"start": 3, "end": 5, "speaker": "B", "language": "ru", "text": "да нет"},
                ],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    (clip,) = ev.load_manifest(tmp_path)
    tr = FakeTranscriber([seg(0, 2, "hello there"), seg(3, 5, "да", "ru")])
    result = ev.run_clip(clip, tr, FakeDiarizer())
    assert not tr.sessions[0].directory.exists()  # temporary copy removed
    assert (tmp_path / "c1.wav").exists()  # the eval clip itself is untouched
    assert result["diarization"]["der"] == 0.0 and result["elapsed_s"] == 0.5
    s = ev.summarize([result])
    assert s["languages"]["en"]["wer"] == 0.0
    assert s["languages"]["ru"]["wer"] == 0.5
    assert s["languages"]["all"] == {"wer": 0.25, "lid_accuracy": 1.0, "words": 4, "utterances": 2}
    assert s["der"] == 0.0 and s["rtf"] == 0.1


def test_der_counts_false_alarms_on_clips_without_reference_speech():
    silent = {"text": {}, "diarization": ev.der([], [(0, 5, "x")])}
    clean = {"text": {}, "diarization": ev.der([(0, 10, "A")], [(0, 10, "x")])}
    assert ev.summarize([silent, clean])["der"] == pytest.approx(0.5)  # 5 s errors / 10 s speech


def test_clip_copies_stay_inside_the_given_root(tmp_path):
    sf.write(tmp_path / "c.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    clip = ev.Clip("c", tmp_path / "c.wav", ev.Track.OTHERS, [utt(0, 1, "a")])
    root = tmp_path / "run"
    root.mkdir()
    tr = FakeTranscriber([])
    ev.run_clip(clip, tr, None, root)
    assert tr.sessions[0].directory.parent == root and list(root.iterdir()) == []


def test_rtf_weighs_clips_by_audio_length():
    results = [
        {"text": {}, "audio_s": 1.0, "elapsed_s": 1.0},  # RTF 1.0 on a 1 s clip
        {"text": {}, "audio_s": 99.0, "elapsed_s": 9.9},  # RTF 0.1 on a 99 s clip
    ]
    assert ev.summarize(results)["rtf"] == pytest.approx(0.109)


@pytest.mark.parametrize("audio", ["../outside.wav", "/etc/hosts"])
def test_manifest_rejects_audio_outside_the_set(tmp_path, audio):
    root = tmp_path / "set"
    root.mkdir()
    sf.write(tmp_path / "outside.wav", np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    u = {"start": 0, "end": 1, "speaker": "A", "language": "en", "text": "a"}
    m = {"clips": [{"id": "x", "audio": audio, "utterances": [u]}]}
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the eval folder"):
        ev.load_manifest(root)


@pytest.mark.parametrize(
    ("start", "end", "msg"),
    [
        (2, 1, "start < end"),
        (-1, 1, "start < end"),
        (0, float("nan"), "start < end"),
        (0, 9, "after the audio"),
    ],
)
def test_manifest_rejects_bad_times(tmp_path, start, end, msg):
    sf.write(tmp_path / "x.wav", np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    u = {"start": start, "end": end, "speaker": "A", "language": "en", "text": "a"}
    bad = {"clips": [{"id": "x", "audio": "x.wav", "utterances": [u]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        ev.load_manifest(tmp_path)


def write_set(root, clip):
    sf.write(root / "c.wav", np.zeros(2 * SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
    (root / "manifest.json").write_text(json.dumps({"clips": [clip]}), encoding="utf-8")


U = {"start": 0, "end": 1, "speaker": "A", "language": "ru-RU", "text": "да"}


def test_manifest_language_tags_are_normalised(tmp_path):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    (clip,) = ev.load_manifest(tmp_path)
    assert clip.utterances[0].language == "ru"


@pytest.mark.parametrize(
    "clip",
    [
        {"audio": "c.wav", "utterances": [U]},  # no id
        {"id": "c", "utterances": [U]},  # no audio
        {"id": "c", "audio": "c.wav", "utterances": [{**U, "who": "x"}]},  # unknown key
        {"id": "c", "audio": "c.wav", "utterances": [{**U, "language": None}]},  # null language
    ],
)
def test_manifest_entry_errors_name_the_clip(tmp_path, clip):
    write_set(tmp_path, clip)
    with pytest.raises(ValueError, match="bad manifest entry"):
        ev.load_manifest(tmp_path)


class NoDiarizer:
    def diarize_file(self, path):
        return None  # model missing or failed


def test_failed_diarization_is_counted(tmp_path):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    (clip,) = ev.load_manifest(tmp_path)
    r = ev.run_clip(clip, FakeTranscriber([]), NoDiarizer(), tmp_path)
    s = ev.summarize([r])
    assert r["diarization_ok"] is False and s["der"] is None and s["clips_without_der"] == 1


def test_main_wires_config_and_writes_scores_only(tmp_path, monkeypatch, capsys):
    import recandcc.config as config
    import recandcc.transcribe as transcribe
    import recandcc.transcribe.diarize as diarize

    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    cfg = config.Config()
    cfg.transcription.keep_audio = False
    seen = {}

    class Tr(FakeTranscriber):
        def __init__(self, c):
            super().__init__([seg(0, 1, "да", "ru")])
            seen["cfg"] = c

        def close(self):
            seen["closed"] = True

    class Di(FakeDiarizer):
        def __init__(self, segmentation=False):
            seen["segmentation"] = segmentation

        def diarize(self, audio):
            seen["warmed"] = True

        def close(self):
            pass

    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(transcribe, "Transcriber", Tr)
    monkeypatch.setattr(diarize, "Diarizer", Di)
    monkeypatch.setattr(ev, "replay_live", lambda audio, *a: [0.5, 1.0])
    out = tmp_path / "r.json"
    assert ev.main([str(tmp_path), "--model", "small", "--live", "--json", str(out)]) == 0
    c = seen["cfg"]
    assert c.model == "small" and c.keep_audio is True and c.diarize is False
    assert seen["warmed"] and seen["closed"] and seen["segmentation"] is False
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["languages"]["ru"]["wer"] == 0.0
    assert report["summary"]["live"]["p90_s"] == 0.95
    printed = capsys.readouterr().out
    assert "live all: p50 0.75 s, p90 0.95 s, max 1.0 s (2 segments)" in printed
    assert "live c: p50 0.75 s" in printed
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: None)  # model not on disk
    with pytest.raises(SystemExit):
        ev.main([str(tmp_path), "--diarize-overlap", "--json", str(tmp_path / "o.json")])
    assert "segmentation model is not on disk" in capsys.readouterr().err
    monkeypatch.setattr(diarize, "model_path", lambda *a, **k: tmp_path)
    assert ev.main([str(tmp_path), "--diarize-overlap", "--json", str(tmp_path / "o.json")]) == 0
    assert seen["segmentation"] is True
    assert "да" not in out.read_text(encoding="utf-8") + printed  # no text


# ------------------------------------------------------------------------- live latency
def test_arrival_is_the_end_of_the_capture_block():
    n = 10 * SAMPLE_RATE
    assert ev.arrival(1.25, n) == pytest.approx(1.3)
    assert ev.arrival(1.3, n) == pytest.approx(1.3)
    assert ev.arrival(9.97, n - 100) == pytest.approx((n - 100) / SAMPLE_RATE)  # last block


def test_latency_summary_percentiles():
    s = ev.latency_summary([float(x) for x in range(10, 0, -1)])
    assert s == {"segments": 10, "p50_s": 5.5, "p90_s": 9.1, "max_s": 10.0}
    assert ev.latency_summary([]) == {"segments": 0}
    assert ev.live_line({"segments": 0}) == "no segments"


def test_summary_pools_live_latencies_across_clips():
    a = {"text": {}, "live_ok": True, "live": {"latencies_s": [1.0, 2.0]}}
    b = {"text": {}, "live_ok": True, "live": {"latencies_s": [3.0]}}
    failed = {"text": {}, "live_ok": False}
    s = ev.summarize([a, b, failed])
    assert s["live"] == {"segments": 3, "p50_s": 2.0, "p90_s": 2.8, "max_s": 3.0}
    assert s["clips_without_live"] == 1
    assert ev.summarize([{"text": {}}])["live"] is None


def test_replay_clock_skips_idle_time_only():
    done, running = Future(), Future()
    done.set_result([])
    clock = ev.ReplayClock()
    clock.wait(5.0, [], [])  # nothing in flight: skipped
    assert clock.skipped == 5.0
    clock.wait(5.0, [done], [])  # a finished region: collected at once, no time passes
    assert clock.skipped == 5.0 and clock() < 5.5
    # in flight: waited for in real time, but only until the region is done (no skip)
    threading.Timer(0.05, running.set_result, [[]]).start()
    t0 = time.monotonic()
    clock.wait(30.0, [running], [running])
    assert running.done() and clock.skipped == 5.0 and time.monotonic() - t0 < 25.0


class VirtualClock(ev.ReplayClock):
    """Skipped time only: deterministic when nothing is ever in flight (inline fake model)."""

    def __call__(self) -> float:
        return self.skipped


def edges_vad(audio):
    """Stub VAD: the exact non-zero runs of the tail, unpadded."""
    d = np.diff(np.concatenate([[0], (audio != 0).astype(int), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(s / SAMPLE_RATE, e / SAMPLE_RATE) for s, e in zip(starts, ends, strict=True)]


def fake_model(delay=0.0):
    from test_live import FakeTranscriber as FakeModel

    from recandcc.config import TranscriptionConfig

    tr = FakeModel(TranscriptionConfig(model="fake", workers=2, diarize=False), sync=delay == 0)
    tr.delay = delay
    return tr


def test_replay_live_latency_is_commit_silence_on_the_step_grid(tmp_path):
    from test_live import make_session

    def part(seconds, value=0.0):
        return np.full(int(seconds * SAMPLE_RATE), value, np.float32)

    # speech 1-3 s and 5-6.4 s
    audio = np.concatenate([part(1), part(2, 0.5), part(2), part(1.4, 0.5), part(1.6)])
    tr, clock, session = fake_model(), VirtualClock(), make_session(tmp_path)
    try:
        lat = ev.replay_live(audio, ev.Track.OTHERS, tr, session, clock, vad=edges_vad)
    finally:
        tr.close()
    # committed at the first 0.3 s step with >= 0.4 s of silence fed after the speech:
    # 3.0 -> 3.6 s and 6.4 -> 6.9 s
    assert lat == pytest.approx([0.6, 0.5])
    assert clock.skipped >= 8.0  # every step was idle: no real time was waited for
    assert not (session.directory / "transcript.json").exists()  # captions only


@pytest.mark.parametrize("delay", [0.0, 0.6])
def test_replay_live_with_silero_counts_model_time(tmp_path, delay):
    from test_live import burst, make_session, silence

    tr = fake_model(delay)  # model time is waited for in real time; idle time is skipped
    audio = np.concatenate([silence(1), burst(2), silence(2), burst(1.5), silence(1.5)])
    clock = ev.ReplayClock()
    try:
        lat = ev.replay_live(audio, ev.Track.OTHERS, tr, make_session(tmp_path), clock)
    finally:
        tr.close()
    assert clock.skipped > 4.0  # most of the 8 s was skipped, not waited for
    assert len(lat) == 2  # one caption per burst
    # commit silence + step granularity (+ slack for slow CI), plus the model's own time
    assert all(0.3 + delay <= x <= 4.0 + delay for x in lat), lat


def test_run_clip_reports_live_latency(tmp_path, monkeypatch):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    (clip,) = ev.load_manifest(tmp_path)
    seen = {}

    def replay(audio, track, tr, session):
        seen["samples"], seen["track"] = len(audio), track
        return [0.5, 1.0]

    monkeypatch.setattr(ev, "replay_live", replay)
    r = ev.run_clip(clip, FakeTranscriber([]), None, tmp_path, live=True)
    assert seen == {"samples": 2 * SAMPLE_RATE, "track": ev.Track.OTHERS}
    assert r["live"] == {
        "segments": 2,
        "p50_s": 0.75,
        "p90_s": 0.95,
        "max_s": 1.0,
        "latencies_s": [0.5, 1.0],
    }
    assert r["live_ok"] is True
    assert "live" not in ev.run_clip(clip, FakeTranscriber([]), None, tmp_path)


def test_failed_live_replay_keeps_the_clip(tmp_path, monkeypatch, capsys):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    (clip,) = ev.load_manifest(tmp_path)

    def replay(*a):
        raise RuntimeError("all 3 live regions failed")

    monkeypatch.setattr(ev, "replay_live", replay)
    r = ev.run_clip(clip, FakeTranscriber([]), None, tmp_path, live=True)
    assert r["live_ok"] is False and "live" not in r and "text" in r
    assert "live replay of clip c failed (RuntimeError)" in capsys.readouterr().out


# ------------------------------------------------------------- backends and resources
def patch_pipeline(monkeypatch, cfg, seen, gpu=True):
    import recandcc.config as config
    import recandcc.transcribe as transcribe
    import recandcc.transcribe.gpu as gpu_mod

    class Tr(FakeTranscriber):
        def __init__(self, c):
            super().__init__([seg(0, 1, "да", "ru")])
            seen["cfg"], self.gpu = c, gpu_mod.use_gpu(c)

        def close(self):
            pass

    def no_save(*a, **k):
        raise AssertionError("eval must never save the config")

    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(config.Config, "save", no_save)
    monkeypatch.setattr(transcribe, "Transcriber", Tr)
    monkeypatch.setattr(gpu_mod, "gpu_available", lambda: gpu)


@pytest.mark.parametrize(
    ("flag", "backend", "device", "label"),
    [("mlx", "mlx", "auto", "mlx"), ("cpu", "faster-whisper", "cpu", "faster-whisper (cpu)")],
)
def test_backend_flag_reaches_the_transcriber_config(
    tmp_path, monkeypatch, capsys, flag, backend, device, label
):
    import recandcc.config as config

    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    cfg, seen = config.Config(), {}
    cfg.transcription.device = "cpu" if flag == "mlx" else "auto"  # the flag wins over config
    patch_pipeline(monkeypatch, cfg, seen)
    out = tmp_path / "r.json"
    assert ev.main([str(tmp_path), "--backend", flag, "--no-diarize", "--json", str(out)]) == 0
    assert (seen["cfg"].backend, seen["cfg"].device) == (backend, device)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["backend"] == label
    assert report["wall_s"] >= 0 and report["peak_memory_mb"] > 0
    assert f"[{label}]" in capsys.readouterr().out


def test_backend_mlx_without_mlx_is_an_error(tmp_path, monkeypatch):
    import recandcc.config as config

    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    patch_pipeline(monkeypatch, config.Config(), {}, gpu=False)
    with pytest.raises(SystemExit):
        ev.main([str(tmp_path), "--backend", "mlx", "--no-diarize", "--json", str(tmp_path / "r")])


def test_languages_flag_sets_detection_languages_in_memory(tmp_path, monkeypatch):
    import recandcc.config as config

    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    cfg, seen = config.Config(), {}
    patch_pipeline(monkeypatch, cfg, seen)  # Config.save raises: never written
    out = tmp_path / "r.json"
    argv = [str(tmp_path), "--languages", "EN, ru,iw,ru", "--no-diarize", "--json", str(out)]
    assert ev.main(argv) == 0
    assert seen["cfg"].languages == ["en", "ru", "he"]  # normalised, aliases, no duplicates
    assert json.loads(out.read_text(encoding="utf-8"))["languages"] == ["en", "ru", "he"]


@pytest.mark.parametrize("value", ["", ",", " ", "en,xx"])
def test_languages_flag_rejects_empty_or_unknown_codes(tmp_path, monkeypatch, capsys, value):
    import recandcc.config as config

    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    seen = {}
    patch_pipeline(monkeypatch, config.Config(), seen)
    with pytest.raises(SystemExit):  # [] would silently mean "any language"
        ev.main(
            [str(tmp_path), "--languages", value, "--no-diarize", "--json", str(tmp_path / "r")]
        )
    assert "cfg" not in seen and "--languages" in capsys.readouterr().err


def test_maxrss_units_and_peak_memory():
    assert ev.maxrss_mb(512 * 2**20, "darwin") == 512.0  # bytes on macOS
    assert ev.maxrss_mb(512 * 1024, "linux") == 512.0  # kilobytes on Linux
    assert ev.peak_memory_mb() > 0  # resource on POSIX, psutil on Windows


# ----------------------------------------------------------------------------- hypotheses
TWO = [
    {"start": 0, "end": 1, "speaker": "A", "language": "en", "text": "hello there"},
    {"start": 1, "end": 2, "speaker": "B", "language": "ru", "text": "да нет"},
]


def hyp_set(tmp_path, files):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": TWO})
    hyp = tmp_path / "hyp"
    hyp.mkdir()
    for name, content in files.items():
        (hyp / name).write_text(content, encoding="utf-8")
    (clip,) = ev.load_manifest(tmp_path)
    return clip, hyp


def test_hypothesis_json_scores_wer_lid_and_der(tmp_path):
    segs = [
        {"start": 0, "end": 1, "text": "Hello there", "language": "EN", "speaker": "0"},
        {"start": 1, "end": 2, "text": "да", "language": "ru", "speaker": "1"},
    ]
    clip, hyp = hyp_set(tmp_path, {"c.json": json.dumps({"segments": segs, "elapsed_s": 0.4})})
    r = ev.score_hypothesis(clip, hyp)
    assert r["diarization"]["der"] == 0.0 and r["elapsed_s"] == 0.4 and r["audio_s"] == 2.0
    s = ev.summarize([r])
    assert s["languages"]["all"] == {"wer": 0.25, "lid_accuracy": 1.0, "words": 4, "utterances": 2}
    assert s["der"] == 0.0 and s["rtf"] == 0.2


def test_hypothesis_without_language_or_speaker_is_na(tmp_path):
    segs = [{"start": 0, "end": 2, "text": "hello there да нет"}]  # bare list
    clip, hyp = hyp_set(tmp_path, {"c.json": json.dumps(segs)})
    s = ev.summarize([ev.score_hypothesis(clip, hyp)])
    assert s["languages"]["all"]["wer"] == 0.0
    assert s["languages"]["all"]["lid_accuracy"] is None and s["der"] is None


def test_hypothesis_txt_is_wer_only_over_the_clip(tmp_path):
    clip, hyp = hyp_set(tmp_path, {"c.txt": "Hello there. Да!"})
    r = ev.score_hypothesis(clip, hyp)
    assert r["text"] == {"mixed": {"words": 4, "edits": 1, "utts": 2, "lid_ok": 0}}
    s = ev.summarize([r])
    assert s["languages"]["all"]["wer"] == 0.25 and s["languages"]["all"]["lid_accuracy"] is None
    assert s["der"] is None and s["rtf"] is None


@pytest.mark.parametrize(
    ("files", "msg"),
    [
        ({}, "no c.json or c.txt"),
        ({"c.json": json.dumps([{"start": 0, "end": 1}])}, r"segment 0 \(KeyError 'text'\)"),
        ({"c.json": json.dumps([{"start": 2, "end": 1, "text": "a"}])}, "start <= end"),
        ({"c.json": json.dumps([{"start": 0, "end": 9, "text": "a"}])}, "after the audio"),
        ({"c.json": json.dumps([{"start": 0, "end": 1, "text": "a", "x": 1}])}, "unknown key"),
        ({"c.json": json.dumps({"segs": []})}, r"file \(KeyError 'segments'\)"),
        ({"c.json": json.dumps({"segments": [], "elapsed_s": -1})}, "elapsed_s"),
        ({"c.json": '{"segments": [], "elapsed_s": NaN}'}, "elapsed_s"),
        ({"c.json": "{secret words"}, r"file \(JSONDecodeError\)"),
        ({"c.json": json.dumps([{"start": "secret words", "end": 1, "text": "a"}])}, "ValueError"),
    ],
)
def test_bad_hypothesis_files_name_the_clip_but_no_values(tmp_path, files, msg):
    clip, hyp = hyp_set(tmp_path, files)
    with pytest.raises(ValueError, match=msg) as e:
        ev.score_hypothesis(clip, hyp)
    assert str(e.value).startswith("clip c: ") and "secret" not in str(e.value)


def test_hypothesis_json_bom_int_speakers_and_json_before_txt(tmp_path):
    segs = [
        {"start": 0, "end": 1, "text": "hello there", "speaker": 0},
        {"start": 1, "end": 2, "text": "да нет", "speaker": 1},
    ]
    files = {"c.json": "﻿" + json.dumps(segs), "c.txt": "something else"}
    clip, hyp = hyp_set(tmp_path, files)
    r = ev.score_hypothesis(clip, hyp)
    assert r["diarization"]["der"] == 0.0 and r["text"]["en"]["edits"] == 0  # .json wins
    assert "diarization" not in ev.score_hypothesis(clip, hyp, diarize=False)


def test_main_scores_hypotheses_without_the_pipeline(tmp_path, monkeypatch, capsys):
    import recandcc.config as config
    import recandcc.transcribe as transcribe

    def boom(*a, **k):
        raise AssertionError("the pipeline must not run")

    monkeypatch.setattr(config, "load_config", boom)
    monkeypatch.setattr(transcribe, "Transcriber", boom)
    _, hyp = hyp_set(tmp_path, {"c.txt": "hello there да нет"})
    out = tmp_path / "r.json"
    assert ev.main([str(tmp_path), "--hypotheses", str(hyp), "--json", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["backend"] == "external" and report["language_id"] == "engine's own"
    assert report["summary"]["languages"]["all"]["wer"] == 0.0
    assert report["wall_s"] is None and report["peak_memory_mb"] is None  # the engine's, not ours
    printed = capsys.readouterr().out
    assert "hypotheses hyp [external]" in printed and "language-ID n/a" in printed
    assert "wall n/a s, peak memory n/a MB" in printed
    assert "1 clip(s) without a time, not in RTF" in printed
    assert "да" not in out.read_text(encoding="utf-8") + printed  # no text
    for flag in (["--backend", "cpu"], ["--model", "small"], ["--live"], ["--languages", "he"]):
        with pytest.raises(SystemExit):
            ev.main([str(tmp_path), "--hypotheses", str(hyp), *flag])


def test_hypothesis_txt_single_language_clip(tmp_path):
    write_set(tmp_path, {"id": "c", "audio": "c.wav", "utterances": [U]})
    (tmp_path / "c.txt").write_text("да", encoding="utf-8")
    (clip,) = ev.load_manifest(tmp_path)
    assert ev.score_hypothesis(clip, tmp_path)["text"] == {
        "ru": {"words": 1, "edits": 0, "utts": 1, "lid_ok": 0}
    }


def test_peak_memory_falls_back_to_psutil_then_na(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "resource", None)  # as on Windows: import fails
    info = SimpleNamespace(peak_wset=300 * 2**20)
    proc = SimpleNamespace(memory_info=lambda: info)
    fake = SimpleNamespace(Process=lambda: proc)
    monkeypatch.setitem(sys.modules, "psutil", fake)
    assert ev.peak_memory_mb() == 300.0
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert ev.peak_memory_mb() is None


def test_mlx_peak_memory(monkeypatch):
    from types import SimpleNamespace

    fake = SimpleNamespace(get_peak_memory=lambda: 2 * 2**30)
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    assert ev.mlx_peak_mb() == 2048.0
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert ev.mlx_peak_mb() is None
