"""Live transcription of a session while it records.

One ``LiveTranscriber`` per session, sharing the app's ``Transcriber`` (its model and worker
pool). Data flow::

    capture thread --feed()--> input queue --> live worker thread
                                                 |  per-track tail buffers (numpy blocks)
                                                 |  every STEP_S: Silero VAD on each tail
                                                 |  ended speech region -> job
                                                 v
                                     Transcriber pool (shared, both tracks in parallel):
                                     language ID (base model, inherit/fallback) -> Whisper
                                                 |
                       live worker <-------------+  results collected in per-track order
                         -> on_segments(new)       (captions)
                         -> transcript.md rewrite  (debounced, atomic)
                         -> at the end: final transcript.{md,json,srt}; speaker
                            diarization of the others track starts with the flush, so it
                            overlaps with transcribing the tail. Captions label Others "Speaker N"
                            (live_speakers.py: display only, embeddings on a pool thread)

A region is committed when it is followed by ``commit_silence_s`` of silence before the end of
the buffer, or force-cut at the quietest point once it is ``max_region_s`` long. Audio before
the commit point is dropped from memory (the WAV on disk has it), so memory stays bounded for
sessions of any length.

The worker thread owns all buffers and job bookkeeping; other threads only touch the input
queue and a few flags, so the only shared state is guarded by ``self._lock``.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..metrics import latency_stats, log_session_metrics, rounded
from ..models import SAMPLE_RATE, Segment, SessionInfo, Track
from .engine import MIN_LANGUAGE_CONFIDENCE, SPEAKERS, Transcriber, diarized_tracks
from .live_speakers import LOAD_GRACE_S, SpeakerLabeler
from .merge import ECHO_TOLERANCE_S, dedupe_echo, merge_tracks
from .output import write_live_markdown

EMBED_WAIT_S = 0.5  # max time a finished Others job waits for its speaker embeddings

log = logging.getLogger("recandcc.transcribe.live")

# --- tuning (see docs/ARCHITECTURE.md "Live transcription" for the measurements) -----------
STEP_S = 0.3  # how often the tails are scanned with VAD
COMMIT_SILENCE_S = 0.4  # silence after a (0.2 s padded) region end -> ~0.6 s after speech
MAX_REGION_S = 15.0  # non-stop speech is cut at the quietest point once a region is this long
MIN_CUT_S = 8.0  # ... but not earlier than this into the region
KEEP_S = 1.0  # tail kept while nothing is speaking (VAD context for a starting word)
MIN_REGION_S = 0.3  # shorter regions are ignored
MIN_SILENCE_MS = 500  # a pause this long ends a VAD region
SPEECH_PAD_S = 0.2
MD_INTERVAL_S = 3.0  # transcript.md is rewritten at most this often
ECHO_HOLD_S = 2.0  # max time a "Me" segment waits for "Others" before it is shown
SHORT_REGION_S = 2.0  # short + uncertain language -> inherit the previous region's language
INHERIT_GAP_S = 3.0  # ... if it ended at most this long before
LIVE_BEAM_SIZE = 5

VadFn = Callable[[np.ndarray], list[tuple[float, float]]]
SegmentsCallback = Callable[[list[Segment]], None]


def live_vad(audio: np.ndarray) -> list[tuple[float, float]]:
    """Silero speech regions (seconds, padded by SPEECH_PAD_S) of a tail buffer. Speech that is
    still going on at the end of `audio` ends at len(audio)."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    if len(audio) < SAMPLE_RATE // 4:
        return []
    opts = VadOptions(
        threshold=0.5,
        min_speech_duration_ms=int(MIN_REGION_S * 1000),
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=int(SPEECH_PAD_S * 1000),
    )
    stamps = get_speech_timestamps(audio, opts, sampling_rate=SAMPLE_RATE)
    return [(s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE) for s in stamps]


def quietest_point(audio: np.ndarray, lo: float, hi: float, win_s: float = 0.1) -> float:
    """Time (s) in [lo, hi] at the centre of the lowest-energy `win_s` window."""
    win = max(1, int(win_s * SAMPLE_RATE))
    a, b = int(lo * SAMPLE_RATE), int(hi * SAMPLE_RATE)
    piece = audio[a:b]
    n = len(piece) // win
    if n < 1:
        return hi
    energy = np.square(piece[: n * win].reshape(n, win), dtype=np.float64).mean(axis=1)
    return (a + int(np.argmin(energy)) * win + win // 2) / SAMPLE_RATE


@dataclass(eq=False)
class _Job:
    track: Track
    start: float  # absolute seconds
    end: float
    audio: np.ndarray | None
    prev: _Job | None  # previous job of the same track (language inheritance)
    language: str | None = None
    lang_ready: threading.Event = field(default_factory=threading.Event)
    future: Future | None = None
    emb_futs: list[Future | None] = field(default_factory=list)  # per segment, Others only
    emb_submitted: float = 0.0  # LiveTranscriber.clock() when the embeddings were requested
    emb_deadline: float = 0.0  # LiveTranscriber.clock() after which a missing embedding is skipped
    echo_noted: bool = (
        False  # raw segments already in _shown_others (set while awaiting embeddings)
    )


def _ready_embeddings(job: _Job, n: int) -> list[np.ndarray | None]:
    out: list[np.ndarray | None] = []
    for i in range(n):
        f = job.emb_futs[i] if i < len(job.emb_futs) else None
        try:
            out.append(f.result(0) if f is not None and f.done() else None)
        except Exception:  # noqa: BLE001 - labels are optional
            out.append(None)
    return out


class _TrackBuf:
    """Not-yet-committed tail of one track. Owned by the worker thread."""

    def __init__(self, track: Track, base: int) -> None:
        self.track = track
        self.blocks: list[np.ndarray] = []
        self.base = base  # absolute sample index of the first buffered sample
        self.size = 0
        self.jobs: deque[_Job] = deque()  # submitted, in time order
        self.last_job: _Job | None = None
        self.segments: list[Segment] = []  # all results (raw, before echo removal)
        self.continued = False  # the buffer starts at a forced cut (mid-speech)

    @property
    def end(self) -> int:
        return self.base + self.size

    def append(self, start: int, frames: np.ndarray) -> None:
        if start < self.end:  # overlap (shouldn't happen): keep only the new part
            frames = frames[self.end - start :]
        if frames.size:
            self.blocks.append(frames)
            self.size += frames.size

    def tail(self) -> np.ndarray:
        if len(self.blocks) > 1:
            self.blocks = [np.concatenate(self.blocks)]
        return self.blocks[0] if self.blocks else np.zeros(0, np.float32)

    def drop_before(self, abs_sample: int) -> None:
        n = min(max(0, abs_sample - self.base), self.size)
        if n:
            rest = self.tail()[n:]
            self.blocks = [rest] if rest.size else []
            self.base += n
            self.size -= n

    def reset(self, base: int) -> None:
        self.blocks, self.base, self.size = [], base, 0

    def done_through(self) -> float:
        """Seconds up to which this track's audio is fully transcribed (or known silent)."""
        t = self.base / SAMPLE_RATE
        return self.jobs[0].start if self.jobs else t


class LiveTranscriber:
    """Transcribes one session while it records. See the module docstring.

    `on_segments(segments)` gets newly finalised segments (time-ordered within the call) on the
    worker thread. `on_end()` is called once on the worker thread when it is done: after the
    final transcript has been written by a flush, or after a failure. Not after ``cancel()``.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        session: SessionInfo,
        on_segments: SegmentsCallback | None = None,
        on_end: Callable[[], None] | None = None,
        *,
        vad: VadFn = live_vad,
        step_s: float = STEP_S,
        commit_silence_s: float = COMMIT_SILENCE_S,
        max_region_s: float = MAX_REGION_S,
        min_cut_s: float = MIN_CUT_S,
        keep_s: float = KEEP_S,
        md_interval_s: float = MD_INTERVAL_S,
        echo_hold_s: float = ECHO_HOLD_S,
        beam_size: int = LIVE_BEAM_SIZE,
        labeler: SpeakerLabeler | None = None,
        clock: Callable[[], float] = time.monotonic,
        start: bool = True,
    ) -> None:
        self.transcriber = transcriber
        self.cfg = transcriber.cfg
        self.session = session
        self.on_segments = on_segments
        self.on_end = on_end
        self.vad = vad
        self.step_s = step_s
        self.commit_silence_s = commit_silence_s
        self.max_region_s = max_region_s
        self.min_cut_s = min(min_cut_s, max_region_s)
        self.keep_s = keep_s
        self.md_interval_s = md_interval_s
        self.echo_hold_s = echo_hold_s
        self.beam_size = beam_size
        self.clock = clock
        # "Speaker N" labels for Others captions (display only, see live_speakers.py)
        # Not when the final transcript also numbers Me's speakers (room / started-as-room
        # sessions): it numbers Me first, so live numbers would not match it. Decided here, at
        # the start, so a call upgraded to a room mid-way keeps its labels (documented limit).
        numbers_me = Track.ME in diarized_tracks(session)
        if numbers_me:
            if labeler is not None:
                labeler.close()
            labeler = None
        elif labeler is None and getattr(self.cfg, "live_speaker_labels", False):
            labeler = SpeakerLabeler(clock=clock)
        self._labeler = labeler

        self._inq: queue.SimpleQueue = queue.SimpleQueue()
        self._wake = threading.Event()
        self._lock = threading.Lock()  # guards the flags below + _write_lock ordering
        self._write_lock = threading.Lock()  # held while writing a transcript file
        self._final_session: SessionInfo | None = None
        self._flush = False
        self._cancelled = False
        self._finished = threading.Event()
        self._error: BaseException | None = None
        self._result: list[Segment] = []
        self._diarization: Future | None = None
        self._diarization_started = False

        self._tracks: dict[Track, _TrackBuf] = {}
        self._held: list[tuple[Segment, float]] = []  # Me segments waiting for Others
        self._shown_others: deque[Segment] = deque()  # recent Others (echo check)
        self._dirty = False
        self._last_md = 0.0
        self._ok_jobs = 0
        self._failed_jobs = 0
        self.latencies: list[float] = []  # seconds from region end (buffer time) to emit
        self._input_lag_max = 0.0  # longest a captured block waited in the input queue
        self._lag_from = float("inf")  # set once the model is loaded
        self._flush_at: float | None = None  # clock() when recording stopped

        self._thread = threading.Thread(target=self._run, name=f"live-{session.id}", daemon=True)
        if start:
            self._thread.start()

    # ------------------------------------------------------------------ public API
    def feed(self, track: Track, start_frame: int, frames: np.ndarray) -> None:
        """Frames of `track` starting at sample `start_frame` of the session timeline. Any
        thread; never blocks (signature matches ``Recorder.add_listener``)."""
        if not self._cancelled and not self._flush:
            self._inq.put((track, start_frame, frames, self.clock()))

    def flush(self, session: SessionInfo | None = None) -> None:
        """No more audio: transcribe the remaining tail and write the final transcript (in the
        background; ``finish`` waits for it). `session`: the stopped session (with ended_at /
        final track list), default the one given at construction."""
        with self._lock:
            if session is not None:
                self._final_session = session
            if self._flush_at is None:
                self._flush_at = self.clock()
            self._flush = True
        self._wake.set()

    def finish(
        self, session: SessionInfo | None = None, timeout: float | None = None
    ) -> list[Segment]:
        """Flush, wait for the final transcript (md/json/srt written, session marked
        transcribed) and return its segments. Raises if live transcription failed as a whole
        (nothing transcribed because the model could not be used)."""
        self.flush(session)
        if not self._finished.wait(timeout):
            raise TimeoutError("live transcription did not finish in time")
        if self._error is not None:
            raise RuntimeError(f"live transcription failed: {self._error}") from self._error
        if self._cancelled:
            raise RuntimeError("live transcription was cancelled")
        return list(self._result)

    def cancel(self) -> None:
        """Stop all work and write nothing more (the session is being discarded). Returns
        once no transcript file is being written."""
        with self._write_lock, self._lock:
            self._cancelled = True
        self._wake.set()
        for tb in list(self._tracks.values()):
            for job in list(tb.jobs):
                if job.future is not None:
                    job.future.cancel()
                job.lang_ready.set()

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def segments(self) -> list[Segment]:
        """Merged transcript so far (echo removed). Worker-thread snapshot; for tests/CLI."""
        with self._lock:
            return self._merged()

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        try:
            if self._labeler is not None:
                # load the speaker model now, off the first caption's critical path; started here
                # (not in __init__) so a transcriber that never runs leaks no thread
                self._labeler.warm()
            self._run_inner()
        finally:
            if self._labeler is not None:
                self._labeler.close()

    def _run_inner(self) -> None:
        try:
            self.transcriber.ensure_loaded()
            self.transcriber.start_prefetch()  # cfg.language_models, on their own thread
        except BaseException as e:  # noqa: BLE001
            log.exception("live transcription unavailable: model failed to load")
            self._error = e
        self._lag_from = self.clock()  # audio queued during the model load doesn't count
        try:
            while True:
                self._wake.wait(self.step_s)
                self._wake.clear()
                with self._lock:
                    cancelled, flush = self._cancelled, self._flush
                if cancelled:
                    break
                if self._error is not None:  # no model: don't buffer audio we can't use
                    while not self._inq.empty():
                        self._inq.get_nowait()
                    if flush:
                        break
                    continue
                if self._step(flush):
                    break
        except BaseException as e:  # noqa: BLE001 - report through finish()
            log.exception("live transcription worker crashed")
            self._error = e
        finally:
            self._finished.set()
            if self.on_end and not self._cancelled:
                try:
                    self.on_end()
                except Exception:
                    log.exception("live on_end callback failed")

    def _step(self, flush: bool) -> bool:
        """One worker iteration: buffer new audio, scan, submit, emit. True when done."""
        if flush and not self._diarization_started:
            self._diarization_started = True
            self._start_diarization()
        self._drain()
        for tb in list(self._tracks.values()):
            self._safe_scan(tb, final=flush)
        self._collect(final=False)
        if flush:
            futures_wait(
                [j.future for tb in self._tracks.values() for j in tb.jobs if j.future is not None]
            )
            self._collect(final=True)
            self._write_final()
            return True
        self._maybe_write_md()
        return False

    def _drain(self) -> None:
        while True:
            try:
                track, start, frames, queued_at = self._inq.get_nowait()
            except queue.Empty:
                return
            if queued_at >= self._lag_from:
                self._input_lag_max = max(self._input_lag_max, self.clock() - queued_at)
            frames = np.asarray(frames, dtype=np.float32).reshape(-1)
            tb = self._tracks.get(track)
            if tb is None:
                tb = self._tracks[track] = _TrackBuf(track, start)
            elif start > tb.end:
                # A gap in the timeline (frames never delivered): finish what is buffered,
                # then continue at the new position.
                self._safe_scan(tb, final=True)
                tb.reset(start)
            tb.append(start, frames)

    def _safe_scan(self, tb: _TrackBuf, final: bool) -> None:
        try:
            self._scan(tb, final)
        except Exception:
            log.exception("live VAD of %s failed; skipping its buffered audio", tb.track)
            tb.drop_before(tb.end)

    def _scan(self, tb: _TrackBuf, final: bool) -> None:
        audio = tb.tail()
        if audio.size == 0:
            return
        dur = audio.size / SAMPLE_RATE
        regions = self.vad(audio)
        if tb.continued and regions and regions[0][0] < 1.0:
            # the tail starts at a forced cut, mid-speech: VAD may miss the onset
            regions[0] = (0.0, regions[0][1])
        tb.continued = False
        commits: list[tuple[float, float]] = []
        resume: float | None = None  # where the next scan starts (relative seconds)
        for s, e in regions:
            if final or dur - e >= self.commit_silence_s:
                commits.append((s, e))
                continue
            if dur - s >= self.max_region_s:  # talking non-stop: cut at the quietest point
                cut = quietest_point(
                    audio, s + self.min_cut_s, min(dur - 0.2, s + self.max_region_s)
                )
                commits.append((s, cut))
                resume = cut
                tb.continued = True
            else:  # open region: wait for it to end (keep some context for the VAD)
                resume = max(commits[-1][1] if commits else 0.0, s - self.keep_s)
            break
        if final:
            resume = dur
        elif resume is None:  # nobody speaking at the end: keep only a little context
            resume = max(commits[-1][1] if commits else 0.0, dur - self.keep_s)
        for s, e in commits:
            if e - s >= MIN_REGION_S:
                self._submit(tb, s, e, audio)
        tb.drop_before(tb.base + int(resume * SAMPLE_RATE))

    def _submit(self, tb: _TrackBuf, s: float, e: float, audio: np.ndarray) -> None:
        a, b = int(s * SAMPLE_RATE), int(e * SAMPLE_RATE)
        t0 = tb.base / SAMPLE_RATE
        job = _Job(tb.track, round(t0 + s, 3), round(t0 + e, 3), audio[a:b].copy(), tb.last_job)
        try:
            job.future = self.transcriber.submit(self._work, job)
        except Exception:  # pool shut down (app quitting)
            log.exception("could not submit live region")
            job.lang_ready.set()
            return
        job.future.add_done_callback(lambda _f: self._wake.set())
        tb.jobs.append(job)
        tb.last_job = job

    # --- pool side (runs on a Transcriber worker thread) -------------------------------
    def _language(self, job: _Job, audio: np.ndarray) -> str:
        t = self.transcriber
        lang, conf = t.detect_language(audio)
        if conf >= MIN_LANGUAGE_CONFIDENCE:
            return lang
        prev = job.prev
        if (
            prev is not None
            and job.end - job.start < SHORT_REGION_S
            and job.start - prev.end <= INHERIT_GAP_S
        ):
            prev.lang_ready.wait(60)  # prev was submitted first, so it is already running
            if prev.language:
                return prev.language
        if t.has_language_model:
            lang, conf2 = t.detect_language(audio, main=True)
            log.debug(
                "live %s %.1fs: main-model language %s (%.2f)", job.track, job.start, lang, conf2
            )
        return lang

    def _work(self, job: _Job) -> list[Segment]:
        audio = job.audio  # set at submit, released (None) only at the end of this call
        try:
            if self._cancelled or audio is None:
                return []
            job.language = self._language(job, audio)
        finally:
            job.lang_ready.set()
            job.prev = None  # don't keep a chain of old jobs (and their audio) alive
        if self._cancelled:
            return []
        try:
            segs = self.transcriber.transcribe_audio(
                audio,
                job.start,
                SPEAKERS[job.track],
                job.language,
                beam_size=self.beam_size,
                max_end=job.end,
                wait_for_model=False,  # a routed model still loading: the main model meanwhile
            )
            if job.track == Track.OTHERS and self._labeler is not None and not self._cancelled:
                # never wait here: the pool is shared. _collect attaches what is ready.
                job.emb_submitted = self.clock()
                job.emb_deadline = job.emb_submitted + EMBED_WAIT_S
                job.emb_futs = [
                    self._labeler.submit(_slice(audio, job, s), job.emb_deadline) for s in segs
                ]
                for f in job.emb_futs:
                    if f is not None:  # wake the worker when an embedding lands
                        f.add_done_callback(lambda _f: self._wake.set())
            return segs
        finally:
            job.audio = None

    # --- results ----------------------------------------------------------------------
    def _collect(self, final: bool) -> None:
        """Move finished jobs (in per-track order) into the track results and emit."""
        new_others: list[Segment] = []
        embeddings: list[np.ndarray | None] = []
        now = self.clock()
        for tb in self._tracks.values():
            while tb.jobs and (fut := tb.jobs[0].future) is not None and fut.done():
                if not self._embeddings_settled(tb.jobs[0], final):
                    self._note_echo(tb.jobs[0])  # the text counts for echo checks right away
                    break  # keep per-track order; the deadline bounds this wait
                job = tb.jobs.popleft()
                try:
                    segs = fut.result()
                    self._ok_jobs += 1
                except BaseException:  # noqa: BLE001 - one bad region must not stop live
                    log.exception("live region %s %.1f-%.1fs failed", job.track, job.start, job.end)
                    self._failed_jobs += 1
                    continue
                # latency: audio captured since the speech ended (~ wall time when real-time)
                spoke_until = job.end - SPEECH_PAD_S
                self.latencies.append(max(0.0, tb.end / SAMPLE_RATE - spoke_until))
                with self._lock:
                    tb.segments.extend(segs)
                if segs:
                    self._dirty = True
                if job.track == Track.ME:
                    self._held.extend((s, now) for s in segs)
                else:
                    new_others.extend(segs)
                    if not job.echo_noted:
                        self._shown_others.extend(segs)
                    embeddings.extend(_ready_embeddings(job, len(segs)))
        emit = self._labelled(new_others, embeddings)
        if new_others:  # only recent Others matter for the echo check: bound memory
            horizon = max(s.end for s in new_others) - 600.0
            while self._shown_others and self._shown_others[0].end < horizon:
                self._shown_others.popleft()
        embed_wait = any(
            tb.jobs and tb.jobs[0].future is not None and tb.jobs[0].future.done()
            for tb in self._tracks.values()
        )  # finished jobs still waiting for embeddings (bounded by EMBED_WAIT_S)
        emit += self._release_me(final, now, embed_wait)
        if emit and self.on_segments:
            emit.sort(key=lambda s: (s.start, s.end))
            try:
                self.on_segments(emit)
            except Exception:
                log.exception("live on_segments callback failed")

    def _note_echo(self, job: _Job) -> None:
        """Add a finished Others job's raw segments to the echo bookkeeping before its caption
        is released: a Me segment released meanwhile must already be checked against them."""
        if job.echo_noted or job.future is None:
            return
        job.echo_noted = True
        # a failed job is logged when it is collected
        with contextlib.suppress(Exception):
            self._shown_others.extend(job.future.result())

    def _embeddings_settled(self, job: _Job, final: bool) -> bool:
        """True when the job's embeddings are ready or their deadline passed (then the
        missing ones fall back to the previous label). Bounded by EMBED_WAIT_S per job."""
        pending = [f for f in job.emb_futs if f is not None and not f.done()]
        if not pending:
            return True
        # a cold model load is not counted against the deadline (bounded by LOAD_GRACE_S)
        deadline = (
            self._labeler.effective_deadline(job.emb_submitted, job.emb_deadline)
            if self._labeler is not None
            else job.emb_deadline
        )
        remaining = deadline - self.clock()
        if final and remaining > 0:
            futures_wait(pending, timeout=min(remaining, EMBED_WAIT_S + LOAD_GRACE_S))
            return True
        return remaining <= 0

    def _labelled(self, segs: list[Segment], embeddings: list[np.ndarray | None]) -> list[Segment]:
        """Copies of the Others segments for display, labelled "Speaker N". The stored
        segments (echo check, transcript.md, final transcript) keep "Others"."""
        labeler = self._labeler
        if labeler is None:
            return list(segs)
        return [
            replace(seg, speaker=labeler.label(emb, seg.start, seg.end))
            for seg, emb in zip(segs, embeddings, strict=True)
        ]

    def _release_me(self, final: bool, now: float, embed_wait: bool = False) -> list[Segment]:
        if not self._held:
            return []
        others = self._tracks.get(Track.OTHERS)
        if not self.cfg.dedupe_echo or others is None:
            out = [s for s, _ in self._held]
            self._held = []
            return out
        through = others.done_through()
        out, keep = [], []
        for seg, t in self._held:
            # an Others job blocked only on embeddings must not use up the hold budget
            expired = now - t >= self.echo_hold_s and not embed_wait
            if final or through >= seg.end or expired:
                near = [o for o in self._shown_others if o.start < seg.end + ECHO_TOLERANCE_S]
                if dedupe_echo([seg], near):
                    out.append(seg)
            else:
                keep.append((seg, t))
        self._held = keep
        return out

    def _merged(self) -> list[Segment]:
        me = self._tracks.get(Track.ME)
        ot = self._tracks.get(Track.OTHERS)
        return merge_tracks(
            list(me.segments) if me else [],
            list(ot.segments) if ot else [],
            dedupe=self.cfg.dedupe_echo,
        )

    def _duration(self) -> float:
        return max((tb.end for tb in self._tracks.values()), default=0) / SAMPLE_RATE

    def _maybe_write_md(self) -> None:
        now = self.clock()
        if not self._dirty or now - self._last_md < self.md_interval_s:
            return
        self._dirty, self._last_md = False, now
        with self._write_lock:
            if self._cancelled or not Path(self.session.directory).exists():
                return
            try:
                with self._lock:
                    segs = self._merged()
                write_live_markdown(self.session, segs, self._duration())
            except Exception:
                log.exception("could not write the live transcript")

    def _start_diarization(self) -> None:
        """At the flush (the WAVs are complete): diarize the others track in the background
        while the tail is transcribed; ``write_final`` picks the result up."""
        with self._lock:
            session = self._final_session or self.session
        try:
            self._diarization = self.transcriber.start_diarization(session)
        except Exception:  # noqa: BLE001 - diarization is optional
            log.exception("could not start speaker diarization")

    def _write_final(self) -> None:
        if self._failed_jobs and not self._ok_jobs:
            raise RuntimeError(f"all {self._failed_jobs} live regions failed")
        with self._write_lock:
            if self._cancelled:
                return
            with self._lock:
                session = self._final_session or self.session
                segs = self._merged()
            duration = self._duration()
            if session.ended_at:
                duration = max(duration, (session.ended_at - session.started_at).total_seconds())
            if not Path(session.directory).exists():
                log.info("session %s was deleted; no live transcript written", session.id)
                return
            self._result = self.transcriber.write_final(session, segs, duration, self._diarization)
            self._save_metrics(session)
        log.info(
            "session %s: live transcript final, %d segments (%d regions, %d failed)",
            session.id,
            len(segs),
            self._ok_jobs,
            self._failed_jobs,
        )

    def _save_metrics(self, session: SessionInfo) -> None:
        """Caption latency (phrase end -> emit), input queue lag, time from stop to the
        final transcript; saved into session.json and logged as one line."""
        values = {
            **latency_stats(self.latencies),
            "failed_regions": self._failed_jobs,
            "input_lag_max_s": rounded(self._input_lag_max),
        }
        with self._lock:
            flush_at = self._flush_at
        if flush_at is not None:
            values["final_after_stop_s"] = rounded(self.clock() - flush_at, 2)
        session.update_metrics("live", values)
        try:
            session.save()
        except OSError:
            log.exception("could not save metrics of session %s", session.id)
        log_session_metrics(session)


def _slice(audio: np.ndarray, job: _Job, seg: Segment) -> np.ndarray:
    a = max(0, int((seg.start - job.start) * SAMPLE_RATE))
    return audio[a : max(a, int((seg.end - job.start) * SAMPLE_RATE))]


def live_title(session: SessionInfo) -> str:
    """Captions window title: the trigger app, or "Manual recording"."""
    return session.trigger_app or "Manual recording"


__all__ = ["LiveTranscriber", "live_title", "live_vad", "quietest_point"]
