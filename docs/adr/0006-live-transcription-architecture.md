# 0006. Live transcription: VAD commit, shared pool, finalize jobs

- Status: Accepted
- Date: 2026-10-26

## Context
In v0.1 the transcript was produced by a full pass after the session stopped, so text was
only available minutes after a meeting. Users want captions while audio plays and the final
transcript right after stopping, without loading a second copy of the models or holding a
whole meeting in memory.

## Decision
- One `LiveTranscriber` per recording session (`transcribe/live.py`), fed by
  `Recorder.add_listener` (the capture thread only enqueues blocks with their absolute track
  offsets, so a mic track joining mid-session lines up).
- **VAD commit**: every 0.3 s Silero VAD scans each track's uncommitted tail. A region is
  committed after ~0.6 s of pause (0.4 s silence after a 0.2 s padded end); non-stop speech is
  force-cut at the quietest point between 8 and 15 s; regions < 0.3 s are ignored. Committed
  audio is dropped from memory (the WAV has it), so at most ~16 s per track is buffered.
- **Shared pool**: each region is a job on the app's single `Transcriber` pool (language ID,
  then Whisper, beam 5), both tracks in parallel; results are released in per-track order.
  `transcript.md` is rewritten at most every 3 s, atomically.
- **Finalize jobs**: on stop the Controller calls `flush()`: the tail region is committed and
  the live worker writes the final `transcript.{md,srt,json}`. The `TranscriptionQueue` gets a
  *finalize* job that waits for that; if live transcription failed it falls back to the full
  batch transcription from the WAVs. Discard calls `cancel()`, which guarantees nothing is
  written before the directory is deleted.

## Alternatives considered
- **Post-session pass only** (`transcription.live = false`, still supported): simpler, but no
  captions and ~15 min (CPU) / ~2.5 min (GPU) of waiting per meeting hour.
- **Streaming Whisper on fixed short windows**: feeding Whisper less than a 30 s window cut a
  pass to ~0.5–1 s but made it repeat or invent text, so phrases are committed by VAD instead.
- **Shorter pause (0.4 s)**: saved ~0.1 s at the cost of more, shorter regions.

## Consequences
- Measured (M5 Max, `large-v3-turbo`): phrase end → caption median 3.4 s on the CPU,
  1.19 s on the GPU ([0004](0004-mlx-gpu-with-faster-whisper-fallback.md)); the final
  transcript is ready ~1 s after stop.
- Live and batch share one pool and one model, so nothing extra is loaded; but a live session
  competes with a queued batch job for the same workers.
- Captions keep `Others`; speaker labels only appear in the final transcript
  ([0009](0009-sherpa-onnx-campp-diarization.md)). Live speaker labels are part of
  [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1).
- Risk: more threads and ordering rules (`live_start` / `live` / `live_end` events, cancel vs
  flush races); covered by tests but subtle to change.

## Links
- `src/recandcc/transcribe/live.py`, `transcribe/queue.py`, `recorder.py`, `ui/captions*.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Concurrency model", "Session lifecycle"; the detailed
  live section is in git history (`git show f35c0ed:docs/ARCHITECTURE.md`)
- [CONFIGURATION.md](../CONFIGURATION.md) `live`, `live_captions`
- Issues [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1), [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13)
- Commits f35c0ed, 275e8a6
