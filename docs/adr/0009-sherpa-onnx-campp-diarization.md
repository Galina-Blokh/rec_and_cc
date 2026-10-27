# 0009. sherpa-onnx CAM++ diarization instead of pyannote

- Status: Accepted (step 3's fixed threshold amended by [0013](0013-adaptive-speaker-count.md);
  step 1's fixed windows by [0014](0014-overlap-aware-segmentation.md))
- Date: 2026-10-26

## Context
The `Others` track ([0003](0003-two-tracks-me-others-echo-dedupe.md)) mixes everyone else on
the call. Users want `Speaker 1`, `Speaker 2`, … in the final transcript. It has to run
locally, offline after a one-time download, without an account, and fast enough not to delay
the transcript ready ~1 s after a live session stops ([0006](0006-live-transcription-architecture.md)).

## Decision
`transcribe/diarize.py`, final transcript only (captions keep `Others`, `Me` is never changed):
1. Silero VAD (the same one used for chunking) finds speech in `others.wav`; regions are cut
   into ~2 s windows.
2. One 192-d embedding per window with **3D-Speaker CAM++** (zh+en "advanced", 28 MB ONNX,
   Apache-2.0) through **sherpa-onnx**, on `min(6, cores // 2)` threads.
3. Average-linkage agglomerative clustering on cosine distance (threshold 0.5) of windows
   ≥ 1 s; clusters with < 2 s of speech and shorter windows join the nearest speaker.
4. Each `Others` segment takes the speaker it overlaps most; numbered by first appearance;
   a single speaker stays `Others`.

It runs on the CPU alongside Whisper (batch) or from `flush()` (live). `write_final` waits up
to 5 s, then writes `Others` and rewrites with labels when diarization finishes.

## Alternatives considered
- **sherpa-onnx's own pyannote pipeline** (segmentation-3.0 + embeddings per 10 s window): on
  the same synthetic en/ru voices it took **20 s for 13 min** of audio and got 141/170 phrases
  right, versus **1.5 s per 10 min** and 170/170 here. It does detect speaker changes without
  a pause and overlapping speech, which the window approach resolves only to ~2 s.
- **pyannote's own Hugging Face models**: need an account/token, which conflicts with
  offline, no-account use.

## Consequences
- Timing (M5 Max, 6 threads): 2 min of audio 0.5 s, 10 min 1.5 s, 60 min 8.5 s (about half
  of it VAD).
- The model is downloaded once, sha256-verified, to the platform cache dir; offline after
  that ([0010](0010-offline-first-model-loading.md)). If the download fails, diarization is
  skipped (retried at most every 10 min) and transcripts keep `Others`.
- Needs the audio: sessions whose WAVs were deleted are not diarized; WAVs are deleted only
  after diarization when `keep_audio = false`.
- Risks: speaker changes without a pause and overlapping speech; no names or cross-session
  identity; the threshold is tuned on synthetic voices. Follow-ups:
  [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1) (names, many speakers),
  [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13) (DER on real speech).

## Links
- `src/recandcc/transcribe/diarize.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Speaker diarization"; [CONFIGURATION.md](../CONFIGURATION.md) `diarize`
- Issues [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1), [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13),
  [#14](https://github.com/Galina-Blokh/rec_and_cc/issues/14) (pinned model revisions)
- Commits 10a9899, 11b9bca
