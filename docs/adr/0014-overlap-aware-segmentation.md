# 0014. Overlap-aware segmentation before the speaker embeddings

- Status: Accepted
- Date: 2026-10-27

## Context
[0009](0009-sherpa-onnx-campp-diarization.md) cuts Silero VAD regions into fixed ~2 s windows,
so a speaker change without a pause is resolved only to ~2 s, and a window where two people
talk at once is embedded as a mix and still shapes the clusters. Issue
[#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1) B.1 asks for overlap-aware
segmentation. Constraints: offline after one pinned, checksum-verified download; Windows; an
hour still diarizes in seconds; keep the clustering of [0013](0013-adaptive-speaker-count.md).

## Decision
- Run **pyannote segmentation-3.0** (MIT; the 6 MB ONNX from the sherpa-onnx
  `speaker-segmentation-models` release, pinned by the archive's SHA-256, `model.onnx`
  extracted to the same cache folder as CAM++) directly through ONNX Runtime (already used by
  Silero VAD) on 10 s chunks every 5 s, only where there is speech. Each chunk is trusted for
  its middle 5 s, so every instant is read once. Per ~17 ms frame the argmax powerset class
  (silence, one of 3 local speakers, or a pair) gives **change points** (another class, runs
  < 0.25 s ignored) and **overlapped stretches** (a pair class).
- VAD regions are cut at the change points (no piece < 0.4 s), then into ~2 s windows as
  before. Windows at least half overlapped get clustering weight 0 (not a clustering unit, not
  in the voice centroid) and take the nearest centroid: the dominant voice.
- A segment with ≥ 0.5 s of overlapped turns gets `"overlap": true` in `transcript.json`; the
  key is absent otherwise; md/srt are unchanged.
- **Opt-in** (`diarize_overlap = false` by default): it beat fixed windows on every
  synthetic test below, but nothing was measured on real meetings yet
  ([#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13)); when it is off the model is
  never downloaded. Without the model (offline) it falls back to fixed windows with an INFO
  line; a broken model (load or run error) logs a WARNING and turns it off for that
  diarizer, so the speaker labels are never lost. Its download failure doesn't delay CAM++
  retries (per-model back-off).

## Alternatives considered
- **sherpa-onnx's `OfflineSpeakerDiarization`**: same model but its own clustering and 10 s
  embedding windows, 20 s per 13 min and 141/170 phrases in 0009's test; drops 0013.
- **Overlapping chunks with averaged scores** (pyannote's 1 s step): ~10x the inferences; local
  speaker ids must be aligned across chunks first.
- **On by default**: at least as good on every synthetic test below, but synthetic `say`
  voices are no evidence for real meetings (codecs, crosstalk, noise); revisit with #13.

## Consequences
Synthetic `say` voices, 3 speakers, 9 phrases (fixed windows → with segmentation):

| track | phrases right | frame accuracy (one speaker) | overlap found |
|---|---|---|---|
| 0.7–1.2 s pauses (the integration test) | 9/9 → 9/9 | 1.000 → 1.000 | none marked |
| 0.15 s pauses | 9/9 → 9/9 | 0.848 → 0.943 | — |
| 1 s overlaps | 8/9 → 9/9 | 0.861 → 1.000 | 7.0 of 8 s (7.2 s marked) |
| 0.5 s overlaps | 9/9 (4 speakers) → 9/9 (3) | 0.736 → 0.995 | 2.2 of 4 s |

- Timing (M5 Max, 6 threads): 58 min of audio 8.8 s → 12.4 s; ~8 ms per chunk.
- Risk: a lone speaker with noise could be marked overlapped (no false marks on the tests).
  Constants are tuned on synthetic speech; DER on real meetings is
  [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13).

## Links
- `src/recandcc/transcribe/diarize.py` (`SEGMENTATION_MODEL`, `Segmenter`, `chunk_events`,
  `cut_regions`, `overlap_flags`, `label_speakers`)
- [CONFIGURATION.md](../CONFIGURATION.md) `diarize_overlap`; story US-12
- Issue [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1) part B.1
