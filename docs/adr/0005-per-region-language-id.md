# 0005. Per-region language ID restricted to configured languages

- Status: Accepted
- Date: 2026-10-26

## Context
The target users switch between English and Russian within one meeting, sometimes within a
minute. Whisper normally detects one language per 30 s window and then transcribes the whole
window in it. With per-chunk detection, English sentences were labelled `ru` and some were
dropped. Unrestricted detection also confuses Russian with Ukrainian or Bulgarian.

## Decision
1. Silero VAD splits each track into speech regions.
2. A **small `base` Whisper model** (`LANGUAGE_ID_MODEL`, on the CPU) detects the language of
   **each region**, with probabilities renormalised over `transcription.languages`
   (default `["en", "ru"]`; `[]` = any language).
3. Short (< 2 s), uncertain (< 0.75) regions such as "yeah" / "да" take the language of the
   neighbouring region within 2 s (previous preferred). Chunks still below 0.75 confidence
   are re-checked with the main model.
4. Same-language regions are merged into chunks of ≤ 30 s (gaps ≤ 2 s) and transcribed with
   a fixed language. Live transcription applies the same rules per phrase.

5. Optionally, the region's language also picks the model that transcribes it
   (`transcription.language_models`, e.g. Hebrew → the ivrit.ai Whisper): the routed model
   loads lazily on first use through the same pinned, offline-first loader
   ([0010](0010-offline-first-model-loading.md)); on MLX it shares the main model's GPU
   thread. A load failure falls back to the main model.

## Alternatives considered
- **Whisper's own per-window detection with the main model**: 2.5 s per detection versus
  0.14 s with `base` on the CPU, and it mislabelled English sentences in mixed audio.
- **Unrestricted language set**: ru→uk/bg misdetections; restricting to the configured
  languages prevents them.
- **Detect once per chunk with the main model** (`language_model=None`): supported, but it
  doubles the encoder work.

## Consequences
- In testing, 26 of 26 alternating en/ru sentences came out with the correct language.
- Language ID runs on CPU workers while the GPU transcribes
  ([0004](0004-mlx-gpu-with-faster-whisper-fallback.md)); it costs ~0.16 s of live latency.
- A second model is loaded (only when detection is needed).
- Risks: a language not in `languages` is forced into one of them; code-switching inside a
  single VAD region (no pause) gets one language; `base` may be weaker for other language
  pairs. Adding Hebrew is [#6](https://github.com/Galina-Blokh/rec_and_cc/issues/6); real-speech
  evaluation is [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13).

## Links
- `src/recandcc/transcribe/engine.py` (`LANGUAGE_ID_MODEL`, `MIN_LANGUAGE_CONFIDENCE`),
  `transcribe/chunking.py`, `transcribe/merge.py` (`smooth_languages`)
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Mixed English/Russian"
- [CONFIGURATION.md](../CONFIGURATION.md) `languages`
- Issues [#6](https://github.com/Galina-Blokh/rec_and_cc/issues/6), [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13)
- Commit fa1e57d
