# 0004. MLX Whisper on the Apple-silicon GPU, faster-whisper CPU fallback

- Status: Accepted
- Date: 2026-10-26

## Context
v0.1 ran `large-v3-turbo` with faster-whisper (CTranslate2) on the CPU. With live
transcription ([0006](0006-live-transcription-architecture.md)) one Whisper pass dominated
latency: the encoder always processes a 30 s window, so a 1 s phrase costs as much as a 15 s
one (~2.45 s with 4 workers × 4 threads, float32). Captions lagged a median 3.4 s behind
speech. Windows (and Macs without MLX) still need a working path.

## Decision
- `transcription.backend = "auto"` (default) runs the main model as **MLX Whisper on the
  GPU** on macOS/arm64 when `mlx_whisper` is installed (`transcribe/gpu.py`); `"mlx"` forces
  it; `"faster-whisper"` or `device = "cpu"` force the CPU path.
- MLX must be driven from one thread, so every GPU call goes through a dedicated
  `whisper-gpu` thread. The CPU worker pool keeps doing VAD, language ID with the small `base`
  model ([0005](0005-per-region-language-id.md)) and bookkeeping in parallel.
- The faster-whisper path is unchanged and remains the fallback: `N = max(1, min(4, cores // 4))`
  workers × `cores // N` threads, float32 on macOS, int8 elsewhere; CUDA on Windows.
- Language restriction (`languages`) is applied on the GPU path too.

## Alternatives considered
Measured on an M5 Max, `large-v3-turbo` (275e8a6):

| | CPU (faster-whisper) | GPU (MLX) |
|---|---|---|
| One Whisper pass | ~2.45 s | ~0.13 s |
| Live: phrase end → text (median) | 3.4 s | 1.19 s |
| 120 s en/ru two-track session | RTF 0.249 (4.0×) | RTF 0.039 (25.8×) |
| 1-hour meeting, post-session | ~15 min | ~2.5 min |

- **Tuning the CPU path** (from f35c0ed): `workers = 1` → 2.6 s latency but the two tracks
  serialise and batch is ~40% slower; `model = "small"` → ~1.7 s but clearly less accurate;
  beam 1 instead of 5 saves only ~0.15 s; a shorter-than-30 s window makes Whisper repeat or
  invent text.
- **whisper.cpp (Metal) / WhisperKit (Core ML/ANE)**: not benchmarked yet; that comparison is
  part of [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3) and
  [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13).

## Consequences
- On the GPU most remaining live latency is the ~0.6 s pause needed to confirm a phrase ended,
  plus ~0.16 s of language ID.
- Two backends to keep in sync (segment format, language handling); the MLX model is a
  separate download (`mlx-community/whisper-large-v3-turbo`) from the CTranslate2 one.
- Risk: the single GPU thread serialises all GPU passes, so many concurrent phrases queue.

## Links
- `src/recandcc/transcribe/gpu.py`, `src/recandcc/transcribe/engine.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Performance: GPU (MLX) with CPU threads alongside"
- [CONFIGURATION.md](../CONFIGURATION.md) `backend`, `device`, `compute_type`, `workers`
- Issues [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3), [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13)
- Commits 275e8a6, f35c0ed
