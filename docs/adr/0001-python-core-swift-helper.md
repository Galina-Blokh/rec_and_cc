# 0001. Python core + small Swift helper on macOS

- Status: Accepted
- Date: 2026-10-26

## Context
RecAndCC needs three kinds of work: OS-level audio plumbing (system-audio capture,
per-process activity, privacy permissions, a global shortcut), local speech models (Whisper,
VAD, speaker embeddings) and the glue around them (detection state machine, recorder,
transcription queue, menu-bar UI, CLI). The speech stack (faster-whisper/CTranslate2, MLX
Whisper, Silero VAD, sherpa-onnx) is most mature from Python. Core Audio process taps, TCC
and Carbon hot keys are only reachable from Swift/Objective-C. Windows support was a goal from
v0.1, so the core had to stay platform-neutral.

## Decision
- The application is **Python 3.12** (`src/recandcc`): engine, recorder, transcription, live
  pipeline, UI (rumps/PyObjC on macOS, pystray on Windows) and CLI.
- macOS-only native work lives in a small **Swift** helper, `rcc-audiotap`
  (`native/macos/AudioTap`), run as child processes with subcommands `monitor` (JSON lines),
  `capture` (raw float32 PCM on stdout), `hotkey` (JSON lines) and permission queries. A
  second Swift target is the `RecAndCC.app` launcher ([0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md)).
- Modules talk through `models.py` and the `AudioSource` / `ActivityMonitor` interfaces, so
  each platform plugs in its own backends (`capture/`, `detect/`).

## Alternatives considered
- **Swift-only macOS app** (WhisperKit, SwiftUI): best native UX, but macOS only and a
  rewrite of the tested pipeline.
- **Rust core + Tauri + whisper.cpp**: one small cross-platform binary, but a rewrite, and
  capture still needs per-OS native code; MLX speed would have to be matched.
- **Native shell per OS + Python ML sidecar**: native UX, but two UIs plus an IPC layer.
- **All in Python via PyObjC/ctypes for Core Audio**: no second language, but process taps,
  TCC SPI calls and Carbon callbacks are fragile from Python, and an audio callback in the
  Python process would compete with the GIL. The helper keeps real-time audio out of Python.

## Consequences
- Heavy work runs in native code that releases the GIL (CTranslate2, MLX, ONNX Runtime), so
  Python is not the bottleneck: live captions ~1.19 s behind speech, a 1-hour meeting in
  ~2.5 min ([0004](0004-mlx-gpu-with-faster-whisper-fallback.md)).
- The helper is a crash boundary: a fault there (e.g. the `TCCAccessRequest` SIGTRAP fixed in
  444a8f7) surfaces as an exit code, not a dead app.
- The Carbon hot key lives in the helper too, which needs no Accessibility permission. Its
  default is ⌃⌥R so that Chrome/Safari's ⌘⇧R keeps working (10a9899).
- Costs: the app runs Python from the repo's `.venv`, so moving the folder breaks it; two UI
  stacks; a large environment plus ~4 GB of models; Windows backends untested on hardware.
  Packaging and a possible rewrite are open in [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3)
  and [#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5).

## Links
- [ARCHITECTURE.md](../ARCHITECTURE.md) (modules, concurrency model)
- `native/macos/AudioTap/Sources/rcc-audiotap/`, `src/recandcc/native.py`, `src/recandcc/hotkey.py`
- Issues [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3) (architecture review),
  [#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5) (Windows)
- Commits fa1e57d (v0.1), 10a9899 (hotkey)
