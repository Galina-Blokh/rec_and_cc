# 0011. Stack and packaging after v0.1

- Status: Proposed
- Date: 2026-10-27

## Context
Issue [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3) asks whether Python plus a
Swift helper ([0001](0001-python-core-swift-helper.md)) is still the right stack. This record
is longer than the usual 25–60 lines because it compares four options and sets up the follow-up
work. The constraints come from [intent.md](../requirements/intent.md): offline, private, macOS
first with Windows by design, TCC grants that survive updates, live captions ~1 s behind
speech, a 1-hour meeting in ~2–3 min on the GPU, and a codebase one owner can maintain with
AI agents.

**Measured** (M5 Max; ADRs 0004/0006/0009, or `du`/`ps` on the dev machine on 2026-10-27):

| What | Value |
|---|---|
| One Whisper pass, `large-v3-turbo`: MLX GPU / faster-whisper CPU | ~0.13 s / ~2.45 s |
| Live caption latency (median): GPU / CPU | 1.19 s / 3.4 s |
| 1-hour meeting, post-session: GPU / CPU | ~2.5 min / ~15 min |
| Diarization, 60 min of audio | 8.5 s (CPU, 6 threads) |
| Code: `src/recandcc` / Swift (`native/macos`) | 7,784 / 1,186 lines |
| `.venv` on disk | 1.3 GB: `torch` 526 MB, `mlx` 238 MB, `llvmlite` 130 MB, `scipy` 83 MB, `onnxruntime` 75 MB |
| Hugging Face model cache (incl. unused tiny/small) | 3.7 GB |
| RSS, app running (one snapshot, model state not checked) | Python ~190 MB, helpers 12 + 27 MB, launcher 7 MB |

`uv tree --invert` shows that `torch`, `numba`/`llvmlite` and `scipy` come in only through
`mlx-whisper`; RecAndCC doesn't import torch itself. Most of the latency is not the language: on
the GPU, the ~0.6 s pause that confirms a phrase has ended plus ~0.16 s of language ID make up
most of the 1.19 s ([0004](0004-mlx-gpu-with-faster-whisper-fallback.md)), and the heavy work
runs in native code that releases the GIL.

The real problems are those listed in #3 and [0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md):
the app runs from the repo's `.venv`, so moving the folder breaks it; the bundle is ad-hoc
signed, so a changed bundle drops the TCC grants; there are two UI stacks (rumps and pystray/Tk)
and no settings window; and Windows has not been run on real hardware
([#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5)). Neither the language nor the
speed is the problem.

## Decision
1. **Keep the Python core and the Swift helper. Package them properly (option A).** Ship a
   self-contained `RecAndCC.app`: the Swift launcher stays the main executable (TCC
   responsible process), a relocatable Python and the locked dependencies go inside the
   bundle, and the helper moves into the bundle. The whole bundle is signed with a
   **Developer ID**, uses the hardened runtime and is notarized. TCC then identifies the app
   by team ID and bundle ID instead of its content hash, so updates keep the grants (general
   knowledge about macOS code signing; the spike has to verify it). Models are still
   downloaded once at first run ([0010](0010-offline-first-model-loading.md)) and are not
   bundled. Windows gets an installer (Inno Setup first) with the same embedded-Python layout,
   tracked in #5.
2. **Keep MLX (GPU) plus faster-whisper (CPU/CUDA) for inference** until a benchmark on the
   eval set ([EVALUATION.md](../EVALUATION.md)) shows an alternative that is better within
   the limits below.
3. **No rewrite now (option C), and no Swift-only app (D).** Revisit option B (native UI
   shell) after packaging, starting with one SwiftUI settings window hosted by the existing
   helper process rather than a second app.
4. Keep the module boundaries (`AudioSource`, `ActivityMonitor`, `Transcriber`, the UI event
   contract) as the seams that allow any later migration to be incremental.

**Switching gate for a different inference engine.** On the same eval set and machine, the
engine must match or beat MLX on WER for en and ru (within 1 point absolute) and on language-ID
accuracy, stay within 1.2× MLX on live latency and RTF, and offer at least one concrete gain:
≥ 30% less peak memory or energy, a working Windows GPU path, or a ≥ 500 MB smaller bundle.

## Alternatives considered
- **A. Python core, packaged** (chosen). Fixes the problems users notice (folder dependency,
  permission resets) without touching the tested pipeline. Costs (estimates, to be measured
  in the spike): a bundle of roughly 0.6–1.3 GB before models, Python start-up of a few
  seconds, and every native `.so`/`.dylib` must be signed. *Tooling options (general
  knowledge):* PyInstaller and Briefcase freeze imports and can miss dynamically loaded
  libraries (MLX, CTranslate2, ONNX Runtime). A relocatable python-build-standalone runtime
  with `uv sync --frozen` into `Contents/Resources` keeps the normal import system and
  reuses `uv.lock`. The spike picks one of the two on measured results.
- **B. Native shell per OS + Python ML sidecar.** Best native UX (settings, notifications,
  captions in SwiftUI/WinUI), but it means two UI codebases in two more languages, a local IPC
  protocol with its own lifecycle, crash and versioning rules, and the Python sidecar still
  has to be packaged, so it needs A's work anyway. Deferred. The low-cost slice is a SwiftUI
  window in `rcc-audiotap`, which already runs an AppKit run loop and talks JSON lines.
- **C. Rust core + Tauri + whisper.cpp / sherpa-onnx.** Could give the smallest binary and
  idle memory, but means rewriting ~7.8k tested lines, the threading rules (AGENTS.md rules
  1–3) and the live/finalize ordering ([0006](0006-live-transcription-architecture.md)).
  Capture still needs Core Audio tap code and WASAPI code per OS. whisper.cpp's Metal speed
  relative to MLX is **not measured** here. A rewrite also gives AI-assisted maintenance a
  less familiar stack. Revisit only if the benchmark passes the gate *and* packaging A fails
  on size or memory.
- **D. Swift-only macOS app (WhisperKit, Core ML/ANE).** Possibly the lowest power use on
  macOS (ANE inference; general knowledge, **not measured**), but it drops Windows (goal 5)
  and needs a second app for it. WhisperKit could still serve as a macOS *backend* behind
  `Transcriber`, through the helper, if it passes the gate.
- **Inference engines.** MLX: measured, the fastest path here, but macOS/arm64 only and the
  source of ~740 MB of `torch`/`numba`/`scipy` dependencies. faster-whisper/CTranslate2:
  measured on CPU; CUDA on Windows is **not measured** (#5). whisper.cpp (Metal / CUDA /
  Vulkan): one engine for both OSes, **not measured**. WhisperKit: macOS only, Swift,
  **not measured**. Russian accuracy and mixed en/ru per-region language ID
  ([0005](0005-per-region-language-id.md)) must hold up in all of them, which is why the
  gate checks language ID as well as WER.

## Consequences
- The repo-folder dependency and the TCC resets go away once option A ships. The [0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md)
  rule "helper outside the bundle" is superseded by the ADR that records the packaging
  result, and AGENTS.md rule 4 is rewritten then.
- Needs an Apple Developer Program membership and a signing identity kept out of the repo.
  Notarization and the hardened runtime may need entitlements for MLX/numba JIT (risk; the
  spike lists the exact set). Windows Authenticode signing is a separate cost (#5).
- Two UI stacks stay for now; the settings-window spike decides how much of option B is
  worth it.
- Risk: the bundle stays large. Trimming the `mlx-whisper`-only dependencies is its own task.

## Benchmark plan
Uses `scripts/eval.py` on the private eval set (10–20 min of en/ru/code-switched speech),
same machine, on AC power, models already cached, network blocked:
1. Add `--backend {auto,mlx,faster-whisper}` to `eval.py` (today it has only `--model`) and
   report peak RSS and wall time next to WER, language-ID accuracy, DER and RTF.
2. Add `--hypotheses DIR` to score JSON segments produced outside the app (`whisper-cli`,
   `whisperkit-cli`, same model size, beam 5, language restricted to en/ru), so external
   engines are scored by the same code. Mark their language ID as "engine's own".
3. Live latency (not measured by `eval.py` today): replay each clip at real time through
   `LiveTranscriber` and record phrase end → caption (median, p90).
4. Energy: `sudo powermetrics --samplers gpu_power,cpu_power,ane_power` during a 10-minute
   replay, as average W per engine.
5. Packaging: bundle size, cold start to menu icon, idle RSS after 10 min, RSS with models
   loaded, and whether the TCC grants survive installing a rebuilt, re-signed bundle.

## Proposed follow-up issues
1. **Eval: compare Whisper backends on the eval set** (part of #13). AC: `eval.py --backend`
   and `--hypotheses` exist with tests; peak RSS and wall time in `eval-results.json`; a table
   for MLX, faster-whisper CPU, whisper.cpp Metal and WhisperKit on one machine, added to
   EVALUATION.md (scores only); a verdict against the gate above.
2. **Measure live caption latency on the eval set**. AC: real-time replay through
   `LiveTranscriber`; median/p90 phrase end → caption reported by `eval.py`; no transcript
   text in its output.
3. **Spike: self-contained, Developer-ID-signed RecAndCC.app**. AC: the app runs from
   `/Applications` with the repo and `.venv` deleted; helper inside the bundle; notarization
   passes; TCC grants survive installing a rebuilt bundle; size, cold start and idle RSS
   recorded; PyInstaller vs standalone Python + `uv sync --frozen` compared; result recorded
   in a new ADR that supersedes 0007.
4. **Trim runtime dependencies pulled by mlx-whisper**. AC: `torch`, `numba` and `scipy`
   are absent from the shipped environment, or the ADR records why they must stay; slow
   en/ru tests and `tests/test_offline.py` pass; the size change is recorded.
5. **Release pipeline: signed, notarized DMG** (after 3, with #17). AC: one script builds,
   signs, notarizes and staples from a tag; secrets come from the keychain/CI secrets, never
   the repo; the checksum is published with the release.
6. **Windows installer with embedded Python** (part of #5). AC: an Inno Setup installer
   installs to `%LOCALAPPDATA%`, launch at login works, the app runs with no system Python,
   uninstall leaves the transcripts in place; tested on real hardware.
7. **Spike: SwiftUI settings window in the helper** (option B, smallest slice). AC: a
   `rcc-audiotap settings` subcommand edits the same TOML keys through JSON lines over a
   pipe; no new TCC prompts; the effort and the line count are recorded so the owner can
   decide on B.

## Links
- Issues [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3), [#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5),
  [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13), [#17](https://github.com/Galina-Blokh/rec_and_cc/issues/17)
- ADRs [0001](0001-python-core-swift-helper.md), [0004](0004-mlx-gpu-with-faster-whisper-fallback.md),
  [0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md), [0010](0010-offline-first-model-loading.md)
- `pyproject.toml`, `native/macos/AudioTap/Sources/RecAndCC/main.swift`, `scripts/eval.py`,
  [ARCHITECTURE.md](../ARCHITECTURE.md), [EVALUATION.md](../EVALUATION.md)
