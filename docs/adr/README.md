# Architecture Decision Records

Short records of the decisions that shape RecAndCC: what was decided, why, what else was
considered and what it costs. The first ten were backfilled from git history and
[ARCHITECTURE.md](../ARCHITECTURE.md) ([#12](https://github.com/Galina-Blokh/rec_and_cc/issues/12));
the numbers in them come from the commits and docs they link to (measured on an M5 Max).

## Index

| # | Decision | Status |
|---|---|---|
| [0001](0001-python-core-swift-helper.md) | Python core + small Swift helper on macOS | Accepted |
| [0002](0002-core-audio-process-tap.md) | Core Audio process tap (global, own process excluded) for system audio | Accepted |
| [0003](0003-two-tracks-me-others-echo-dedupe.md) | Two tracks (mic = `Me`, system = `Others`) + echo dedupe | Accepted |
| [0004](0004-mlx-gpu-with-faster-whisper-fallback.md) | MLX Whisper on the Apple-silicon GPU, faster-whisper CPU fallback | Accepted |
| [0005](0005-per-region-language-id.md) | Per-region language ID restricted to configured languages | Accepted |
| [0006](0006-live-transcription-architecture.md) | Live transcription: VAD commit, shared pool, finalize jobs | Accepted |
| [0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md) | Ad-hoc signing, launcher as TCC responsible process, helper outside the bundle | Accepted |
| [0008](0008-in-app-toast-notifications.md) | In-app toast instead of `osascript` notifications | Accepted |
| [0009](0009-sherpa-onnx-campp-diarization.md) | sherpa-onnx CAM++ diarization instead of pyannote | Accepted |
| [0010](0010-offline-first-model-loading.md) | Offline-first model loading | Accepted |

## Adding an ADR

1. Copy the template below to `docs/adr/NNNN-short-title.md` with the next free number.
2. Fill it in (aim for 25–60 lines). State only what you measured or can point to; link the
   code, the issue and the PR.
3. Add a row to the index above and open the PR together with the change it describes
   (or before it, with status `Proposed`).
4. ADRs are not rewritten later. To change a decision, add a new ADR and set the old one's
   status to `Superseded by NNNN`.

Statuses: `Proposed` · `Accepted` · `Deprecated` · `Superseded by NNNN`.

## Template (MADR-lite)

```markdown
# NNNN. Title in the imperative or as a noun phrase

- Status: Proposed | Accepted | Deprecated | Superseded by NNNN
- Date: YYYY-MM-DD

## Context
The problem, the forces and constraints, what we knew at the time.

## Decision
What we do, stated plainly.

## Alternatives considered
- **Option A**: why not (trade-offs, measurements).
- **Option B**: ...

## Consequences
What gets easier, what gets harder, known risks and follow-ups.

## Links
Code, docs, issues, PRs, commits.
```
