# 0010. Offline-first model loading

- Status: Accepted
- Date: 2026-10-26

## Context
RecAndCC records private conversations and promises local processing. Model loading,
however, contacted Hugging Face on every load (a revision check per model), so the app
needed the network to start transcribing and contacted a third party every time it did
(5847503).

## Decision
Resolve every model from the local cache first and download only when it is missing:
- **faster-whisper** (main and `base` language-ID model): `WhisperModel(name,
  local_files_only=True)`, falling back to a normal load (download once) if the model is not
  on disk (`transcribe/engine.py`).
- **MLX Whisper**: `huggingface_hub.snapshot_download(repo, local_files_only=True)` and pass
  the **local snapshot path** to `mlx_whisper` instead of the repo id; download once if it
  is missing (`transcribe/gpu.py`, `local_model_path`).
- **Diarization model**: a single 28 MB ONNX file fetched once over HTTPS, **sha256-verified**,
  stored in `platformdirs.user_cache_dir("RecAndCC")/models`; a failed download is retried at
  most every 10 minutes and diarization is skipped meanwhile ([0009](0009-sherpa-onnx-campp-diarization.md)).

## Alternatives considered
- **Default library behaviour (revision check on every load)**: always up to date, but needs
  the network and contacts Hugging Face on each start.
- **`HF_HUB_OFFLINE=1` globally**: also offline, but then a missing model can never be
  downloaded and first-run setup breaks.
- **Bundling models in the app**: fully offline from install, but ~4 GB in the bundle and
  conflicts with the ad-hoc signing approach ([0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md));
  part of the packaging question in [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3).

## Consequences
- Verified: the real-model en/ru tests pass with all network access blocked and zero HTTP
  requests; diarization was likewise tested offline.
- The first run still needs the network to download the models.
- Risks: cached Whisper models are never updated automatically and are not pinned to a
  revision or checksum (only the diarization model is sha256-checked); a corrupt cache entry
  fails the local load and triggers a re-download attempt. Pinned model revisions are
  [#14](https://github.com/Galina-Blokh/rec_and_cc/issues/14).

## Links
- `src/recandcc/transcribe/engine.py` (`_load`), `src/recandcc/transcribe/gpu.py`
  (`local_model_path`), `src/recandcc/transcribe/diarize.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Speaker diarization" (offline download)
- Issues [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3), [#14](https://github.com/Galina-Blokh/rec_and_cc/issues/14),
  [#17](https://github.com/Galina-Blokh/rec_and_cc/issues/17) (privacy)
- Commits 5847503, 10a9899
