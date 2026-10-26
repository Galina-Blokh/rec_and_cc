# Changelog

All notable changes to RecAndCC are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security
- CI installs exactly what `uv.lock` pins (`uv sync --locked` / `uv run --locked`), and its
  GitHub Actions are pinned to commit SHAs.
- Dependabot opens weekly, grouped update PRs for GitHub Actions and Python (uv) dependencies.
- Whisper models (MLX and faster-whisper) are downloaded at pinned Hugging Face commits
  (`src/recandcc/transcribe/revisions.py`); offline loading uses the same pinned snapshot.
- Ruff's bandit rules (`S`) run in CI; every exception is justified inline or scoped per file.
- Folders created for recordings and transcripts are owner-only (`0700`).
- Speaker-model downloads refuse non-HTTPS URLs (in addition to the SHA-256 check).

### Added
- `SECURITY.md`: how to report a vulnerability and a threat-model summary.
- README "Privacy & consent": consent laws, where data lives, retention, deleting everything.
- This changelog.

## [0.1.0] — 2026-10-26

First version: a local, private, always-on recorder and transcriber for calls, meetings and
videos.

### Added
- **Recording**: system audio ("Others") and microphone ("Me") as separate tracks, on macOS
  through a small Swift helper (`rcc-audiotap`, Core Audio process taps) and on Windows
  through WASAPI loopback (untested on real hardware).
- **Detection**: recognises calls and media playback per app, with modes Off / Ask / Auto,
  per-app overrides and ignore lists. Ask mode starts recording immediately and asks whether
  to keep it. A call that starts during a media recording is followed.
- **Transcription** with Whisper (`large-v3-turbo` by default): VAD chunking, per-region
  language detection among the configured languages, mic echo removal, output as
  `transcript.md`, `.json` and `.srt`.
- **Live transcription** while recording, with a floating live-captions window.
- **Apple-silicon GPU** transcription via MLX, with CPU threads (language ID, VAD) alongside.
- **Speaker diarization** of the "Others" track (`Speaker 1`, `Speaker 2`, …), offline, with
  a checksum-verified speaker model.
- **Global start/stop shortcut** ⌃⌥R (Ctrl+Alt+R on Windows), configurable as `hotkey`.
- **Audio retention**: delete a recording's audio N days after transcription
  (`audio_retention_days`, menu → Delete audio after); transcripts are kept.
- **Offline**: once models are downloaded, nothing contacts the network.
- Menu-bar app (macOS) / tray icon (Windows), in-app toast notifications, first-run
  permissions dialog, launch at login, command line (`recandcc run|record|transcribe|monitor|
  devices|autostart|config`).
- macOS permissions survive code updates (reproducible, ad-hoc signed app bundle).
- **CI** on macOS and Windows, and an advisory AI review of every pull request by the Cursor
  agent (four parallel reviewers), hardened against prompt injection from PR content.

### Fixed
- UTF-8 file I/O on Windows.
- Crash when requesting the system-audio permission.
- Media sessions stop sooner after playback ends; no false "permission missing" warning.

[Unreleased]: https://github.com/Galina-Blokh/rec_and_cc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Galina-Blokh/rec_and_cc/releases/tag/v0.1.0
