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
- Every Whisper model the docs offer (`tiny` to `large-v3-turbo`, both backends, and the
  `large`/`turbo` aliases) is pinned to a commit; an unpinned or custom repo that isn't on
  disk is refused unless `transcription.allow_unpinned_models = true` (#48).
- A model already on disk that fails to load no longer triggers an online re-download; only
  a genuine cache miss goes online (#47).
- Offline is enforced end to end: a test runs whole sessions with sockets, DNS, UDP and
  subprocesses blocked (#49).
- A warning when the output folder is synced to iCloud, OneDrive, Dropbox, Google Drive or
  Box, in the log, the menu and `recandcc config`; the default folder is unchanged (#43).
- "Open transcripts folder" creates the folder owner-only, like session folders (#32).

### Added
- Per-language model routing, opt-in: `transcription.language_models`, e.g.
  `{ he = "ivrit-ai/whisper-large-v3-turbo-ct2" }`, transcribes Hebrew regions (live and
  final) with the Hebrew-tuned ivrit.ai Whisper, pinned for faster-whisper and MLX, loaded on
  first use; other languages keep the default model (#6).
- `SECURITY.md`: how to report a vulnerability and a threat-model summary.
- README "Privacy & consent": consent laws, where data lives, retention, deleting everything.
- This changelog.
- Search across all transcripts with a local SQLite FTS5 index: `recandcc search` and a
  menu item (#59).
- "Meeting notes…" in the menu and the Windows tray: past sessions newest first, with
  open transcript / summary, reveal, rename (`title` in `session.json`) and delete to the
  Trash (#4).
- Opt-in local meeting summaries (`[summary] enabled = true`, Apple silicon): `summary.md`
  with a short summary and action items after each transcript, from a pinned Qwen2.5 1.5B
  Instruct model (Apache-2.0) via mlx-lm; `recandcc summarize <session_dir>` (#4). The
  larger Qwen2.5 3B is also pinned and selectable with `[summary] model`, but its Qwen
  Research licence allows non-commercial use only: it also needs
  `accept_qwen_research_licence = true`, otherwise a config saved while it was the default
  moves to the 1.5B, and the old ~1.7 GB cache can be deleted.
- Speaker names that carry over between sessions, from opt-in local voice profiles; name a
  speaker once from the CLI (#62).
- `recandcc speakers` suggests a name when a speaker introduces themself in their first
  2 minutes (English, Russian, Hebrew); only a hint, nothing is stored (#1).
- Hebrew in `transcription.languages`, with right-to-left captions and transcripts (#60).
- Record only the app that triggered a session with `detection.system_audio_scope = "app"`
  (macOS per-app audio tap) (#63, #65).
- `scripts/eval.py`: WER per language, language-ID accuracy, DER and RTF on a local,
  consented speech set; see `docs/EVALUATION.md` (#34). `--live` adds live caption latency
  (p50 / p90 / max) by replaying each clip through the live transcriber (#13).
- Requirements docs: intent, user stories with acceptance criteria, and a traceability table
  from stories to modules and tests, checked by a test (#33, #35).
- ADR 0011: stack and packaging after v0.1 (Proposed) (#57).

### Changed
- The macOS environment no longer installs torch (#69) or numba, llvmlite and scipy (#77),
  which mlx-whisper declares but RecAndCC never uses: `.venv` shrinks from 1.3 GB to
  559 MB (part of #3).
- Menu-bar presentation logic moved out of the AppKit/pystray shells into `ui/present.py`,
  so it is unit-tested on every platform (#67).
- CI type-checks `src/` with pyright and enforces a 70 % coverage floor on non-UI code
  (#36, #54, #55).

### Fixed
- With `keep_audio = false`, audio left behind by a quit during slow diarization or a failed
  delete is removed at the next start (#45).
- A discarded session is never transcribed later, even if deleting it failed; the delete is
  retried at startup (#46, #56).

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
- **Crash reports** in a local `faults.log`, a **Reveal logs** menu item, and per-session
  pipeline metrics in `session.json` (#21).

### Fixed
- UTF-8 file I/O on Windows.
- Crash when requesting the system-audio permission.
- Media sessions stop sooner after playback ends; no false "permission missing" warning.

[Unreleased]: https://github.com/Galina-Blokh/rec_and_cc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Galina-Blokh/rec_and_cc/releases/tag/v0.1.0
