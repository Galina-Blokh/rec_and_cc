# AGENTS.md

Shared context and rules for AI coding agents (Claude Code, Cursor, Codex, ...) working in
this repo. `CLAUDE.md` is a symlink to this file; edit `AGENTS.md` only.

## What this is

RecAndCC is an always-on menu-bar app (macOS 14.4+, Windows by design) that notices when the
computer plays or captures voice, records audio only (mic = `Me`, system audio = `Others`),
and turns it into a local transcript with speaker labels (en/ru, Whisper on MLX/CPU, offline
diarization). Python 3.12 plus a small Swift helper (`rcc-audiotap`) and app launcher.

## Docs: read before changing things

- [README.md](README.md): user-facing behaviour, install, privacy.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): modules, **concurrency model**, design decisions.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): setup, what to rebuild, testing, PR workflow.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): every config key. Document new keys here.
- [docs/ROADMAP.md](docs/ROADMAP.md): what's planned.

## Module map (`src/recandcc/`)

```
models.py, config.py        shared contract: types + TOML settings (the only cross-module deps)
detect/                     ActivityMonitor: who does audio I/O (mac.py via helper, win.py pycaw)
capture/                    AudioSource: float32 mono 16 kHz frames (mic, system_mac, system_win)
recorder.py                 one session: sources -> queues -> WAV writer threads, levels, frame tap
engine.py                   detection state machine (debounce, off/ask/auto, stop conditions)
transcribe/                 VAD, language ID, parallel Whisper (gpu.py = MLX), merge, outputs,
                            queue, live.py (live captions), diarize.py (Others -> Speaker N)
hotkey.py, permissions.py   global shortcut; TCC permissions (native.py finds the helper)
ui/                         common.Controller + mac.py (rumps) / tray.py (pystray) + captions*
native/macos/AudioTap/      SwiftPM: rcc-audiotap (monitor/capture/permissions/hotkey) + launcher
```

## Hard rules

1. **Real-time audio threads never block.** Capture callbacks and pipe readers only copy and
   enqueue. Disk, resampling, inference and UI work go behind a queue on their own thread.
2. **AppKit / rumps only on the main thread.** Worker threads post events to the UI event
   queue; the main-thread timer applies them. Same for the captions `NSPanel`.
3. **Engine side effects happen outside its lock.** The engine holds one lock and never makes
   a blocking call (recorder start/stop, callbacks, I/O) while holding it. UI actions are
   queued as engine commands; only the engine thread touches the recorder.
4. **Never rebuild or reinstall `RecAndCC.app`** (`build_app.sh --install`) unless the
   launcher, `Info.plist` or icon changed. A changed bundle resets the user's TCC
   permissions. Python changes need only an app restart; helper changes need
   `./native/macos/build.sh` (the helper lives in `bin/`, outside the bundle).
5. **Offline guarantee.** No network access except the one-time, checksum-verified model
   download. Once models are on disk, nothing contacts the network (`tests/test_offline.py`).
   No telemetry, update checks or new remote dependencies at runtime.
6. **Privacy.** Audio and transcripts never leave the device. Don't log transcript text or
   audio content at INFO or above; don't upload test fixtures made from real recordings.
7. **Windows must keep working.** CI runs the fast suite on Windows. Import platform code
   lazily through the `create_*()` factories, guard macOS-only code by platform, and use
   `pathlib`/`platformdirs` for paths.
8. Modules depend only on `models.py`, `config.py` and the `base.py` interfaces, never on
   each other's internals. Logging via `logging` (`recandcc.*`), never `print` outside `cli.py`.

## Commands

```bash
uv sync                                   # deps (incl. dev) into .venv
uv run pytest -q                          # fast suite: no models, no network; coverage floor
uv run pytest -m slow --no-cov           # + real Whisper / diarization integration tests
uv run ruff check . && uv run ruff format --check src tests
uv run ruff format src tests              # apply formatting
./native/macos/build.sh                   # rebuild bin/rcc-audiotap after Swift helper changes
uv run recandcc monitor                   # watch detection live
```

Run `ruff check` and the fast suite before every commit. Run `-m slow` when touching
`transcribe/` or `capture/`.

## Workflow

- Never commit straight to `main`. Branch (`feat/...`, `fix/...`, `docs/...`), open a PR.
- Keep PRs small: aim for **≲ 400 changed lines**; split larger work into stacked PRs.
- Fill in [.github/pull_request_template.md](.github/pull_request_template.md), including
  `Closes #<issue>`.
- CI and the Cursor review must be addressed before merge: fix the finding, or reply in the
  PR why it does not apply.
- Commit messages in conventional style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `ci:`, `chore:`), imperative, explaining *why*. AI-authored commits end with a trailer, e.g.
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Add or update tests with every behaviour change (engine tests use fakes and a fake clock).
  Update the docs when behaviour, config keys or commands change.

## Don't touch without a reason (say why in the PR)

- `native/macos/build_app.sh`, the launcher sources, `Info.plist`, the icon: TCC reset (rule 4).
- `.github/workflows/*` and `scripts/review_lanes.sh`: CI / review hardening (pinned SHAs,
  checksums, sandboxing).
- `transcribe/diarize.py` model URLs and checksums; model names/defaults in `config.py`.
- `uv.lock`: change it only via `uv add` / `uv lock`, never by hand.
- `bin/`, `dist/`, `.venv/`: build output, not source.
- Never commit secrets (`.env`, tokens) or recordings (`*.wav`, transcripts).
