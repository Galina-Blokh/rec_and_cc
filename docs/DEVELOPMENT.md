# Development

## Setup

```bash
brew install uv                  # or see https://docs.astral.sh/uv/
uv sync                          # creates .venv with Python 3.12 + all deps (incl. dev)
./native/macos/build.sh          # builds bin/rcc-audiotap (macOS helper)
./native/macos/build_app.sh      # builds dist/RecAndCC.app   (--install → ~/Applications)
```

### What to rebuild after a change (and why it matters for permissions)

macOS remembers RecAndCC's permissions by the app bundle's code signature. The app is ad-hoc
signed, so that identity is a hash of the bundle's contents. **If the bundle changes, macOS
sees a new app and you have to grant Microphone and System audio again.** That's why the
bundle holds only the launcher, `Info.plist`, `launcher.json` and the icon:

| You changed | Do this | Permissions |
|---|---|---|
| Python code (`src/`) | Quit and reopen RecAndCC (the app runs the repo's code) | kept |
| Swift helper (`Sources/rcc-audiotap`) | `./native/macos/build.sh`, then restart the app | kept (helper lives in `bin/`, outside the bundle) |
| Launcher, `Info.plist`, icon, or moved the repo | `./native/macos/build_app.sh --install` | **re-grant** (the script tells you) |

`build_app.sh --install` compares a fingerprint of the bundle contents and leaves the
installed app untouched when nothing changed. The build is reproducible: the same sources
give the same cdhash. The repo path is baked into `launcher.json`; `RCC_PYTHON` overrides
the Python path at runtime.

A self-signed code-signing certificate would make permissions survive bundle changes too, by
pinning the identity to the certificate instead of the content hash. That needs a keychain
change and isn't set up.

## Project layout

```
src/recandcc/
  models.py config.py           shared contract (types + settings)
  capture/  detect/             platform backends behind AudioSource / ActivityMonitor
  recorder.py engine.py         session recording + detection state machine
  transcribe/                   chunking, language ID, parallel Whisper, merge, outputs, queue
  permissions.py native.py      TCC permissions + helper discovery (macOS)
  ui/                           common Controller + rumps (mac) / pystray (windows) + live captions
  autostart.py cli.py logging_setup.py
native/macos/AudioTap/          SwiftPM: rcc-audiotap (monitor/capture/permissions/hotkey/settings) + RecAndCC launcher
tests/                          pytest suite
scripts/                        dev tools (captions_demo.py, eval.py)
docs/                           this documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Pull requests and AI review

Changes go through pull requests, not straight to `main`:

```bash
git switch -c feature/short-name
# ... commit ...
git push -u origin feature/short-name
gh pr create --fill
```

- **Keep PRs small: aim for ≲ 400 changed lines.** Split bigger work into a series of
  PRs (for example: refactor first, then the feature, then docs), each green on its own.
  Lock files and generated files don't count, but say so in the description.
- Fill in the PR template ([.github/pull_request_template.md](../.github/pull_request_template.md)):
  linked issue (`Closes #…`), acceptance criteria, tests, AI review findings addressed,
  risk/rollback.
- Rules and commands for AI coding agents (also a good checklist for humans) are in
  [AGENTS.md](../AGENTS.md) at the repo root; `CLAUDE.md` is a symlink to it.

Every PR gets two automatic checks:
- **CI** (`.github/workflows/ci.yml`): lint + tests on macOS and Windows.
- **Cursor review** (`.github/workflows/cursor-review.yml`): **four Cursor agents review in
  parallel**, one per focus area (correctness & threading, security & privacy, platforms,
  tests & performance; see `scripts/review_lanes.sh`), using **GPT-5.6 Sol** (`gpt-5.6-sol-high`;
  change it with the repo variable `CURSOR_REVIEW_MODEL`, list models with
  `cursor-agent models`). A final job merges them into one PR comment. Needs the repo secret
  `CURSOR_API_KEY`; without it (and on fork PRs) the jobs are skipped and stay green, so the
  review is advisory. Parallel lanes buy coverage, not speed: wall time is the slowest lane
  (use `gpt-5.6-sol-high-fast` for quicker reviews).
  Hardening: read-only agent (`--mode ask`, sandboxed) on the trusted base tree with the PR as
  a diff file, instruction files removed, pinned + checksummed agent build, SHA-pinned actions,
  secret-leak guard before posting. See the workflow header for the accepted risk.

Run the same parallel review locally (Cursor CLI installed and logged in, or
`CURSOR_API_KEY` set):

```bash
git diff main...HEAD > /tmp/pr.diff
scripts/review_lanes.sh /tmp/pr.diff > /tmp/review.md        # 4 lanes in parallel
scripts/review_lanes.sh /tmp/pr.diff security                # one lane
```

## Testing

```bash
uv run pytest                    # fast suite (~11 s): no models, no network; coverage report
uv run pytest -m slow            # + real Whisper integration test (en/ru speech made with `say`)
RECANDCC_TEST_MODEL=large-v3-turbo uv run pytest -m slow   # same with the production model
uv run ruff check . && uv run ruff format --check src tests
uv run pyright                   # type check src/ (basic mode; CI runs it in the lint job)
```

- **Engine tests** use a fake monitor, fake recorder and fake clock, so detection scenarios
  (debounce, ask/always/ignore/timeout, stop conditions) run in milliseconds.
- **Recorder tests** feed synthetic sine/silence sources.
- **Transcription unit tests** use a fake model. The `slow` test runs real inference.
- **Accuracy on real speech** (WER, language ID, DER, RTF) is measured with
  `scripts/eval.py` on a local consented set; see [EVALUATION.md](EVALUATION.md).
- **`test_offline_e2e.py`** runs whole sessions, live and batch, through the real
  Controller, Engine, Recorder, queue, outputs and diarization fallback with the network
  blocked: socket connects (loopback too), UDP sends, DNS lookups and launching programs
  raise and fail the test, naming the calling app frame. The only attempt allowed is the
  speaker-model download's connection to its own host, which must give up. C extensions
  that open sockets themselves aren't covered. The `slow` variants load the real Whisper
  models from disk (skipped if not cached); with the speaker model on disk too, they must
  make no attempt at all.
- **`test_mac_native.py`** runs only on macOS when `bin/rcc-audiotap` exists. It plays
  audio with `afplay`/`say` and checks that it is detected and captured.
- **Type checking**: `pyright` in basic mode over `src/` (config in `pyproject.toml`). CI runs it
  on Linux, so Windows- and macOS-only imports carry narrow `# pyright: ignore[...]` comments
  (file-level for the PyObjC modules, which have no stubs). Fix real errors; suppress only
  with a rule name and a reason.
- **Traceability:** each test file's docstring names the user stories it covers
  (`Stories: US-03, US-06.`), matching [requirements/traceability.md](requirements/traceability.md).
  Every test file needs that line; tooling tests say `Stories: none.`.
  `test_traceability.py` fails when a story has no tests, a listed test doesn't exist, or a
  docstring disagrees with the table. A new behaviour gets a story in
  [requirements/stories.md](requirements/stories.md) (or a criterion in an existing one).

CI (`.github/workflows/ci.yml`) runs lint on Linux and the fast suite on macOS and Windows.
It uses `--locked`, so commit `uv.lock` with every dependency change (`uv add` / `uv lock`);
Dependabot (`.github/dependabot.yml`) proposes weekly updates.

**Coverage.** Every `pytest` run prints a coverage report (`pytest-cov`, configured in
`pyproject.toml`). CI fails the fast suite below the floor (`--cov-fail-under=70` in
`ci.yml`); local runs only report, so a full local run can pass below the floor. The AppKit/pystray shells
(`ui/mac.py`, `ui/tray.py`, `ui/captions_mac.py`, `ui/toast_mac.py`) and `cli.py` are not
counted; keep their logic in `ui/common.py` / `ui/captions.py` so it's testable. Raise the
floor when you add tests; never lower it to get a PR green. Check it locally with
`uv run pytest --cov-fail-under=70`. Windows sits lower than macOS (macOS-only tests skip), so
watch the Windows number in CI.

## Useful commands while developing

```bash
uv run recandcc monitor                       # watch detection/classification live
uv run recandcc record --seconds 10 --no-mic  # end-to-end check of capture + transcription
bin/rcc-audiotap permissions                  # TCC status (as seen by the calling app!)
bin/rcc-audiotap hotkey "<ctrl>+<alt>+f9"     # register a shortcut, prints a line per press
bin/rcc-audiotap capture | head -c 64000 > x.raw   # 1 s of f32le 16 kHz mono
tail -f ~/Library/Logs/RecAndCC/recandcc.log
```

### Live captions demo

`scripts/captions_demo.py` shows the macOS captions panel (`ui/captions_mac.py`) without
recording anything. It posts fake `live_start` / `live` / `live_end` events (English and
Russian lines, one per second for ~15 s) through a queue drained by a 0.5 s rumps timer,
the same way `ui/mac.py` does, then quits after the panel auto-hides.

```bash
uv run python scripts/captions_demo.py                       # watch it on screen
uv run python scripts/captions_demo.py --seconds 30 --interval 0.5
uv run python scripts/captions_demo.py --snapshot /tmp/cc.png           # PNG after the end
uv run python scripts/captions_demo.py --snapshot-live --snapshot /tmp/cc.png  # PNG while live
```

`--snapshot` renders the panel window in-process, so it works without Screen Recording
permission for the terminal. The line formatting, ordering and trimming logic lives in
`ui/captions.py` and is covered by `tests/test_captions.py`.

Permissions are tracked per app. When you run things from a terminal, the terminal app is the
one asking. Test permission flows through `RecAndCC.app`.

## Build report

`scripts/report/` turns the repository's own record into one self-contained HTML page: pull
requests, issues, CI and Cursor review runs from GitHub, Claude trailers from `git log`, and
token usage from the local Claude Code session logs (counts only, never prompt text).

```bash
./scripts/report/refresh.sh        # writes .report/report-data.json, report.html, artifact.html
```

`collect.py` gathers everything into `report-data.json` (GitHub responses are cached under
`~/.cache/recandcc-report/`, so a refresh only fetches what changed); `render.py` builds the
page from that file alone, so every number and sentence comes from one place. `refresh.sh`
reads `GH_TOKEN`, or `git_api_key_0307` from `~/.env`, and prints what moved since the last
run. `artifact.html` is the same page without its `<html>` wrapper, for hosts that add
their own. Refresh after each merge; `.report/` is git-ignored. Claude cost is an estimate at
Anthropic list prices (the table is in `collect.py`).

## Conventions

- Python 3.12, type hints, docstrings; `ruff` (line length 100) for lint and format.
- Modules depend only on `models.py`, `config.py` and the `base.py` interfaces, never on each
  other's internals. Platform code is imported lazily through `create_*()` factories.
- Never block a real-time audio thread or the UI main thread. Use queues and worker threads.
- Logging goes through `logging` (`recandcc.*` loggers), never `print`, outside `cli.py`.
