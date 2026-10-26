# Development

## Setup

```bash
brew install uv                  # or see https://docs.astral.sh/uv/
uv sync                          # creates .venv with Python 3.12 + all deps (incl. dev)
./native/macos/build.sh          # builds bin/rcc-audiotap (macOS helper)
./native/macos/build_app.sh      # builds dist/RecAndCC.app   (--install → ~/Applications)
```

Rebuild the helper after changing Swift code. Rebuild the app if you move the repo, because
the repo path is baked into `RecAndCC.app/Contents/Resources/launcher.json`. You can override
it at runtime with the `RCC_PYTHON` environment variable.

## Project layout

```
src/recandcc/
  models.py config.py           shared contract (types + settings)
  capture/  detect/             platform backends behind AudioSource / ActivityMonitor
  recorder.py engine.py         session recording + detection state machine
  transcribe/                   chunking, language ID, parallel Whisper, merge, outputs, queue
  permissions.py native.py      TCC permissions + helper discovery (macOS)
  ui/                           common Controller + rumps (mac) / pystray (windows)
  autostart.py cli.py logging_setup.py
native/macos/AudioTap/          SwiftPM: rcc-audiotap (monitor/capture/permissions) + RecAndCC launcher
tests/                          pytest suite
docs/                           this documentation
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.

## Testing

```bash
uv run pytest                    # fast suite (~11 s): no models, no network
uv run pytest -m slow            # + real Whisper integration test (en/ru speech made with `say`)
RECANDCC_TEST_MODEL=large-v3-turbo uv run pytest -m slow   # same with the production model
uv run ruff check . && uv run ruff format --check src tests
```

- **Engine tests** use a fake monitor, fake recorder and fake clock, so detection scenarios
  (debounce, ask/always/ignore/timeout, stop conditions) run in milliseconds.
- **Recorder tests** feed synthetic sine/silence sources.
- **Transcription unit tests** use a fake model. The `slow` test runs real inference.
- **`test_mac_native.py`** runs only on macOS when `bin/rcc-audiotap` exists. It plays
  audio with `afplay`/`say` and checks that it is detected and captured.

CI (`.github/workflows/ci.yml`) runs lint on Linux and the fast suite on macOS and Windows.

## Useful commands while developing

```bash
uv run recandcc monitor                       # watch detection/classification live
uv run recandcc record --seconds 10 --no-mic  # end-to-end check of capture + transcription
bin/rcc-audiotap permissions                  # TCC status (as seen by the calling app!)
bin/rcc-audiotap capture | head -c 64000 > x.raw   # 1 s of f32le 16 kHz mono
tail -f ~/Library/Logs/RecAndCC/recandcc.log
```

Permissions are tracked per app. When you run things from a terminal, the terminal app is the
one asking. Test permission flows through `RecAndCC.app`.

## Conventions

- Python 3.12, type hints, docstrings; `ruff` (line length 100) for lint and format.
- Modules depend only on `models.py`, `config.py` and the `base.py` interfaces, never on each
  other's internals. Platform code is imported lazily through `create_*()` factories.
- Never block a real-time audio thread or the UI main thread. Use queues and worker threads.
- Logging goes through `logging` (`recandcc.*` loggers), never `print`, outside `cli.py`.
