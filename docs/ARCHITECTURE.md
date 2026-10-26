# Architecture

RecAndCC is a Python 3.12 application with a small native helper on macOS. It is organised
as independent modules that only talk to each other through the types in
`src/recandcc/models.py` and the interfaces in `capture/base.py` and `detect/base.py`. Each
platform implements those interfaces, so the engine, recorder, transcriber and UI are shared
across platforms.

```
                ┌──────────────────────── RecAndCC.app (macOS) ────────────────────────┐
                │  Contents/MacOS/RecAndCC   Swift launcher: TCC "responsible process" │
                │      └─ spawns ─► .venv/bin/python -m recandcc run                   │
                │                        ├─ spawns ─► rcc-audiotap monitor   (JSON/s)  │
                │                        └─ spawns ─► rcc-audiotap capture   (PCM)     │
                └──────────────────────────────────────────────────────────────────────┘

  ActivityMonitor ──snapshot()──► Engine ──start/stop──► Recorder ──SessionInfo──► TranscriptionQueue
  (detect/*)          1/s         (state machine)        (capture/* → WAV)         (transcribe/*)
                                     ▲    │ events          │ frame tap                ▲      │
                                     │    ▼                 ▼                          │      ▼
                                     │  session start/end  LiveTranscriber ──finalize──┘   on_done
                                     │  ──────────────►    (transcribe/live.py)
                                     │                      │ live segments
                                   UI Controller (ui/common.py) ◄──────────────── notify, refresh
                                   + rumps (ui/mac.py) / pystray (ui/tray.py) + captions window
```

## Modules

| Module | Responsibility |
|---|---|
| `models.py` | Shared types: `Mode`, `Track`, `TriggerKind`, `AudioProcess`, `Segment`, `SessionInfo`, `SAMPLE_RATE` |
| `config.py` | Dataclass config ⇄ TOML, defaults and app lists, platform paths |
| `detect/` | `ActivityMonitor`: which processes are doing audio input/output right now. `mac.py` (Core Audio process objects via helper), `win.py` (pycaw sessions + registry mic-in-use) |
| `capture/` | `AudioSource`: produces float32 mono 16 kHz frames. `mic.py` (sounddevice), `system_mac.py` (process tap via helper), `system_win.py` (WASAPI loopback) |
| `recorder.py` | One session: sources → queues → WAV writer threads, live levels, silence tracking, frame tap for live transcription |
| `engine.py` | Detection state machine: classify apps, debounce, ask/auto/off, stop conditions |
| `transcribe/` | VAD chunking, language ID, parallel Whisper, merge + echo removal, md/srt/json output, background queue, live transcription (`live.py`) |
| `permissions.py` | Query, request and open settings for mic and system-audio permissions |
| `ui/` | `common.Controller` (platform-independent glue) + `mac.py` (rumps) / `tray.py` (pystray) |
| `ui/captions*.py` | Live captions window: `captions.py` (pure logic: ordering, trimming, header) + `captions_mac.py` (floating non-activating NSPanel); Windows uses a Tk window in `tray.py` |
| `autostart.py` | LaunchAgent (macOS) / HKCU Run key (Windows) |
| `cli.py` | `recandcc` command line |
| `native/macos/AudioTap` | SwiftPM package: `rcc-audiotap` helper + `RecAndCC` app launcher |

## Concurrency model

The app is built around one rule: **real-time audio threads never block**. Every slow step
(resampling, disk, inference, UI) sits behind a queue on its own thread.

| Thread / process | Does | Talks via |
|---|---|---|
| `rcc-audiotap capture` (process) | Core Audio IO proc → copy → conversion queue → stdout; silence-filler timer | pipe (raw f32le) |
| `rcc-audiotap monitor` (process) | Polls Core Audio process list | pipe (JSON lines) |
| capture reader (per source) | Reads the pipe / PortAudio callback → `on_frames` (+ frame listeners, which only enqueue) | `queue.Queue` |
| WAV writer (per track) | Writes PCM_16 WAV, so disk I/O never stalls capture | owns its file |
| monitor reader | Parses JSON, keeps the latest snapshot (lock-protected) | `snapshot()` |
| engine | `tick()` loop: the only thread that starts/stops the recorder | command queue + one lock |
| transcription dispatcher | Processes sessions in order, one at a time: a full transcription, or the "finalize" job of a live session (waits for `LiveTranscriber.finish()`) | `TranscriptionQueue` |
| live worker (per recording session) | Buffers the tap's frames, VAD every 0.3 s, submits ended phrases, emits results, rewrites `transcript.md`, writes the final transcript | input `SimpleQueue`; callbacks |
| transcription workers (N) | Whisper chunks / live phrases from **both tracks concurrently** (shared by batch and live) | `ThreadPoolExecutor` |
| UI main thread | rumps/AppKit; drains an event queue every 0.5 s | event queue |
| UI workers | Permission checks, recent-transcript scan, `osascript` dialogs | event queue |

- The engine holds one lock and never makes a blocking call while holding it. UI actions
  (`start_manual`, `stop`, `respond`) are queued as commands, so only the engine thread
  touches the recorder.
- AppKit objects are touched only on the main thread. Worker threads post events that the
  main-thread timer applies.
- Engine → Controller: `on_session_start(session)` / `on_session_end(session | None, kept)`
  run on the engine thread, outside the lock. They create / flush / cancel the session's
  `LiveTranscriber`; `on_session_end` runs before the session is queued or deleted.
- CTranslate2 releases the GIL, so the transcription workers run in true parallel. One
  `WhisperModel(num_workers=N, cpu_threads=T)` is shared by all of them.

## Key design decisions

### macOS system audio: Core Audio process taps (macOS 14.4+)
No virtual audio driver (BlackHole) and no screen-recording permission are needed. A
`CATapDescription(stereoGlobalTapButExcludeProcesses:)` captures everything the Mac plays
except the app itself, through a private aggregate device. The helper converts the audio to
16 kHz mono with `AVAudioConverter` and streams raw float32 to Python.

A process tap delivers **no IO cycles while nothing is playing**. To keep the "others" track
aligned with the mic track and continuous on disk, the helper fills those gaps with zeros,
paced by the wall clock.

### Permissions belong to RecAndCC, not Terminal/python
macOS (TCC) attributes permission requests to the *responsible process*. `RecAndCC.app`'s
executable is a tiny Swift launcher that `posix_spawn`s Python as a child, so the app stays
responsible, the prompts say "RecAndCC", and the grant survives restarts. The launcher sets
`RCC_APP_BUNDLE`. The helper deliberately lives **outside** the bundle in `<repo>/bin`: TCC
charges its audio access to the responsible app wherever the binary is. Keeping it out means
helper updates don't change the app's ad-hoc signature, which would make macOS treat
RecAndCC as a new app and drop its permissions.
System-audio permission status/request uses the TCC SPI (`TCCAccessPreflight` /
`TCCAccessRequest`, loaded with `dlopen`). If that SPI is missing, the helper falls back to
starting a tap, which triggers the system prompt.

### Detection = who is doing audio I/O, not "is there sound"
Core Audio reports per process whether input or output is running
(`kAudioProcessPropertyIsRunningInput/Output`); Windows reports active audio sessions. The
engine matches bundle ids against call / media / ignore lists, requires 3 s of continuous
activity (debounce), and applies the per-app or global mode. Level-based silence (recorder
dBFS) is a second stop condition, for apps that keep the audio device open while paused.

### Two tracks → speaker labels for free
Mic → `Me`, system → `Others`. A mic track that joins mid-session (a media recording upgraded
to a call) starts with silence up to that moment, so both tracks share one timeline. If you use speakers instead of headphones, the mic also picks
up the other side. Echo removal drops a `Me` segment that overlaps an `Others` segment in time
and has similar text (`difflib` ratio ≥ 0.6).

### Mixed English/Russian
Whisper normally detects one language per 30 s window. Instead:
1. Silero VAD splits each track into speech regions.
2. A small `base` model detects the language of **each region**, restricted to
   `languages = ["en", "ru"]`. This prevents ru→uk/bg misdetection and costs 0.14 s per call,
   versus 2.5 s with the large model. Short, uncertain regions take the language of the
   region next to them; low-confidence chunks (< 0.75) are re-checked with the main model.
3. Same-language regions are merged into chunks of ≤ 30 s (gaps ≤ 2 s) and transcribed with
   a fixed language.

In testing, 26 of 26 alternating en/ru sentences came out with the correct language. With
per-chunk detection, English sentences were labelled `ru` and some were dropped.

### Performance: GPU (MLX) with CPU threads alongside

On Apple silicon the main model runs as **MLX Whisper on the GPU** (`transcribe/gpu.py`,
`backend = "auto"`). MLX must be driven from one thread, so every GPU call goes through a
dedicated `whisper-gpu` thread. The CPU worker pool keeps doing VAD, language ID with the
small `base` model and bookkeeping in parallel, and hands phrases to the GPU queue. Measured
on an M5 Max, `large-v3-turbo`:

| | CPU (faster-whisper, 4 × 4 threads, float32) | GPU (MLX) |
|---|---|---|
| One Whisper pass (any phrase ≤ 30 s) | ~2.45 s | **~0.13 s** |
| Live: phrase end → text on screen (median) | 3.4 s | **1.19 s** |
| Full session, 120 s en/ru two-track worst case | RTF 0.249 (4.0× realtime) | **RTF 0.039 (25.8× realtime)** |
| 1-hour meeting, post-session pass (`live = false`) | ~15 min | **~2.5 min** |

On the GPU most of the remaining live latency is waiting ~0.6 s to be sure the phrase ended,
plus ~0.16 s of language ID. CPU-only machines (Windows, `backend = "faster-whisper"`) use
`N = max(1, min(4, cores // 4))` workers × `cores // N` threads, float32 on macOS / int8
elsewhere; CPU numbers for other worker splits are in git history (docs/ARCHITECTURE.md
before the GPU backend).

## Session lifecycle

```
detected ─(active 3 s)─► recording ──────────────────────────────────────────────┐
   ask mode: "Keep it?" pop-up → Keep/Always/no answer: keep · Discard: stop+delete │
   call app appears during a media recording → upgrade: + mic track, padded        │
stop: app inactive 20 s │ silent 3 min │ manual stop ◄─────────────────────────────┘
  stop(): flush writers, save session.json ─► (< 3 s audio? delete) ─► TranscriptionQueue
  ─► transcript.md/.srt/.json, session.transcribed = true ─► (keep_audio = false? delete WAVs)
```

With `transcription.live` (default):

```
recording ── frames ──► LiveTranscriber: phrases → captions + transcript.md (every ≤ 3 s)
stop ─► flush(): tail phrase, final transcript.md/.srt/.json, transcribed = true ─► live_end
     └► TranscriptionQueue "finalize" job waits for it (⏳) ─► transcribed event
discard / too short ─► cancel() (nothing written) ─► live_end ─► directory deleted
```

At startup, sessions with `transcribed = false` (for example after quitting mid-queue) are
queued again.
