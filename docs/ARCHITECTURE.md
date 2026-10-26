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

### Performance (M5 Max, 18 cores, 120 s en/ru worst case, large-v3-turbo)

| workers × threads | int8 RTF | float32 RTF |
|---|---|---|
| 1 × 18 | 0.358 | 0.497 |
| 2 × 9 | 0.307 | 0.295 |
| 4 × 4 | 0.315 | **0.249** |
| 6 × 3 | 0.305 | 0.234 |

RTF = processing time ÷ audio duration (lower is better). The default is
`N = max(1, min(4, cores // 4))` workers with `cores // N` threads each, and **float32 on
macOS** (Apple Accelerate) / int8 elsewhere. A 1-hour meeting is ready about 15 minutes after
it ends. Real meetings with longer same-language stretches are faster than this test.

### Live transcription (`transcribe/live.py`)
One `LiveTranscriber` per recording session, using the app's single `Transcriber` (same
models, same pool), so nothing extra is loaded.

1. **Tap.** `Recorder.add_listener(cb)` delivers each block as `cb(track, start_frame, frames)`
   on the capture thread; the listener only enqueues. `start_frame` is the block's position in
   the track's WAV, so a mic track that joins mid-session (padded with silence) arrives with
   its true offset without the zeros being sent. A listener added after the start first gets
   the last 10 s replayed, so nothing is lost between `start()` and attaching.
2. **Commit.** Every 0.3 s the worker runs Silero VAD over each track's *uncommitted tail* only.
   A speech region (padded 0.2 s) is committed once ≥ 0.4 s of silence follows its padded end
   (≈ 0.6 s pause). Someone talking non-stop is cut after 15 s at the quietest 100 ms between
   8 and 15 s. Regions < 0.3 s are ignored. Committed/silent audio is dropped from memory (the
   WAV has it): per track at most ~16 s is buffered, whatever the session length.
3. **Transcribe.** Each region is a pool job: language ID with the small model among
   `languages`; if unsure (< 0.75), a short region (< 2 s) within 3 s of the previous region
   of its track inherits that region's language (it waits for that job's language decision,
   which is safe because the pool is FIFO), otherwise the main model decides. Then Whisper
   with that language, beam 5. Regions of both tracks run in parallel; results are released
   in per-track order.
4. **Emit.** New `Others` segments go out at once. `Me` segments wait until the `Others` track
   is transcribed past their end (max 2 s) and are dropped from the captions if they are
   echo of `Others` (same check as the final merge). `transcript.md` is rewritten at most
   every 3 s (merged, echo removed, atomic) with a "recording in progress" note.
5. **End.** On stop the Controller calls `flush(session)` right away: the trailing region is
   committed even without silence after it, in-flight jobs finish, and the final
   `transcript.{md,json,srt}` are written by the live worker (session marked transcribed,
   `keep_audio` honoured) — typically within one Whisper pass (~2.5 s) of the stop. The
   queue's finalize job just waits for that. If live transcription failed as a whole (model
   could not load, every region failed), the job falls back to the full transcription. On
   Discard, `cancel()` stops the work and guarantees no file is written afterwards, before
   the engine deletes the directory.

Measured on the M5 Max, `large-v3-turbo`, float32, audio fed in real time (9 en/ru phrases
on two tracks): phrase end → text on screen **median 3.4 s** (range 3.2–3.9 s): ~0.8 s
until the pause is confirmed and the next scan, 0.16 s language ID, **~2.45 s Whisper**. The
Whisper pass costs the same for a 1 s or a 15 s phrase, because the encoder always processes a
30 s window; feeding a shorter window cuts it to ~0.5–1 s but makes Whisper repeat or invent
text, so it is not used. Beam 1 instead of 5 saves only ~0.15 s. Shortening the pause (0.4 s)
saves ~0.1 s at the cost of more, shorter regions. `workers = 1` (one worker with all 18
threads) brings it to **2.6 s** (1.7 s Whisper pass) but serialises the two tracks. The
`small` model takes ~0.7 s per pass (~1.7 s latency) but is clearly less accurate.

### Ordering of UI events
`live_start` is posted when the recording has started and the live transcriber is attached
(`engine.state.session_dir` is already set). `live` events follow as phrases are
transcribed. `live_end` is posted after the final transcript has been written for a kept
session (so the tail is shown first), or right after `cancel()` on Discard / a session too
short to keep. The usual `transcribed` event follows when the queue's finalize job ends.

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
