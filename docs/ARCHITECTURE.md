# Architecture

> Why things are built this way, with the alternatives and measurements: [Architecture Decision Records](adr/README.md) (`docs/adr/`).

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
                │                        ├─ spawns ─► rcc-audiotap capture   (PCM)     │
                │                        └─ spawns ─► rcc-audiotap hotkey    (JSON/s)  │
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
| `transcribe/diarize.py` | Offline speaker diarization of the `Others` track (speaker embeddings + clustering) and `Others` → `Speaker N` relabelling of the final transcript; one-time, checksum-verified model download |
| `hotkey.py` | Global start/stop shortcut: parses `hotkey`, runs `rcc-audiotap hotkey` (macOS) or `RegisterHotKey` (Windows) |
| `cloudsync.py` | Is `output_dir` synced to iCloud / OneDrive / File Provider? Local path checks only; the controller warns once per folder |
| `permissions.py` | Query, request and open settings for mic and system-audio permissions |
| `ui/` | `common.Controller` (platform-independent glue) + `present.py` (pure presenter logic: menu titles and check marks, notice texts, the ask dialog script and its answers, toast placement) + `mac.py` (rumps) / `tray.py` (pystray) / `toast_mac.py`, thin shells that only apply it |
| `ui/notes.py` | Meeting notes (pure logic, no UI imports): scan session folders newest first, row texts, `summary.md` excerpt, rename (`title` in `session.json`), delete to the Trash (Foundation / `SHFileOperationW`) or via `discard_session`, the macOS dialog scripts; the Windows window is `TkNotes` in `tray.py` |
| `ui/captions*.py` | Live captions window: `captions.py` (pure logic: ordering, trimming, header, panel frame, speaker colours) + `captions_mac.py` (floating non-activating NSPanel); Windows uses a Tk window in `tray.py` |
| `autostart.py` | LaunchAgent (macOS) / HKCU Run key (Windows) |
| `logging_setup.py`, `metrics.py` | Log file + crash reporting (`sys`/`threading.excepthook` → log with traceback, thread crash → `notice` toast at most once per thread name per minute; `faulthandler` → `<log dir>/faults.log`); per-session pipeline metrics in `session.json` `metrics` + one INFO summary line |
| `transcribe/summary.py` | Opt-in `summary.md` (summary + action items) with a local LLM via mlx-lm: prompts in the meeting's main language, map-reduce over chunks, pinned offline-first model load (`gpu.local_model_path`) |
| `search.py` | Transcript search: incremental SQLite FTS5 index of `transcript.json` segments in the app-data dir (owner-only, never in `output_dir`); `LIKE` fallback without FTS5. With `[search] semantic`, segment embeddings (`seg_vec`) are fused with the BM25 ranking by reciprocal rank |
| `semantic.py` | Opt-in sentence embeddings: pinned, sha256-verified one-time model download, ONNX Runtime + `tokenizers` (already installed via faster-whisper); `embed()` blocks, so only worker threads call it |
| `cli.py` | `recandcc` command line |
| `native/macos/AudioTap` | SwiftPM package: `rcc-audiotap` helper + `RecAndCC` app launcher |

## Concurrency model

The app is built around one rule: **real-time audio threads never block**. Every slow step
(resampling, disk, inference, UI) sits behind a queue on its own thread.

| Thread / process | Does | Talks via |
|---|---|---|
| `rcc-audiotap capture` (process) | Core Audio IO proc → copy → conversion queue → stdout; silence-filler timer | pipe (raw f32le) |
| `rcc-audiotap monitor` (process) | Polls Core Audio process list | pipe (JSON lines) |
| `rcc-audiotap hotkey` (process) | Carbon `RegisterEventHotKey` + AppKit run loop; one line per press; exits on stdin EOF | pipe (JSON lines) |
| hotkey reader | Reads the helper (restart with backoff) / Windows: `GetMessage` loop for `WM_HOTKEY`; calls `Controller.hotkey_pressed()` (debounced 0.5 s → `start_manual`/`stop` commands + `notice` event) | engine commands, event queue |
| capture reader (per source) | Reads the pipe / PortAudio callback → `on_frames` (+ frame listeners, which only enqueue) | `queue.Queue` |
| WAV writer (per track) | Writes PCM_16 WAV, so disk I/O never stalls capture | owns its file |
| monitor reader | Parses JSON, keeps the latest snapshot (lock-protected) | `snapshot()` |
| engine | `tick()` loop: the only thread that starts/stops the recorder; on exit it calls the activity monitor's `release_thread()`, so per-thread resources (Windows COM) are released on the thread that created them; `shutdown()` then calls `close()` | command queue + one lock |
| transcription dispatcher | Processes sessions in order, one at a time: a full transcription, or the "finalize" job of a live session (waits for `LiveTranscriber.finish()`) | `TranscriptionQueue` |
| live worker (per recording session) | Buffers the tap's frames, VAD every 0.3 s, submits ended phrases, emits results, rewrites `transcript.md`, writes the final transcript | input `SimpleQueue`; callbacks |
| transcription workers (N) | Whisper chunks / live phrases from **both tracks concurrently** (shared by batch and live) | `ThreadPoolExecutor` |
| diarizer (1) + embedding workers (`min(6, cores // 2)`) | Speaker diarization of `others.wav`, one session at a time; then (if it finished late) rewrites the transcript with speaker labels | `Future` + done-callback |
| UI main thread | rumps/AppKit; drains an event queue every 0.5 s | event queue |
| UI workers | Permission checks, recent-transcript scan, `osascript` dialogs, transcript search (syncs the index, posts a `search` event), meeting notes scan / rename / delete (post a `notes` event; a delete also drops the session from the index) | event queue |
| search indexer (per finished transcript) | `TranscriptIndex.index_session` after `on_done`; with semantic search it wakes the single background `search-embed` worker per index (non-blocking guard; embeds outside the write lock, re-checks the transcript stamp and the setting before each insert; `search` never waits for it); errors logged | own SQLite connection |
| summary (1, only with `summary.enabled`) | Loads the summary LLM, writes `summary.md`, frees the model (and MLX's buffer cache); one session at a time. It never competes with a recording: while the engine is not idle (recording or asking) it waits before a session and pauses between model calls (chunk notes, summary, action items). Each call holds `gpu.MLX_LOCK`, which every MLX Whisper call also holds, for its prompt prefill (≤ 8,000 characters, ~1–2 s, in 512-token steps) and then per generated token, so the GPU is never driven from two threads at once and a post-session Whisper pass waits at most one prefill | `SummaryQueue` |

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

With `detection.system_audio_scope = "app"` the engine passes the trigger app as an
`AppTarget` (pattern + pids) through `Recorder.start` to `create_system_source(target)`, and
the helper taps only those processes (`CATapDescription(stereoMixdownOfProcesses:)`). Each
second it re-matches the pattern and updates the running tap's description, so browser or
Electron helpers that start later are included; `pid` / `match` lines on its stdin add a
second app when a call upgrades a media session. No process object yet, or a failing
per-app tap, falls back to the global tap
([ADR 0012](adr/0012-per-app-system-audio-tap.md)).

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

### Global shortcut without extra permissions
NSEvent global monitors, CGEventTaps and pynput see every keystroke, so macOS requires
Accessibility / Input Monitoring for them. Carbon's `RegisterEventHotKey` only delivers the
one registered combination and needs no permission. It is still the supported API for this
(no modern replacement) and works in a background process with an AppKit run loop, which is
why it lives in the helper (`rcc-audiotap hotkey "<cmd>+<shift>+r"`): registered with
`kEventHotKeyExclusive`, so a combination another app already holds fails with a clear
`{"error":"taken"}` (exit 2) instead of silently sharing it. A registered hot key takes
precedence over the focused app's own shortcut. The helper matches the physical key, so the
shortcut doesn't depend on the keyboard layout (en/ru). Its stdin is a pipe from Python and
EOF ends it, so it never outlives the app. The Windows backend uses `RegisterHotKey`
(`MOD_NOREPEAT`) on a thread with a `GetMessage` loop.

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

### Speaker diarization: `Others` → `Speaker 1`, `Speaker 2`, …
The system-audio track mixes everyone else on the call. Only the **final** transcript splits
it; live captions keep `Others`, and `Me` is never touched (`diarize.py`):

1. Silero VAD (the one used for chunking) finds speech in `others.wav`; regions are cut into
   ~2 s windows.
2. One 192-d speaker embedding per window with **3D-Speaker CAM++** (zh+en "advanced",
   28 MB ONNX, Apache-2.0) through **sherpa-onnx**, on `min(6, cores // 2)` threads (ONNX
   Runtime releases the GIL).
3. Average-linkage agglomerative clustering on cosine distance of the windows ≥ 1 s. The
   number of speakers is picked adaptively ([ADR 0013](adr/0013-adaptive-speaker-count.md)):
   the k ≤ 20 with the best cluster silhouette (read off the linkage matrix as it merges),
   one speaker if no split scores ≥ 0.3, within the optional `min_speakers` / `max_speakers`
   hints; with fewer than 10 windows the fixed threshold 0.5 decides. Clusters with < 2 s of
   speech and shorter windows join the nearest speaker. Clustering 3000 windows (the
   subsample cap; an hour is ~1800) takes ~0.15 s.
4. Each `Others` segment takes the speaker it overlaps most (nearest one if none), numbered by
   first appearance. If only one speaker is found the label stays `Others`.

Why not sherpa-onnx's own pyannote pipeline (segmentation-3.0 + embeddings per 10 s window):
on the same synthetic en/ru voices it took **20 s for 13 min** of audio and got 141/170
phrases right, against **1.5 s per 10 min** and 170/170 here. It does find speaker changes
without a pause and overlapping speech, which this window approach resolves only to ~2 s.
pyannote's own HF models need an account/token, so they are out.

Timing (M5 Max, 6 threads): 2 min of audio 0.5 s, 10 min 1.5 s, 60 min 8.5 s (about half
of it VAD). It runs on the CPU and overlaps other work: the batch pass starts it together
with Whisper; a live session starts it at `flush()`, while the tail phrase is transcribed.
`write_final` waits up to `DIARIZE_WAIT_S` (5 s) for it; after that, the transcript is
written with `Others` straight away and rewritten with labels when diarization ends (WAVs
are deleted only after that, if `keep_audio = false`). No WAV (deleted) or `diarize = false`
→ skipped. The model is downloaded once (sha256-checked) to
`platformdirs.user_cache_dir("RecAndCC")/models`; if that fails (offline) diarization is
skipped, retried at most every 10 minutes, and transcripts keep `Others`. Once it is on
disk, nothing touches the network (tested with all network access blocked).

**Speaker names (opt-in, `speaker_names`, `voices.py`).** Diarization also returns each
speaker's centroid (unit mean of its window embeddings). With the option on, the final pass
names a speaker whose centroid matches a voice profile (cosine ≥ 0.6 and 0.1 ahead of the
next profile; each name once per session, `Me` never; numbers skip names) and saves the
session's centroids, so `recandcc name-speaker` can name a speaker later without re-running
diarization. Profiles (sum + count + embedding model, one JSON file per name, so a corrected
name can take a session's centroid out again) and the session centroids (`sessions/`, keyed
by session id, with the session folder: pruned once it is deleted) live in the owner-only (0700/0600)
`platformdirs.user_data_dir("RecAndCC")/voices`, never in `output_dir`, which may be synced
to a cloud drive. A transcript written without labels or with the option off deletes its
session's centroids. `name-speaker` marks the session as folded into the profile before it
changes the profile, so a failure partway can't count one session twice. All of it runs on the diarizer / writer threads, after the audio threads.
`intro_hints.py` only reads a transcript: `recandcc speakers` suggests a name for a
`Speaker N` who introduces themself in their first 2 minutes ("I'm Anna", "Меня зовут Анна",
"קוראים לי דנה"); strict patterns, nothing stored, renamed or logged.

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
  ─► transcript.md/.srt/.json (+ Speaker N labels), session.transcribed = true ─► (keep_audio = false? delete WAVs)
```

With `transcription.live` (default):

```
recording ── frames ──► LiveTranscriber: phrases → captions + transcript.md (every ≤ 3 s)
stop ─► flush(): tail phrase ∥ diarization, final transcript.md/.srt/.json, transcribed = true ─► live_end
     └► TranscriptionQueue "finalize" job waits for it (⏳) ─► transcribed event
discard / too short ─► cancel() (nothing written) ─► live_end ─► directory deleted
```

At startup, sessions with `transcribed = false` (for example after quitting mid-queue) are
queued again. Discarding a session drops a `.discarded` marker in its folder before
`recorder.stop()` (or, if that fails, renames `session.json` to it); the files are then
deleted one by one on a background thread, the marker last. If some file can't be deleted
(Windows locks an open WAV) the user gets a notice, the leftover is never transcribed or
listed, and deletion is retried at the next start, after the re-queueing. `session.json` is
written atomically (temp file + `os.replace`, retried briefly and then written in place if
another process holds it open on Windows).
