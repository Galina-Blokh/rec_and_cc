# Configuration

File: `~/Library/Application Support/RecAndCC/config.toml` (macOS) or
`%LOCALAPPDATA%\RecAndCC\config.toml` (Windows). It is created with defaults on first run.
Open it from **menu → Edit settings**, or print it with `recandcc config`. Unknown keys are
ignored, and missing keys take their default values.

Changes to `[detection]` are picked up while the app runs. Restart the app after changing
`[transcription]` or `output_dir`.

## Top level

| Key | Default | Description |
|---|---|---|
| `output_dir` | `~/Documents/Transcribes` | Where sessions are saved (`<date>/<time>_<app>/`) |
| `autostart` | `false` | Informational; use **menu → Launch at login** or `recandcc autostart on` |
| `hotkey` | `"<ctrl>+<alt>+r"` (⌃⌥R) | Global start/stop shortcut (toggle; stops any recording). Modifiers `<cmd>`, `<shift>`, `<alt>`/`<option>`, `<ctrl>` (`<win>` = Windows key) + one key: `a`–`z`, `0`–`9`, `f1`–`f12`, `space`. Keys other than F1–F12 need a modifier. On Windows `<cmd>` means Ctrl. `""` disables it. Applied live when the file changes |
| `live_captions` | `true` | Show the floating captions window while recording (needs `transcription.live`) |

## `[detection]`

| Key | Default | Description |
|---|---|---|
| `mode` | `"ask"` | `off` never records · `ask` records and asks whether to keep it · `auto` records automatically |
| `poll_interval_s` | `1.0` | How often audio activity is checked |
| `min_active_s` | `3.0` | An app must be doing audio this long before a recording starts or a prompt appears |
| `stop_after_inactive_s` | `20.0` | Calls: stop when the trigger app has stopped doing audio for this long |
| `stop_after_silence_s` | `180.0` | Calls: stop when all tracks have been below `silence_dbfs` for this long |
| `media_stop_after_inactive_s` | `8.0` | Same as `stop_after_inactive_s`, for media (videos end crisply; calls have pauses) |
| `media_stop_after_silence_s` | `30.0` | Same as `stop_after_silence_s`, for media (e.g. a paused tab that keeps its audio open) |
| `silence_dbfs` | `-50.0` | Level below which audio counts as silence |
| `ask_timeout_s` | `30.0` | The "Keep it?" pop-up closes after this |
| `ask_timeout_action` | `"record"` | What an unanswered pop-up means: `record` keeps the recording (nothing is lost if you're away), `ignore` discards it. In Ask mode recording always starts right away; the pop-up only decides whether to keep it |
| `record_mic` | `"calls"` | `always` · `calls` (calls and manual recordings only) · `never` |
| `room_listening` | `false` | Opt-in **room listening**: while nothing records, keep the mic open and start a `room` session when people talk near the laptop. Menu: **Listen for conversations in the room**. macOS shows the orange mic indicator the whole time. Only when no app is doing audio and no other app uses the mic; a call that starts during a room session turns it into a call session. Needs `mode` other than `off` and `record_mic` other than `never`. Ask/Auto apply as for apps; override with `app_modes = {"@room" = "auto"}` |
| `room_speech_s` | `6.0` | Room listening starts a session once speech totals this many seconds … |
| `room_window_s` | `15.0` | … within this sliding window, so a short remark doesn't start one |
| `room_preroll_s` | `15.0` | Audio from before the trigger put at the start of a room session, so the first sentence is kept (at least `room_window_s`; kept only in memory) |
| `room_stop_after_silence_s` | `90.0` | A room session stops after this long without speech |
| `room_vad_threshold` | `0.5` | Silero speech probability (0–1) that counts as speech; raise it if noise starts sessions |
| `call_apps` | Zoom, Teams, FaceTime, Slack, Discord, Telegram, WhatsApp, Skype, Webex | Patterns for call apps |
| `media_apps` | browsers, QuickTime, VLC, Podcasts, TV | Patterns for media apps |
| `ignore_apps` | Spotify, Music, system sounds, Siri, Voice Memos | Never trigger |
| `app_modes` | `{}` | Per-app mode overrides, e.g. `{"us.zoom" = "auto", "com.google.chrome" = "ask"}` |

**Patterns** are case-insensitive substrings matched against the app's bundle id (macOS,
e.g. `us.zoom.xos`) or process/exe name (Windows, e.g. `zoom.exe`). To find the right
pattern, run `uv run recandcc monitor` while the app plays audio.

Classification order: ignore → call → media → unknown. An unknown app using the mic counts
as a call, and an unknown app that only plays audio counts as media. A media app (a browser)
using the mic is treated as a call (Google Meet). The global `mode = "off"` overrides all
`app_modes`.

## `[transcription]`

| Key | Default | Description |
|---|---|---|
| `model` | `"large-v3-turbo"` | faster-whisper model name (`tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo`), each pinned to an exact repo commit; or a local CTranslate2 model path. A custom repo id needs `allow_unpinned_models` to download |
| `backend` | `"auto"` | `auto` = MLX on the Apple-silicon GPU when available (one pass ~0.13 s vs ~2.5 s on the CPU), else faster-whisper · `mlx` · `faster-whisper` (CPU/CUDA) |
| `device` | `"auto"` | `auto` · `cpu` · `cuda` (NVIDIA on Windows) |
| `compute_type` | `"float32"` on macOS, `"int8"` elsewhere | Quantisation. On Apple silicon float32 is ~20% faster (Accelerate) at ~3 GB RAM; int8 uses ~1 GB |
| `languages` | `["en", "ru"]` | Detect only among these Whisper language codes, e.g. `["en", "ru", "he"]`; `[]` = any language. Codes are lowercased, `"iw"` is read as `"he"`, and unknown codes are dropped with a warning in the log (all invalid, or not a list → the default). Your own spelling stays in `config.toml` when the app saves settings. Hebrew (and other right-to-left languages) is shown right-aligned and right to left in live captions; `transcript.md` wraps its text in Unicode bidi isolates and `transcript.srt` starts its lines with RLM, while `transcript.json` keeps the raw text. Hebrew accuracy with the stock model is not yet tuned (#6) |
| `workers` | `0` | Parallel transcription workers; `0` = auto (`min(4, cores // 4)`) |
| `dedupe_echo` | `true` | Remove mic segments that are speaker echo of the other side |
| `diarize` | `true` | Split `Others` into `Speaker 1`, `Speaker 2`, … (numbered by first appearance) in the final transcript (md/srt/json), with offline speaker diarization of the system-audio track. One other speaker stays `Others`; `Me` is never changed; live captions keep `Others`. Costs ~1.5 s per 10 min of audio (CPU, overlapped with transcription). Needs the audio, so sessions whose WAVs were deleted are not diarized. The 28 MB model is downloaded once to `~/Library/Caches/RecAndCC/models` (Windows: `%LOCALAPPDATA%\RecAndCC\Cache\models`); without it (offline) transcripts keep `Others` |
| `keep_audio` | `true` | `false` deletes the WAVs after a successful transcription (~115 MB per track-hour). WAVs a quit left behind (e.g. before a slow diarization finished) are deleted at the next startup; audio of sessions not yet transcribed is never deleted |
| `audio_retention_days` | `0` | Delete a recording's audio (`*.wav`) this many days after it was transcribed; `0` = keep forever. Transcripts are never deleted. Checked at startup and hourly. Also in **menu → Delete audio after** (Never / 1 / 7 / 30 / 90 days) |
| `live` | `true` | Transcribe while recording: each phrase appears ~1 s after it is spoken (~3 s on the CPU backend) (captions, and `transcript.md` is rewritten every few seconds), and the final transcript is written about a second after the session stops. `false` = transcribe the whole session after it stops |
| `allow_unpinned_models` | `false` | Allow downloading a Whisper model with no pinned revision (a custom Hugging Face repo id). Every offered `model` name is pinned to an exact repo commit for both backends ([`revisions.py`](../src/recandcc/transcribe/revisions.py)); other repos are refused with a logged error unless this is `true`, because upstream could swap their weights. Already-downloaded models and local paths load either way |

With `live = true`, the latency is dominated by one Whisper pass per phrase (~2.5 s for
`large-v3-turbo` with the default 4 workers on an M5 Max). `workers = 1` gives every pass all
cores (~2.6 s phrase-to-text instead of ~3.4 s), but the two tracks then wait for each other
and a post-session transcription is ~40% slower. `model = "small"` gets ~1.7 s, at a clear
loss of accuracy in the final transcript too. Sessions recorded with `live = false`, sessions
left untranscribed when the app quit, and `recandcc transcribe` always use the full
post-session pass.

## Example

```toml
output_dir = "/Users/me/Documents/Transcribes"

[detection]
mode = "ask"
record_mic = "calls"
ignore_apps = ["com.spotify.client", "com.apple.music", "coreaudiod", "recandcc", "rcc-audiotap"]

[detection.app_modes]
"us.zoom" = "auto"
"com.microsoft.teams" = "auto"

[transcription]
model = "large-v3-turbo"
languages = ["en", "ru"]
keep_audio = false
```
