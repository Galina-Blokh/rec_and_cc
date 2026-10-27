# Configuration

File: `~/Library/Application Support/RecAndCC/config.toml` (macOS) or
`%LOCALAPPDATA%\RecAndCC\config.toml` (Windows). It is created with defaults on first run.
Open it from **menu → Edit config.toml** (macOS) or **Edit settings** (Windows tray), or
print it with `recandcc config`. On macOS, **menu → Settings…** edits the common keys in a
window. Unknown keys are ignored, and missing keys take their default values.

A menu change (or `recandcc autostart on/off`) writes only its own key, so other hand edits in
the file are kept; like any save, it drops comments and rewrites the keys in a fixed order.
These hand edits are picked up while the app runs (a few seconds after you save the file):
`[detection]`, `[summary]`, `output_dir`, `hotkey`, `live_captions`, and in `[transcription]`
`languages`, `keep_audio`, `audio_retention_days` and `speaker_names`. Restart the app after
changing any other `[transcription]` key (models, device, ...). Reload and menu saves never
interleave, so a menu change made at the same moment is not lost. Saves take a lock file next
to config.toml (shared with the CLI) and give up after 10 s if another process holds it
("Settings not saved"). A failed menu save puts the setting back to what the file says, except
for settings whose effect already happened (launch at login, live captions, detection mode and
per-app modes, room listening, names from the call app). Those keep the new value, and the app
retries the save every few seconds until it succeeds; meanwhile a reload of config.toml leaves
them alone (a hand edit of that setting is overwritten by the retry). If the app quits before
the retry succeeds, the file's value applies at the next start. A retention change deletes
audio only once it is saved.

## Top level

| Key | Default | Description |
|---|---|---|
| `output_dir` | `~/Documents/Transcribes` | Where sessions are saved (`<date>/<time>_<app>/`). If it is in a known cloud-synced folder (iCloud Drive incl. "Desktop & Documents", `~/Library/CloudStorage`, `~/Dropbox`, `~/Google Drive`, `~/OneDrive*`, `~/Box`; case-insensitive, also through symlinks; Windows: OneDrive, `%USERPROFILE%\Dropbox`, also through junctions; see README → Privacy), the app warns once per folder (log + notification; `recandcc config` too): recordings would be uploaded. Pick a local folder, e.g. `~/Transcribes`, to keep them on this device. The same one-time warning appears for a network volume (SMB/AFP/NFS/WebDAV mount, UNC path or mapped network drive): recordings would leave this device over the network |
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
| `system_audio_scope` | `"all"` | What the `Others` track records. `all`: everything the Mac plays. `app` (macOS): only the app that triggered the session, including helper processes it starts later; a call joining a video session adds the call app. A room session has no app to follow and records `all` until a call joins it; from then on `Others` records only the call app (the moment of the switch is silence; `session.json` notes `all→app`). If the call app has no audio process yet, the room session keeps recording `all`. Manual sessions, apps with no audio process yet, an outdated `bin/rcc-audiotap` and FaceTime (its calls play through the `avconferenced` daemon) record `all`, with a warning in the log; `session.json` notes the scope actually used ([ADR 0012](adr/0012-per-app-system-audio-tap.md)) |
| `room_listening` | `false` | Opt-in **room listening**: while nothing records, keep the mic open and start a `room` session when people talk near the laptop. Menu: **Listen for conversations in the room**. macOS shows the orange mic indicator the whole time. Only when no app is doing audio and no other app uses the mic; a call that starts during a room session turns it into a call session. Needs `mode` other than `off` and `record_mic` other than `never`. Ask/Auto apply as for apps; override with `app_modes = {"@room" = "auto"}` |
| `room_speech_s` | `6.0` | Room listening starts a session once speech totals this many seconds … (must be positive: zero, negative or non-numbers fall back to the default with a warning) |
| `room_window_s` | `15.0` | … within this sliding window, so a short remark doesn't start one. A `room_speech_s` longer than this is clamped to it when used (with a warning in the log; the file keeps your value). With `outside_sensitivity` set the window widens instead. Must be positive, like `room_speech_s` |
| `room_preroll_s` | `15.0` | Audio from before the trigger put at the start of a room session, so the first sentence is kept (at least `room_window_s`; kept only in memory) |
| `room_stop_after_silence_s` | `90.0` | A room session stops after this long without speech |
| `room_vad_threshold` | `0.5` | Silero speech probability (0–1) that counts as speech; raise it if noise starts sessions |
| `outside_sensitivity` | unset | 0–100: how readily the room (mic, room listening) starts a recording. Scales `room_speech_s` by 4× at 0, 1× at 50, 0.25× at 100; the window widens so that speech is at most half of it (0 = 24 s within 48 s). A half-second step only counts if its loud parts are 12 dB (at 0) to 9 dB (at 100) above the room's noise floor: the 35th percentile of the last 20 s, which sits under a conversation's pauses but on top of a fan or café babble. The floor is relearnt when the mic opens, after a pause in the audio of over 5 s (sleep/wake), and when the room suddenly gets louder (carried into a café): the first ~3 s after the mic opens or the Mac wakes are used to learn the room, and speech in them doesn't count. Known limit of level-based gating: voices must be clearly louder than the background, about 10 dB at 0–25 (*Noisy place*) and 8 dB at 50–75. Normal and Noisy place ignore 2–8 talker babble; 100 is for quiet rooms (walking from a quiet room into chatter can start a session). Unset = plain `room_speech_s` / `room_window_s`, no level gate. *Normal* (50/50) keeps those thresholds but is not the same as *Off*: it adds the noise-floor gate and the ~3 s warm-up after each mic open. The gate only decides what starts a session; while a room session records, plain voice detection decides when the talk has ended (`room_stop_after_silence_s`). Non-numbers are ignored with a warning. Apps always take precedence: while any app does audio the room holds off, whatever this is set to. Menu: **Sensitivity** |
| `inside_sensitivity` | unset | 0–100: how readily app audio (calls, videos) starts a recording. Scales `min_active_s` the same way (12 s at 0, 0.75 s at 100 with the default 3 s). Unset = plain `min_active_s`. Presets: Quiet room = 75/50, Normal = 50/50, Noisy place = 25/50 (outside/inside), Off = both unset |
| `call_apps` | Zoom, Teams, FaceTime, Slack, Discord, Telegram, WhatsApp, Skype, Webex | Patterns for call apps |
| `media_apps` | browsers, QuickTime, VLC, Podcasts, TV | Patterns for media apps |
| `ignore_apps` | Spotify, Music, system sounds (`com.apple.systemsound`, `coreaudiod`), Notification Center, Siri, Voice Memos, RecAndCC itself, `afplay`, `rcc-audiotap` | Never trigger |
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
| `languages` | `["en", "ru"]` | Detect only among these Whisper language codes, e.g. `["en", "ru", "he"]`; `[]` = any language. Codes are lowercased, `"iw"` is read as `"he"`, and unknown codes are dropped with a warning in the log (all invalid, or not a list → the default). Your own spelling stays in `config.toml` when the app saves settings. Hebrew (and other right-to-left languages) is shown right-aligned and right to left in live captions; `transcript.md` wraps its text in Unicode bidi isolates and `transcript.srt` starts its lines with RLM, while `transcript.json` keeps the raw text. For Hebrew-tuned transcription see `language_models`. Also in **menu → Languages** (English / Russian / Hebrew / Any language); a change there or in `config.toml` applies from the next piece of audio transcribed, without a restart |
| `workers` | `0` | Parallel transcription workers; `0` = auto (`min(4, cores // 4)`) |
| `dedupe_echo` | `true` | Remove mic segments that are speaker echo of the other side |
| `diarize` | `true` | Split `Others` into `Speaker 1`, `Speaker 2`, … (numbered by first appearance) in the final transcript (md/srt/json), with offline speaker diarization of the system-audio track. One other speaker stays `Others`; `Me` is never changed; live captions keep `Others`. Costs ~1.5 s per 10 min of audio (CPU, overlapped with transcription). Needs the audio, so sessions whose WAVs were deleted are not diarized. The 28 MB model is downloaded once to `~/Library/Caches/RecAndCC/models` (Windows: `%LOCALAPPDATA%\RecAndCC\Cache\models`); without it (offline) transcripts keep `Others` |
| `split_on_speaker_change` | `false` | Opt-in, not yet measured: split an `Others` segment at the word where the diarized speaker changes, so each part gets the right `Speaker N` (runs under 1 s and 3 words are ignored). Needs word timestamps, which cost extra CPU time, so it applies only to batch transcription (`live = false`, `recandcc transcribe`, sessions re-queued at startup) on the faster-whisper backend, or on the MLX (Apple GPU) backend with mlx-whisper 0.4.3 (RecAndCC's numpy word aligner; another mlx-whisper version logs a warning and keeps whole segments). On MLX each window costs an extra forward pass plus CPU alignment while the GPU is held, so it can delay live captions and summaries running at the same time, and its word timings are approximate: they use untuned alignment heads (at most 64), so split points can be 0.3–1 s off and are less precise than faster-whisper's. It does not apply to live sessions (their phrases are transcribed without words); those keep whole segments. Needs `diarize = true`; `Me` is never split |
| `min_speakers` | `0` | Diarization hint: at least this many other speakers (`Others` only, `Me` not counted). `0` = no hint: the count is estimated per session (up to 20). Useful when you know the size of the meeting. Both hints apply to the remote side (the `Others` track) only. A room session diarizes only the room mic, so they are ignored there by design; in a room session a call joined, they apply to its `Others` track |
| `max_speakers` | `0` | Diarization hint: at most this many other speakers (`Others` only, as `min_speakers`); `0` = no hint (estimated, up to 20). If both are set and `min_speakers` is larger, `max_speakers` wins |
| `diarize_overlap` | `false` | Opt-in, experimental: tuned on synthetic voices only, not yet measured on real meetings (#13). Before the speaker embeddings, find speaker changes without a pause and overlapped speech with pyannote segmentation-3.0 (6 MB, downloaded once next to the speaker model only when this is `true`; ~4 s per hour of audio). Sharper speaker boundaries; overlapped speech goes to the dominant speaker and the segment gets `"overlap": true` in `transcript.json` (md/srt unchanged). `false`, the model missing (offline) or broken: fixed ~2 s windows |
| `speaker_names` | `false` | Remember voices so a speaker named once is named in later sessions. Opt-in because voice profiles are biometric data; while `false` nothing is stored. When `true`, each diarized session's speaker voice centroids are kept in the owner-only `~/Library/Application Support/RecAndCC/voices/sessions/` (Windows: `%LOCALAPPDATA%\RecAndCC\voices\sessions`), never in `output_dir`, and `recandcc name-speaker <session_dir> "Speaker 2" "Anna"` saves Anna's profile in `voices/` and rewrites that transcript. Names can't be `Me`, `Others` or `Speaker N`. Later sessions use a name when a speaker's voice matches its profile clearly (cosine ≥ 0.75 and 0.1 ahead of any other profile); each name once per session, `Me` never. `recandcc voices list` / `voices delete <name>` / `voices delete --all` (profiles and session voices). Turning the option off stops storing new voice data; run `recandcc voices delete --all` to remove what is already stored |
| `names_from_call_app` | `false` | macOS only, opt-in, effective only with `speaker_names`. While a call is recorded, read the other participants' display names from the call app's **visible** participants panel (Zoom, Teams, Meet in a browser) through the Accessibility API, every ~15 s, and keep them in the owner-only `voices/sessions/` store (never in the session folder or `output_dir`; pruned with the session, removed by `voices delete --all`). Only the Zoom and Teams apps and, in a browser, a window whose title shows a Meet/Teams/Zoom call page are read (Slack, Discord, WhatsApp, FaceTime, Telegram and others are skipped; with more than one call window open in browsers, or more than one participant list anywhere, nothing is read; the app is matched by bundle id or name at every sample, never by a saved process id); a name counts only when it is in two samples in a row; trust is re-checked at every sample, so granting Accessibility mid-call works. After diarization a `Speaker N` is renamed (the transcript only, no voice profile is created) only when certain: the number of names equals the number of speakers besides `Me`, every voice-matched speaker matches a name (by shared word, case-insensitive), exactly one `Speaker N` and one name remain, and no self-introduction disagrees. Otherwise nothing is guessed (`recandcc speakers` lists the names). `Me` is never renamed. Needs the Accessibility permission, which RecAndCC never asks for by itself: **menu → Use names from the call app** explains it and opens System Settings > Privacy & Security > Accessibility. Without the permission, or without the panel open, nothing is read. Names are never logged. Unused names stay in the owner-only `voices/` store on this Mac (until the session is deleted); a name that was applied is part of the transcript and therefore follows `output_dir`, including any cloud sync of it |
| `keep_audio` | `true` | `false` deletes the WAVs after a successful transcription (~115 MB per track-hour). WAVs a quit left behind (e.g. before a slow diarization finished) are deleted at the next startup; audio of sessions not yet transcribed is never deleted |
| `audio_retention_days` | `0` | Delete a recording's audio (`*.wav`) this many days after it was transcribed; `0` = keep forever. Transcripts are never deleted. Checked at startup and hourly. Also in **menu → Delete audio after** (Never / 1 / 7 / 30 / 90 days) |
| `live` | `true` | Transcribe while recording: each phrase appears ~1 s after it is spoken (~3 s on the CPU backend) (captions, and `transcript.md` is rewritten every few seconds), and the final transcript is written about a second after the session stops. `false` = transcribe the whole session after it stops |
| `language_models` | `{}` | Per-language model routing (opt-in): `{ he = "ivrit-ai/whisper-large-v3-turbo-ct2" }` transcribes every speech region detected as Hebrew with that model and all other languages with `model`, in live captions and the final transcript. The key is a Whisper language code (`"iw"` is read as `"he"`; unknown codes and empty names are dropped with a warning; a language missing from `languages` is never detected, so its entry is unused and logged). The value takes the same forms as `model`. The routed model is loaded in the background when a session starts (if its language is in `languages`), offline-first at a pinned revision like `model`; until it is ready (e.g. during its one-time download) live captions use `model` for that language, while the final transcript waits for it. Pinned Hebrew-tuned [ivrit.ai](https://huggingface.co/ivrit-ai) builds: `ivrit-ai/whisper-large-v3-turbo-ct2` (recommended) and `ivrit-ai/whisper-large-v3-ct2` are CTranslate2 (faster-whisper) builds; on the MLX backend the app loads their MLX conversions `mlx-community/ivrit-ai-whisper-large-v3-turbo-mlx` / `…-large-v3-mlx` instead (either name works on both backends). Other repos follow `allow_unpinned_models`. **Memory:** both models stay loaded, so Whisper needs about twice the RAM: a turbo build adds ~1.6 GB on the GPU (MLX, fp16), ~3 GB on the CPU at `float32` or ~1 GB at `int8`; `large-v3` builds roughly double that. If the routed model can't be loaded (refused unpinned repo, failed download, corrupt cache), its language falls back to `model` with a warning in the log and a one-time notice; the load is retried every 5 minutes (a refused unpinned repo is not) |
| `allow_unpinned_models` | `false` | Allow downloading a Whisper model with no pinned revision (a custom Hugging Face repo id). Every offered `model` name is pinned to an exact repo commit for both backends ([`revisions.py`](../src/recandcc/transcribe/revisions.py)); other repos are refused with a logged error unless this is `true`, because upstream could swap their weights. Already-downloaded models and local paths load either way |

With `live = true`, the latency is dominated by one Whisper pass per phrase (~2.5 s for
`large-v3-turbo` with the default 4 workers on an M5 Max). `workers = 1` gives every pass all
cores (~2.6 s phrase-to-text instead of ~3.4 s), but the two tracks then wait for each other
and a post-session transcription is ~40% slower. `model = "small"` gets ~1.7 s, at a clear
loss of accuracy in the final transcript too. Sessions recorded with `live = false`, sessions
left untranscribed when the app quit, and `recandcc transcribe` always use the full
post-session pass.

## `[summary]`

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | After each session's final transcript, write `summary.md` next to `transcript.md`: a short summary and an **Action items** list, in the meeting's main language (English, Russian and Hebrew headings; other languages get English headings). Runs a local LLM through mlx-lm on its own background thread, never while a recording is in progress (it waits, and pauses between chunks if a recording starts), Apple silicon only; a "Downloading the summary model" notice is shown before the one-time download; elsewhere, or when the model is missing and can't be downloaded (offline), the summary is skipped with a log line and a one-time "Summary skipped" notice. Long meetings are summarised in chunks and then merged. `recandcc summarize <session_dir>` writes one on demand, also when this is `false`. Summaries are drafts from a small model: check names and dates against the transcript. Summaries support English and Russian only; Hebrew needs a different, Hebrew-capable model, which is being researched in #6 |
| `model` | `"mlx-community/Qwen2.5-1.5B-Instruct-4bit"` | mlx-lm model: a Hugging Face repo id or a local path. The default (Qwen2.5 1.5B Instruct, 4-bit, **Apache-2.0**; ~0.9 GB download; RAM while it runs is estimated at ~1-1.5 GB, not yet measured) is pinned to an exact repo commit in [`revisions.py`](../src/recandcc/transcribe/revisions.py) and downloaded once on first use, then loaded from disk without the network; another repo needs `transcription.allow_unpinned_models` to download. `"mlx-community/Qwen2.5-3B-Instruct-4bit"` (the previous default, ~1.7 GB, ~2 GB RAM) is also pinned and may write better summaries, but its Qwen Research licence allows **non-commercial use only**. It needs `accept_qwen_research_licence = true` too |
| `accept_qwen_research_licence` | `false` | Must be the TOML boolean `true` (a string such as `"true"` or a number doesn't count: the app says so in a notice). `true` keeps `model = "mlx-community/Qwen2.5-3B-Instruct-4bit"` (non-commercial use only). With `false`, that model is read as the old default, which earlier versions wrote into every `config.toml`, and the current default is used instead: the switch is written to `config.toml` (that key only) and, with summaries on, shown once as a notice. Exception: while the 3B (its pinned commit) is already downloaded and the new default isn't, the 3B is kept, so offline summaries keep working and nothing is downloaded by surprise. Nothing downloads the new default in that case, so the 3B stays until you set `model` to the 1.5B (the Settings window has no summary-model choice); with summaries on, the app says so once ("Keeping the cached Qwen2.5 3B summary model, which is non-commercial; …", remembered by a `.summary-3b-kept-noticed` file next to `config.toml`). The switch is saved and noticed by the app at startup; CLI commands apply it in memory only. A hand edit of `[summary]` is picked up by the running app's config reload, with the same in-memory switch (only the next start saves it and shows the notice). After the switch, its cached copy (~1.7 GB, `~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit`) can then be deleted |
| `max_tokens` | `400` | Longest generated answer (the summary, the action list, each chunk's notes) |

## `[search]`

| Key | Default | Description |
|---|---|---|
| `semantic` | `false` | Also rank transcript search by meaning, so `payment` finds "settle the invoice" (English and Russian). Off: nothing is downloaded or stored and results are the full-text ones. On: a small multilingual sentence model (`paraphrase-multilingual-MiniLM-L12-v2`, int8 ONNX, 118 MB + 17 MB tokenizer, Apache-2.0, pinned to an exact Hugging Face commit and sha256s in [`semantic.py`](../src/recandcc/semantic.py)) is downloaded once (by the background worker, never while a recording or prompt is active: it pauses or stops if one starts) to `~/Library/Caches/RecAndCC/models` (Windows: `%LOCALAPPDATA%\RecAndCC\Cache\models`) when first needed, then runs offline on the CPU. Segment embeddings are kept in the owner-only `search.sqlite3` next to the text index (never in `output_dir`), added by one background worker as transcripts are indexed (it pauses while a recording, a prompt or the transcription backlog is active, like summaries, so a backlog never competes with live captions; search never waits for it or for the download: until the model is loaded, while the indexer is embedding a chunk, or while a recording, a prompt or the backlog is active, you get full-text results only), tagged with the model that made them, and merged with the full-text ranking by reciprocal rank fusion. Meaning-only matches need a cosine similarity of at least 0.45 and rank after exact word matches. If the model can't be downloaded (offline) search stays full-text and retries later. Switching it off (a hand edit of `config.toml` is applied by the running app at once) stops the embedding worker and any download in progress, deletes the vectors and unloads the model from memory (turning it on reloads it). Only `true` or `false` is accepted: any other value (e.g. the string `"false"`) is logged as a warning and treated as `false`. A query scores the stored vectors in chunks of 2,048 rows, so memory stays flat as the index grows. `recandcc search` is a one-shot process, so it does the work itself: it downloads the model if needed (with a notice on stderr, once), indexes, and waits up to 120 s for the embeddings before searching (on timeout it prints `still indexing; showing what's ready`). While the app is running and the model isn't on disk yet, `recandcc search` leaves the download to the app (which pauses it during recordings) and shows full-text matches; it waits at most 15 s for another process's download, then prints `Semantic search model not available yet: full-text matches only.` The download refuses redirects to anything but https, writes a per-process temp file under an exclusive file lock (`models/.download.lock`, shared by the app and `recandcc search`: the second process waits up to 10 minutes, then uses the finished file). Applies without a restart |

Saving from the **Settings…** window or the **Languages** menu rewrites `config.toml`
through a TOML library: values are kept, comments are not.

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
languages = ["en", "ru", "he"]
keep_audio = false

[transcription.language_models]
he = "ivrit-ai/whisper-large-v3-turbo-ct2"   # Hebrew regions: Hebrew-tuned ivrit.ai Whisper

[summary]
enabled = true
```
