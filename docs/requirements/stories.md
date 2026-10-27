# User stories

What RecAndCC does for its user, with stable IDs and acceptance criteria. Reference the IDs
in PR bodies ("Story: US-07") and test docstrings, so every behaviour can be traced to the
tests that check it. The *why* is in [intent.md](intent.md).

**Status:** ✅ done (v0.1) · 🟡 partial · ⬜ planned. Planned stories from the future-work
issues (#1–#7) get their acceptance criteria copied in here when they are picked up.

| ID | Story | Status |
|---|---|---|
| US-01 | Auto-record calls | ✅ |
| US-02 | Auto-record videos without the microphone | ✅ |
| US-03 | Ask mode: record now, decide later | ✅ |
| US-04 | Off mode and per-app modes | ✅ |
| US-05 | Manual recording and global shortcut | ✅ |
| US-06 | Stop automatically when the audio ends | ✅ |
| US-07 | A call during a video recording becomes a call | ✅ |
| US-08 | Separate `Me` / `Others` tracks without echo | ✅ |
| US-09 | English and Russian detected automatically | ✅ |
| US-10 | Fast local transcription | ✅ GPU · 🟡 CPU slower |
| US-11 | Live captions and live transcript | ✅ |
| US-12 | Speaker labels for the other side | ✅ |
| US-13 | Readable transcripts in private folders | ✅ |
| US-14 | Works offline, nothing leaves the device | ✅ |
| US-15 | Delete old audio automatically | ✅ |
| US-16 | Permissions set up from the app | ✅ |
| US-17 | Always on, visible state | ✅ |
| US-18 | Runs on Windows | 🟡 untested on hardware (#5) |
| US-19 | Hebrew transcripts and right-to-left captions | 🟡 no Hebrew test set, ivrit.ai routing not evaluated (#6) |
| US-20 | Search my transcripts | 🟡 search only; summaries, export planned (#4) |
| US-21 | Speaker names remembered across sessions | 🟡 CLI only; UI naming, call-app names, overlaps planned (#1) |
| US-27 | Meeting summaries and action items | 🟡 opt-in, Apple silicon only; no quality evaluation yet (#4) |
| US-29 | Right-to-left text readable in the Windows captions | 🟡 unit-tested, no hardware check (#6) |

---

### US-01 Auto-record calls
*As the owner, I want calls (Zoom, Teams, Meet, FaceTime, Telegram…) recorded without
pressing anything, so that I never forget to record one.*

- **Given** Auto mode, **when** a call app (or an unknown app, or a browser) uses the
  microphone for 3 s, **then** a session starts with both tracks (mic and system audio).
- **Given** a gap in audio activity shorter than 3 s, **when** activity resumes, **then** the
  3 s debounce restarts instead of starting a session.
- **Given** an app in `ignore_apps` or RecAndCC itself, **when** it does audio, **then**
  nothing is recorded.

### US-02 Auto-record videos without the microphone
*As the owner, I want videos and podcasts recorded too, but without my room's sound.*

- **Given** `record_mic = "calls"` (default), **when** a media app (browser, video player)
  plays audio for 3 s, **then** only system audio is recorded.
- **Given** `record_mic = "always"` or `"never"`, **then** the mic follows that policy.

### US-03 Ask mode: record now, decide later
*As the owner, I want to be asked whether to keep a recording, without losing the start
while I decide.*

- **Given** Ask mode, **when** an app becomes active, **then** recording starts immediately
  and a "Keep it?" prompt appears.
- **When** I click **Keep**, **then** the session continues; **Discard** stops and deletes
  it and doesn't ask again until that app's audio stops; **Always** keeps it and sets that
  app to Auto in the settings.
- **Given** no answer within 30 s, **then** the recording is kept (or discarded when
  `ask_timeout_action = "ignore"`).
- **Given** the audio ends or I stop manually before answering, **then** the recording is
  kept.

### US-04 Off mode and per-app modes
*As the owner, I want to switch recording off, or treat specific apps differently.*

- **Given** Off mode, **then** nothing is ever recorded automatically; switching to Off
  stops an automatic recording.
- **Given** `[detection.app_modes]` overrides, **then** the app's own mode wins over the
  global one (an `off` override blocks it, `auto` skips the prompt).

### US-05 Manual recording and global shortcut
*As the owner, I want to record in-person meetings, where no app does audio.*

- **When** I press ⌃⌥R (Ctrl+Alt+R on Windows) or use the menu, **then** a manual session
  starts, or the current session stops; a toast confirms it.
- **Given** a manual session, **then** it is never stopped automatically.
- **Given** the shortcut is taken by another app or invalid, **then** I am told once and can
  change it with `hotkey`.

### US-06 Stop automatically when the audio ends
*As the owner, I want recordings to end by themselves.*

- **When** the active app stops doing audio, **then** the session stops after 8 s (media)
  or 20 s (calls).
- **When** the tracks are silent while the app stays active (a paused tab), **then** the
  session stops after 30 s (media) or 3 min (calls).
- **When** I stop an automatic session manually, **then** it doesn't restart until that
  app's audio stops.

### US-07 A call during a video recording becomes a call
- **Given** a media session is recording, **when** a call starts, **then** the session
  becomes a call: the mic joins, aligned in time, and the call's stop rules apply.

### US-08 Separate `Me` / `Others` tracks without echo
- **Then** the mic is labelled `Me` and system audio `Others`, as separate WAV tracks.
- **Given** the mic picks up the speakers, **then** echoed `Others` speech is not shown again
  as `Me`, live or in the final transcript.
- **Given** `system_audio_scope = "app"` on macOS, **when** an app triggers a session,
  **then** `Others` records only that app's processes (also helpers that start later), and a
  call joining a media session adds the call app. Manual sessions, FaceTime, an app without
  an audio process, an outdated helper or a failing per-app tap record all system audio
  instead, and session.json records which scope was used.

### US-09 English and Russian detected automatically
- **Given** `languages = ["en", "ru"]`, **then** each speech region gets its own language,
  so a mid-meeting switch is transcribed correctly.
- **Given** a short, uncertain region, **then** it inherits the previous region's language.

### US-10 Fast local transcription
- **Then** Whisper runs on the device: MLX on the Apple-silicon GPU with CPU workers
  alongside, faster-whisper on CPU elsewhere. A 1-hour meeting takes about 2–3 minutes
  on the GPU (M5 Max); the CPU-only path (Windows) takes about 15 minutes.
- **Given** a transcription failure, **then** the session is still saved and the error
  reported; other queued sessions continue.

### US-11 Live captions and live transcript
- **Given** `transcription.live = true`, **when** a phrase ends, **then** it appears in the
  floating captions window about 1 s later, labelled `Me` / `Others`.
- **Then** `transcript.md` is rewritten every few seconds during the recording, and is final
  a few seconds after the stop, with no second pass.
- **Given** live transcription fails or is off, **then** the whole recording is transcribed
  after it stops.
- The captions window stays on top (also over full-screen video), never takes focus,
  remembers its position, auto-scrolls unless I scrolled up, and hides 10 s after the stop.

### US-12 Speaker labels for the other side
- **Given** several people on the `Others` track, **then** the final transcript labels them
  `Speaker 1`, `Speaker 2`… in order of first appearance; `Me` is never relabelled.
- **Given** one speaker, or no speaker model, **then** the label stays `Others`.
- **Given** a large meeting (up to 20 other speakers, some with similar voices) or one speaker
  with noisy audio, **then** the number of speakers is estimated per session instead of fixed
  by a threshold, within `min_speakers` / `max_speakers` if set, and an hour still
  diarizes in seconds.

### US-13 Readable transcripts in private folders
- **Then** each session is saved under `<output_dir>/<date>/<time>_<app>/` with
  `transcript.md`, `.srt`, `.json`, WAVs and `session.json`.
- **Then** the folders RecAndCC creates are readable only by the owner: `0700` on macOS; on
  Windows a protected access list (owner and SYSTEM only, nothing inherited from the parent).
- **Then** the menu lists the 10 most recent transcripts.

### US-14 Works offline, nothing leaves the device
- **Given** the models are on disk, **when** all network access is blocked, **then** a
  session records and transcribes end to end.
- **Then** models download once: every offered Whisper model (`tiny` to `large-v3-turbo`,
  both backends) pinned to an exact revision, the speaker model verified by SHA-256 (a bad
  checksum is rejected). A custom repo without a pin is refused unless
  `allow_unpinned_models = true`.
- **Then** RecAndCC uploads nothing itself. The default output folder may still be synced
  by iCloud or OneDrive; the app detects that locally and warns once (log, notification,
  `recandcc config`), suggesting a local folder.
- **Then** this is enforced end to end: `test_offline_e2e.py` records and transcribes a
  session (live and batch) with sockets and DNS blocked; only the speaker-model download
  is attempted, and it fails fast, leaving "Others".
- **Then** logs contain events, app names and paths, never transcript text or audio.

### US-15 Delete old audio automatically
- **Given** `audio_retention_days = N` (menu: Never / 1 / 7 / 30 / 90), **then** WAVs of
  transcribed sessions older than N days are deleted; transcripts are always kept.
- **Given** `keep_audio = false`, **then** WAVs are deleted right after transcription. WAVs
  left behind (the app quit before a slow diarization finished, or a delete failed) are
  deleted at the next startup; audio of sessions not yet transcribed is never deleted.

### US-16 Permissions set up from the app
- **Given** first run on macOS, **then** a setup dialog requests Microphone and System audio.
- **Then** the Permissions menu shows granted / not asked / denied and opens the right
  System Settings page.
- **Given** System audio is missing, **then** I am warned before recording and after a
  completely silent recording.

### US-17 Always on, visible state
- **Then** the menu-bar icon shows Off / Ask / Auto / recording time / queued transcriptions /
  missing permission.
- **Given** "Launch at login" is on, **then** RecAndCC starts at every login.
- **Given** a worker thread crashes, **then** it is logged and I get one notice per minute
  at most, and the app keeps running.

### US-18 Runs on Windows
- **Then** WASAPI loopback capture, pycaw detection, a tray icon and registry autostart
  implement the same interfaces; CI runs the fast suite on Windows.
- **Not yet:** a real-hardware test pass, per-app capture, installer (#5).

### US-19 Hebrew transcripts and right-to-left captions
- **Given** `languages = ["en", "ru", "he"]`, **then** each speech region gets one of the
  three languages (never Arabic or Yiddish), and Hebrew regions are transcribed in Hebrew.
- **Given** a Hebrew line, **when** it is shown in the live captions, **then** it is
  right-aligned and reads right to left, with the speaker label on the right; English words
  inside it stay readable.
- **Given** a Hebrew segment, **then** `transcript.md` wraps its text in a right-to-left
  isolate and `transcript.srt` starts its line with RLM, so editors and players render mixed
  Hebrew/English correctly; en/ru lines and `transcript.json` are unchanged.
- **Given** `"iw"` or an unknown code in `languages`, **then** it is mapped to `he` or
  dropped with a log warning.
- **Given** the menu **Languages** (English, Russian, Hebrew, Any language), **when** I tick
  or untick one, **then** `languages` is saved (codes added by hand and their spelling kept)
  and the next piece of audio is transcribed with it; unticking the last language is refused
  unless **Any language** is chosen.
- **Given** `language_models = { he = "ivrit-ai/whisper-large-v3-turbo-ct2" }`, **then**
  Hebrew regions (live and final) are transcribed by that model, loaded once in the
  background at a pinned revision, and en/ru regions by the default model; **given** it
  can't be loaded, **then** Hebrew falls back to the default model with a log warning and a
  one-time notice, and the load is retried after 5 minutes.
- **Not yet:** a Hebrew evaluation set and WER (#6).
- **Not yet:** the Windows (Tk) captions only right-justify Hebrew lines; Tk does no bidi
  reordering, so the label stays left of the Hebrew text and the glyph order is unverified
  on real hardware (#6).

### US-20 Search my transcripts
*As the owner, I want to find what was said in any past meeting, so that I don't have to
open transcripts one by one.*

- **Given** transcripts in `output_dir`, **when** I search for words (menu **Search
  transcripts…** in the macOS menu or the Windows tray, or `recandcc search "<words>"`), **then** I get the segments that
  contain all of them, best match first, each with its session, time in the recording,
  speaker and a snippet; choosing one opens its transcript.
- **Given** English or Russian text, **then** matching ignores case and diacritics and
  matches word prefixes (`бюдж` finds `бюджет`).
- **Given** 100+ sessions, **then** a search returns in < 200 ms.
- **When** a transcript is final, **then** it is indexed in the background; a changed
  transcript is re-indexed and a deleted or discarded session leaves the index.
- **Then** the index is local and owner-only, outside `output_dir` (never synced), and logs
  never contain the searched or found text.
- **Not yet:** summaries and action items with a local LLM, DOCX/PDF export, a notes window
  (#4).

### US-21 Speaker names remembered across sessions
*As the owner, I want to name a speaker once and see that name in later transcripts, so that
I don't have to work out who `Speaker 2` was every time.*

- **Given** `speaker_names = false` (default), **then** no voice data is stored anywhere
  (voice profiles are biometric data).
- **Given** `speaker_names = true`, **when** a final transcript is diarized, **then**
  each speaker's voice centroid is kept in the owner-only app-data `voices/` folder (never
  in `output_dir`), and
  `recandcc speakers <session>` lists the speakers with their first lines.
- **When** I run `recandcc name-speaker <session> "Speaker 2" "Anna"`, **then** that
  session's `transcript.md`, `.srt` and `.json` say `Anna`, and Anna's voice profile is saved
  (or updated as a running mean) in the owner-only app-data `voices/` folder, never in
  `output_dir`.
- **Given** a profile, **when** a later session has a speaker whose voice matches it clearly
  (cosine similarity ≥ threshold and a margin over the next profile), **then** that speaker
  is labelled `Anna`; otherwise `Speaker N`. Each name is used once per session; `Me` is
  never relabelled.
- **Then** `recandcc voices list`, `voices delete <name>` and `voices delete --all` show and
  delete the profiles (`--all` also deletes every session's voices).
- **Given** a `Speaker N` who introduces themself in their first 2 minutes ("I'm Anna",
  "Меня зовут Анна", "קוראים לי דנה"), **when** I run `recandcc speakers <session>`,
  **then** it shows `suggested: Anna (said: …)` next to that label (at most one per label,
  never for `Me`; "I'm sure", "Я думаю", "This is Paris." or "אני מדבר" are not names), with
  `speaker_names` on or off, and
  nothing is renamed, stored or logged until I run `name-speaker`.
- **Not yet:** naming from the menu with a voice sample, names from the call app, overlap
  handling and an accuracy measurement on real speech (#1, #13).

### US-27 Meeting summaries and action items
*As the owner, I want a short summary and the action items of each meeting written next to
its transcript, so that I know what was decided and who does what without rereading it.*

- **Given** `summary.enabled = false` (default), **then** no summary is generated and no
  summary model is downloaded or loaded.
- **Given** `summary.enabled = true` on Apple silicon, **when** a session's final transcript
  is written, **then** a background job (never on the audio or UI threads, and never while
  a recording is in progress: it waits, and pauses between chunks) writes
  `summary.md` next to `transcript.md`: a short summary and an **Action items** list, in the
  meeting's main language (English, Russian or Hebrew headings; Hebrew right to left).
- **Given** a transcript longer than the model's context, **then** it is summarised in
  chunks and the chunk notes are merged (map-reduce).
- **Then** the model runs locally (mlx-lm), pinned to an exact Hugging Face commit, loaded
  from the local cache without the network, and downloaded once only on a genuine cache miss.
- **Given** another platform, or the model missing while offline, **then** the summary is
  skipped with an INFO log line and a one-time **Summary skipped** notice.
- **When** I run `recandcc summarize <session_dir>`, **then** that session's `summary.md` is
  (re)written, also with `summary.enabled = false`.
- **Then** logs never contain transcript or summary text.
- **Given** a 1-hour en/ru meeting on an M-series Mac, **then** the summary is ready within
  ~1–2 min, offline.
- **Not yet:** Windows/CPU backend (llama.cpp), a quality evaluation on real meetings,
  summaries in the meeting-notes window (#4).

### US-29 Right-to-left text readable in the Windows captions
*As a Hebrew speaker on Windows, I want mixed Hebrew/English/number captions to read
correctly, even though Tk does no bidi.*

- **Given** a Hebrew line in the Tk captions, **then** its text is reordered to visual order
  with the speaker label at the right edge, and the line is right-justified.
- **Given** English words or numbers inside Hebrew, **then** they keep their reading order
  (`3.5`, `1,000` stay whole), brackets are mirrored, and no Unicode control characters
  reach Tk.
- **Given** an English or Russian line, **then** it is unchanged (`label<TAB>text`); macOS
  captions and the transcript files are unchanged.
- **Not yet:** a visual check on Windows hardware (#6); long RTL lines that wrap are reordered
  as one line, so the wrap point may read oddly.
