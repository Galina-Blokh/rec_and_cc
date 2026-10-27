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
| US-31 | Speaker names from the call app's participant list | 🟡 claimed (#1 A.3): opt-in, macOS, one-to-one only |
| US-25 | Export a transcript to DOCX and PDF | 🟡 CLI and macOS menu; Windows tray, templates planned (#4) |
| US-27 | Meeting summaries and action items | 🟡 opt-in, Apple silicon only; no quality evaluation yet (#4) |
| US-22 | Room listening (opt-in) | 🟡 locked screen, CPU and battery not measured (#2) |
| US-23 | Sensitivity: room vs system audio | 🟡 presets only, no live meters (#7) |
| US-28 | Browse past meetings | 🟡 Windows window untested on hardware (#5) |
| US-29 | Right-to-left text readable in the Windows captions | 🟡 unit-tested, no hardware check (#6) |
| US-30 | Change common settings in a window | 🟡 claimed (#3): macOS spike, not on Windows |
| US-34 | Warn when recordings go to a network volume | 🟡 claimed (#3): detection is fake-tested, no real share or Windows drive checked |

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
- **Given** `split_on_speaker_change = true` (opt-in) and a batch transcription
  (`live = false`, `recandcc transcribe`, re-queued session; not live sessions) on the
  faster-whisper backend, with a segment in which the speaker changes mid-way, **then** it is split at that word boundary and each part gets its
  own speaker label, with flicker under about 1 s / 3 words ignored, the text kept word for
  word, `Me` never split, and segments without word timings (live, MLX backend, option off) unchanged.
- **Given** `diarize_overlap = true` and a speaker change without a pause, or two speakers
  at once, **then** the turn
  boundary follows the change (segmentation model), overlapped speech gets the dominant
  speaker, and that segment has `"overlap": true` in `transcript.json` (only there); without
  the segmentation model (off, offline or broken), fixed ~2 s windows as before.

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
  both backends) pinned to an exact revision, the speaker and segmentation models verified by SHA-256 (a bad
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
- **Not yet:** a Hebrew evaluation set and WER (#6); `tests/test_hebrew_integration.py`
  covers he/en/ru switching, RTL outputs and a two-speaker dialogue on synthetic `say` audio.
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
- **Given** 100+ sessions, **then** a search returns in < 200 ms (`test_search_latency.py`).
- **When** a transcript is final, **then** it is indexed in the background; a changed
  transcript is re-indexed and a deleted or discarded session leaves the index.
- **Then** the index is local and owner-only, outside `output_dir` (never synced), and logs
  never contain the searched or found text.
- **Not yet:** summaries and action items with a local LLM, DOCX/PDF export
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

### US-31 Speaker names from the call app's participant list
*As the owner, I want `Speaker 2` named after the person shown in Zoom/Teams/Meet, so that I
don't have to name them myself, without the app ever asking for more permission on its own.*

- **Given** `names_from_call_app = false` (default) or `speaker_names = false` or Windows,
  **then** nothing is read from any other app.
- **Given** it is on and Accessibility is granted, **when** a call is recorded and the call
  app's participants panel is open, **then** the visible names are sampled every ~15 s on a
  worker thread and kept in the owner-only voices store (not in the session folder); in a
  only Zoom, Teams and browser call windows are read, and a name must be in two samples in a row; the panel is never opened or clicked for me.
- **Given** Accessibility is not granted, **then** nothing is read and macOS never shows its
  prompt because of RecAndCC: trust is only checked silently. Only **menu → Use names from the
  call app** explains why and opens System Settings > Accessibility.
- **Given** the participant names are exactly as many as the speakers besides `Me`, every
  voice-matched speaker matches one of them, and one `Speaker N` and one name remain,
  **then** that speaker is relabelled in the transcript (no voice profile is created from it).
  Otherwise (several unknowns, mismatched spellings such as "Robert" vs "Bob Smith", extra
  names, a disagreeing self-introduction) nothing is assigned and `recandcc speakers` lists the
  names. `Me` is never renamed.
- **Then** names appear nowhere in logs at INFO or above and nowhere outside the owner-only
  voices store (and a transcript they were used in).
- **Not yet:** matching several speakers to several names (needs active-speaker highlighting),
  self-introductions (A.4), tested layouts for each app version (see ARCHITECTURE).

### US-25 Export a transcript to DOCX and PDF
*As the owner, I want a transcript as a Word or PDF file, so that I can share or print it.*

- **Given** a session with a `transcript.json`, **when** I run `recandcc export <session>
  [--format docx|pdf|both]` (or choose **Export latest transcript** on macOS), **then**
  `transcript.docx` and/or `transcript.pdf` are written next to it: title, date, speakers,
  and one paragraph per segment with its timestamp.
- **Given** English, Russian and Hebrew segments, **then** all are readable; Hebrew segments
  are right to left (bidi paragraphs in DOCX, shaped and reordered text in the PDF).
- **Then** export needs no network, the files are owner-only and logs never contain the
  transcript text.
- **Not yet:** the Windows tray entry, custom templates; the PDF needs a system font with
  Latin, Cyrillic and Hebrew (Arial, DejaVu Sans), otherwise only DOCX is produced (#4).

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
- **Not yet:** Windows/CPU backend (llama.cpp), a quality evaluation on real meetings (#4).

### US-22 Room listening (opt-in)
*As the owner, I want a conversation next to the laptop (an in-person meeting) recorded
without pressing anything, like a call, and only when I have turned that on.* (#2, with #58)

- **Given** a fresh install (`room_listening = false`), **then** the microphone is never
  opened while nothing records.
- **Given** room listening is on and nothing records, **when** nobody speaks or I make a
  short remark (less than `room_speech_s` of speech within `room_window_s`), **then**
  nothing is recorded; **when** people talk for `room_speech_s` (6 s) within
  `room_window_s` (15 s), **then** a `room` session starts (Auto), or starts and asks whether
  to keep it (Ask).
- **Then** the session's audio begins up to `room_preroll_s` (15 s) before the trigger, so
  the first sentence is kept, with no gap or overlap, and the other track stays aligned.
- **Then** the mic is never opened twice: a room session records the monitor's own stream,
  and the monitor's mic is closed before any other session opens one.
- **Given** an app is doing audio or another app uses the mic (even one I switched off,
  stopped or that is ignored), **then** room speech starts nothing, the room mic is closed
  while the other app uses it, and nothing it heard reaches a later room session; a call
  that starts during a room session turns it into that call's session, without a second
  session, and an excluded app starting during a room session (or turning room listening
  off) stops it.
- **Given** a room session, **then** it stops after `room_stop_after_silence_s` (90 s)
  without speech; after a manual stop or a discard it doesn't restart until the room was
  quiet for `stop_after_inactive_s`.
- **Given** a room session, **then** the final transcript numbers the speakers on the mic
  track (`Speaker 1`, `Speaker 2`, …).
- **Given** the menu item "Listen for conversations in the room", **then** it switches the
  setting (saved) and closes or reopens the mic; turning it on explains the always-on mic
  indicator, the battery cost and consent, and warns if mic access is denied.
- **Then** speech detection runs on its own thread (the audio thread only copies frames);
  nothing about the room audio is logged beyond on/off states, and nothing leaves the device.
- **Not yet:** verified with the screen locked; idle CPU and battery impact measured.

### US-23 Sensitivity: room vs system audio
*As the owner, I want to tell the app how readily the room (outside, microphone) and the
laptop's own audio (inside, apps) start a recording, so that it fits a quiet office or a
noisy café.* (#7)

- **Given** `outside_sensitivity` and `inside_sensitivity` are unset (the default), **then**
  detection behaves exactly as before: `min_active_s`, `room_speech_s`, no level gate.
- **Given** *Normal* (50/50), **then** the thresholds are the plain ones, but unlike *Off* the
  noise-floor gate and its ~3 s warm-up after each mic open apply.
- **Given** a room session is recording, **then** the gate and floor are paused (frozen) and
  plain voice detection decides when the talk has ended, so a long lecture never becomes
  "background" and stops the session.
- **Given** a sensitivity 0–100, **then** the threshold is scaled 4× at 0, 1× at 50 and
  0.25× at 100: inside scales `min_active_s`; outside scales `room_speech_s` (the window widens
  so speech is at most half of it, so even 0 triggers on a real conversation) and sets a
  12–9 dB margin above the room's noise floor, judged per ~0.5 s step by its loud parts (so
  the dips between syllables still count). A non-numeric value counts as unset.
- **Given** inside low, **then** short app audio starts nothing yet; **given** inside favoured,
  **then** app audio starts a recording within a second. Apps always take precedence over the
  room: while any app does audio the room holds off (by design), and the outside setting only
  changes how much room speech starts a recording when no app does.
- **Given** a turn-taking conversation (1–3 s turns, 0.3–1.5 s pauses), **then** settings 0,
  25, 50 and 75 all start a recording, in a quiet room and in a café with the voices 10 dB
  (0, 25) or 8 dB (50, 75) over the babble; quieter voices are a known limit.
- **Given** steady background noise (a fan, 2–8 talker café babble, even if the voice detector
  calls it speech), **then** it becomes the noise floor (35th percentile of the last 20 s) and
  Normal, Noisy place and Quiet room neither start nor, after a room session, restart a
  recording; 100 is meant for quiet rooms. A mic opened mid-conversation still counts it.
- **Given** the place changes with the mic open (carried into a café, or after sleep), **then**
  the floor is relearnt (a level jump of the quieter moments, or a gap of over 5 s in the
  audio) and the new background doesn't start a recording at 25, 50 or 75.
- **Given** the laptop slept, **then** the speech and audio from before the gap are forgotten:
  nothing triggers on them and a session never starts with pre-sleep audio. **Given** an app
  played audio (the room holds off), **then** its playback never becomes the room's floor.
- **Then** a room session's live transcript starts with the whole pre-roll, even when a low
  outside sensitivity widened it beyond `room_preroll_s`.
- **Given** the menu "Sensitivity" (Off / Quiet room / Normal / Noisy place), **then** the
  preset applies at once and only its two keys are written to `config.toml`, so hand edits
  to other keys are kept; "Off" removes them.
- **Not yet:** a settings window with live level meters, a per-source speech score, caption
  priority between the tracks.

### US-28 Browse past meetings
*As the owner, I want to see my past meetings in one place and open, rename or delete them,
so that I don't have to dig through dated folders.*

- **When** I choose **Meeting notes…** (macOS menu or Windows tray), **then** I see the
  sessions in `output_dir`, newest first, each with its title (the triggering app, "Manual
  recording", or the title I gave it), date and time, duration, speakers, languages and
  whether it is transcribed; discarded sessions are not listed. The folder is read off the
  UI thread.
- **Given** a session folder with a `summary.md`, **then** its first lines show as an
  excerpt and I can open it; without one the listing works the same, minus that action.
- **When** I pick a session, **then** I can open its transcript, reveal its folder, rename
  it or delete it. A rename is stored as `title` in `session.json` (older files without it
  still load) and an empty title goes back to the default.
- **When** I delete a session and confirm, **then** its folder goes to the Trash / Recycle
  Bin (permanently via `discard_session` where there is none), it leaves the search index,
  the recent list and the stored speaker voices (those don't come back from the Trash).
  If the Trash refuses it, nothing is deleted and I am told. If the Trash takes only part
  of it, the rest is deleted for good (files in use: at the next start) and I am told. A
  folder that is a link, or not inside `output_dir`, is never deleted.
- **Given** a session from its start until its transcription is done, **then** rename and
  delete are refused with a notice.
- **Then** logs never contain titles, excerpts or transcript text.
- **Not yet:** a native macOS window (it is a pair of dialogs) (#4).

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

### US-30 Change common settings in a window
*As the owner, I want a settings window with the options I change most, so that I don't have
to edit `config.toml` by hand.*

- **Given** macOS and a helper that has the `settings` subcommand, **when** I choose
  **Settings…** in the menu, **then** a window shows mode (off/ask/auto), transcripts folder,
  languages (English, Russian, Hebrew), keep audio, live captions, speaker names and the
  shortcut (read-only), without blocking the menu bar.
- **When** I press Save, **then** the values are validated by the app and written to
  `config.toml` (other keys are kept) and applied at once; **when** I cancel or close the
  window, **then** nothing changes.
- **Given** invalid values, **then** nothing is written and a notice says so.
- **Given** a missing or older helper, **then** the app says so and opens `config.toml`
  instead.
- **Then** the window shows no transcript or audio content, uses no network and asks for no
  new permission.
- **Not yet:** Windows, the other keys (detection tuning, hotkey editing), a two-way protocol.

### US-34 Warn when recordings go to a network volume
*As the owner, I want a warning when my transcripts folder is on a network share, so that I
know recordings and transcripts leave this computer.*

- **Given** `output_dir` is on an SMB, AFP, NFS or WebDAV mount (macOS), a UNC path or a
  mapped network drive (Windows), or an NFS/CIFS/SSHFS mount (Linux), **when** the app starts,
  `output_dir` changes (also from Settings) or the menu refreshes, **then** a notification
  says so and suggests a local folder; it is shown once per folder, and `recandcc config`
  prints the same warning.
- **Then** recording is never blocked, the check is local (no network, no subprocess) and
  logs contain the kind of share but never the path.
- **Not yet:** shares that a sync or FUSE tool presents as a local disk are not detected.
