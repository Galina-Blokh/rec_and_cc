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
| US-14 | Works offline, nothing leaves the device | 🟡 no warning for synced folders yet (#41) |
| US-15 | Delete old audio automatically | ✅ |
| US-16 | Permissions set up from the app | ✅ |
| US-17 | Always on, visible state | ✅ |
| US-18 | Runs on Windows | 🟡 untested on hardware (#5) |
| US-19 | Hebrew transcripts and right-to-left captions | 🟡 stock model, no Hebrew test set (#6) |
| US-20 | Search my transcripts | 🟡 search only; summaries, export planned (#4) |

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
  by iCloud or OneDrive; the README warns, the app doesn't yet (#41).
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
- **Not yet:** a Hebrew-tuned model (ivrit.ai) and per-language model routing, language
  checkboxes in the menu, a Hebrew evaluation set and WER (#6).
- **Not yet:** the Windows (Tk) captions only right-justify Hebrew lines; Tk does no bidi
  reordering, so the label stays left of the Hebrew text and the glyph order is unverified
  on real hardware (#6).

### US-20 Search my transcripts
*As the owner, I want to find what was said in any past meeting, so that I don't have to
open transcripts one by one.*

- **Given** transcripts in `output_dir`, **when** I search for words (menu **Search
  transcripts…** on macOS, or `recandcc search "<words>"`), **then** I get the segments that
  contain all of them, best match first, each with its session, time in the recording,
  speaker and a snippet; choosing one opens its transcript.
- **Given** English or Russian text, **then** matching ignores case and diacritics and
  matches word prefixes (`бюдж` finds `бюджет`).
- **Given** 100+ sessions, **then** a search returns in < 200 ms.
- **When** a transcript is final, **then** it is indexed in the background; a changed
  transcript is re-indexed and a deleted or discarded session leaves the index.
- **Then** the index is local and owner-only, outside `output_dir` (never synced), and logs
  never contain the searched or found text.
- **Not yet:** summaries and action items with a local LLM, DOCX/PDF export, a notes window,
  a search window on Windows (#4).
