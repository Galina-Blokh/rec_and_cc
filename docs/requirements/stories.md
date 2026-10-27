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
| US-13 | Readable transcripts in private folders | 🟡 owner-only not enforced on Windows (#37) |
| US-14 | Works offline, nothing leaves the device | 🟡 no warning for synced folders yet (#41) |
| US-15 | Delete old audio automatically | 🟡 `keep_audio = false` can miss a crash (#38) |
| US-16 | Permissions set up from the app | ✅ |
| US-17 | Always on, visible state | ✅ |
| US-18 | Runs on Windows | 🟡 untested on hardware (#5) |
| US-22 | Room listening (opt-in) | 🟡 locked screen, CPU and battery not measured (#2) |

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
- **Then** the folders RecAndCC creates are readable only by the owner (`0700` on macOS).
- **Not yet:** on Windows, `chmod` doesn't restrict NTFS ACLs, so folders inherit the
  parent's permissions; an owner-only ACL is needed (#37).
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
- **Given** `keep_audio = false`, **then** WAVs are deleted right after transcription.
- **Not yet:** when diarization is slow, deletion waits for it in memory; if the app quits
  or crashes first, the WAVs stay (until `audio_retention_days` removes them, if set) (#38).

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
  stopped or that is ignored), **then** room speech starts nothing; a call that starts
  during a room session turns it into that call's session, without a second session.
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
