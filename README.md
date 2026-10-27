# RecAndCC: Record & Closed-Captions, locally

A small always-on menu-bar app that notices when your computer is playing or capturing voice
(Zoom/Teams/Meet calls, YouTube, podcasts, any video), records **audio only**, and turns it
into a **text transcript with speaker labels**. Everything runs on your machine: no cloud, no
account, nothing uploaded.

- 🎙️ Records **you** (microphone) and **everyone else** (system audio) as separate tracks,
  so the transcript knows who spoke: `Me` / `Others`.
- 🗣️ **Speaker labels** for the other side: in a 5-person call, `Others` is split into
  `Speaker 1`, `Speaker 2`, … in the final transcript (offline speaker diarization, a few
  seconds per hour of audio). Live captions keep showing `Others`. Opt in to
  `speaker_names` and name a speaker once (`recandcc name-speaker`): later sessions
  recognize that voice and write `Anna` instead of `Speaker 2`.
- 🌍 **English and Russian** are detected automatically, even when the language switches
  mid-meeting. **Hebrew** can be added (`languages = ["en", "ru", "he"]`): captions and
  transcripts show it right to left (Windows captions reorder RTL text themselves, as Tk does no
  bidi; verified by unit tests only, no hardware check yet). Hebrew accuracy with the stock model
  is not yet tuned; for better Hebrew, opt in to the Hebrew-tuned
  [ivrit.ai](https://huggingface.co/ivrit-ai) Whisper: with
  `language_models = { he = "ivrit-ai/whisper-large-v3-turbo-ct2" }` Hebrew regions are
  transcribed by it (its MLX build on the Apple-silicon GPU), English and Russian by the
  default model. It is downloaded once, pinned, in the background when a session starts, and needs about
  as much memory again as the default model.
- 🧠 Local Whisper (`large-v3-turbo`) on the **Apple-silicon GPU** (MLX), with CPU threads
  working alongside: a 1-hour meeting transcribes in about 2–3 minutes. Windows uses the CPU.
  and the final transcript is ready a moment after the recording stops.
- 🟡 Three modes: **Off**, **Ask** (starts recording and asks "Keep it?"), and **Auto**. You
  can also set a different mode per app.
- 🔐 Permissions are granted from inside the app, with a first-run setup dialog.
- 💻 macOS today (14.4+). Windows is supported by design (see [Windows](#windows)).

Transcripts are saved to **`~/Documents/Transcribes/<date>/<time>_<app>/`**:

```
2026-10-26/
  143005_zoom-us/
    transcript.md    ← readable transcript
    transcript.srt   ← subtitles
    transcript.json  ← machine-readable (segments, timestamps, languages)
    me.wav, others.wav, session.json
```

Sample `transcript.md`:

```markdown
# Transcript 2026-10-26 14:30
- **Trigger:** call (zoom.us)
- **Duration:** 00:42:10
- **Languages:** en, ru

**[00:00:02] Speaker 1:** Hello team, this is the weekly sync. We will discuss the release schedule.
**[00:00:09] Me:** Привет всем, давайте обсудим бюджет на следующий квартал.
**[00:00:15] Speaker 2:** Sounds good. I have the numbers from the last quarter ready.
**[00:00:21] Speaker 1:** Great, let's start with those.
```

---

## Install (macOS)

Requirements: macOS 14.4 or later, Xcode Command Line Tools (`xcode-select --install`),
[Homebrew](https://brew.sh) and [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
git clone https://github.com/Galina-Blokh/rec_and_cc.git
cd rec_and_cc
uv sync                               # Python 3.12 + dependencies into .venv
./native/macos/build_app.sh --install # builds RecAndCC.app → ~/Applications
open ~/Applications/RecAndCC.app
```

The first transcription downloads the Whisper models once (~1.6 GB `large-v3-turbo` +
~145 MB `base` for language detection) into `~/.cache/huggingface`. After that, everything
works offline.

> The app runs the Python code from the folder you cloned into, so keep that folder where it
> is. If you move it, rebuild the app with `./native/macos/build_app.sh --install`.

## First run: permissions

macOS requires your explicit consent before any app can hear the microphone or what your Mac
plays. RecAndCC asks for both in a setup dialog. Click **Grant now**, then **Allow** in each
system prompt.

You can check or fix permissions at any time in **menu bar icon → Permissions**:

| Shows | Meaning | Click "Grant…" to |
|---|---|---|
| ✅ granted | all good | (nothing) |
| ⚠️ not asked | macOS hasn't asked yet | show the macOS prompt |
| ❌ denied | you declined earlier | open the right page in System Settings |

Without **System audio** permission, macOS gives the app silence instead of an error. The
app warns you before recording, and again after a recording that came out completely silent.
After setup, test once: play a YouTube clip in Ask mode, click **Keep**, stop it, and check
that the transcript has text.

## Using it

The menu-bar icon shows the state:

| Icon | State |
|---|---|
| ⚪ | Off: never records |
| 🟡 | Ask mode, idle |
| 🎧 | Auto mode, idle |
| 🔴 03:12 | Recording (elapsed time), including while a "Keep it?" pop-up is open |
| ⏳ | Transcriptions in progress (number = queued sessions) |
| ⚠️ (prefix) | A needed permission is missing |

Menu items:
- **Mode → Off / Ask / Auto**
- **Start recording / Stop recording  ⌃⌥R**: manual control. A manual recording only stops
  when you stop it. The global shortcut **⌃⌥R** (Control+Option+R; Ctrl+Alt+R on Windows) does the same from
  any app without opening the menu: it stops whatever is recording, otherwise it starts a
  manual recording, and a small toast confirms it. Change or disable it with `hotkey` in the
  settings.
- **Show live captions**: on/off switch for the floating captions window (saved to settings).
- **Listen for conversations in the room** (off by default): while nothing records, the mic
  stays on (macOS shows the orange dot the whole time; it costs some battery, and Bluetooth
  headsets switch to their call profile) and about 6 s of speech within 15 s near the laptop
  starts a recording, as in Auto/Ask mode, including the 15 s before it (system audio is
  recorded too, as for calls). It stops after 90 s without speech. Apps take precedence:
  nothing starts while another app plays or uses the mic (music too), and a room recording
  stops if an app you switched off, stopped or ignore starts. Tell the people in the room: recording them may need
  their consent.
- **Open live transcript**: opens the current (or last) live session's `transcript.md`.
- **Recent transcripts**: the last 10, click to open.
- **Search transcripts…**: full-text search across all your transcripts (English, Russian,
  any language); pick a match to open its transcript. Same as `recandcc search`. On
  Windows the tray item opens a small window: type the words, press Enter, and
  double-click a match (time, speaker, snippet) to open it.
- **Open transcripts folder**
- **Permissions**: status and "Grant…" actions.
- **Launch at login**: starts RecAndCC every time you log in. Turn it on once.
- **Languages**: English / Russian / Hebrew checkboxes, plus **Any language**: which
  languages speech is detected among (`languages` in the settings). Applies at once, also to
  the rest of a recording in progress; at least one stays ticked unless Any is chosen. Codes
  you added in the settings by hand are kept and listed as "Also from settings".
- **Delete audio after**: Never / 1 / 7 / 30 / 90 days. Audio of transcribed recordings is
  deleted after that many days; transcripts are always kept. Any number of days can be set as
  `audio_retention_days` in the settings.
- **Edit settings**: opens `config.toml` (changes to detection settings, `hotkey` and
  `languages` apply live).
- **Quit RecAndCC**

### Live captions

While a session is being transcribed live, a small floating window at the bottom of the
screen shows the text as it is spoken, one line per phrase, labelled `Me` / `Others`
(English and Russian mixed freely). It stays on top of other windows, including full-screen
video and Zoom, on every desktop, and never takes keyboard focus. Drag or resize it and it
remembers where you put it. It scrolls along with new text unless you scroll up to reread
something. When the recording stops it shows "Recording finished — transcript saved" and
hides itself after 10 seconds. The close button hides it until the next recording; to turn
captions off for good, untick **Show live captions** in the menu. Captions need live
transcription (`transcription.live = true`, the default).

While recording, `transcript.md` in the session folder is rewritten every few seconds, so
you can open it mid-meeting (**Open live transcript**). Text arrives about 1 s after each
phrase on an M5 Max with `large-v3-turbo`: almost all of that is one Whisper pass. For
faster captions see the `workers` / `model` trade-offs in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md). The very first recording after the app starts
shows captions only once the model has loaded (a few seconds).

### In-person meetings: press ⌃⌥R

A meeting in a room doesn't make any app play or capture audio, so nothing triggers a
recording unless **Listen for conversations in the room** is on. Otherwise press **⌃⌥R** (Control+Option+R) when it starts and again when it ends. It works in every app, needs
no extra permission (no Accessibility / Input Monitoring), and uses the physical R key, so it
works with a Russian keyboard layout too. While RecAndCC runs, the shortcut belongs to it, so
the default avoids ⌘⇧R (hard reload in Chrome, Reader in Safari). If another app already
registered the shortcut, RecAndCC tells you once; pick another one, e.g.
`hotkey = "<cmd>+<shift>+r"`.

### When does it record?

1. Every second the app checks which apps are playing or capturing audio.
2. Apps are classified using the lists in your settings: **call apps** (Zoom, Teams,
   FaceTime, Slack, Discord, Telegram, WhatsApp…), **media apps** (browsers, video players),
   and **ignored apps** (Spotify, Music, system sounds). An unknown app that uses the
   microphone counts as a call; a browser that uses the microphone (Google Meet) counts as a
   call too.
3. When an app has been active for 3 s, recording **starts**. In Auto mode that's all. In
   Ask mode a pop-up also asks whether to keep the recording, so no audio is lost while you
   decide or if you're away from the screen:
   - **Keep**: keep recording this session (also the default if you don't answer in 30 s;
     set `ask_timeout_action = "ignore"` to discard unanswered recordings instead).
   - **Discard**: stop and delete the recording, and don't ask again until this app's audio
     stops.
   - **Always**: keep it, and record this app automatically from now on (saved to settings).
4. Recording stops when the app stops doing audio: after 8 s for videos, 20 s for calls
   (calls have natural pauses). It also stops after 30 s of silence for videos (e.g. a
   paused tab), 3 minutes for calls. The transcript has been built while recording (see
   [Live captions](#live-captions)), so it is final a few seconds after the stop. With
   `transcription.live = false` the whole recording is transcribed after it stops instead.

The microphone is recorded only for **calls** by default, so watching YouTube doesn't
capture your room. If a call starts while a video is already being recorded, the session
becomes a call: the microphone joins in, lined up in time, and the recording follows the call
instead of stopping when the video ends. You can change this with `record_mic` in the settings.

System audio (`Others`) is everything the Mac plays, so a notification sound or a video in
another tab ends up in the recording too. On macOS, `system_audio_scope = "app"` records only
the app that started the session (Zoom, the browser tab's browser…) and its helper
processes; if a call starts during a video, both apps are recorded. Manual recordings always
take everything, and so does FaceTime (its calls play through a system process). It is off
by default until it has been checked with Zoom, Teams and Meet calls.

## Settings

`~/Library/Application Support/RecAndCC/config.toml` (Windows:
`%LOCALAPPDATA%\RecAndCC\config.toml`). It is created on first run. All options are
documented in **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**. The most useful ones:

```toml
output_dir = "/Users/you/Documents/Transcribes"

[detection]
mode = "ask"                 # off | ask | auto
record_mic = "calls"         # always | calls | never
[detection.app_modes]        # per-app overrides
"us.zoom" = "auto"
"com.google.chrome" = "ask"

[transcription]
model = "large-v3-turbo"     # tiny | base | small | medium | large-v3 | large-v3-turbo
languages = ["en", "ru"]     # auto-detect among these, e.g. ["en", "ru", "he"] ([] = any)
language_models = {}         # per-language model, e.g. { he = "ivrit-ai/whisper-large-v3-turbo-ct2" }
keep_audio = true            # false = delete WAVs after transcription (~115 MB per track-hour)
live = true                  # transcribe while recording (false = after the session stops)

[summary]
enabled = false              # true = summary.md with action items after each meeting (local LLM)
```

### Meeting summaries (opt-in)

With `[summary] enabled = true` (Apple silicon), each finished transcript also gets a
`summary.md` in its folder: a few sentences about the meeting and an **Action items** list
(who does what, by when), written in the meeting's main language (English, Russian or
Hebrew). A small local model (Qwen2.5 3B, 4-bit, ~1.7 GB) does it in the background after
the transcript is final, and waits while you are recording, so captions are never slowed; it is downloaded once, pinned to an exact version, and nothing is
sent anywhere. It is a draft: check names and dates against the transcript. On Windows, or
offline before the model is downloaded, the summary is skipped with a notice.

## Command line

Everything the app does is also available from the terminal (`uv run recandcc …` in the
repo):

| Command | What it does |
|---|---|
| `recandcc run` | Start the menu-bar app (what RecAndCC.app runs) |
| `recandcc record --seconds 60 [--no-mic]` | Record now without the UI; prints the text live as it is spoken, then the transcript path |
| `recandcc transcribe <session_dir>` | (Re-)transcribe a recorded session |
| `recandcc summarize <session_dir>` | Write the session's `summary.md` (summary + action items) with the local model |
| `recandcc monitor` | Live view of which apps do audio and how the rules classify them. Use it to tune the app lists |
| `recandcc search "<words>" [--limit N]` | Search all transcripts; prints each match's date, time in the recording, speaker, the text and the transcript path |
| `recandcc devices` | List audio devices |
| `recandcc autostart on\|off\|status` | Launch at login |
| `recandcc config` | Show the config path and contents |
| `recandcc speakers <session_dir>` | List a transcript's speakers with their first lines. A speaker who introduces themself in their first 2 minutes ("I'm Anna", "Меня зовут Анна", "קוראים לי דנה") gets a hint, `suggested: Anna (said: …)`. Only a hint: nothing is renamed or saved until you run `name-speaker` |
| `recandcc name-speaker <session_dir> "Speaker 2" "Anna"` | Name a speaker: rewrites that transcript and saves Anna's voice profile, so later sessions say `Anna` (needs `speaker_names = true`) |
| `recandcc voices list\|delete <name>\|delete --all` | List or delete saved voice profiles |

## Privacy & consent

> This section is general information, not legal advice. Laws differ between countries and
> states and change over time; check the rules that apply to you and to everyone you record.

### Recording people is regulated

- **One-party consent** (e.g. US federal law and most US states): you may record a
  conversation you take part in without telling the others.
- **All-party ("two-party") consent** (e.g. California, Florida, Illinois, Washington, and
  several countries): **everyone** on the call must agree before you record. When people are
  in different places, assume the strictest rule applies.
- **GDPR (EU/EEA, UK GDPR)**: a recording or transcript of an identifiable person is personal
  data. Recording for anything beyond purely personal use (e.g. at work) needs a lawful basis,
  telling people what you record and why, keeping it no longer than needed, and deleting it
  on request.
- In practice: **say that you are recording** at the start of a call, and stop if someone
  objects. You are responsible for using RecAndCC lawfully. In **Ask** mode a recording is
  only kept if you confirm it.

### Where your data lives

Audio and transcripts never leave your computer. The only network access is the one-time
model download: Whisper (and, with summaries on, the summary model) from Hugging Face, pinned to exact repo commits
([`revisions.py`](src/recandcc/transcribe/revisions.py); unpinned custom repos are refused
unless you opt in), and the 28 MB speaker model from the
sherpa-onnx GitHub releases, verified by its SHA-256. After that the app works fully offline:
models load from disk without contacting anything (tested with all network access blocked).

| What | macOS | Windows |
|---|---|---|
| Recordings (`*.wav`), transcripts (`transcript.md/.json/.srt`), summaries (`summary.md`), `session.json` | `~/Documents/Transcribes/` (or your `output_dir`) | `Documents\Transcribes\` |
| Settings (`config.toml`) | `~/Library/Application Support/RecAndCC/` | `%LOCALAPPDATA%\RecAndCC\` |
| Search index (`search.sqlite3`: a copy of the transcript text, owner-only) | `~/Library/Application Support/RecAndCC/` | `%LOCALAPPDATA%\RecAndCC\` |
| Logs (events, app names, file paths) | `~/Library/Logs/RecAndCC/` | `%LOCALAPPDATA%\RecAndCC\Logs\` |
| Voice profiles (only with `speaker_names = true`; biometric data, owner-only) | `~/Library/Application Support/RecAndCC/voices/` | `%LOCALAPPDATA%\RecAndCC\voices\` |
| Each session's speaker voices (only with `speaker_names = true`) | `…/RecAndCC/voices/sessions/` | `%LOCALAPPDATA%\RecAndCC\voices\sessions\` |
| Speaker-diarization model | `~/Library/Caches/RecAndCC/models/` | `%LOCALAPPDATA%\RecAndCC\Cache\models\` |
| Whisper models (shared Hugging Face cache) | `~/.cache/huggingface/hub/models--*whisper*` | `%USERPROFILE%\.cache\huggingface\hub\` |
| Summary model (only with summaries on) | `~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-3B-Instruct-4bit` | not used |
| Launch at login | `~/Library/LaunchAgents/com.gblokh.recandcc.plist` | registry `HKCU\…\CurrentVersion\Run\RecAndCC` |

The folders RecAndCC creates for recordings (the transcripts folder, each day and each
session) are readable only by you: `0700` on macOS; on Windows an access list with only your
account and SYSTEM, set when the folder is created and not inherited from its parent. A
transcripts folder that already existed keeps its permissions; tighten it with
`chmod 700 ~/Documents/Transcribes`. On Windows, the simplest fix is to let RecAndCC create a
new folder (point `output_dir` at a path that doesn't exist yet); `icacls <folder>` shows who
has access. Access lists need an NTFS drive: on a FAT/exFAT USB stick or some network
shares anyone with access to the drive can read the recordings (RecAndCC warns you once per drive), so keep `output_dir` on a
local NTFS drive.
`~/Documents` may be synced by iCloud Drive ("Desktop & Documents Folders") on macOS, and
`Documents` by OneDrive folder backup on Windows. RecAndCC checks `output_dir` locally (no network)
while it runs: if it lies in iCloud Drive (including "Desktop & Documents"),
`~/Library/CloudStorage` (OneDrive, Dropbox, Google Drive, Box), `~/Dropbox`, `~/Google Drive`,
`~/OneDrive*`, `~/Box`, `~/Box Sync` (case-insensitive, also through symlinks),
or on Windows a OneDrive folder (including `Documents` / `Desktop` moved by OneDrive folder
backup) or `%USERPROFILE%\Dropbox`, also through symlinks and junctions, it logs a warning and shows a
one-time notification, and `recandcc config` prints the same warning. Other sync tools (for
example a Google Drive drive letter on Windows) are not detected. Set `output_dir` to a
local folder (for example `~/Transcribes`) if recordings must stay on this device only.

### Retention

- `keep_audio = false` deletes a session's WAVs as soon as it is transcribed (any left behind
  by a quit are deleted at the next startup).
- `audio_retention_days = N` (menu → **Delete audio after**: Never / 1 / 7 / 30 / 90 days)
  deletes the audio N days after transcription; `0` keeps it forever.
- Transcripts are never deleted automatically. Delete a session by deleting its folder.
  Its text leaves the search index at the next search, the next app start, or the app's
  hourly check, whichever comes first. Removed or replaced text is overwritten in
  `search.sqlite3`, so it can't be recovered from the file. To remove the whole index at
  once, delete `search.sqlite3`; it is rebuilt from the transcripts on the next search.
  With `speaker_names` on, its stored speaker voices are deleted the next time a transcript
  is written or a `recandcc voices` command runs.
- `recandcc voices delete <name>` deletes that profile and removes the named voice from every
  session's stored voices too; renaming a speaker (`Anna` → `Bob`) takes that session's voice
  out of Anna's profile.

### Delete everything

Quit RecAndCC and turn off **Launch at login** (or `uv run recandcc autostart off`). Then, on
macOS:

```bash
rm -rf ~/Documents/Transcribes                    # recordings + transcripts (or your output_dir)
uv run recandcc voices delete --all               # voice data only (also removed below)
rm -f  ~/Library/Application\ Support/RecAndCC/search.sqlite3  # search index
rm -rf ~/Library/Application\ Support/RecAndCC    # settings, voice profiles and the search index
rm -rf ~/Library/Logs/RecAndCC                    # logs
rm -rf ~/Library/Caches/RecAndCC                  # speaker model
rm -rf ~/.cache/huggingface/hub/models--*whisper* # Whisper models (if no other app uses them)
rm -f  ~/Library/LaunchAgents/com.gblokh.recandcc.plist
rm -rf ~/Applications/RecAndCC.app                # the app (and the repo folder, if you like)
```

Remove the Microphone / System audio permissions in System Settings → Privacy & Security. On
Windows, delete the folders listed in the table above.

## Troubleshooting

| Problem | Fix |
|---|---|
| Transcript is empty / "recording was silent" warning | Menu → Permissions → grant **System audio** (and Microphone). Then quit and reopen RecAndCC. |
| Permission prompts show "Terminal" instead of RecAndCC | You ran `recandcc run` from a terminal. Launch `RecAndCC.app` instead. |
| An app isn't detected / something is detected that shouldn't be | Run `uv run recandcc monitor`, note the `bundle_id`, and add it to `call_apps` / `media_apps` / `ignore_apps`. |
| App doesn't start | Look at `~/Library/Logs/RecAndCC/app.log` and `recandcc.log`. |
| Moved the repo folder | Run `./native/macos/build_app.sh --install` again, then re-grant permissions (macOS sees a changed app as a new one). |
| Permissions reset after an update | Only updates to the app bundle itself do that, and the build script warns you. Menu → Permissions → Grant… again. Normal code updates just need a restart of RecAndCC. |
| Transcription is too slow | Use `model = "small"` in settings, or lower `workers`. |

## Windows

The code includes Windows backends: WASAPI loopback capture (PyAudioWPatch), audio session
detection (pycaw), a tray icon (pystray), and autostart through the registry. They are
**not yet tested on real hardware**; CI checks that the code runs on Windows.

```powershell
git clone https://github.com/Galina-Blokh/rec_and_cc.git; cd rec_and_cc
uv sync
uv run recandcc run          # tray icon; enable "Launch at login" from its menu
```

Windows needs no system-audio permission. The microphone is controlled in
*Settings → Privacy → Microphone*, and the tray menu has a shortcut to that page.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it works: components, threads,
  data flow, design decisions
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): every setting explained
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): building, testing, project layout, contributing
- [docs/requirements/](docs/requirements/intent.md): intent, goals and non-goals, and
  user stories with acceptance criteria
- [docs/ROADMAP.md](docs/ROADMAP.md): what's next
- [CHANGELOG.md](CHANGELOG.md): what changed in each version
- [SECURITY.md](SECURITY.md): reporting security issues, threat model
