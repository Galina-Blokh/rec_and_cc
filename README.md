# RecAndCC: Record & Closed-Captions, locally

A small always-on menu-bar app that notices when your computer is playing or capturing voice
(Zoom/Teams/Meet calls, YouTube, podcasts, any video), records **audio only**, and turns it
into a **text transcript with speaker labels**. Everything runs on your machine: no cloud, no
account, nothing uploaded.

- 🎙️ Records **you** (microphone) and **everyone else** (system audio) as separate tracks,
  so the transcript knows who spoke: `Me` / `Others`.
- 🌍 **English and Russian** are detected automatically, even when the language switches
  mid-meeting.
- 🧠 Local Whisper (`large-v3-turbo`) via faster-whisper. On an M-series Mac it transcribes
  about 3–4× faster than realtime, in parallel.
- 🟡 Three modes: **Off**, **Ask** ("Audio detected in Zoom. Record?"), and **Auto**. You
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

**[00:00:02] Others:** Hello team, this is the weekly sync. We will discuss the release schedule.
**[00:00:09] Me:** Привет всем, давайте обсудим бюджет на следующий квартал.
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
After setup, test once: play a YouTube clip in Ask mode, click **Record**, stop it, and check
that the transcript has text.

## Using it

The menu-bar icon shows the state:

| Icon | State |
|---|---|
| ⚪ | Off: never records |
| 🟡 | Ask mode, idle |
| 🎧 | Auto mode, idle |
| 🟠 | Asking you whether to record |
| 🔴 03:12 | Recording (elapsed time) |
| ⏳ | Transcriptions in progress (number = queued sessions) |
| ⚠️ (prefix) | A needed permission is missing |

Menu items:
- **Mode → Off / Ask / Auto**
- **Start recording / Stop recording**: manual control. A manual recording only stops when
  you stop it.
- **Recent transcripts**: the last 10, click to open.
- **Open transcripts folder**
- **Permissions**: status and "Grant…" actions.
- **Launch at login**: starts RecAndCC every time you log in. Turn it on once.
- **Edit settings**: opens `config.toml` (changes to detection settings apply live).
- **Quit RecAndCC**

### When does it record?

1. Every second the app checks which apps are playing or capturing audio.
2. Apps are classified using the lists in your settings: **call apps** (Zoom, Teams,
   FaceTime, Slack, Discord, Telegram, WhatsApp…), **media apps** (browsers, video players),
   and **ignored apps** (Spotify, Music, system sounds). An unknown app that uses the
   microphone counts as a call; a browser that uses the microphone (Google Meet) counts as a
   call too.
3. When an app has been active for 3 s, the app records (Auto) or asks you (Ask). In the Ask
   dialog:
   - **Record**: record this session.
   - **Ignore**: don't ask again until this app's audio stops.
   - **Always**: record this app automatically from now on (saved to settings).
4. Recording stops 20 s after the app stops doing audio, or after 3 minutes of total
   silence. Then the recording is transcribed in the background.

The microphone is recorded only for **calls** by default, so watching YouTube doesn't
capture your room. If a call starts while a video is already being recorded, the session
becomes a call: the microphone joins in, lined up in time, and the recording follows the call
instead of stopping when the video ends. You can change this with `record_mic` in the settings.

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
languages = ["en", "ru"]     # auto-detect among these ([] = any language)
keep_audio = true            # false = delete WAVs after transcription (~115 MB per track-hour)
```

## Command line

Everything the app does is also available from the terminal (`uv run recandcc …` in the
repo):

| Command | What it does |
|---|---|
| `recandcc run` | Start the menu-bar app (what RecAndCC.app runs) |
| `recandcc record --seconds 60 [--no-mic]` | Record now without the UI, then transcribe |
| `recandcc transcribe <session_dir>` | (Re-)transcribe a recorded session |
| `recandcc monitor` | Live view of which apps do audio and how the rules classify them. Use it to tune the app lists |
| `recandcc devices` | List audio devices |
| `recandcc autostart on\|off\|status` | Launch at login |
| `recandcc config` | Show the config path and contents |

## Privacy & legal

- Audio and transcripts never leave your computer. The only network access is the one-time
  model download from Hugging Face.
- Recording people can require their consent (some US states and the EU require everyone on
  a call to agree). You are responsible for using this lawfully. Tell people when you record
  them.

## Troubleshooting

| Problem | Fix |
|---|---|
| Transcript is empty / "recording was silent" warning | Menu → Permissions → grant **System audio** (and Microphone). Then quit and reopen RecAndCC. |
| Permission prompts show "Terminal" instead of RecAndCC | You ran `recandcc run` from a terminal. Launch `RecAndCC.app` instead. |
| An app isn't detected / something is detected that shouldn't be | Run `uv run recandcc monitor`, note the `bundle_id`, and add it to `call_apps` / `media_apps` / `ignore_apps`. |
| App doesn't start | Look at `~/Library/Logs/RecAndCC/app.log` and `recandcc.log`. |
| Moved the repo folder | Run `./native/macos/build_app.sh --install` again. |
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
- [docs/ROADMAP.md](docs/ROADMAP.md): what's next
