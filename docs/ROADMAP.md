# Roadmap

## Done: v0.1 (MVP)
- [x] Menu-bar app with Off / Ask / Auto modes, per-app rules, manual start/stop
- [x] macOS system-audio capture via Core Audio process taps + mic capture, as separate tracks
- [x] Audio-activity detection per app (Core Audio process objects / Windows audio sessions)
- [x] Local parallel transcription (faster-whisper large-v3-turbo), en/ru auto-detected per speech region
- [x] Speaker labels `Me` / `Others` with echo removal
- [x] Offline speaker diarization of "Others" → `Speaker 1/2/…` in the final transcript
      (3D-Speaker CAM++ via sherpa-onnx, ~8.5 s per hour of audio)
- [x] Markdown / SRT / JSON transcripts in `~/Documents/Transcribes`
- [x] RecAndCC.app with in-app permission requests and first-run setup
- [x] Launch at login; Windows backends (untested on hardware)
- [x] Live transcription while recording + floating live captions window (~1 s behind speech on the GPU)
- [x] MLX Whisper on the Apple-silicon GPU: live captions ~1.2 s behind speech, 1-hour meeting in ~2.5 min
- [x] Audio retention: delete audio N days after transcription (menu + config)
- [x] Global start/stop shortcut ⌃⌥R (`hotkey`), for in-person meetings; no extra permissions
- [x] Ask mode records immediately and asks whether to keep it; calls during media recordings add the mic

## Future work

Each item is tracked as a GitHub issue (label `future work`) with a design proposal and
acceptance criteria.

- [ ] **Speaker names & many speakers** ([#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1)): Shipped: naming
      `Speaker N` and recognising the voice later (local voice profiles, "Name speakers" and
      "Forget voices" in the macOS menu, #62, #76, #81), names from Zoom/Teams participant lists
      (#80), overlap-aware segmentation (`diarize_overlap`, #96), adaptive speaker counting
      (`min_speakers` / `max_speakers`, #89), splitting a segment at a mid-segment speaker
      change (#97), MLX word timestamps (#116). Left: live speaker labels (PR #88), per-track
      labels, an audio sample in "Name speakers", and measuring DER on real meetings.
- [ ] **Semantic search at scale**: shipped in #87 as opt-in; a query scores the stored vectors
      in chunks of 2,048 rows (`SCAN_CHUNK`). Left: an approximate index only if archives grow
      far past what a linear scan handles quickly (memory use: see `[search]` in
      CONFIGURATION.md).
- [x] **Room listening** ([#2](https://github.com/Galina-Blokh/rec_and_cc/issues/2)): auto-record conversations near the laptop, also with
      the screen locked: always-on mic VAD, pre-roll buffer, mic-track diarization; opt-in
      (the orange mic indicator stays on). Shipped in #64 (opt-in, off by default); locked-screen
      behaviour, idle CPU and battery still to verify on hardware.
- [ ] **Architecture review** ([#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3)): languages and packaging: self-contained,
      Developer-ID-signed app now; benchmark whisper.cpp / WhisperKit vs MLX; possible Rust +
      Tauri or native-UI rewrite. Review and staged plan: [ADR 0011](adr/0011-stack-and-packaging-after-v0-1.md)
      (keep the Python core, package it properly, switch engines only past a benchmark gate).
- [ ] **Summaries, action items, search, export** ([#4](https://github.com/Galina-Blokh/rec_and_cc/issues/4)): local LLM (e.g. Qwen via
      MLX / llama.cpp), SQLite FTS search, DOCX/PDF export. Done: search, summaries (Apple
      silicon), the meeting notes view and DOCX/PDF export (#75); left: export from the Windows
      tray, a Windows summary backend.
- ~~**Windows for real**~~ ([#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5)): hardware test pass, per-app capture, installer,
      signing. Not planned: closed, no Windows hardware available; the Windows backends stay
      as they are and CI keeps running the fast suite on Windows.
- [ ] **Hebrew** ([#6](https://github.com/Galina-Blokh/rec_and_cc/issues/6)): he/en/ru language ID, Hebrew-tuned Whisper (e.g. ivrit.ai),
      right-to-left captions and transcripts. Shipped: `he` in `transcription.languages`,
      `language_models` routing to a pinned ivrit.ai Whisper (#85), right-to-left captions and
      transcripts (#60). Left: wrapping of long right-to-left lines in the Windows (Tk) captions.
- [x] **Sensitivity slider** ([#7](https://github.com/Galina-Blokh/rec_and_cc/issues/7)): balance outside (room/mic) vs inside (system
      audio) and trigger recording on whichever side passes its threshold; live level meters.
      Shipped in #102 as menu presets plus `outside_sensitivity` / `inside_sensitivity`; no
      live level meters.
- [x] Per-app (instead of global) system-audio tap on macOS, so only the call app is recorded
      (opt-in: `system_audio_scope = "app"`, [ADR 0012](adr/0012-per-app-system-audio-tap.md)).
