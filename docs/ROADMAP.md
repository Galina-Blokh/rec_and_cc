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

- [ ] **Speaker names & many speakers** ([#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1)): name `Speaker N` once and recognise
      the voice later (local voice profiles; optionally names from Zoom/Teams); overlap-aware
      segmentation and adaptive speaker counting for large meetings; live speaker labels.
- [ ] **Room listening** ([#2](https://github.com/Galina-Blokh/rec_and_cc/issues/2)): auto-record conversations near the laptop, also with
      the screen locked: always-on mic VAD, pre-roll buffer, mic-track diarization; opt-in
      (the orange mic indicator stays on).
- [ ] **Architecture review** ([#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3)): languages and packaging: self-contained,
      Developer-ID-signed app now; benchmark whisper.cpp / WhisperKit vs MLX; possible Rust +
      Tauri or native-UI rewrite.
- [ ] **Summaries, action items, search, export** ([#4](https://github.com/Galina-Blokh/rec_and_cc/issues/4)): local LLM (e.g. Qwen via
      MLX / llama.cpp), SQLite FTS search, DOCX/PDF export.
- [ ] **Windows for real** ([#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5)): hardware test pass, per-app capture, installer,
      signing.
- [ ] Per-app (instead of global) system-audio tap on macOS, so only the call app is recorded.
