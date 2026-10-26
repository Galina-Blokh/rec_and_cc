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

## Next
- [ ] **Speaker names**: assign a name to `Speaker N` once and recognise the voice in later
      calls (the speaker embeddings are already computed)
- [ ] **Summaries & action items** with a local LLM (e.g. Qwen 2.5/3 7B quantised via
      llama.cpp / MLX), off by default
- [ ] **Search** across all transcripts from the menu
- [ ] Self-contained app bundle (embedded Python, signed + notarised) so the repo folder isn't needed
- [ ] Verified Windows build + installer; per-app capture on Windows
- [ ] Per-app (instead of global) system-audio tap on macOS, so only the call app is recorded
- [ ] Export to DOCX
