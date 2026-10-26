# 0002. Core Audio process tap (global, own process excluded) for system audio

- Status: Accepted
- Date: 2026-10-26

## Context
To transcribe the other side of a call, RecAndCC has to record what the Mac plays. Before
macOS 14.4 this meant a virtual audio driver or screen capture. The capture also has to stay
time-aligned with the microphone track ([0003](0003-two-tracks-me-others-echo-dedupe.md)) and
continuous on disk.

## Decision
- Capture system audio with a **Core Audio process tap** (macOS 14.4+) in `rcc-audiotap
  capture`: `CATapDescription(stereoGlobalTapButExcludeProcesses:)`, i.e. everything the Mac
  plays **except the helper's own process** (plus any extra PIDs passed in), through a private
  aggregate device. The tap is private (only this process sees it) and unmuted (the user
  keeps hearing the audio).
- The helper converts to 16 kHz mono float32 with `AVAudioConverter` and streams raw PCM to
  Python over a pipe.
- **Silence filler**: a process tap delivers no IO cycles while nothing plays. A 100 ms timer
  writes zeros, paced by the wall clock, whenever no real audio arrived for > 200 ms, so the
  `Others` track stays aligned with the mic track and the reader can tell silence from a stall.
  Real audio and filler go through one serial queue, so they never interleave.

## Alternatives considered
- **Virtual driver (BlackHole) / multi-output device**: needs a driver install and changes
  the user's output routing.
- **ScreenCaptureKit audio**: needs the screen-recording permission, which is more invasive
  than "System audio recording only".
- **Per-app tap (only the call app)**: cleaner recordings, but needs tap re-targeting as apps
  come and go; left as future work in [ROADMAP.md](../ROADMAP.md).

## Consequences
- No driver and no screen-recording permission; one TCC grant ("System audio recording")
  charged to RecAndCC ([0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md)).
- Risk: the global tap records **everything** that plays during a session (notification
  sounds, a video in another tab), not just the call app.
- Because idle periods are padded with zeros, an all-zero `Others` track is normal when nothing
  played; the "permission missing" warning only fires when the permission isn't granted
  (c0864d2).
- Requires macOS 14.4+. Windows uses WASAPI loopback (`capture/system_win.py`), untested on
  hardware ([#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5)).
- Hosted CI runners have no audio device or TCC grant, so live tap tests only run on real Macs
  (9140311).

## Links
- `native/macos/AudioTap/Sources/rcc-audiotap/Capture.swift`, `src/recandcc/capture/system_mac.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "macOS system audio: Core Audio process taps"
- Issues [#5](https://github.com/Galina-Blokh/rec_and_cc/issues/5),
  [#7](https://github.com/Galina-Blokh/rec_and_cc/issues/7) (balancing mic vs system audio)
- Commits fa1e57d, c0864d2, 9140311
