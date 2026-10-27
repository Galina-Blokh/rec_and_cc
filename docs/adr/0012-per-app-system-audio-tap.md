# 0012. Optional per-app system-audio tap that follows the trigger app

- Status: Accepted
- Date: 2026-10-27

## Context
The global tap ([0002](0002-core-audio-process-tap.md)) records everything the Mac plays
during a session: notification sounds, a video in another tab. Recording only the call app
gives cleaner `Others` tracks and keeps unrelated audio out of transcripts. An app's audio
often comes from several processes (browser renderers, Electron helpers), and those start
and stop during a session. Detection already groups them under one pattern (`engine.py`
`group_activity`: substring of bundle id or name).

## Decision
- `detection.system_audio_scope = "app"` makes the `Others` track record only the app that
  triggered the session. The engine passes an `AppTarget` (the app's pattern + pids, in
  `models.py`) via `Recorder.start` to `create_system_source(target)`.
- `rcc-audiotap capture --pid PID --match TEXT` builds
  `CATapDescription(stereoMixdownOfProcesses:)` over the matching process objects. Every
  second it re-checks the set and updates the running tap through `kAudioTapPropertyDescription`
  (no new aggregate device, no gap), so helpers that start later are recorded. An empty set
  (the app quit) keeps the last one.
- Stdin lines `pid N` / `match TEXT` add targets to a running tap. Used when a call starts
  during a media session (US-07): the tap then records **both** apps, so the video's audio
  up to that point stays and the call is recorded too.
- Falls back to the global tap, with a log line: manual sessions, no process object for
  the target at start, the per-app tap can't be created, or the source can't be made.
- **Default stays `"all"`.** Some apps don't play call audio from their own process: FaceTime
  uses the `avconferenced` daemon, so a per-app tap on FaceTime would record silence. A
  silent `Others` track in a call loses the other side, which is worse than a stray
  notification. Switch the default once Zoom, Teams, Meet and FaceTime are verified.

## Alternatives considered
- **Rebuild tap + aggregate device on every change**: a gap of tens of ms per change and more
  Core Audio churn; updating the description in place worked on macOS 26.7.
- **Switch to the global tap on a call upgrade**: needs restarting the helper mid-session
  (a gap) and loses the point of the option.
- **Pids only, no pattern**: misses helper processes that start after the session.

## Consequences
- Windows ignores the target (WASAPI loopback is global; per-app is #5).
- Known gap: apps whose audio comes from a system daemon (FaceTime) record silence with
  `"app"`; keep the global default for them.
- Measured on this Mac (macOS 26.7, `tests/test_mac_native.py`): a per-app tap on `afplay`
  playing a 1 kHz tone kept > 98 % of the energy in the tone band while `say` spoke; the
  global tap recorded both. A process added through stdin was picked up within ~1 s.

## Links
- `native/macos/AudioTap/Sources/rcc-audiotap/Capture.swift`, `src/recandcc/capture/system_mac.py`,
  `src/recandcc/recorder.py`, `src/recandcc/engine.py`
- [ROADMAP.md](../ROADMAP.md), [CONFIGURATION.md](../CONFIGURATION.md) `system_audio_scope`
