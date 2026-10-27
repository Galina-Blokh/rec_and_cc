# Intent

The *why* behind RecAndCC: the problem it solves, what it must and must not do, and the
constraints every change has to respect. User stories with acceptance criteria are in
[stories.md](stories.md); design decisions are in [../adr/](../adr/README.md).

## Problem

The owner spends much of the day in calls, meetings and videos (English and Russian, often
mixed) and wants to remember what was said without taking notes. Existing tools either upload
audio to a cloud service, need to be started by hand for every call, join meetings as a bot,
or handle only one language per recording.

## Goals

1. **Never miss a conversation.** Notice on its own when the computer plays or captures
   voice (calls, YouTube, podcasts, any video) and record it, or offer to, without manual
   steps. Allow manual start/stop for everything else (in-person meetings).
2. **A useful transcript, quickly.** Text with timestamps and speaker labels (`Me`, `Others`,
   `Speaker N`), English and Russian detected automatically even when the language switches
   mid-meeting. Live captions while recording; the final transcript a few seconds after the
   stop.
3. **Completely private.** RecAndCC itself never uploads audio or transcripts, and works
   offline once the models are downloaded. (A synced folder is another matter: the default
   `~/Documents/Transcribes` may be uploaded by iCloud "Desktop & Documents" or OneDrive
   folder backup; the app warns when `output_dir` is cloud-synced.)
4. **Invisible when idle.** An always-on menu-bar / tray app that starts at login, uses
   little CPU while waiting, and needs no attention.
5. **Portable.** macOS first; Windows supported by design so the owner can switch laptops.

## Non-goals (for now)

- Video recording or screenshots: audio only.
- Cloud transcription, accounts, sync, sharing or collaboration.
- Meeting bots or integrations that join calls on the owner's behalf.
- Multi-user / team deployment: it is a personal tool for one person.
- Summaries, search and speaker *names* are future work ([ROADMAP](../ROADMAP.md),
  issues #1–#7), not part of v0.1.

## Constraints

| Constraint | Means | Enforced by |
|---|---|---|
| Local only | Whisper (MLX on Apple-silicon GPU, faster-whisper on CPU) and diarization run on the device | ADR 0004, 0009 |
| Offline | No network except the one-time model download (default models pinned to exact revisions, speaker model SHA-256-verified); no telemetry or update checks | AGENTS.md rule 5; `tests/test_offline.py` checks model loading only, no end-to-end blocked-network test yet |
| Privacy | No audio or transcript content in logs at INFO+; recording folders owner-only (POSIX `0700`; not yet enforced by Windows ACLs) | AGENTS.md rule 6, `tests/test_recorder.py` |
| Languages | English and Russian, auto-detected per speech region | ADR 0005 |
| Platforms | macOS 14.4+ (Core Audio process taps); Windows through the same interfaces | ADR 0001, 0002; CI on Windows |
| Responsiveness | Audio threads never block; UI only on the main thread | AGENTS.md rules 1–3 |
| Legal | Recording others is regulated; the user is told and stays in control (Ask mode, consent notes) | README "Privacy & consent" |

## Success measures

- A call or video with voice is recorded within ~3 s of audio starting, with no user action
  in Auto mode.
- Live captions appear about 1 s after each phrase on Apple silicon.
- A 1-hour meeting transcribes in about 2–3 minutes on the Apple-silicon GPU (about
  15 minutes on CPU-only machines such as Windows); the live transcript is final a few
  seconds after the recording stops.
- With all network access blocked, a session records and transcribes end to end (target;
  not yet covered by an automated test).
