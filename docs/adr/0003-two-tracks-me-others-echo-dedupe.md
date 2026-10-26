# 0003. Two tracks (mic = `Me`, system = `Others`) + echo dedupe

- Status: Accepted
- Date: 2026-10-26

## Context
A call transcript is only useful if it says who said what. Full speaker diarization of a
mixed recording is expensive and error-prone, but on a computer the two sides of a call
already arrive on different paths: the user through the microphone, everyone else through
system audio.

## Decision
- Record the **mic** and **system audio** as two separate WAV tracks of one session
  (`mic.wav` → `Me`, `others.wav` → `Others`) and transcribe both, in parallel, on the shared
  worker pool.
- Both tracks share one timeline. A mic track that joins mid-session (a media recording
  upgraded to a call, ef8e8ba) is padded with silence up to that moment; the system track is
  padded by the helper's silence filler ([0002](0002-core-audio-process-tap.md)).
- **Echo removal** (`transcribe/merge.py`, `dedupe_echo = true`): on speakers the mic also
  hears the other side. A `Me` segment is dropped when it overlaps an `Others` segment in time
  (with 1.5 s tolerance for echo delay and timestamp jitter) and its text is similar
  (`difflib` ratio ≥ 0.6). Live captions apply the same check; a `Me` segment waits at most
  2 s for `Others` to catch up before it is shown.
- `record_mic = "calls"` by default: media sessions (a video) record only the system track.

## Alternatives considered
- **One mixed track + diarization only**: loses the free, reliable `Me` label and costs a
  diarization pass for every session.
- **Acoustic echo cancellation on the mic signal**: would need a reference-aligned AEC in the
  capture path; text-level dedupe is simpler and works after the fact on any recording.

## Consequences
- `Me` / `Others` labels for free, and the two tracks transcribe concurrently.
- `Others` still mixes everyone else; splitting it is [0009](0009-sherpa-onnx-campp-diarization.md).
- Risks: with speakers, echo that Whisper transcribes differently (ratio < 0.6) survives as a
  duplicate `Me` line; a real `Me` reply that repeats the other side's words while they are
  still talking can be dropped. In-room conversations go to one mic track (room listening is
  [#2](https://github.com/Galina-Blokh/rec_and_cc/issues/2)).

## Links
- `src/recandcc/recorder.py`, `src/recandcc/transcribe/merge.py`, `src/recandcc/transcribe/live.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Two tracks → speaker labels for free"
- [CONFIGURATION.md](../CONFIGURATION.md) `dedupe_echo`, `record_mic`
- Issues [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1), [#2](https://github.com/Galina-Blokh/rec_and_cc/issues/2),
  [#7](https://github.com/Galina-Blokh/rec_and_cc/issues/7)
- Commits fa1e57d, ef8e8ba
