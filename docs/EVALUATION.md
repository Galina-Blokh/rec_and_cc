# Evaluation

The slow tests check transcription with macOS `say` voices, which are clean and
easy to recognise, so passing them overstates accuracy. `scripts/eval.py` measures the real
pipeline on a small set of real, consented speech, so that model, VAD and diarization changes
(#1, #6, #7) can be compared with numbers.

## Metrics

| Metric | What it measures |
|---|---|
| **WER** per language | word edits / reference words, after lowercasing, removing punctuation and mapping ё → е. A hypothesis segment spanning several utterances has its words split between them by overlap; one overlapping none counts toward the nearest. |
| **Language-ID accuracy** | share of reference utterances whose overlapping speech was mostly (by overlap time) tagged with the right language |
| **DER** | missed + false-alarm + wrong-speaker time / reference speech time; 10 ms frames, no collar, best one-to-one speaker mapping (exact up to 12 speakers on the smaller side, greedy above) |
| **RTF** | total transcription time / total audio duration (batch pipeline) |
| **Diarization time** | seconds per clip |
| **Live latency** (`--live`) | per live caption: seconds from the moment the capture block holding the segment's end reached the live transcriber to the moment the caption was emitted; p50, p90 and max per clip and overall |

## The eval set

The set never goes into git. The reference text in `manifest.json` is as private as the audio.
Keep the folder outside the repo (for example `~/RecAndCC-eval`). If a copy has to live in
the clone, name its folder `eval` or `RecAndCC-eval`: folders with those names are ignored at
any depth, like `*.wav` and `eval-results.json`. Under any other name, git would pick up the
manifest text.
Audio paths in the manifest must stay inside the set's folder. Back the set up privately, for
example on an encrypted disk.

```
~/RecAndCC-eval/
  manifest.json
  call-en-ru-01.wav        any rate / channels; resampled to 16 kHz mono
  ...
```

```json
{
  "clips": [
    {
      "id": "call-en-ru-01",
      "audio": "call-en-ru-01.wav",
      "track": "others",
      "utterances": [
        {"start": 0.4, "end": 3.1, "speaker": "A", "language": "en", "text": "..."},
        {"start": 3.5, "end": 6.0, "speaker": "B", "language": "ru", "text": "..."}
      ]
    }
  ]
}
```

- `track` is `others` (system audio; diarized and scored for DER) or `me` (microphone; no DER).
- `speaker` names are any strings; only their grouping matters.
- Times are seconds from the start of the clip, within the audio. Overlapping utterances are
  allowed. Without word timestamps, though, a segment over simultaneous speech is split between
  the utterances in start-time order, so WER on crosstalk is approximate. DER handles overlap
  exactly.

Aim for 10–20 minutes overall: en and ru, code-switching within a call, a few accents, 2–6
speakers, and speaker-phone echo. Hebrew goes in later (#6).

**Consent.** Only record people who agreed to the recording being used to test this app.
Note who agreed and when, next to the set (not in the repo). Don't use clips from real calls
without everyone's consent.

## Running

```bash
uv run python scripts/eval.py ~/RecAndCC-eval                       # config's model
uv run python scripts/eval.py ~/RecAndCC-eval --model small --json small.json
uv run python scripts/eval.py ~/RecAndCC-eval --no-diarize          # skip DER
uv run python scripts/eval.py ~/RecAndCC-eval --live                # + live caption latency
```

With `--live`, each clip is also replayed through the live transcriber (`LiveTranscriber`,
the same one that drives captions in the app) after the batch run, on the same models. The
audio is fed in 100 ms blocks, each one when its audio would have been captured. While a
region is being transcribed, the replay runs in real time, so the model's speed counts in
full; while nothing is in flight, idle time is skipped, so a clip usually takes much less than
its own length. The models and the VAD are loaded before the replay starts, as in the running
app. The tail of a clip is emitted when the replay stops, like captions when a recording ends.

Each clip is copied to a temporary session folder and run through the session pipeline
(`Transcriber.transcribe_session`), and the diarizer runs on the same copy. The copies go in
a new owner-only `rcc-eval-*` folder in the system temp directory, created for that run alone and
deleted when it ends. A run killed with SIGKILL can leave that folder behind; delete it by hand. The script prints scores only, never transcript text. It writes the full
results, including the counts for each clip, to `eval-results.json`. Like the app, it runs
offline once the models are downloaded.

## Baseline

Re-run after changing the model, VAD or diarization settings, and add a row here (results only,
never audio or text):

| Date | Model | Set | WER en | WER ru | LID | DER | RTF | Live p90 |
|---|---|---|---|---|---|---|---|---|
| (none yet: needs the consented set) | | | | | | | | |
