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
| **Wall time, peak memory** | seconds for the whole run (model loading included) and the process's peak resident memory (`getrusage` on macOS/Linux, `psutil` on Windows; n/a if neither works). On the MLX path, MLX's own peak GPU allocation is reported too, since Metal buffers may not show in the RSS |
| **Live latency** (`--live`) | per live caption: seconds from the nominal capture time of the 100 ms block holding the segment's end (the moment that audio ended) to the moment the caption was emitted; p50, p90 (linear interpolation) and max per clip and over all captions |

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
uv run python scripts/eval.py ~/RecAndCC-eval --backend mlx         # force MLX on the GPU
uv run python scripts/eval.py ~/RecAndCC-eval --backend cpu         # force faster-whisper, CPU
uv run python scripts/eval.py ~/RecAndCC-eval --hypotheses ~/RecAndCC-eval/whisper-cpp
```

`--backend` changes the transcription settings in memory for that run only; the config on
disk is never written. `mlx` fails at once if MLX is not available (not Apple silicon, or
mlx-whisper not installed). `auto` (the default) uses the config's `backend` and `device`.
The chosen backend is printed and stored in `eval-results.json` as `backend`, with
`wall_s`, `peak_memory_mb` and, on MLX, `gpu_peak_memory_mb`. Run one backend per process,
so that the peak memory belongs to that backend alone. Peak memory is this process's high-water
mark, so it includes the diarizer (add `--no-diarize` to leave it out). Wall time includes
model loading and, with `--live`, the live replay, so compare backends' wall times without
`--live`.

With `--live`, each clip is also replayed through the live transcriber (`LiveTranscriber`,
the same one that drives captions in the app) after the batch run, on the same models. The
audio is fed in 100 ms blocks. Latency counts from each block's nominal capture time, so a
step that runs late is counted, not hidden. While a region is being transcribed, the replay
runs in real time, so the model's speed counts in full. While nothing is in flight, idle time
is skipped, so a clip usually takes much less than its own length. The models and the VAD are
loaded before the replay starts, as in the running app. A region still open when the clip ends
is cut and emitted without waiting for the trailing silence, like captions when a recording
stops. End clips with a second or two of silence, or that last caption looks faster than the
rest. A clip whose replay fails is counted in `clips_without_live` and left out of the latency.

Each clip is copied to a temporary session folder and run through the session pipeline
(`Transcriber.transcribe_session`), and the diarizer runs on the same copy. The copies go in
a new owner-only `rcc-eval-*` folder in the system temp directory, created for that run alone and
deleted when it ends. A run killed with SIGKILL can leave that folder behind; delete it by hand. The script prints scores only, never transcript text. It writes the full
results, including the counts for each clip, to `eval-results.json`. Like the app, it runs
offline once the models are downloaded.

## Scoring other engines (`--hypotheses`)

To compare engines that don't run inside RecAndCC (whisper.cpp, WhisperKit, …), run them by
hand on the same clips (same model size, beam 5, language restricted to en/ru) and put one
file per clip in a folder, named after the clip's `id`. That folder holds transcript text,
so keep it inside the eval set's folder (or another git-ignored eval path), never in the repo.
`eval.py --hypotheses DIR` then scores those files with the same code as a pipeline run;
it loads no models and can't be combined with `--model`, `--backend` or `--live`.

`<clip-id>.json`, segments in seconds from the start of the clip:

```json
{
  "elapsed_s": 41.2,
  "segments": [
    {"start": 0.4, "end": 3.1, "text": "...", "language": "en", "speaker": "0"},
    {"start": 3.5, "end": 6.0, "text": "...", "language": "ru", "speaker": "1"}
  ]
}
```

- A bare list of segments works too. `start`, `end` and `text` are required; segments must
  end within the audio (0.5 s of slack). A UTF-8 BOM is accepted.
- `language` (optional): language-ID accuracy is scored only if every segment has one; it
  is the engine's own language ID, not RecAndCC's (`language_id` in the results says so).
- `speaker` (optional, string or number): DER is scored only on `others` clips where every
  segment has one, and not with `--no-diarize`.
- `elapsed_s` (optional): the engine's wall time for the clip, used for RTF. Clips without
  it are left out of RTF, with a warning.

`<clip-id>.txt`: plain text, no times. Scored for WER only, over the whole clip: listed
under the clip's language, or `mixed` for a code-switched clip. Language ID and DER are n/a.
If both files exist, the `.json` is used.

The report's `wall_s` and `peak_memory_mb` are n/a here: this script's own time and memory
say nothing about the engine. Measure the engine's run yourself, e.g. with `/usr/bin/time -l`
(macOS: "maximum resident set size"), and put those numbers in the table below. Error
messages name the clip and segment, never a value from the file.

## Backend comparison

One machine, same set, models cached, network blocked, on AC power (ADR 0011, follow-up 1;
the switching gate is in [ADR 0011](adr/0011-stack-and-packaging-after-v0-1.md)). Scores only.

| Engine | How | WER en | WER ru | LID | RTF | Live p90 | Wall s | Peak MB |
|---|---|---|---|---|---|---|---|---|
| MLX (GPU) | `--backend mlx` | | | | | | | |
| faster-whisper (CPU) | `--backend cpu` | | | | | | | |
| whisper.cpp (Metal) | `--hypotheses` | | | | | n/a | | |
| WhisperKit | `--hypotheses` | | | | | n/a | | |

## Baseline

Re-run after changing the model, VAD or diarization settings, and add a row here (results only,
never audio or text):

| Date | Model | Set | WER en | WER ru | LID | DER | RTF | Live p90 |
|---|---|---|---|---|---|---|---|---|
| (none yet: needs the consented set) | | | | | | | | |
