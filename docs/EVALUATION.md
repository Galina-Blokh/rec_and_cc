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

**Synthetic interim set.** Until the consented set exists, `scripts/make_synthetic_evalset.py`
(macOS only) renders 7 clips (~2.7 min) of fixed, neutral sentences with `say` voices: en, ru
and he alone, en+ru, en+he and ru+he code-switching, and a 3-speaker English dialogue. It
writes the WAVs and `manifest.json` to the system temp folder (never into a git checkout,
~/Documents or a non-empty folder that holds no earlier set). The baseline below was measured
with the voices Samantha, Reed (English (US)) and Flo (English (US)) for en, Milena for ru and
Carmit for he. Some macOS versions need Milena or Carmit downloaded first (System Settings >
Accessibility > Spoken Content > System voice > Manage Voices); a clip whose voice is missing
is skipped with a message, so a set without it is not comparable with the baseline. Clips with
one voice are `me` tracks (no DER); clips with two or three voices, including the
code-switching ones, are `others` tracks and scored for DER. Studio TTS is far cleaner than a call, so these
scores overstate real-world accuracy; use them to catch regressions, not to choose models.

**Consent.** Only record people who agreed to the recording being used to test this app.
Note who agreed and when, next to the set (not in the repo). Don't use clips from real calls
without everyone's consent.

## Running

```bash
uv run python scripts/eval.py ~/RecAndCC-eval                       # config's model
uv run python scripts/eval.py ~/RecAndCC-eval --model small --json small.json
uv run python scripts/eval.py ~/RecAndCC-eval --no-diarize          # skip DER
uv run python scripts/eval.py ~/RecAndCC-eval --diarize-overlap     # DER with the segmentation model
uv run python scripts/eval.py ~/RecAndCC-eval --live                # + live caption latency
uv run python scripts/eval.py ~/RecAndCC-eval --backend mlx         # force MLX on the GPU
uv run python scripts/eval.py ~/RecAndCC-eval --backend cpu         # force faster-whisper, CPU
uv run python scripts/eval.py ~/RecAndCC-eval --languages en,ru,he  # detect among these
uv run python scripts/eval.py ~/RecAndCC-eval --hypotheses ~/RecAndCC-eval/whisper-cpp
```

`--backend` and `--languages` change the transcription settings in memory for that run only;
the config on disk is never written. The languages used are stored as `languages`; an unknown
or empty `--languages` is an error. `mlx` fails at once if MLX is not available (not Apple
silicon, or mlx-whisper not installed). `auto` (the default) uses the config's `backend` and `device`.
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

**The filled rows (†) are synthetic TTS (macOS say), not the consented set: interim,
overstates real-world accuracy.** 2026-10-27, `large-v3-turbo`, languages en,ru,he, Apple M5
Max (36 GB, macOS 26.7.1, AC power), models cached, `HF_HUB_OFFLINE=1`, diarizer on. Wall
and peak memory come from the runs without `--live`; the MLX run went first, with a cold
model load. Peak MB is the process RSS; MLX's own GPU peak was 2449 MB. Live p90 is from a
separate `--live` MLX run; live was not run on the CPU backend.

| Engine | How | WER en | WER ru | WER he | LID | RTF | Live p90 | Wall s | Peak MB |
|---|---|---|---|---|---|---|---|---|---|
| MLX (GPU) † | `--backend mlx` | 0.000 | 0.025 | 0.030 | 1.00 | 0.040 | 1.24 s | 24.3 | 2747 |
| faster-whisper (CPU) † | `--backend cpu` | 0.000 | 0.025 | 0.030 | 1.00 | 0.263 | not run | 43.4 | 5468 |
| whisper.cpp (Metal) | `--hypotheses` | | | | | | n/a | | |
| WhisperKit | `--hypotheses` | | | | | | n/a | | |

## Baseline

Re-run after changing the model, VAD or diarization settings, and add a row here (results only,
never audio or text):

| Date | Model | Set | WER en | WER ru | WER he | LID | DER | RTF | Live p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-27 | large-v3-turbo, MLX, M5 Max | **synthetic TTS (macOS say), not the consented set: interim, overstates real-world accuracy** (7 clips, 331 words) | 0.000 | 0.025 | 0.030 | 1.00 | 0.240 | 0.040 | 1.12 / 1.24 s |
| (consented set: none yet) | | | | | | | | | |

On the synthetic set, all DER is false alarm (~2.2–2.7 s per clip, the pauses between
sentences counted as speech) except on ru+he: Milena and Carmit, two female voices, merge
into one speaker (DER 0.64 there; 0.13 on the other three `others` clips). Diarization takes
~0.08 s per clip. Both backends give the same WER and language ID on this set.

To regenerate: `uv run python scripts/make_synthetic_evalset.py "$TMPDIR/rcc-synth"`, then
`uv run python scripts/eval.py "$TMPDIR/rcc-synth" --languages en,ru,he --backend mlx --json
"$TMPDIR/mlx.json"` (and `--backend cpu`, and `--backend mlx --live`).

## Summary timing (synthetic 1-hour transcript)

Acceptance check of issue #4: "a 1-hour meeting summary in 1-2 minutes". Input: the transcript
from `scripts/make_synthetic_meeting_transcript.py` at its default seed 0 (232 segments, 8551
words, four speakers, English; fixed templates, not real data). The default `[summary]` config
(`mlx-community/Qwen2.5-3B-Instruct-4bit`, 4-bit MLX, `max_tokens = 400`) ran through
`summarize_session` -> `summarize` (the `recandcc summarize` path; timed by calling `summarize`
with `MlxGenerator` directly) with `HF_HUB_OFFLINE=1` and the model from the local cache. Only
timings are recorded here, never text.

| Date | Machine | Model | Load (1st in process) | Load (reload) | Generation | Total |
|---|---|---|---|---|---|---|
| 2026-10-27 | Mac17,6 (Apple M5 Max, 36 GB), macOS 26.7.1 | Qwen2.5-3B-Instruct-4bit | 1.70 / 1.93 / 1.64 s | 0.75 / 0.59 / 0.52 s | 30.1 / 26.2 / 27.2 s | ~28-32 s |

Three runs, each in a fresh process. Map-reduce made 10 model calls (8 chunk notes, one
summary, one action list): 14,866 prompt tokens prefilled and 2,947 tokens generated. That is
about 100 generated tokens/s counting the prefill time in the denominator (2,947 tokens in
~28 s); generation-only throughput is higher. The result had a 606-character summary and 8
action items.

Verdict: **meets the 1-2 minute target on an M5 Max** at roughly 30 s, about 4x inside the 2
minute limit; slower Macs are untested. Generation (prefill plus decoding) dominates; loading is
under 2 s. The load times have the model files in the OS page cache; a cold-disk load was not
measured (needs `sudo purge`). Timings are without a recording running: the summary thread
waits while recording, and shares the GPU lock with Whisper.

Cross-check: an independent run by another session on the same machine class (M5 Max,
Qwen2.5-3B-Instruct-4bit, 8 chunks, about 9,000 en/ru words) took 30.7 s, in line with the
numbers above. That run is `tests/test_summary_timing.py` (`uv run pytest -m slow -k
summary_timing -s`, offline, fails above 180 s).

To regenerate: `uv run python scripts/make_synthetic_meeting_transcript.py "$TMPDIR/rcc-meeting"`,
then time `summarize` on that folder.

## Interim synthetic speaker baseline

For issue #1 ("6-speaker test meeting: DER measurably better than v0.1; no regression on 2–3
speakers").
**Synthetic TTS (macOS `say`), not the consented set: interim, overstates real-world accuracy.**
Real calls have room noise, codecs, crosstalk and voices that differ far less than six distinct
TTS voices do, so treat the numbers as a regression check, not as the acceptance result.

DER here is scored with 10 ms frames and no collar.

Set: `scripts/make_synthetic_evalset.py --meetings` (en only; voices Samantha, Reed, Daniel,
Karen, Moira, Rishi). Three `others` clips of turn-taking meetings with turns of 1 to ~15
words, varied pauses (0.15-1.0 s) and quick replies: 2 speakers (71 s, 24 turns), 3 speakers
(83 s, 26 turns), 6 speakers (224 s, 77 turns). No overlapping speech. 2026-10-27, Apple M5 Max,
macOS 26.7.1, offline (`HF_HUB_OFFLINE=1`), diarizer models `3dspeaker_speech_campplus_sv_zh_en_16k`
(embedding) and, where noted, `pyannote-segmentation-3.0` (ONNX), both from the local cache.
Transcription (large-v3-turbo, MLX, `--languages en`) gives WER 0.003 on the 997 words.

| Diarizer | Clip | Speakers (found / true) | DER | Missed s | False alarm s | Confusion s |
|---|---|---|---|---|---|---|
| v0.1.0 | 2 speakers | 3 / 2 | 0.257 | 0.0 | 7.6 | 7.4 |
| v0.1.0 | 3 speakers | 3 / 3 | 0.210 | 0.0 | 8.6 | 5.9 |
| v0.1.0 | 6 speakers | 6 / 6 | 0.357 | 0.0 | 25.3 | 41.4 |
| 281bd90, `diarize_overlap` off | 2 speakers | 2 / 2 | 0.231 | 0.0 | 7.6 | 5.9 |
| 281bd90, `diarize_overlap` off | 3 speakers | 3 / 3 | 0.210 | 0.0 | 8.6 | 5.9 |
| 281bd90, `diarize_overlap` off | 6 speakers | 6 / 6 | 0.357 | 0.0 | 25.3 | 41.4 |
| 281bd90, `diarize_overlap` on | 2 speakers | 2 / 2 | 0.130 | 0.0 | 7.6 | 0.0 |
| 281bd90, `diarize_overlap` on | 3 speakers | 3 / 3 | 0.169 | 0.0 | 8.6 | 3.1 |
| 281bd90, `diarize_overlap` on | 6 speakers | 5 / 6 | 0.309 | 0.0 | 25.3 | 32.3 |

These numbers are not comparable with the 7-clip baseline above (DER 0.240): the clip set
differs (long meetings with 2, 3 and 6 distinct voices, shorter pauses).

Reading it:

- Missed speech is 0 everywhere. False alarm is the same in every row because every
  diarizer fills the pauses between turns (about 72-75% of the pause time in the three clips).
  Confusion is the part that differs.
- 281bd90 with `diarize_overlap` off matches v0.1.0 at 3 and 6 speakers (consistent with
  the same windows and clustering) and is slightly better at 2 (v0.1.0 found a third speaker;
  the cause was not investigated).
  So: no regression at 2-3 speakers, but **no gain at 6 speakers either**.
- With `diarize_overlap` on, DER improves on all three clips (6 speakers: 0.357 to 0.309), but it
  merged two of the six voices (5 found). The "better than v0.1" criterion is met only in that
  mode and only by 5 points on TTS; it stays open until measured on a real, consented
  6-speaker meeting.
- The 6 / 6 with `diarize_overlap` off is a coincidence, not better separation: it merges the
  same two voices, and its sixth "speaker" is a ~4 s cluster of two stray windows. See
  [Why two of the six voices merge](#why-two-of-the-six-voices-merge).

How the rows were made: the diarizer at commit 281bd90 (the commit measured, pinned here; not
the current main), scored with `scripts/eval.py` from #111, with and without its
`--diarize-overlap` flag (which builds `Diarizer(segmentation=True)`, like config
`diarize_overlap`); `eval.py` also stores `speakers` and `ref_speakers` per clip. v0.1.0 has
no `eval.py` and no segmentation, so its rows come from a throwaway driver that put a
`git worktree` of the `v0.1.0` tag (commit 5126a7c) first on `sys.path`, ran its
`Diarizer().diarize_file` on each clip and scored the turns with the current `eval.py`'s `der`
(same references, same 10 ms frames, no collar), with the same cached embedding model
(`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`) as 281bd90. Each clip was run
once, so there is no variance estimate. `--diarize-overlap` exits with an error if the
segmentation model is not on disk (it never downloads or silently falls back). The audio was
generated into a temp folder and is not committed.

To regenerate: `uv run python scripts/make_synthetic_evalset.py --meetings "$TMPDIR/rcc-meetings"`,
then `uv run python scripts/eval.py "$TMPDIR/rcc-meetings" --languages en [--diarize-overlap]
--json "$TMPDIR/out.json"`.

### Why two of the six voices merge

Investigated on main at 8097620, which reproduces the rows above: the same speaker counts, and
DER 0.357 off and 0.309 on, as in the table (a rerun at 8097620 gave 0.308 on). The diarizer
is unchanged since 281bd90. The figures below (cosines, leave-one-out counts, the k splits,
the forced-count DER and the variants table) come from an uncommitted scratch analysis on the
regenerated meetings, not from a committed script.

- **The merged voices are Samantha (`en1`) and Karen (`en5`), in both modes.** The cosine
  similarity of their voice centroids (the unit mean of a speaker's clean windows) is 0.74.
  Every other pair is 0.25-0.58. Within one voice, windows average 0.83-0.87 to their own
  centroid. To CAM++ these two TTS voices are much closer than any other pair, but they are not
  the same: a leave-one-out nearest-centroid test with the true labels separates their windows
  perfectly (31 of 31 off, 33 of 33 on).
- **Average linkage merges them before it absorbs outliers.** Their windows' average
  cross-similarity is about 0.55. A few stray windows (a short reply, the end of a turn) are
  less similar than that to everything. So the dendrogram keeps 1-2-window outlier clusters
  as separate "speakers" longer than it keeps en1 and en5 apart. They first split at k = 12
  clusters off and k = 9 on. Both are well past the silhouette's pick (k = 6 off, 5 on), and
  the extra clusters at that point are 1-3-window outliers.
- **The speaker count is a near tie either way.** Silhouette off: k = 4: 0.470, 5: 0.439,
  6: 0.478. On: k = 4: 0.507, 5: 0.515, 6: 0.478. Off picks k = 6 over k = 4 by 0.008, and its sixth
  cluster is two stray windows (one en5, one en2), about 4 s, just above `MIN_SPEAKER_S`
  (2 s). On, segmentation turns those strays into cleaner pieces, so k = 5 wins, and the
  segmentation's own gain shows as lower confusion (32 s vs 41 s). Segmentation is not
  what merges the voices. The embedding windows are about as long with it as without it
  (mean clustered window 1.90 s vs 2.05 s).
- **Forcing the right count does not fix it either.** With `min_speakers = max_speakers = 6`,
  DER is 0.357 off and 0.313 on, because the sixth cluster is still an outlier, not en5.

**No principled fix found, so the clustering is unchanged.** Changes that were tried on the
saved embeddings:

| Change | 6 speakers, off / on | 2-3 speakers |
|---|---|---|
| Silhouette over cuts whose clusters of ≥ N windows count as speakers (smaller ones are outliers) | N = 2: 7 / 6; N = 3-4: 5 / 7; N = 5: 6 / 6 | unchanged |
| Spectral clustering, pruned cosine affinity, eigengap count | p = 0.1: 6 / 6; p = 0.15: 6 / 6; p = 0.2: 5 / 6; p = 0.3: 5 / 5 | p = 0.1: 16-18 speakers found with segmentation on; else unchanged |

Each fixes the 6-speaker clip only at one setting of its own parameter, and moving the
parameter one step breaks it again (or breaks the 2-3-speaker clips). One synthetic clip
can't justify that choice; it would be tuning to two TTS voices. Even with 6 / 6, DER stays
at 0.19-0.25 (false alarm in the pauses plus window-boundary confusion). Revisit with the
consented set. The options in order: a minimum cluster size from real meetings, so outliers
don't count as speakers; re-assigning outlier windows after the cut; a stronger embedding
model.
