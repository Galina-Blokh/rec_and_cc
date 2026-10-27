# 0013. Adaptive speaker count instead of a fixed clustering threshold

- Status: Accepted
- Date: 2026-10-27

## Context
[0009](0009-sherpa-onnx-campp-diarization.md) clusters ~2 s CAM++ window embeddings with
average linkage and stops at a fixed cosine distance of 0.5. One threshold for every session
over-merges similar voices in big meetings and over-splits one speaker whose windows are
noisy (bad mic, codec). Issue [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1) B.2
asks to estimate the number of speakers and to accept `min_speakers` / `max_speakers` hints.
Constraints: offline, no new dependencies or models, an hour must still diarize in seconds.

## Decision
- Run the average linkage to a single cluster and record the merges. For every k ≤ 20 (or
  `max_speakers`) on the way, score a size-weighted **cluster silhouette**: per cluster
  `(b - a) / max(a, b)`, `a` = mean within-cluster cosine distance, `b` = average-linkage
  distance to the nearest other cluster. Both come from the linkage matrix the algorithm
  keeps anyway (plus a running within-cluster sum), so a score is O(k²), not O(n²).
- Pick the best-scoring k within `[min_speakers, max_speakers]`; one speaker if no split
  scores ≥ 0.3 (`MIN_SPLIT_SCORE`) and no minimum is set.
- With fewer than 10 windows (`MIN_ADAPTIVE_UNITS`) the old threshold cut is kept (clamped to
  the hints): too little data to estimate k. Tiny clusters are still dissolved, but never
  below `min_speakers`. The 3000-window subsample stays.

## Alternatives considered
- **Eigengap of the cosine affinity**: needs an n×n eigendecomposition (~1.4 s for 3000
  windows in numpy; partial solvers need scipy, a new direct dependency) and a k = 1 rule.
- **Point-level silhouette over a threshold sweep**: O(n²k) per candidate (~2·10⁹ flops for k ≤ 20
  at 3000 windows, more than the linkage itself); the cluster score reuses its matrix.

## Consequences
- Synthetic embeddings (tests/test_diarize.py): 2/4/8/12 noisy voices recovered exactly; one
  speaker with within-cosine ~0.5 stays one (the fixed threshold made 4–27 speakers of it);
  two voices whose windows are ~0.55 cosine apart are separated (the threshold merged them).
- Timing: clustering 6000 windows (subsampled to 3000) ~0.15 s on an M5 Max; the
  embedding pass still dominates.
- Risk: one speaker with two clearly different acoustic conditions may score as two. The
  constants are tuned on synthetic data; DER on real meetings is
  [#13](https://github.com/Galina-Blokh/rec_and_cc/issues/13).

## Links
- `src/recandcc/transcribe/diarize.py` (`cluster_embeddings`, `_pick_k`, `_dendrogram`)
- [CONFIGURATION.md](../CONFIGURATION.md) `min_speakers`, `max_speakers`; story US-12
- Issue [#1](https://github.com/Galina-Blokh/rec_and_cc/issues/1) part B
