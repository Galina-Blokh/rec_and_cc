# Security policy

## Reporting a vulnerability

This is a private repository with a single maintainer. Please report security issues
**privately** and directly to the repository owner,
[@Galina-Blokh](https://github.com/Galina-Blokh) (GitHub's private vulnerability reporting
isn't available for private repositories). If the repository becomes public, use
"Report a vulnerability" on its Security tab instead.

Don't open a public issue or pull request for a vulnerability, and don't attach real
recordings or transcripts. Include what you found, how to reproduce it, and the affected
version or commit. You'll get an answer as soon as possible; fixes go to `main` and are noted
in [CHANGELOG.md](CHANGELOG.md).

Only the latest version on `main` is supported.

## Threat model (summary)

RecAndCC records calls and transcribes them. What it protects, and how:

### Local-only data
- Audio, transcripts, settings and logs stay on the computer
  (paths: README → [Privacy & consent](README.md#privacy--consent)). Nothing is uploaded; there
  is no telemetry and no account.
- Recording and transcript folders created by the app are owner-only (`0700`). Other local
  users and processes running as **you** are out of scope: anything running under your
  account can read your files, and macOS permissions (Microphone, System audio) are what gate
  recording itself.
- Retention is up to the user (`keep_audio`, `audio_retention_days`); transcripts are never
  deleted automatically. Disk encryption (FileVault / BitLocker) is recommended.

### Model downloads (the only network access)
- Whisper models come from Hugging Face at **pinned commits**
  ([`src/recandcc/transcribe/revisions.py`](src/recandcc/transcribe/revisions.py)), so a
  changed upstream repo can't swap the weights. Once downloaded they load with
  `local_files_only`, without network access.
- The speaker-diarization model is downloaded over HTTPS from a fixed GitHub release URL and
  checked against a pinned SHA-256 and size before use.
- Models are data, but loaders are complex parsers: pinning limits this to versions we have
  tested. A custom `model` repo id in the settings is downloaded unpinned (logged).

### Dependencies and CI
- Python dependencies are locked in `uv.lock`; CI installs with `--locked`.
- GitHub Actions are pinned to commit SHAs; Dependabot proposes weekly updates.
- Ruff's bandit rules (`S`) run in CI. Subprocess calls use fixed argument lists, never a
  shell.

### Secrets
- The app itself needs no secrets, API keys or accounts.
- `.env` files are git-ignored. The only CI secret is `CURSOR_API_KEY`, for the AI review.

### AI code review (Cursor agent) and its accepted risk
The [Cursor review workflow](.github/workflows/cursor-review.yml) reads untrusted PR content,
so it is hardened:
- the agent runs read-only (`--mode ask`, sandbox on), on the trusted **base** tree, with the
  PR diff as a data file and agent instruction files (`AGENTS.md`, `.cursor`, …) removed;
- the agent build is pinned and verified by SHA-256, and actions are pinned to commit SHAs;
- no GitHub token is persisted; the API key only exists in the review step, and a review that
  contains it is withheld;
- the review is advisory, and fork PRs get no secrets (the job is skipped).

**Accepted risk:** on `pull_request`, a branch in this repo can modify the workflow itself and
thus reach the `CURSOR_API_KEY` secret. That is acceptable while the repository is private
with a single maintainer. With outside collaborators, the privileged part must move to a
`workflow_run` job on the default branch.
