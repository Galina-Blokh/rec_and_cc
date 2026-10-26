## Summary

<!-- What changes and why, in 1–3 sentences. Keep PRs ≲ 400 changed lines (see AGENTS.md). -->

## Linked issue / story

Closes #

## Acceptance criteria met

<!-- Copy the issue's acceptance criteria and tick them, or explain any gap. -->

- [ ]

## Tests added / updated

- Fast (`uv run pytest`):
- Slow (`uv run pytest -m slow`), if `transcribe/` or `capture/` changed:
- Manual checks (app, helper, Windows), if any:

## AI review findings addressed

<!-- Cursor review (and any other AI review): what was fixed, and for each finding not fixed,
why not. -->

## Risk / rollback

<!-- What could break (threads, permissions/TCC, Windows, offline/privacy) and how to roll
back (revert this PR; any config/data migration?). -->

## Checklist

- [ ] `uv run ruff check .` and `uv run ruff format --check src tests` pass
- [ ] `uv run pytest` passes (and `-m slow` where relevant)
- [ ] Docs updated (README / docs/*.md / CONFIGURATION.md for new keys), or not needed
- [ ] No `RecAndCC.app` rebuild needed, or the PR says why (TCC permissions reset)
- [ ] Hard rules in AGENTS.md respected (non-blocking audio threads, main-thread UI, offline, privacy, Windows)
