# 0007. Ad-hoc signing, launcher as TCC responsible process, helper outside the bundle

- Status: Accepted
- Date: 2026-10-26

## Context
macOS privacy (TCC) attributes microphone and system-audio access to the *responsible
process*. Run from Terminal, the prompts would name Terminal or python3, and the grants would
belong to them. The app is not signed with a Developer ID; an **ad-hoc** signature identifies
it by its content hash, so any change to the bundle makes macOS treat it as a new app and
silently drop its grants. Early builds embedded `rcc-audiotap` in the bundle, so every helper
rebuild reset permissions (c18cf6c).

## Decision
- `RecAndCC.app/Contents/MacOS/RecAndCC` is a tiny **Swift launcher** that `posix_spawn`s the
  repo's `.venv/bin/python -m recandcc run` as a child, forwards signals and sets
  `RCC_APP_BUNDLE`. The app stays the responsible process, so the prompts say "RecAndCC" and
  the grants survive restarts.
- The bundle holds only the launcher, `Info.plist`, `launcher.json` and the icon, and is
  **ad-hoc signed** (`codesign -s -`). The build is reproducible (same sources → same cdhash);
  `build_app.sh --install` fingerprints the contents and leaves an unchanged installed app
  untouched, warning when a re-grant is required.
- The helper lives **outside** the bundle in `<repo>/bin`; `find_helper()` prefers it. TCC
  charges its audio access to the responsible app wherever the binary is.
- System-audio permission uses the TCC SPI (`TCCAccessPreflight` / `TCCAccessRequest` via
  `dlopen`), falling back to starting a tap, which triggers the system prompt.

## Alternatives considered
- **Run Python directly / from Terminal**: permissions attributed to Terminal or python3.
- **Helper inside the bundle**: every helper update changed the signature and dropped grants.
- **Self-signed code-signing certificate**: pins identity to the certificate so bundle
  changes keep grants, but needs a keychain change; not set up.
- **Developer ID signing + notarization, self-contained app**: the proper fix, proposed in
  [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3); not done yet.

## Consequences
- Python and helper changes keep permissions; only launcher / `Info.plist` / icon changes or
  moving the repo need a re-grant (table in [DEVELOPMENT.md](../DEVELOPMENT.md)).
- Risks: the app depends on the repo folder staying in place; it relies on a private TCC SPI
  that may change between macOS versions (its completion block had to be declared escaping
  after a SIGTRAP crash, 444a8f7); a helper outside the bundle is not covered by the app's
  signature.

## Links
- `native/macos/AudioTap/Sources/RecAndCC/main.swift`, `native/macos/build_app.sh`,
  `src/recandcc/native.py` (`find_helper`), `src/recandcc/permissions.py`,
  `native/macos/AudioTap/Sources/rcc-audiotap/Permission*.swift`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Permissions belong to RecAndCC, not Terminal/python"
- Issues [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3), [#17](https://github.com/Galina-Blokh/rec_and_cc/issues/17) (release hygiene)
- Commits fa1e57d, 444a8f7, c18cf6c
