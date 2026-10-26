# 0008. In-app toast instead of `osascript` notifications

- Status: Accepted
- Date: 2026-10-26

## Context
The app tells the user about events such as "Transcript ready" or a hotkey start/stop.
v0.1 posted them with `osascript -e 'display notification …'`. macOS attributes those
notifications to Script Editor, which then popped up an empty Script Editor window on every
notification. The Python child of `RecAndCC.app` is not an app bundle, so it cannot post real
User Notifications of its own either ([0007](0007-ad-hoc-signing-launcher-helper-outside-bundle.md)).

## Decision
Show notifications as an **in-app toast** (`ui/toast_mac.py`): a small floating,
non-activating, click-through `NSPanel` in the top-right corner that hides by itself after
6 s. It never takes focus, joins all Spaces and shows over full-screen apps. It is created
and updated only on the main thread (the menu-bar app's 0.5 s timer calls `tick`).

## Alternatives considered
- **`osascript display notification`**: no extra code, but the Script Editor window on every
  notification made it unusable.
- **`UNUserNotificationCenter` from the launcher or a helper app**: real Notification Center
  entries, but needs a bundled process that owns the notifications plus IPC to it, and a
  bundle change re-triggers the permission problem of the ad-hoc signature.
- **Menu-bar title/icon changes only**: easy to miss.

## Consequences
- No stray windows; toasts are visible over full-screen calls and never steal focus.
- Toasts are not in Notification Center: no history, no Focus/Do-Not-Disturb integration,
  and a missed toast is gone after 6 s.
- Dialogs (the "Keep it?" prompt, onboarding) still use `osascript` on worker threads; only
  notifications moved.
- Windows uses the tray backend's own notifications (`ui/tray.py`).

## Links
- `src/recandcc/ui/toast_mac.py`, `src/recandcc/ui/mac.py`
- [ARCHITECTURE.md](../ARCHITECTURE.md) "Concurrency model" (UI main thread, UI workers)
- Issue [#3](https://github.com/Galina-Blokh/rec_and_cc/issues/3) (native shell would give real notifications)
- Commit 399b783
