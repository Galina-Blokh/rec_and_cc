"""macOS menu-bar UI (rumps).

All rumps/AppKit objects are touched only on the main thread: a ``rumps.Timer`` fires every
0.5 s, drains ``Controller.events`` and re-renders the title/menu from ``engine.state``.
Dialogs (record prompt, onboarding) run ``osascript`` on worker threads and route answers
back to the engine, whose API is thread-safe.

Global shortcut (default ⌃⌥R, ``cfg.hotkey``): owned by the Controller; its toasts arrive as
``notice`` events and the Start/Stop item shows the combination.

Live captions: the Controller posts ``live_start`` / ``live`` / ``live_end`` events; the
same timer feeds them to a floating ``CaptionsPanel`` (``captions_mac.py``) when
``cfg.live_captions`` is on.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading
from pathlib import Path

import rumps  # pyright: ignore[reportMissingImports] - macOS-only dependency

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import RECORDING
from ..models import Mode, TriggerKind
from ..retention import RETENTION_CHOICES
from .common import (
    ONBOARDING_TEXT,
    PERM_KINDS,
    PERM_LABELS,
    PERM_TITLES,
    PROMPT_BUTTONS,
    Controller,
    as_str,
    keep_prompt_text,
    open_output_folder,
    open_path,
    reveal_logs,
    transcript_label,
)

try:
    from .captions_mac import CaptionsPanel
except Exception:  # pragma: no cover - captions are optional; never break the menu bar app
    CaptionsPanel = None

log = logging.getLogger(__name__)

TICK_S = 0.5


class MacApp(rumps.App):
    def __init__(self, ctrl: Controller):
        # rumps infers quit_button: str from its default; None (documented) drops the Quit item
        super().__init__("RecAndCC", title="🟡", quit_button=None)  # pyright: ignore[reportArgumentType]
        self.ctrl = ctrl
        self.cfg = ctrl.cfg
        self._dialog: tuple[int | None, subprocess.Popen] | None = None  # (prompt_id, osascript)
        self._dialog_lock = threading.Lock()
        self._onboarded = False
        self._delegate = None
        self._recent_sig: tuple = ()
        # live captions (main thread only)
        self.captions = CaptionsPanel() if CaptionsPanel is not None else None
        self._live_id: str | None = None  # session currently being captioned
        self._live_dir: str | None = None  # its directory (kept after it ends)
        self._live_open_enabled: bool | None = None

        self.status_item = rumps.MenuItem("Starting…")
        self.mode_items = {
            m: rumps.MenuItem(label, callback=self._set_mode_cb(m))
            for m, label in ((Mode.OFF, "Off"), (Mode.ASK, "Ask"), (Mode.AUTO, "Auto"))
        }
        self.mode_menu = rumps.MenuItem("Mode")
        for item in self.mode_items.values():
            self.mode_menu.add(item)
        self.rec_item = rumps.MenuItem("Start recording", callback=self._toggle_rec)
        self.recent_menu = rumps.MenuItem("Recent transcripts")
        self.recent_menu.add(rumps.MenuItem("(none yet)"))
        self.perm_menu = rumps.MenuItem("Permissions")
        self.perm_status_items = {k: rumps.MenuItem(f"{PERM_TITLES[k]}: …") for k in PERM_KINDS}
        for item in self.perm_status_items.values():
            self.perm_menu.add(item)
        self.perm_menu.add(rumps.separator)
        self.perm_menu.add(
            rumps.MenuItem(
                "Grant microphone access…", callback=lambda _: ctrl.grant_async("microphone")
            )
        )
        self.perm_menu.add(
            rumps.MenuItem(
                "Grant system audio access…", callback=lambda _: ctrl.grant_async("system_audio")
            )
        )
        self.login_item = rumps.MenuItem("Launch at login", callback=self._toggle_login)
        self.retention_menu = rumps.MenuItem("Delete audio after")
        for days in RETENTION_CHOICES:
            label = "Never (keep audio)" if days == 0 else f"{days} day" + ("s" if days > 1 else "")
            item = rumps.MenuItem(label, callback=self._set_retention)
            # ad-hoc attribute on rumps.MenuItem, read back by the menu callback
            item.retention_days = days  # pyright: ignore[reportAttributeAccessIssue]
            self.retention_menu.add(item)
        self._render_retention()
        self.captions_item = rumps.MenuItem("Show live captions", callback=self._toggle_captions)
        self.captions_item.state = 1 if self.cfg.live_captions else 0
        self.live_open_item = rumps.MenuItem("Open live transcript")

        self.menu = [
            self.status_item,
            None,
            self.mode_menu,
            self.rec_item,
            self.captions_item,
            self.live_open_item,
            None,
            rumps.MenuItem("Open transcripts folder", callback=self._open_folder),
            self.recent_menu,
            None,
            self.perm_menu,
            self.login_item,
            self.retention_menu,
            rumps.MenuItem("Edit settings", callback=lambda _: open_path(CONFIG_PATH, True)),
            rumps.MenuItem("Reveal logs", callback=lambda _: reveal_logs()),
            None,
            rumps.MenuItem("Quit RecAndCC", callback=self._quit),
        ]
        self._toast = None
        self.timer = rumps.Timer(self._tick, TICK_S)
        self.timer.start()

    def _notify(self, title: str, message: str) -> None:
        """In-app toast (main thread). Not osascript: macOS attributes those notifications to
        Script Editor, which then pops up an empty Script Editor window."""
        try:
            if self._toast is None:
                from .toast_mac import ToastPanel

                self._toast = ToastPanel()
            self._toast.show(title, message)
        except Exception:
            log.exception("could not show notification: %s — %s", title, message)

    # ------------------------------------------------------------------ main thread
    def _tick(self, _timer) -> None:
        try:
            self._install_menu_delegate()
            self._drain_events()
            if self._toast is not None:
                self._toast.tick()
            self._render()
            self.ctrl.refresh_async()
        except Exception:
            log.exception("UI tick failed")

    def _drain_events(self) -> None:
        while True:
            try:
                ev = self.ctrl.events.get_nowait()
            except Exception:
                return
            kind = ev[0]
            if kind == "prompt":
                _, app, tkind, prompt_id = ev
                self._show_prompt(app, tkind, prompt_id)
            elif kind == "notice":
                self._notify(ev[1], ev[2])
            elif kind == "transcribed":
                _, session, result = ev
                if isinstance(result, Exception):
                    self._notify("Transcription failed", f"{session.directory.name}: {result}")
                else:
                    self._notify("Transcript ready", session.trigger_app or session.directory.name)
            elif kind == "live_start":
                self._live_started(ev[1], ev[2])
            elif kind == "live":
                if ev[1] != self._live_id:
                    self._live_started(ev[1], "")
                if self.captions is not None:
                    self.captions.append(ev[2])
            elif kind == "live_end":
                if ev[1] == self._live_id:
                    self._live_id = None
                    if self.captions is not None:
                        self.captions.end()
            elif kind == "refreshed":
                self._render_recent()
                self._render_permissions()
                self._maybe_onboard()

    def _render(self) -> None:
        st = self.ctrl.engine.state
        title = self.ctrl.title(st)
        if self.title != title:
            self.title = title
        self.status_item.title = self.ctrl.status_line(st)
        for m, item in self.mode_items.items():
            item.state = 1 if st.mode == m else 0
        rec_title = "Stop recording" if st.status == RECORDING else "Start recording"
        if shortcut := self.ctrl.hotkey_label:
            rec_title += f"  {shortcut}"
        if self.rec_item.title != rec_title:
            self.rec_item.title = rec_title
        self.login_item.state = 1 if autostart.is_enabled() else 0
        self._render_live(st)
        # close a stale prompt dialog (answered elsewhere, expired, mode changed, ...)
        with self._dialog_lock:
            if self._dialog and st.prompt_id != self._dialog[0]:
                with contextlib.suppress(Exception):
                    self._dialog[1].terminate()
                self._dialog = None

    def _live_started(self, session_id: str, title: str) -> None:
        self._live_id = session_id
        self._live_dir = self.ctrl.engine.state.session_dir
        if self.captions is not None:
            self.captions.show(title, display=self.cfg.live_captions)

    def _render_live(self, st) -> None:
        if self._live_id is not None and st.session_dir:
            self._live_dir = st.session_dir
        if self.captions is not None:
            live = self._live_id is not None and st.status == RECORDING
            self.captions.tick(st.elapsed if live else None)
        enabled = bool(self._live_dir)
        if enabled != self._live_open_enabled:
            self._live_open_enabled = enabled
            # rumps greys out an item that has no callback
            self.live_open_item.set_callback(self._open_live if enabled else None)

    def _render_recent(self) -> None:
        recent = self.ctrl.recent
        sig = tuple(recent)
        if sig == self._recent_sig:
            return
        self._recent_sig = sig
        self.recent_menu.clear()
        if not recent:
            self.recent_menu.add(rumps.MenuItem("(none yet)"))
            return
        for path in recent:
            item = rumps.MenuItem(transcript_label(path), callback=self._open_recent)
            item.path = path  # pyright: ignore[reportAttributeAccessIssue] - read by _open_recent
            self.recent_menu.add(item)

    def _render_permissions(self) -> None:
        perm = self.ctrl.permissions
        for k, item in self.perm_status_items.items():
            st = perm.get(k, "unknown")
            item.title = f"{PERM_TITLES[k]}: {PERM_LABELS.get(st, st)}"

    def _install_menu_delegate(self) -> None:
        """Refresh permissions when the menu opens (NSMenuDelegate.menuWillOpen:)."""
        if self._delegate is not None:
            return
        try:
            from Foundation import NSObject  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

            ctrl = self.ctrl

            class _Delegate(NSObject):
                def menuWillOpen_(self, _menu):
                    ctrl.refresh_async(force=True)

            d = _Delegate.alloc().init()
            self._menu._menu.setDelegate_(d)
            self._delegate = d
        except Exception:
            log.debug("could not install menu delegate", exc_info=True)
            self._delegate = False

    # ------------------------------------------------------------------ callbacks
    def _set_mode_cb(self, mode: Mode):
        def cb(_sender):
            self.ctrl.engine.set_mode(mode)
            self._render()

        return cb

    def _toggle_rec(self, _sender) -> None:
        self.ctrl.toggle_recording()

    def _toggle_login(self, sender) -> None:
        try:
            if autostart.is_enabled():
                autostart.disable()
                self.cfg.autostart = False
            else:
                autostart.enable()
                self.cfg.autostart = True
            threading.Thread(target=self.cfg.save, daemon=True).start()
        except Exception as e:
            log.exception("autostart toggle failed")
            self._notify("Launch at login", f"Failed: {e}")
        sender.state = 1 if autostart.is_enabled() else 0

    def _render_retention(self) -> None:
        days = self.cfg.transcription.audio_retention_days
        for item in self.retention_menu.values():
            item.state = 1 if getattr(item, "retention_days", None) == days else 0

    def _set_retention(self, sender) -> None:
        self.cfg.transcription.audio_retention_days = sender.retention_days
        self._render_retention()

        def apply():
            self.cfg.save()
            self.ctrl.purge_audio_now()

        threading.Thread(target=apply, daemon=True, name="retention-set").start()

    def _toggle_captions(self, sender) -> None:
        self.cfg.live_captions = not self.cfg.live_captions
        sender.state = 1 if self.cfg.live_captions else 0
        threading.Thread(target=self.cfg.save, daemon=True).start()
        if self.captions is None:
            return
        if self.cfg.live_captions:
            if self._live_id is not None:
                self.captions.reveal()
        else:
            self.captions.hide()

    def _open_live(self, _sender) -> None:
        if not self._live_dir:
            return
        d = Path(self._live_dir)
        md = d / "transcript.md"
        open_path(md if md.exists() else d)

    def _open_folder(self, _sender) -> None:
        if err := open_output_folder(self.cfg.output_dir):
            self._notify("Can't open transcripts folder", err)

    def _open_recent(self, sender) -> None:
        open_path(sender.path)

    def _quit(self, _sender) -> None:
        self.timer.stop()
        self.title = "…"
        try:
            self.ctrl.shutdown()
        finally:
            rumps.quit_application()

    # ------------------------------------------------------------------ dialogs (workers)
    def _show_prompt(self, app: str, kind: TriggerKind, prompt_id: int | None) -> None:
        what = "a call" if kind == TriggerKind.CALL else "audio"
        msg = keep_prompt_text(app, kind, self.cfg.detection.ask_timeout_action)
        self._notify("Recording — keep it?", f"{app}: {what} detected")
        timeout = max(5, int(self.cfg.detection.ask_timeout_s))
        buttons = ", ".join(f'"{label}"' for label, _ in PROMPT_BUTTONS)
        script = (
            f'display dialog {as_str(msg)} with title "RecAndCC" '
            f'buttons {{{buttons}}} default button "Keep" '
            f"with icon note giving up after {timeout}"
        )

        def run():
            proc = subprocess.Popen(
                ["osascript", "-e", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            with self._dialog_lock:
                self._dialog = (prompt_id, proc)
            out, _ = proc.communicate()
            with self._dialog_lock:
                if self._dialog and self._dialog[1] is proc:
                    self._dialog = None
            out = out or ""
            if "gave up:true" in out:
                return  # the engine expires the prompt itself
            for label, choice in PROMPT_BUTTONS:
                if f"button returned:{label}" in out:
                    self.ctrl.engine.respond(choice, prompt_id)
                    return
            # cancelled/killed: leave it to the engine's timeout

        threading.Thread(target=run, daemon=True, name="prompt-dialog").start()

    def _maybe_onboard(self) -> None:
        if self._onboarded:
            return
        self._onboarded = True  # once per session; "Later" isn't persisted
        missing = self.ctrl.missing_permissions()
        if not missing:
            return
        script = (
            f'display dialog {as_str(ONBOARDING_TEXT)} with title "Welcome to RecAndCC" '
            f'buttons {{"Later", "Grant now"}} default button "Grant now" '
            f"with icon note"
        )

        def run():
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if "button returned:Grant now" in r.stdout:
                self.ctrl.grant_all_missing()

        threading.Thread(target=run, daemon=True, name="onboarding").start()


def run_app(ctrl: Controller) -> None:
    import signal

    from AppKit import (  # pyright: ignore[reportMissingImports]
        NSApplication,  # pyright: ignore[reportAttributeAccessIssue]
        NSApplicationActivationPolicyAccessory,  # pyright: ignore[reportAttributeAccessIssue]
    )

    # The Python child of RecAndCC.app is not itself a bundle, so without this it would get a
    # Dock icon. Accessory = menu bar only (the equivalent of LSUIElement).
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = MacApp(ctrl)
    # Python handlers run between bytecodes; the 0.5 s rumps timer keeps that happening.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app._quit(None))
    app.run()
