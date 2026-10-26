"""macOS menu-bar UI (rumps).

All rumps/AppKit objects are touched only on the main thread: a ``rumps.Timer`` fires every
0.5 s, drains ``Controller.events`` and re-renders the title/menu from ``engine.state``.
Dialogs (record prompt, onboarding) run ``osascript`` on worker threads and route answers
back to the engine, whose API is thread-safe.
"""

from __future__ import annotations

import contextlib
import logging
import subprocess
import threading

import rumps

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import RECORDING
from ..models import Mode, TriggerKind
from .common import (
    ONBOARDING_TEXT,
    PERM_KINDS,
    PERM_LABELS,
    PERM_TITLES,
    Controller,
    as_str,
    mac_notify,
    open_path,
    transcript_label,
)

log = logging.getLogger(__name__)

TICK_S = 0.5


class MacApp(rumps.App):
    def __init__(self, ctrl: Controller):
        super().__init__("RecAndCC", title="🟡", quit_button=None)
        self.ctrl = ctrl
        self.cfg = ctrl.cfg
        self._dialog: tuple[int, subprocess.Popen] | None = None  # (prompt_id, osascript)
        self._dialog_lock = threading.Lock()
        self._onboarded = False
        self._delegate = None
        self._recent_sig: tuple = ()

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

        self.menu = [
            self.status_item,
            None,
            self.mode_menu,
            self.rec_item,
            None,
            rumps.MenuItem("Open transcripts folder", callback=self._open_folder),
            self.recent_menu,
            None,
            self.perm_menu,
            self.login_item,
            rumps.MenuItem("Edit settings", callback=lambda _: open_path(CONFIG_PATH, True)),
            None,
            rumps.MenuItem("Quit RecAndCC", callback=self._quit),
        ]
        self.timer = rumps.Timer(self._tick, TICK_S)
        self.timer.start()

    # ------------------------------------------------------------------ main thread
    def _tick(self, _timer) -> None:
        try:
            self._install_menu_delegate()
            self._drain_events()
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
                mac_notify(ev[1], ev[2])
            elif kind == "transcribed":
                _, session, result = ev
                if isinstance(result, Exception):
                    mac_notify("Transcription failed", f"{session.directory.name}: {result}")
                else:
                    mac_notify("Transcript ready", session.trigger_app or session.directory.name)
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
        self.rec_item.title = "Stop recording" if st.status == RECORDING else "Start recording"
        self.login_item.state = 1 if autostart.is_enabled() else 0
        # close a stale prompt dialog (answered elsewhere, expired, mode changed, ...)
        with self._dialog_lock:
            if self._dialog and st.prompt_id != self._dialog[0]:
                with contextlib.suppress(Exception):
                    self._dialog[1].terminate()
                self._dialog = None

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
            item.path = path
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
            from Foundation import NSObject

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
            mac_notify("Launch at login", f"Failed: {e}")
        sender.state = 1 if autostart.is_enabled() else 0

    def _open_folder(self, _sender) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        open_path(self.cfg.output_dir)

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
        msg = f"RecAndCC detected {what} in {app}.\n\nRecord and transcribe it?"
        mac_notify("Record?", f"{app} is playing {what}")
        timeout = max(5, int(self.cfg.detection.ask_timeout_s))
        script = (
            f'display dialog {as_str(msg)} with title "RecAndCC" '
            f'buttons {{"Ignore", "Always", "Record"}} default button "Record" '
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
            for label, choice in (("Record", "record"), ("Always", "always"), ("Ignore", "ignore")):
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

    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

    # The Python child of RecAndCC.app is not itself a bundle, so without this it would get a
    # Dock icon. Accessory = menu bar only (the equivalent of LSUIElement).
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app = MacApp(ctrl)
    # Python handlers run between bytecodes; the 0.5 s rumps timer keeps that happening.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app._quit(None))
    app.run()
