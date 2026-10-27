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

import rumps  # pyright: ignore[reportMissingImports] - macOS-only dependency

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import RECORDING
from ..models import Mode, TriggerKind
from ..retention import RETENTION_CHOICES
from . import notes as notes_ui
from .common import (
    PERM_KINDS,
    PERM_TITLES,
    ROOM_ITEM,
    SENSITIVITY_ITEM,
    SENSITIVITY_PRESETS,
    Controller,
    as_str,
    keep_prompt_text,
    open_output_folder,
    open_path,
    osascript,
    reveal_logs,
    search_hit_label,
    transcript_label,
)
from .present import (
    MODE_LABELS,
    check,
    dialog_timeout,
    language_menu,
    live_transcript_target,
    onboarding_accepted,
    onboarding_script,
    permission_title,
    prompt_dialog_choice,
    prompt_dialog_script,
    prompt_notice,
    record_item_title,
    retention_label,
    search_notice,
    search_open_target,
    transcribed_notice,
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
        self._notes_lock = threading.Lock()  # held while the meeting notes dialogs are up
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
            m: rumps.MenuItem(label, callback=self._set_mode_cb(m)) for m, label in MODE_LABELS
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
            item = rumps.MenuItem(retention_label(days), callback=self._set_retention)
            # ad-hoc attribute on rumps.MenuItem, read back by the menu callback
            item.retention_days = days  # pyright: ignore[reportAttributeAccessIssue]
            self.retention_menu.add(item)
        self._render_retention()
        self.lang_menu = rumps.MenuItem("Languages")
        self._lang_sig: list | None = None
        self._render_languages()
        self.captions_item = rumps.MenuItem("Show live captions", callback=self._toggle_captions)
        self.captions_item.state = check(self.cfg.live_captions)
        self.room_item = rumps.MenuItem(ROOM_ITEM, callback=self._toggle_room)
        self.room_item.state = check(self.cfg.detection.room_listening)
        self.sensitivity_menu = rumps.MenuItem(SENSITIVITY_ITEM)
        self._sensitivity_shown: str | None = None
        for name in SENSITIVITY_PRESETS:
            self.sensitivity_menu.add(rumps.MenuItem(name, callback=self._set_sensitivity))
        self._render_sensitivity()
        self.live_open_item = rumps.MenuItem("Open live transcript")

        self.menu = [
            self.status_item,
            None,
            self.mode_menu,
            self.rec_item,
            self.captions_item,
            self.room_item,
            self.sensitivity_menu,
            self.live_open_item,
            None,
            rumps.MenuItem("Open transcripts folder", callback=self._open_folder),
            self.recent_menu,
            rumps.MenuItem("Search transcripts…", callback=self._search),
            rumps.MenuItem("Meeting notes…", callback=lambda _: ctrl.notes_async()),
            None,
            self.perm_menu,
            self.login_item,
            self.lang_menu,
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
                self._notify(*transcribed_notice(ev[1], ev[2]))
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
            elif kind == "search":
                self._show_hits(ev[1], ev[2])
            elif kind == "notes":
                self._show_notes(ev[1])
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
            item.state = check(st.mode == m)
        rec_title = record_item_title(st.status == RECORDING, self.ctrl.hotkey_label)
        if self.rec_item.title != rec_title:
            self.rec_item.title = rec_title
        self.login_item.state = check(autostart.is_enabled())
        self.room_item.state = check(self.cfg.detection.room_listening)  # hand edits too
        if self.ctrl.sensitivity_preset() != self._sensitivity_shown:
            self._render_sensitivity()  # hand edits and config reloads too
        self._render_languages()
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
            item.title = permission_title(k, perm.get(k, "unknown"))

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
            item.state = check(getattr(item, "retention_days", None) == days)

    def _render_languages(self) -> None:
        """(Re)build the Languages submenu when the setting changed (menu or config.toml)."""
        rows = language_menu(self.cfg.transcription.languages)
        if rows == self._lang_sig:
            return
        if self._lang_sig is not None:
            self.lang_menu.clear()
        self._lang_sig = rows
        for code, title, on in rows:
            # rumps greys out an item that has no callback: the "Also from settings" row
            item = rumps.MenuItem(title, callback=None if code is None else self._set_language)
            item.language_code = code  # pyright: ignore[reportAttributeAccessIssue]
            item.state = check(on)
            self.lang_menu.add(item)

    def _set_language(self, sender) -> None:
        self.ctrl.toggle_language(sender.language_code)
        self._render_languages()

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

    def _toggle_room(self, sender) -> None:
        self.ctrl.set_room_listening(not self.cfg.detection.room_listening)
        sender.state = check(self.cfg.detection.room_listening)

    def _render_sensitivity(self) -> None:
        preset = self._sensitivity_shown = self.ctrl.sensitivity_preset()
        for name, item in self.sensitivity_menu.items():
            item.state = check(name == preset)

    def _set_sensitivity(self, sender) -> None:
        self.ctrl.set_sensitivity(sender.title)
        self._render_sensitivity()

    def _open_live(self, _sender) -> None:
        if not self._live_dir:
            return
        open_path(live_transcript_target(self._live_dir))

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
        msg = keep_prompt_text(app, kind, self.cfg.detection.ask_timeout_action)
        self._notify(*prompt_notice(app, kind))
        timeout = dialog_timeout(self.cfg.detection.ask_timeout_s)
        script = prompt_dialog_script(msg, timeout)

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
            # gave up / cancelled / killed: the engine expires the prompt itself
            if choice := prompt_dialog_choice(out):
                self.ctrl.engine.respond(choice, prompt_id)

        threading.Thread(target=run, daemon=True, name="prompt-dialog").start()

    def _search(self, _sender) -> None:
        """Ask for a query (osascript, worker thread); the search itself runs on another
        worker and its hits come back as a ``search`` event."""
        script = (
            'display dialog "Search transcripts for:" default answer "" with title "RecAndCC" '
            'buttons {"Cancel", "Search"} default button "Search"'
        )

        def run():
            query = osascript(script).partition("text returned:")[2].strip()
            if query:
                self.ctrl.search_async(query)

        threading.Thread(target=run, daemon=True, name="search-dialog").start()

    def _show_hits(self, query: str, hits) -> None:
        if notice := search_notice(hits):
            self._notify(*notice)
            return
        labels = [f"{i}. {search_hit_label(h)}"[:200] for i, h in enumerate(hits, 1)]
        script = (
            f'choose from list {{{", ".join(map(as_str, labels))}}} with title "RecAndCC" '
            f'with prompt {as_str(f"Matches for “{query}”:")} OK button name "Open"'
        )

        def run():
            number = osascript(script).partition(".")[0]
            if number.isdigit() and 0 < int(number) <= len(hits):
                open_path(search_open_target(hits[int(number) - 1]))

        threading.Thread(target=run, daemon=True, name="search-results").start()

    def _show_notes(self, notes: list) -> None:
        """Meeting notes as AppleScript dialogs on a worker thread (pick a meeting, then an
        action); rename and delete run on the Controller, whose fresh list reopens the
        dialog. One dialog at a time: a list arriving while one is open is dropped."""
        if not notes:
            self._notify("Meeting notes", "No meetings yet")
            return
        if not self._notes_lock.acquire(blocking=False):
            return

        def run():
            try:
                notes_ui.run_dialogs(notes, osascript, self.ctrl.note_action_async, open_path)
            finally:
                self._notes_lock.release()

        threading.Thread(target=run, daemon=True, name="notes-dialog").start()

    def _maybe_onboard(self) -> None:
        if self._onboarded:
            return
        self._onboarded = True  # once per session; "Later" isn't persisted
        missing = self.ctrl.missing_permissions()
        if not missing:
            return
        script = onboarding_script()

        def run():
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if onboarding_accepted(r.stdout):
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
