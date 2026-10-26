"""Windows tray UI (pystray). UNTESTED (written without access to Windows).

pystray runs its own message loop in ``icon.run()``; a poll thread drains
``Controller.events`` every 0.5 s and updates the icon image/tooltip and menu
(``icon.update_menu()``), which pystray's win32 backend supports from other threads.
Record prompts: a tray notification plus a small always-on-top tkinter window with
Record / Always / Ignore buttons (own Tk root on its own thread, auto-closes after
``ask_timeout_s``), and the "Record now" menu item while a prompt is pending.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import pystray
from PIL import Image, ImageDraw

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import PROMPTING, RECORDING
from ..models import Mode, TriggerKind
from .common import (
    ONBOARDING_TEXT,
    PERM_KINDS,
    PERM_LABELS,
    PERM_TITLES,
    Controller,
    open_path,
    transcript_label,
)

log = logging.getLogger(__name__)

COLORS = {
    "off": (150, 150, 150),
    "ask": (240, 190, 0),
    "auto": (60, 160, 255),
    "prompt": (255, 140, 0),
    "rec": (220, 30, 30),
}


def make_icon(color: tuple[int, int, int], pending: bool, warn: bool) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((6, 6, 58, 58), fill=color + (255,), outline=(40, 40, 40, 255), width=3)
    if pending:
        d.ellipse((40, 40, 62, 62), fill=(255, 255, 255, 255), outline=(40, 40, 40, 255), width=3)
    if warn:
        d.polygon([(2, 26), (14, 2), (26, 26)], fill=(255, 220, 0, 255), outline=(0, 0, 0, 255))
    return img


class TrayApp:
    def __init__(self, ctrl: Controller):
        self.ctrl = ctrl
        self.cfg = ctrl.cfg
        self._stop = threading.Event()
        self._last_look = None
        self._prompt_windows: dict[int | None, object] = {}
        self.icon = pystray.Icon(
            "RecAndCC", make_icon(COLORS["ask"], False, False), "RecAndCC", menu=self._menu()
        )

    # ------------------------------------------------------------------ menu
    def _menu(self) -> pystray.Menu:
        st = lambda: self.ctrl.engine.state  # noqa: E731
        Item = pystray.MenuItem

        def mode_item(label, mode):
            return Item(
                label,
                lambda: self.ctrl.engine.set_mode(mode),
                checked=lambda _i: st().mode == mode,
                radio=True,
            )

        def recent():
            items = [
                Item(transcript_label(p), (lambda p=p: lambda: open_path(p))())  # noqa: B023 — p is bound by the default arg
                for p in self.ctrl.recent
            ]
            return items or [Item("(none yet)", None, enabled=False)]

        def perms():
            perm = self.ctrl.permissions
            items = [
                Item(
                    f"{PERM_TITLES[k]}: {PERM_LABELS.get(perm.get(k, 'unknown'))}",
                    None,
                    enabled=False,
                )
                for k in PERM_KINDS
                if k in perm
            ]
            items.append(
                Item(
                    "Microphone privacy settings…",
                    lambda: os.startfile("ms-settings:privacy-microphone"),
                )
            )  # type: ignore[attr-defined]
            return items

        return pystray.Menu(
            Item(lambda _i: self.ctrl.status_line(), None, enabled=False),
            pystray.Menu.SEPARATOR,
            Item(
                "Record now",
                lambda: self._answer(st().prompt_id, "record"),
                visible=lambda _i: st().status == PROMPTING,
            ),
            Item(
                "Mode",
                pystray.Menu(
                    mode_item("Off", Mode.OFF),
                    mode_item("Ask", Mode.ASK),
                    mode_item("Auto", Mode.AUTO),
                ),
            ),
            Item(
                lambda _i: "Stop recording" if st().status == RECORDING else "Start recording",
                lambda: self.ctrl.toggle_recording(),
            ),
            pystray.Menu.SEPARATOR,
            Item("Open transcripts folder", self._open_folder),
            Item("Recent transcripts", pystray.Menu(recent)),
            Item("Permissions", pystray.Menu(perms)),
            Item("Launch at login", self._toggle_login, checked=lambda _i: autostart.is_enabled()),
            Item("Edit settings", lambda: open_path(CONFIG_PATH, True)),
            pystray.Menu.SEPARATOR,
            Item("Quit", self._quit),
        )

    def _open_folder(self) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        open_path(self.cfg.output_dir)

    def _toggle_login(self) -> None:
        try:
            if autostart.is_enabled():
                autostart.disable()
                self.cfg.autostart = False
            else:
                autostart.enable()
                self.cfg.autostart = True
            self.cfg.save()
        except Exception:
            log.exception("autostart toggle failed")

    def _quit(self) -> None:
        self._stop.set()
        try:
            self.ctrl.shutdown()
        finally:
            self.icon.stop()

    # ------------------------------------------------------------------ polling
    def _poll(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._drain()
                self._render()
                self.ctrl.refresh_async()
            except Exception:
                log.exception("tray poll failed")

    def _notify(self, title: str, msg: str) -> None:
        try:
            self.icon.notify(msg, title)
        except Exception:
            log.debug("notify failed", exc_info=True)

    def _drain(self) -> None:
        while True:
            try:
                ev = self.ctrl.events.get_nowait()
            except Exception:
                return
            if ev[0] == "prompt":
                _, app, kind, pid = ev
                what = "a call" if kind == TriggerKind.CALL else "audio"
                self._notify("Record?", f"{app} is playing {what}. Use the tray menu or dialog.")
                threading.Thread(
                    target=self._prompt_window, args=(app, what, pid), daemon=True
                ).start()
            elif ev[0] == "notice":
                self._notify(ev[1], ev[2])
            elif ev[0] == "transcribed":
                _, s, res = ev
                self._notify(
                    "Transcription failed" if isinstance(res, Exception) else "Transcript ready",
                    s.trigger_app or s.directory.name,
                )

    def _render(self) -> None:
        st = self.ctrl.engine.state
        if st.status == RECORDING:
            color = COLORS["rec"]
        elif st.status == PROMPTING:
            color = COLORS["prompt"]
        else:
            color = COLORS[str(st.mode)] if str(st.mode) in COLORS else COLORS["ask"]
        try:
            pending = self.ctrl.tq.pending() > 0
        except Exception:
            pending = False
        warn = bool(self.ctrl.missing_permissions())
        look = (color, pending, warn)
        if look != self._last_look:
            self._last_look = look
            self.icon.icon = make_icon(color, pending, warn)
        self.icon.title = f"RecAndCC — {self.ctrl.title(st)}"[:127]
        self.icon.update_menu()

    # ------------------------------------------------------------------ prompts
    def _answer(self, prompt_id, choice: str) -> None:
        self.ctrl.engine.respond(choice, prompt_id)

    def _prompt_window(self, app: str, what: str, prompt_id) -> None:
        try:
            import tkinter as tk
        except Exception:
            return  # tray menu "Record now" still works
        root = tk.Tk()
        root.title("RecAndCC")
        root.attributes("-topmost", True)
        root.resizable(False, False)
        tk.Label(
            root,
            text=f"RecAndCC detected {what} in {app}.\nRecord and transcribe it?",
            padx=20,
            pady=12,
            justify="left",
        ).pack()
        row = tk.Frame(root, pady=8)
        row.pack()

        def choose(c):
            self._answer(prompt_id, c)
            root.destroy()

        for label, c in (("Record", "record"), ("Always", "always"), ("Ignore", "ignore")):
            tk.Button(row, text=label, width=10, command=lambda c=c: choose(c)).pack(
                side="left", padx=4
            )
        deadline = time.monotonic() + max(5, self.cfg.detection.ask_timeout_s)

        def check():
            st = self.ctrl.engine.state
            if time.monotonic() > deadline or st.prompt_id != prompt_id:
                root.destroy()
            else:
                root.after(500, check)

        root.after(500, check)
        root.mainloop()

    def _onboard(self) -> None:
        # Windows has no TCC-style prompts; just point at the privacy settings if needed.
        if not self.ctrl.missing_permissions():
            return
        try:
            from tkinter import messagebox

            if messagebox.askyesno(
                "Welcome to RecAndCC", ONBOARDING_TEXT + "\n\nOpen microphone privacy settings now?"
            ):
                os.startfile("ms-settings:privacy-microphone")  # type: ignore[attr-defined]
        except Exception:
            log.debug("onboarding dialog failed", exc_info=True)

    def run(self) -> None:
        threading.Thread(target=self._poll, daemon=True, name="tray-poll").start()
        threading.Thread(target=self._onboard, daemon=True, name="onboarding").start()
        self.icon.run()


def run_app(ctrl: Controller) -> None:
    TrayApp(ctrl).run()
