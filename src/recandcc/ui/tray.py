"""Windows tray UI (pystray). UNTESTED (written without access to Windows).

pystray runs its own message loop in ``icon.run()``; a poll thread drains
``Controller.events`` every 0.5 s and updates the icon image/tooltip and menu
(``icon.update_menu()``), which pystray's win32 backend supports from other threads.
Record prompts: a tray notification plus a small always-on-top tkinter window with
Record / Always / Ignore buttons (own Tk root on its own thread, auto-closes after
``ask_timeout_s``), and the "Record now" menu item while a prompt is pending.
Live captions: ``TkCaptions``, a topmost tkinter window on its own Tk thread, fed through a
queue from the ``live_start`` / ``live`` / ``live_end`` events (UNTESTED as well).
Global shortcut (default Ctrl+Shift+R): ``hotkey.WinHotkeyListener``, owned by the Controller;
the Start/Stop item shows it after a tab (Win32 menus right-align text after ``\t``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
from pathlib import Path

import pystray  # pyright: ignore[reportMissingImports] - Windows-only dependency
from PIL import Image, ImageDraw  # pyright: ignore[reportMissingImports] - Windows-only dependency

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import PROMPTING, RECORDING
from ..models import Mode, TriggerKind
from .captions import CaptionBuffer, header_text, should_auto_hide
from .common import (
    ONBOARDING_TEXT,
    PERM_KINDS,
    PERM_LABELS,
    PERM_TITLES,
    PROMPT_BUTTONS,
    ROOM_ITEM,
    SENSITIVITY_ITEM,
    SENSITIVITY_PRESETS,
    Controller,
    keep_prompt_text,
    open_output_folder,
    open_path,
    reveal_logs,
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


class TkCaptions:
    """Topmost live-captions window. UNTESTED (written without access to Windows).

    All tkinter calls happen on one dedicated thread that owns its own ``Tk`` root; other
    threads only ``post()`` commands onto a queue that the Tk thread polls. Commands:
    ``("show", title, display)``, ``("append", segments)``, ``("end",)``, ``("hide",)``,
    ``("reveal",)``. Closing the window hides it until the next session.
    """

    POLL_MS = 150

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def post(self, *cmd) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="captions-tk")
            self._thread.start()
        self._q.put(cmd)

    def _run(self) -> None:
        try:
            import tkinter as tk
        except Exception:
            log.info("tkinter unavailable; no live captions window")
            return
        root = tk.Tk()
        root.withdraw()
        win = tk.Toplevel(root)
        win.withdraw()
        win.attributes("-topmost", True)
        with contextlib.suppress(Exception):
            win.attributes("-alpha", 0.92)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(760, int(sw * 0.8)), 190
        win.geometry(f"{w}x{h}+{(sw - w) // 2}+{sh - h - 120}")
        text = tk.Text(
            win,
            wrap="word",
            bg="#1c1c1e",
            fg="#f2f2f2",
            font=("Segoe UI", 13),
            padx=10,
            pady=8,
            borderwidth=0,
            highlightthickness=0,
        )
        text.tag_configure("Me", foreground="#8cdc9e", font=("Segoe UI", 10, "bold"))
        text.tag_configure("Others", foreground="#85bdff", font=("Segoe UI", 10, "bold"))
        text.tag_configure("other", foreground="#d9bf80", font=("Segoe UI", 10, "bold"))
        text.tag_configure("line", lmargin2=64, tabs=(64,))
        text.pack(fill="both", expand=True)
        text.configure(state="disabled")

        buf = CaptionBuffer()
        st = {"title": "", "started": None, "ended": None, "visible": False}

        def render(force_scroll: bool = False) -> None:
            follow = force_scroll or text.yview()[1] >= 0.999
            text.configure(state="normal")
            text.delete("1.0", "end")
            for i, ln in enumerate(buf.lines):
                tag = ln.label if ln.label in ("Me", "Others") else "other"
                text.insert("end", ln.label + "\t", (tag, "line"))
                text.insert("end", ln.text + ("\n" if i < len(buf) - 1 else ""), ("line",))
            text.configure(state="disabled")
            if follow:
                text.see("end")

        def set_visible(on: bool) -> None:
            st["visible"] = on
            if on:
                win.deiconify()
                win.attributes("-topmost", True)
            else:
                win.withdraw()

        win.protocol("WM_DELETE_WINDOW", lambda: set_visible(False))

        def poll() -> None:
            try:
                while True:
                    cmd = self._q.get_nowait()
                    op = cmd[0]
                    if op == "show":
                        buf.clear()
                        st.update(title=cmd[1] or "", started=time.monotonic(), ended=None)
                        render(True)
                        set_visible(bool(cmd[2]))
                    elif op == "append":
                        if buf.add(cmd[1]):
                            render()
                    elif op == "end":
                        if st["started"] is not None:
                            st["ended"] = time.monotonic()
                    elif op == "hide":
                        set_visible(False)
                    elif op == "reveal":
                        if st["started"] is not None:
                            render(True)
                            set_visible(True)
            except queue.Empty:
                pass
            except Exception:
                log.exception("captions window update failed")
            if st["started"] is not None:
                if should_auto_hide(st["ended"], time.monotonic()):
                    set_visible(False)
                    st.update(started=None, ended=None)
                else:
                    elapsed = time.monotonic() - st["started"]
                    win.title(header_text(st["title"], elapsed, finished=st["ended"] is not None))
            root.after(self.POLL_MS, poll)

        root.after(self.POLL_MS, poll)
        root.mainloop()


class TrayApp:
    def __init__(self, ctrl: Controller):
        self.ctrl = ctrl
        self.cfg = ctrl.cfg
        self._stop = threading.Event()
        self._last_look = None
        self._prompt_windows: dict[int | None, object] = {}
        self.captions = TkCaptions()
        self._live_id: str | None = None
        self._live_dir: str | None = None
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

        def sensitivity_item(name):
            return Item(
                name,
                lambda: self.ctrl.set_sensitivity(name),
                checked=lambda _i: self.ctrl.sensitivity_preset() == name,
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
                    lambda: os.startfile("ms-settings:privacy-microphone"),  # type: ignore[attr-defined]
                )
            )
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
                lambda _i: (
                    ("Stop recording" if st().status == RECORDING else "Start recording")
                    + (f"\t{self.ctrl.hotkey_label}" if self.ctrl.hotkey_label else "")
                ),
                lambda: self.ctrl.toggle_recording(),
            ),
            pystray.Menu.SEPARATOR,
            Item(
                "Show live captions",
                self._toggle_captions,
                checked=lambda _i: bool(self.cfg.live_captions),
            ),
            Item(
                ROOM_ITEM,
                lambda: self.ctrl.set_room_listening(not self.cfg.detection.room_listening),
                checked=lambda _i: bool(self.cfg.detection.room_listening),
            ),
            Item(
                SENSITIVITY_ITEM,
                pystray.Menu(*(sensitivity_item(n) for n in SENSITIVITY_PRESETS)),
            ),
            Item(
                "Open live transcript",
                self._open_live,
                enabled=lambda _i: bool(self._live_dir),
            ),
            pystray.Menu.SEPARATOR,
            Item("Open transcripts folder", self._open_folder),
            Item("Recent transcripts", pystray.Menu(recent)),
            Item("Permissions", pystray.Menu(perms)),
            Item("Launch at login", self._toggle_login, checked=lambda _i: autostart.is_enabled()),
            Item("Edit settings", lambda: open_path(CONFIG_PATH, True)),
            Item("Reveal logs", lambda: reveal_logs()),
            pystray.Menu.SEPARATOR,
            Item("Quit", self._quit),
        )

    def _open_folder(self) -> None:
        if err := open_output_folder(self.cfg.output_dir):
            self._notify("Can't open transcripts folder", err)

    def _toggle_captions(self) -> None:
        self.cfg.live_captions = not self.cfg.live_captions
        try:
            self.cfg.save()
        except Exception:
            log.exception("saving settings failed")
        if not self.cfg.live_captions:
            self.captions.post("hide")
        elif self._live_id is not None:
            self.captions.post("reveal")

    def _open_live(self) -> None:
        if self._live_dir:
            d = Path(self._live_dir)
            md = d / "transcript.md"
            open_path(md if md.exists() else d)

    def _live_started(self, session_id: str, title: str) -> None:
        self._live_id = session_id
        self._live_dir = self.ctrl.engine.state.session_dir or self._live_dir
        self.captions.post("show", title, bool(self.cfg.live_captions))

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
                self._notify("Recording — keep it?", f"{app}: {what} detected")
                threading.Thread(
                    target=self._prompt_window, args=(app, kind, pid), daemon=True
                ).start()
            elif ev[0] == "notice":
                self._notify(ev[1], ev[2])
            elif ev[0] == "live_start":
                self._live_started(ev[1], ev[2])
            elif ev[0] == "live":
                if ev[1] != self._live_id:
                    self._live_started(ev[1], "")
                self.captions.post("append", ev[2])
            elif ev[0] == "live_end":
                if ev[1] == self._live_id:
                    self._live_id = None
                    self.captions.post("end")
            elif ev[0] == "transcribed":
                _, s, res = ev
                self._notify(
                    "Transcription failed" if isinstance(res, Exception) else "Transcript ready",
                    s.trigger_app or s.directory.name,
                )

    def _render(self) -> None:
        st = self.ctrl.engine.state
        if self._live_id is not None and st.session_dir:
            self._live_dir = st.session_dir
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

    def _prompt_window(self, app: str, kind: TriggerKind, prompt_id) -> None:
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
            text=keep_prompt_text(app, kind, self.cfg.detection.ask_timeout_action),
            padx=20,
            pady=12,
            justify="left",
        ).pack()
        row = tk.Frame(root, pady=8)
        row.pack()

        def choose(c):
            self._answer(prompt_id, c)
            root.destroy()

        for label, c in PROMPT_BUTTONS:
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
