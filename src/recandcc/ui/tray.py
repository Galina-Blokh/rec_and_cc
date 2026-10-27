"""Windows tray UI (pystray). UNTESTED (written without access to Windows).

pystray runs its own message loop in ``icon.run()``; a poll thread drains
``Controller.events`` every 0.5 s and updates the icon image/tooltip and menu
(``icon.update_menu()``), which pystray's win32 backend supports from other threads.
Record prompts: a tray notification plus a small always-on-top tkinter window with
Record / Always / Ignore buttons (own Tk root on its own thread, auto-closes after
``ask_timeout_s``), and the "Record now" menu item while a prompt is pending.
Live captions: ``TkCaptions``, a topmost tkinter window on its own Tk thread, fed through a
queue from the ``live_start`` / ``live`` / ``live_end`` events (UNTESTED as well).
Meeting notes: ``TkNotes``, the same pattern (list, open/rename/delete via the Controller).
Transcript search: ``TkSearch``, a small window on its own Tk thread (query entry, results
list); the search runs on a Controller worker and its hits arrive as a ``search`` event,
which the poll thread forwards to the window's queue (UNTESTED on Windows as well).
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
from collections.abc import Callable

import pystray  # pyright: ignore[reportMissingImports] - Windows-only dependency
from PIL import Image, ImageDraw  # pyright: ignore[reportMissingImports] - Windows-only dependency

from .. import autostart
from ..config import CONFIG_PATH
from ..engine import PROMPTING, RECORDING
from ..models import Mode, TriggerKind
from . import notes as notes_ui
from .captions import (
    MAX_LINES,
    CaptionBuffer,
    geometry_width,
    header_text,
    should_auto_hide,
    text_area_width,
    tk_parts,
)
from .common import (
    ONBOARDING_TEXT,
    PERM_KINDS,
    PERM_LABELS,
    PERM_TITLES,
    PROMPT_BUTTONS,
    ROOM_ITEM,
    SENSITIVITY_HINT,
    SENSITIVITY_ITEM,
    SENSITIVITY_PRESETS,
    Controller,
    keep_prompt_text,
    open_output_folder,
    open_path,
    reveal_logs,
    transcript_label,
)
from .present import (
    SEARCHING,
    language_menu,
    live_transcript_target,
    prompt_notice,
    record_item_title,
    search_hit_row,
    search_open_target,
    search_status,
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
            from tkinter import font as tkfont
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
        # Right-to-left (Hebrew) lines: right-justified; tk_parts() reorders the text to visual
        # order and puts the label last, because Tk does not apply the bidi algorithm itself.
        text.tag_configure("rtl", justify="right", rmargin=10)
        text.pack(fill="both", expand=True)
        text.configure(state="disabled")

        buf = CaptionBuffer()
        st = {"title": "", "started": None, "ended": None, "visible": False}

        fonts: dict = {}  # one Font per kind, reused for every measure
        wrapped: dict = {}  # (line, width) -> pieces, so an append doesn't re-measure all lines

        def wrap_limit() -> tuple[float | None, Callable[[str], float], Callable[[str], float]]:
            # Usable text width and font measures; RTL lines are wrapped before reordering.
            # A withdrawn window has no real width yet: use the one it will get from geometry.
            try:
                rmargin = float(text.tag_cget("rtl", "rmargin") or 0)
            except ValueError:  # a unit string such as '10c'
                rmargin = 0.0
            widget_w = text.winfo_width()
            if widget_w < 100:
                widget_w = geometry_width(win.geometry())
            width = text_area_width(
                widget_w,
                float(text.cget("padx")),
                float(text.cget("borderwidth")),
                float(text.cget("highlightthickness")),
                rmargin,
            )
            if width is None:  # no usable size at all: fall back to a character count
                return None, len, len
            if not fonts:
                fonts["text"] = tkfont.Font(font=text.cget("font"))
                fonts["label"] = tkfont.Font(font=("Segoe UI", 10, "bold"))
            return width, fonts["text"].measure, fonts["label"].measure

        def render(force_scroll: bool = False) -> None:
            st["width"] = text.winfo_width()
            width, measure, label_measure = wrap_limit()  # once per render, not per line
            if len(wrapped) > 2 * MAX_LINES:
                wrapped.clear()
            follow = force_scroll or text.yview()[1] >= 0.999
            text.configure(state="normal")
            text.delete("1.0", "end")
            for i, ln in enumerate(buf.lines):
                tag = ln.label if ln.label in ("Me", "Others") else "other"
                para = ("rtl",) if ln.rtl else ("line",)
                key = (ln, width)
                if key not in wrapped:
                    wrapped[key] = tk_parts(ln, width, measure, label_measure)
                for piece, is_label in wrapped[key]:
                    text.insert("end", piece, (tag, *para) if is_label else para)
                if i < len(buf) - 1:
                    text.insert("end", "\n", para)
            text.configure(state="disabled")
            if follow:
                text.see("end")

        st["width"] = 0
        st["resize"] = None

        def on_resize(_event=None) -> None:
            # Re-wrap RTL lines when the width changes (debounced); this also covers the first
            # real width after the window was unmapped (reveal).
            if st["resize"] is not None:
                text.after_cancel(st["resize"])

            def apply() -> None:
                st["resize"] = None
                w = text.winfo_width()
                if w != st["width"]:
                    render()

            st["resize"] = text.after_idle(apply)

        text.bind("<Configure>", on_resize)

        def set_visible(on: bool) -> None:
            st["visible"] = on
            if on:
                win.deiconify()
                win.attributes("-topmost", True)
                win.update_idletasks()
                render()
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


class TkSearch:
    """ "Search transcripts…" window. UNTESTED (written without access to Windows).

    Like ``TkCaptions``: one thread owns its own ``Tk`` root and all tkinter calls; others
    only ``post()`` ``("ask",)`` (show the query entry) or ``("results", query, hits)``.
    ``submit`` (a query, from the Tk thread) hands it to ``search``, which must not block;
    results for a query other than the one in the entry are dropped. Double-clicking a match
    opens it on a worker thread (the file check may be slow). If Tk fails, ``post`` stops.
    """

    POLL_MS = 150
    COLUMNS = (  # id, heading, width; the order of present.search_hit_row
        ("session", "Session", 150),
        ("at", "Time", 50),
        ("who", "Speaker", 80),
        ("text", "Snippet", 420),
    )

    def __init__(self, search):
        self._search = search
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.hits: list = []
        self.failed = False
        self._status = None  # set by the Tk thread: shows a status line

    def post(self, *cmd) -> None:
        if self.failed:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="search-tk")
            self._thread.start()
        self._q.put(cmd)

    def submit(self, query: str) -> None:
        if query := query.strip():
            if self._status is not None:
                self._status(SEARCHING)
            self._search(query)

    def open(self, index: int) -> None:
        if 0 <= index < len(self.hits):
            hit = self.hits[index]
            threading.Thread(
                target=lambda: open_path(search_open_target(hit)), daemon=True, name="search-open"
            ).start()

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.withdraw()
            win = tk.Toplevel(root)
            win.withdraw()
            win.title("Search transcripts")
            win.geometry("720x360")
            win.protocol("WM_DELETE_WINDOW", win.withdraw)
            query = tk.StringVar()
            top = tk.Frame(win, padx=8, pady=8)
            top.pack(fill="x")
            entry = tk.Entry(top, textvariable=query)
            entry.pack(side="left", fill="x", expand=True)
            tk.Button(top, text="Search", command=lambda: self.submit(query.get())).pack(
                side="left"
            )
            entry.bind("<Return>", lambda _e: self.submit(query.get()))
            status = tk.Label(win, anchor="w", padx=8)
            status.pack(fill="x")
            tree = ttk.Treeview(win, columns=[c[0] for c in self.COLUMNS], show="headings")
            tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            for col, head, width in self.COLUMNS:
                tree.heading(col, text=head)
                tree.column(col, width=width, stretch=col == "text")
            tree.bind(
                "<Double-1>", lambda _e: self.open(tree.index(f) if (f := tree.focus()) else -1)
            )
            self._status = lambda text: status.configure(text=text)
        except Exception as e:
            self.failed = True
            log.warning("no search window (tkinter unavailable): %s", type(e).__name__)
            return

        def poll() -> None:
            try:
                while True:
                    cmd = self._q.get_nowait()
                    if cmd[0] == "ask":
                        win.deiconify()
                        win.lift()
                        entry.focus_set()
                    elif cmd[0] == "results" and cmd[1] == query.get().strip():
                        self.hits = [] if isinstance(cmd[2], Exception) else list(cmd[2])
                        tree.delete(*tree.get_children())
                        for hit in self.hits:
                            tree.insert("", "end", values=search_hit_row(hit))
                        status.configure(text=search_status(cmd[1], cmd[2]))
            except queue.Empty:
                pass
            except Exception as e:  # its text may quote a transcript: log the type only
                log.error("search window update failed: %s", type(e).__name__)
            root.after(self.POLL_MS, poll)

        root.after(self.POLL_MS, poll)
        root.mainloop()


class TkNotes:
    """ "Meeting notes…" window. UNTESTED (written without access to Windows).

    Like ``TkSearch``: one thread owns its own ``Tk`` root and all tkinter calls; others only
    ``post()`` ``("notes", notes)`` (show the window with a fresh list). Scans, opening files,
    rename and delete go to ``ctrl`` (worker threads), which posts the next list as an event.
    """

    POLL_MS = 150
    COLUMNS = (  # id, heading, width; the order of notes.note_columns
        ("title", "Title", 200),
        ("when", "Date", 120),
        ("dur", "Duration", 70),
        ("who", "Speakers", 150),
        ("lang", "Languages", 70),
        ("status", "Status", 80),
    )

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.notes: list = []
        self.failed = False

    def post(self, *cmd) -> None:
        if self.failed:
            return
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="notes-tk")
            self._thread.start()
        self._q.put(cmd)

    def act(self, index: int, action: str, ask=None) -> None:
        """Run ``action`` (one of ``notes.ACTIONS``) on row ``index``; ``ask(kind, note)``
        (Tk thread) returns the new title / True to delete, or None to cancel."""
        if not 0 <= index < len(self.notes):
            return
        note = self.notes[index]
        if action == "Rename…":
            if (title := ask and ask("rename", note)) is not None:
                self.ctrl.note_action_async(note.directory, "rename", title)
        elif action == "Delete…":
            if ask and ask("delete", note):
                self.ctrl.note_action_async(note.directory, "delete")
        elif action in notes_ui.note_actions(note):
            target = notes_ui.action_target(note, action)
            threading.Thread(target=open_path, args=(target,), daemon=True).start()

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox, simpledialog, ttk

            root = tk.Tk()
            root.withdraw()
            win = tk.Toplevel(root)
            win.withdraw()
            win.title("Meeting notes")
            win.geometry("760x400")
            win.protocol("WM_DELETE_WINDOW", win.withdraw)
            tree = ttk.Treeview(win, columns=[c[0] for c in self.COLUMNS], show="headings")
            tree.pack(fill="both", expand=True, padx=8, pady=8)
            for col, head, width in self.COLUMNS:
                tree.heading(col, text=head)
                tree.column(col, width=width, stretch=col == "title")
            excerpt = tk.Label(win, anchor="w", justify="left", padx=8, wraplength=740)
            excerpt.pack(fill="x")
            bar = tk.Frame(win, padx=8, pady=8)
            bar.pack(fill="x")

            def ask(kind, note):
                if kind == "rename":
                    return simpledialog.askstring(
                        "Rename meeting", "New title (empty: the default):",
                        initialvalue=note.title, parent=win,
                    )  # fmt: skip
                return messagebox.askyesno(
                    "Delete meeting", notes_ui.delete_prompt(note), parent=win
                )

            def selected() -> int:
                return tree.index(f) if (f := tree.focus()) else -1

            for action in notes_ui.ACTIONS:
                tk.Button(
                    bar, text=action, command=lambda a=action: self.act(selected(), a, ask)
                ).pack(side="left", padx=2)
            tk.Button(bar, text="Refresh", command=self.ctrl.notes_async).pack(side="right")
            tree.bind(  # the row under the pointer, not a stale focus (header, empty space)
                "<Double-1>",
                lambda e: (
                    (row := tree.identify_row(e.y)) and self.act(tree.index(row), "Open transcript")
                ),
            )
            tree.bind(
                "<<TreeviewSelect>>",
                lambda _e: excerpt.configure(
                    text=self.notes[i].excerpt if 0 <= (i := selected()) < len(self.notes) else ""
                ),
            )
        except Exception as e:
            self.failed = True
            log.warning("no meeting notes window (tkinter unavailable): %s", type(e).__name__)
            return

        def poll() -> None:
            try:
                while True:
                    cmd = self._q.get_nowait()
                    if cmd[0] == "notes":
                        self.notes = list(cmd[1])
                        tree.delete(*tree.get_children())
                        for note in self.notes:
                            tree.insert("", "end", values=notes_ui.note_columns(note))
                        excerpt.configure(text="" if self.notes else "No meetings yet")
                        win.deiconify()
                        win.lift()
            except queue.Empty:
                pass
            except Exception as e:  # titles and excerpts are meeting text: log the type only
                log.error("meeting notes update failed: %s", type(e).__name__)
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
        self.search = TkSearch(ctrl.search_async)
        self.notes = TkNotes(ctrl)
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

        def language_item(code: str | None, title: str, on: bool):
            action = None if code is None else lambda: self.ctrl.toggle_language(code)
            return Item(title, action, checked=lambda _i: on, enabled=code is not None)

        def languages():
            rows = language_menu(self.cfg.transcription.languages)
            return [language_item(*row) for row in rows]

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
                lambda _i: record_item_title(
                    st().status == RECORDING, self.ctrl.hotkey_label, sep="\t"
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
                pystray.Menu(
                    *(sensitivity_item(n) for n in SENSITIVITY_PRESETS),
                    pystray.Menu.SEPARATOR,
                    Item(SENSITIVITY_HINT, None, enabled=False),
                ),
            ),
            Item(
                "Open live transcript",
                self._open_live,
                enabled=lambda _i: bool(self._live_dir),
            ),
            pystray.Menu.SEPARATOR,
            Item("Open transcripts folder", self._open_folder),
            Item("Recent transcripts", pystray.Menu(recent)),
            Item("Search transcripts…", lambda: self.search.post("ask")),
            Item("Meeting notes…", lambda: self.ctrl.notes_async()),
            Item("Permissions", pystray.Menu(perms)),
            Item("Languages", pystray.Menu(languages)),
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
        self.ctrl.save_settings("live_captions")
        if not self.cfg.live_captions:
            self.captions.post("hide")
        elif self._live_id is not None:
            self.captions.post("reveal")

    def _open_live(self) -> None:
        if self._live_dir:
            open_path(live_transcript_target(self._live_dir))

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
            self.ctrl.save_settings("autostart")
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
                self._notify(*prompt_notice(app, kind))
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
            elif ev[0] == "search":
                self.search.post("results", ev[1], ev[2])  # failures and no matches too
            elif ev[0] == "notes":
                self.notes.post("notes", ev[1])
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
