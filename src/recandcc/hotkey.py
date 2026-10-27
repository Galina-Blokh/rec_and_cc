"""Global keyboard shortcut (``Config.hotkey``, default ``<ctrl>+<alt>+r``).

Neither backend needs an extra permission, and neither sees any other keystrokes:

* **macOS**: ``rcc-audiotap hotkey COMBO`` registers the combination with Carbon's
  ``RegisterEventHotKey`` (no Accessibility / Input Monitoring needed) and prints one JSON line
  per press. ``HotkeyListener`` keeps that helper running as a child process (restarting it with
  a backoff if it dies) and calls ``on_press`` from its reader thread. The helper exits when we
  close its stdin, so it never outlives the app.
* **Windows** (UNTESTED, written without access to Windows): ``WinHotkeyListener`` calls
  ``user32.RegisterHotKey`` on a dedicated thread that runs a ``GetMessage`` loop and gets
  ``WM_HOTKEY`` for each press.

Combo format (case-insensitive, ``<...>`` optional): modifiers ``<cmd>`` / ``<command>`` /
``<super>``, ``<ctrl>`` / ``<control>``, ``<alt>`` / ``<option>`` / ``<opt>``, ``<shift>``,
``<win>``, plus one key: ``a``-``z``, ``0``-``9``, ``f1``-``f12``, ``space``. An empty string
disables the shortcut. On Windows ``<cmd>`` means Ctrl (so the default is Ctrl+Shift+R) and
``<win>`` is the Windows key; on macOS ``<win>`` is ⌘.

``create_listener(text, on_press, on_error)`` returns the right listener for the platform (or
None). ``on_press()`` and ``on_error(kind, message)`` run on the listener's thread; ``kind`` is
``"taken"`` (another app has the combination), ``"invalid"`` or ``"failed"``. ``on_error`` is
called at most once per listener.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

log = logging.getLogger(__name__)

MODIFIER_ALIASES = {
    "cmd": "cmd",
    "command": "cmd",
    "super": "cmd",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "option": "alt",
    "opt": "alt",
    "shift": "shift",
    "win": "win",
}
FUNCTION_KEYS = tuple(f"f{i}" for i in range(1, 13))
KEYS = tuple("abcdefghijklmnopqrstuvwxyz") + tuple("0123456789") + FUNCTION_KEYS + ("space",)
# Apple's canonical modifier order: ⌃⌥⇧⌘.
MAC_ORDER = ("ctrl", "alt", "shift", "cmd")
MAC_SYMBOLS = {"ctrl": "⌃", "alt": "⌥", "shift": "⇧", "cmd": "⌘"}
WIN_ORDER = ("ctrl", "alt", "shift", "win")
WIN_NAMES = {"ctrl": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win"}

# Windows RegisterHotKey constants.
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
WIN_MODS = {"alt": MOD_ALT, "ctrl": MOD_CONTROL, "shift": MOD_SHIFT, "win": MOD_WIN}
VK_SPACE, VK_F1 = 0x20, 0x70
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012
ERROR_HOTKEY_ALREADY_REGISTERED = 1409

MAX_BACKOFF_S = 30.0
TAKEN_RETRY_S = 60.0  # the other app may quit; retry quietly


class HotkeyError(ValueError):
    """The combo string can't be parsed."""


@dataclass(frozen=True)
class Hotkey:
    """A parsed combo. ``modifiers`` holds canonical names: cmd, ctrl, alt, shift, win."""

    modifiers: frozenset[str]
    key: str

    def _platform_mods(self, platform: str) -> set[str]:
        mods = set(self.modifiers)
        if platform == "win32":
            if "cmd" in mods:  # <cmd> is the "primary" modifier: Ctrl on Windows
                mods.discard("cmd")
                mods.add("ctrl")
        elif "win" in mods:  # same physical key as ⌘
            mods.discard("win")
            mods.add("cmd")
        return mods

    def mac_combo(self) -> str:
        """Normalised string for ``rcc-audiotap hotkey``, e.g. ``<shift>+<cmd>+r``."""
        mods = self._platform_mods("darwin")
        return "+".join([f"<{m}>" for m in MAC_ORDER if m in mods] + [self.key])

    def label(self, platform: str | None = None) -> str:
        """``⇧⌘R`` style on macOS (Apple's order ⌃⌥⇧⌘), ``Ctrl+Shift+R`` elsewhere."""
        platform = platform or sys.platform
        key = "Space" if self.key == "space" else self.key.upper()
        mods = self._platform_mods(platform)
        if platform == "darwin":
            return "".join(MAC_SYMBOLS[m] for m in MAC_ORDER if m in mods) + key
        return "+".join([WIN_NAMES[m] for m in WIN_ORDER if m in mods] + [key])

    def win32(self) -> tuple[int, int]:
        """(fsModifiers, virtual key code) for ``RegisterHotKey`` (MOD_NOREPEAT included)."""
        mods = MOD_NOREPEAT
        for m in self._platform_mods("win32"):
            mods |= WIN_MODS[m]
        if self.key == "space":
            vk = VK_SPACE
        elif self.key in FUNCTION_KEYS:
            vk = VK_F1 + int(self.key[1:]) - 1
        else:
            vk = ord(self.key.upper())  # VK codes of A-Z / 0-9 are their ASCII codes
        return mods, vk


def parse_hotkey(text: str | None) -> Hotkey | None:
    """``"<cmd>+<shift>+r"`` -> Hotkey; ``""`` / None -> None (disabled).

    Raises HotkeyError for anything else that isn't exactly one key plus modifiers. Keys other
    than F1-F12 need at least one modifier, so a plain letter is never grabbed system-wide.
    """
    text = (text or "").strip()
    if not text:
        return None
    mods: set[str] = set()
    key: str | None = None
    for raw in text.lower().split("+"):
        tok = raw.strip()
        if len(tok) >= 2 and tok.startswith("<") and tok.endswith(">"):
            tok = tok[1:-1].strip()
        if not tok:
            raise HotkeyError(f"empty part in {text!r}")
        if tok in MODIFIER_ALIASES:
            mods.add(MODIFIER_ALIASES[tok])
        elif tok in KEYS:
            if key is not None:
                raise HotkeyError(f"more than one key in {text!r}")
            key = tok
        else:
            raise HotkeyError(f"unknown key {tok!r} in {text!r}")
    if key is None:
        raise HotkeyError(f"no key in {text!r}")
    if not mods and key not in FUNCTION_KEYS:
        raise HotkeyError(f"{text!r} needs a modifier (<cmd>, <ctrl>, <alt>, <shift>)")
    return Hotkey(frozenset(mods), key)


def _call(cb: Callable | None, *args) -> None:
    if cb is None:
        return
    try:
        cb(*args)
    except Exception:
        log.exception("hotkey callback failed")


# ------------------------------------------------------------------------- macOS
class HotkeyListener:
    """Runs ``rcc-audiotap hotkey COMBO`` and calls ``on_press`` for each press (macOS).

    ``command`` replaces ``[<helper>, "hotkey"]`` (tests use a fake helper script).
    """

    def __init__(
        self,
        hotkey: Hotkey,
        on_press: Callable[[], None],
        on_error: Callable[[str, str], None] | None = None,
        *,
        command: Sequence[str] | None = None,
        backoff_s: float = 1.0,
        max_backoff_s: float = MAX_BACKOFF_S,
        taken_retry_s: float = TAKEN_RETRY_S,
    ) -> None:
        self.hotkey = hotkey
        self._on_press = on_press
        self._on_error = on_error
        self._command = list(command) if command is not None else None
        self._backoff0 = backoff_s
        self._max_backoff = max_backoff_s
        self._taken_retry = taken_retry_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._closed = threading.Event()
        self._error_reported = False
        self.registered = threading.Event()  # set while the helper holds the shortcut
        self.starts = 0  # helper launches (for tests / logs)
        self._thread = threading.Thread(target=self._run, name="hotkey", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        with self._lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            with contextlib.suppress(Exception):
                if proc.stdin is not None:
                    proc.stdin.close()  # the helper exits on EOF
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=2)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _report(self, kind: str, message: str) -> None:
        if self._error_reported:
            return
        self._error_reported = True
        _call(self._on_error, kind, message)

    def _run(self) -> None:
        backoff = self._backoff0
        while not self._closed.is_set():
            started = time.monotonic()
            error: tuple[str, str] | None = None
            try:
                error = self._run_once()
            except Exception as exc:
                log.exception("hotkey helper failed")
                error = ("failed", str(exc))
                if isinstance(exc, FileNotFoundError):
                    self._report("failed", str(exc))
            self.registered.clear()
            if self._closed.is_set():
                break
            if error is not None and error[0] in ("taken", "invalid"):
                log.warning("hotkey %s: %s", self.hotkey.mac_combo(), error[1])
                self._report(*error)
                if error[0] == "invalid":
                    return  # won't get better until the config changes
                self._closed.wait(self._taken_retry)
                continue
            if time.monotonic() - started > 60:
                backoff = self._backoff0
            log.warning("hotkey helper exited; restarting in %.1f s", backoff)
            self._closed.wait(backoff)
            backoff = min(backoff * 2, self._max_backoff)

    def _run_once(self) -> tuple[str, str] | None:
        if self._command is not None:
            base = self._command
        else:
            from .native import find_helper

            base = [str(find_helper()), "hotkey"]
        proc = subprocess.Popen(  # noqa: S603 - fixed argv: our helper (or a test stub), no shell
            [*base, self.hotkey.mac_combo()],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.starts += 1
        with self._lock:
            self._proc = proc
        if self._closed.is_set():  # close() raced with the start
            proc.kill()
        threading.Thread(target=_pump_stderr, args=(proc,), daemon=True).start()
        error: tuple[str, str] | None = None
        assert proc.stdout is not None  # noqa: S101 - type narrowing: the pipe was requested in Popen
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                log.debug("bad hotkey line %r", line)
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("hotkey") == "pressed":
                if not self._closed.is_set():
                    _call(self._on_press)
            elif msg.get("hotkey") == "registered":
                log.info("global shortcut %s registered", self.hotkey.label("darwin"))
                self.registered.set()
            elif "error" in msg:
                error = (str(msg["error"]), str(msg.get("message") or msg["error"]))
        proc.wait()
        with self._lock:
            self._proc = None
        return error


def _pump_stderr(proc: subprocess.Popen[str]) -> None:
    assert proc.stderr is not None  # noqa: S101 - type narrowing: the pipe was requested in Popen
    for line in proc.stderr:
        if line.strip():
            log.info("%s", line.rstrip())


# ------------------------------------------------------------------------- Windows
class WinHotkeyListener:
    """``RegisterHotKey`` + ``GetMessage`` loop on a dedicated thread (Windows).

    UNTESTED (written without access to Windows). The hotkey is registered for the thread
    (``hWnd = NULL``), so ``WM_HOTKEY`` arrives in that thread's message queue; ``close()``
    posts ``WM_QUIT`` to it.
    """

    def __init__(
        self,
        hotkey: Hotkey,
        on_press: Callable[[], None],
        on_error: Callable[[str, str], None] | None = None,
    ) -> None:
        self.hotkey = hotkey
        self._on_press = on_press
        self._on_error = on_error
        self._tid: int | None = None
        self._ready = threading.Event()  # thread has a message queue (or gave up)
        self._closed = threading.Event()
        self.registered = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hotkey", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._ready.wait(2)
        if self._tid is not None:
            with contextlib.suppress(Exception):
                import ctypes

                ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)  # type: ignore[attr-defined]
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
            kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]
            user32.RegisterHotKey.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                wintypes.UINT,
                wintypes.UINT,
            ]
            user32.RegisterHotKey.restype = wintypes.BOOL
            user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetMessageW.argtypes = [
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
            ]
            user32.GetMessageW.restype = ctypes.c_int  # BOOL, but -1 on error
            user32.PeekMessageW.argtypes = [
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
                wintypes.UINT,
            ]
            msg = wintypes.MSG()
            # Force creation of this thread's message queue before anyone posts WM_QUIT to it.
            user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
            self._tid = int(kernel32.GetCurrentThreadId())
        except Exception as exc:
            log.exception("hotkey: Windows API unavailable")
            self._ready.set()
            _call(self._on_error, "failed", str(exc))
            return
        self._ready.set()
        if self._closed.is_set():
            return
        mods, vk = self.hotkey.win32()
        hk_id = 1
        if not user32.RegisterHotKey(None, hk_id, mods, vk):
            err = ctypes.get_last_error()  # type: ignore[attr-defined]
            label = self.hotkey.label("win32")
            if err == ERROR_HOTKEY_ALREADY_REGISTERED:
                _call(self._on_error, "taken", f"{label} is already registered by another app")
            else:
                _call(self._on_error, "failed", f"RegisterHotKey({label}) failed: error {err}")
            return
        self.registered.set()
        log.info("global shortcut %s registered", self.hotkey.label("win32"))
        try:
            while not self._closed.is_set():
                r = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r <= 0:  # 0 = WM_QUIT, -1 = error
                    break
                if msg.message == WM_HOTKEY and msg.wParam == hk_id:
                    _call(self._on_press)
        finally:
            user32.UnregisterHotKey(None, hk_id)
            self.registered.clear()


# ------------------------------------------------------------------------- factory
def create_listener(
    text: str | None,
    on_press: Callable[[], None],
    on_error: Callable[[str, str], None] | None = None,
) -> HotkeyListener | WinHotkeyListener | None:
    """Listener for ``text`` on this platform; None if disabled or unsupported.

    Raises HotkeyError if ``text`` can't be parsed.
    """
    hk = parse_hotkey(text)
    if hk is None:
        return None
    if sys.platform == "darwin":
        return HotkeyListener(hk, on_press, on_error)
    if sys.platform == "win32":
        return WinHotkeyListener(hk, on_press, on_error)
    log.info("global shortcuts are not supported on %s", sys.platform)
    return None
