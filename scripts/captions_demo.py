"""Demo of the macOS live-captions panel, fed with fake live events (no recording).

    uv run python scripts/captions_demo.py              # ~15 s of en/ru lines, then exits
    uv run python scripts/captions_demo.py --seconds 40 --interval 0.5
    uv run python scripts/captions_demo.py --snapshot /tmp/captions.png   # + PNG of the panel

``--snapshot`` renders the panel window itself (no Screen Recording permission needed) a few
seconds after the session ends, so the "finished" header is visible too.

A background thread posts ``("live_start" | "live" | "live_end", ...)`` events onto a queue,
exactly like ``Controller.events``; a 0.5 s rumps timer drains it on the main thread (as
``ui/mac.py`` does) and drives ``CaptionsPanel``. The panel stays up for the auto-hide delay
after ``live_end``, then the app quits.
"""

from __future__ import annotations

import argparse
import itertools
import queue
import threading
import time

import rumps
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

from recandcc.models import Segment
from recandcc.ui.captions import AUTO_HIDE_AFTER_S
from recandcc.ui.captions_mac import CaptionsPanel

LINES = [
    ("Others", "en", "Hello team, this is the weekly sync. Can everyone hear me?"),
    ("Me", "en", "Yes, loud and clear."),
    ("Others", "en", "Great. Let's start with the release status for the captions feature."),
    ("Me", "ru", "Привет! Релиз почти готов, осталось проверить окно субтитров."),
    ("Others", "ru", "Отлично. А как оно выглядит поверх полноэкранного видео?"),
    ("Me", "en", "It floats above everything and doesn't steal focus."),
    (
        "Others",
        "en",
        "Nice. One more thing: long sentences should wrap nicely under the text column and "
        "not under the speaker label, so it stays readable.",
    ),
    ("Me", "ru", "Да, перенос строк выровнен по колонке текста."),
]


def producer(events: queue.Queue, seconds: float, interval: float) -> None:
    sid = "demo-session"
    events.put(("live_start", sid, "Zoom (demo)"))
    t0 = time.monotonic()
    for speaker, lang, text in itertools.cycle(LINES):
        time.sleep(interval)
        t = time.monotonic() - t0
        if t >= seconds:
            break
        events.put(("live", sid, [Segment(t, t + 2.0, text, speaker, lang)]))
    events.put(("live_end", sid))


def snapshot(panel: CaptionsPanel, path: str) -> None:
    """Render the panel (title bar included) to a PNG, in-process."""
    from AppKit import NSBitmapImageFileTypePNG

    view = panel._panel.contentView().superview()  # the window frame view
    rect = view.bounds()
    rep = view.bitmapImageRepForCachingDisplayInRect_(rect)
    view.cacheDisplayInRect_toBitmapImageRep_(rect, rep)
    rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}).writeToFile_atomically_(
        path, True
    )
    print(f"snapshot: {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seconds", type=float, default=15.0, help="how long to post lines")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between lines")
    ap.add_argument("--snapshot", metavar="PNG", help="save a PNG of the panel")
    ap.add_argument(
        "--snapshot-live", action="store_true", help="snapshot while live, not after end"
    )
    args = ap.parse_args()

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    events: queue.Queue = queue.Queue()
    panel = CaptionsPanel()
    app = rumps.App("CaptionsDemo", title="CC", quit_button="Quit demo")
    done: dict[str, float] = {}
    t0 = time.monotonic()

    def tick(_timer) -> None:
        while True:
            try:
                ev = events.get_nowait()
            except queue.Empty:
                break
            if ev[0] == "live_start":
                panel.show(ev[2])
            elif ev[0] == "live":
                panel.append(ev[2])
            elif ev[0] == "live_end":
                panel.end()
                done["at"] = time.monotonic()
        panel.tick()
        if args.snapshot and "snap" not in done:
            ready = (
                time.monotonic() - t0 > args.seconds * 0.7
                if args.snapshot_live
                else "at" in done and time.monotonic() - done["at"] > 1
            )
            if ready:
                done["snap"] = 1
                snapshot(panel, args.snapshot)
        if "at" in done and time.monotonic() - done["at"] > AUTO_HIDE_AFTER_S + 1:
            rumps.quit_application()

    timer = rumps.Timer(tick, 0.5)
    timer.start()
    threading.Thread(
        target=producer, args=(events, args.seconds, args.interval), daemon=True
    ).start()
    app.run()


if __name__ == "__main__":
    main()
