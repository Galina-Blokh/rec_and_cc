"""Command line entry point: ``recandcc <command>``."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import CONFIG_DIR, CONFIG_PATH, LOG_DIR, load_config, save_keys
from .logging_setup import LOG_FILE, set_crash_notifier, setup_logging

if TYPE_CHECKING:
    from .transcribe import Transcriber
    from .transcribe.live import LiveTranscriber

log = logging.getLogger("recandcc")


# ---------------------------------------------------------------------- run
class _SingleInstance:
    """Advisory lock so two tray apps don't record the same audio twice."""

    def __init__(self, path: Path):
        self.path = path
        self._f = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a+")  # noqa: SIM115 — held open: it is the lock
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        self._f.seek(0)
        self._f.truncate()
        self._f.write(str(os.getpid()))
        self._f.flush()
        return True


def cmd_run(args) -> int:
    setup_logging()
    lock = _SingleInstance(CONFIG_DIR / "run.lock")
    if not lock.acquire():
        log.error("RecAndCC is already running (lock %s)", lock.path)
        print("RecAndCC is already running.", file=sys.stderr)
        return 1
    cfg = load_config()
    log.info(
        "starting RecAndCC (mode=%s, output=%s, pid=%d)",
        cfg.detection.mode,
        cfg.output_dir,
        os.getpid(),
    )
    from .ui.common import Controller

    ctrl = Controller(cfg)
    set_crash_notifier(ctrl.post_notice)  # thread crashes -> toast (via the event queue)
    try:
        if sys.platform == "darwin":
            from .ui.mac import run_app
        elif sys.platform == "win32":
            from .ui.tray import run_app
        else:
            raise NotImplementedError(sys.platform)
        run_app(ctrl)
    finally:
        set_crash_notifier(None)
        ctrl.shutdown()  # idempotent enough: second engine/queue shutdown is a no-op
    return 0


# ---------------------------------------------------------------------- record
def _transcribe_now(cfg, session) -> None:
    from .transcribe import Transcriber

    t = Transcriber(cfg.transcription)
    try:
        t0 = time.monotonic()
        segs = t.transcribe_session(session)
        print(f"Transcribed {len(segs)} segments in {time.monotonic() - t0:.1f}s")
    finally:
        t.close()
    md = Path(session.directory) / "transcript.md"
    if md.exists():
        print(f"Transcript: {md}")


def cmd_record(args) -> int:
    setup_logging(to_file=False, level=logging.INFO if args.verbose else logging.WARNING)
    cfg = load_config()
    from .models import TriggerKind
    from .recorder import Recorder

    rec = Recorder(cfg)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    session = rec.start(TriggerKind.MANUAL, "manual", record_mic=False if args.no_mic else None)
    print(f"Recording to {session.directory} (tracks: {', '.join(map(str, session.tracks))})")
    live: LiveTranscriber | None = None
    transcriber: Transcriber | None = None
    print_lock = threading.Lock()
    if cfg.transcription.live and not args.no_transcribe:
        from .transcribe import Transcriber
        from .transcribe.live import LiveTranscriber
        from .transcribe.output import fmt_hms

        def show(segs) -> None:
            with print_lock:
                for s in segs:
                    print(f"\r\033[K[{fmt_hms(s.start)}] {s.speaker}: {s.text}", flush=True)

        transcriber = Transcriber(cfg.transcription)
        live = LiveTranscriber(transcriber, session, on_segments=show)
        rec.add_listener(live.feed)
        print("Live transcription on (the model loads in the background).")
    print("Press Ctrl-C to stop." if args.seconds is None else f"Recording {args.seconds}s…")
    t0 = time.monotonic()
    while not stop.is_set():
        if args.seconds is not None and time.monotonic() - t0 >= args.seconds:
            break
        levels = " ".join(f"{t}:{v:6.1f}dB" for t, v in rec.levels().items())
        with print_lock:
            print(f"\r  {time.monotonic() - t0:6.1f}s  {levels}   ", end="", flush=True)
        stop.wait(0.25)
    print()
    result = rec.stop()
    silent = rec.silent_tracks()
    if silent:
        print(
            f"WARNING: track(s) {', '.join(map(str, silent))} were completely silent — "
            "check microphone / system audio permissions."
        )
    if result is None:
        if live is not None and transcriber is not None:  # set together
            live.cancel()
            transcriber.close()
        print("Session too short; discarded.")
        return 1
    print(f"Saved {result.directory}")
    if args.no_transcribe:
        return 0
    if live is None or transcriber is None:  # set together
        _transcribe_now(cfg, result)
        return 0
    try:
        t1 = time.monotonic()
        segs = live.finish(result)
        print(f"Final transcript ({len(segs)} segments) {time.monotonic() - t1:.1f}s after stop")
        print(f"Transcript: {Path(result.directory) / 'transcript.md'}")
    except Exception as e:
        print(f"Live transcription failed ({e}); transcribing the recording instead.")
        _transcribe_now(cfg, result)
    finally:
        transcriber.close()
    return 0


def cmd_transcribe(args) -> int:
    setup_logging(to_file=False)
    cfg = load_config()
    from .models import SessionInfo

    session = SessionInfo.load(Path(args.session_dir).expanduser())
    _transcribe_now(cfg, session)
    return 0


def cmd_summarize(args) -> int:
    setup_logging(to_file=False)
    from .transcribe.summary import SummaryUnavailable, clear_pending, summarize_session

    directory = Path(args.session_dir).expanduser()
    try:
        path = summarize_session(directory, load_config())
    except SummaryUnavailable as e:
        print(f"No summary: {e}.", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as e:
        # the type only: the message could quote the transcript
        print(f"Can't summarize the session: {type(e).__name__}", file=sys.stderr)
        return 1
    clear_pending(directory)
    print(path)
    return 0


# ---------------------------------------------------------------------- monitor
def cmd_monitor(args) -> int:
    setup_logging(to_file=False, level=logging.WARNING)
    cfg = load_config()
    from .detect import create_monitor
    from .engine import classify, effective_mode

    mon = create_monitor()
    det = cfg.detection
    own = {os.getpid()}
    print(
        "Watching audio activity (Ctrl-C to stop). "
        "Columns: in/out, classification, effective mode, pid, bundle id, name\n"
    )
    try:
        while True:
            procs = mon.snapshot()
            stamp = time.strftime("%H:%M:%S")
            if not procs:
                print(f"{stamp}  (no audio activity)")
            for p in sorted(procs, key=lambda p: p.bundle_id):
                c = classify(p, det, own)
                io = ("I" if p.input_active else "-") + ("O" if p.output_active else "-")
                if c is None:
                    what, mode = "ignored", "-"
                else:
                    what = f"{c[1]}:{c[0]}"
                    mode = str(effective_mode(det, c[0], [p]))
                print(f"{stamp}  {io}  {what:<32} {mode:<5} {p.pid:>6}  {p.bundle_id}  ({p.name})")
            print()
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        pass
    finally:
        mon.close()
    return 0


# ---------------------------------------------------------------------- misc
def cmd_devices(args) -> int:
    import sounddevice as sd

    print(sd.query_devices())
    try:
        din, dout = sd.default.device
        print(f"\nDefault input:  {din}  {sd.query_devices(din)['name'] if din >= 0 else '-'}")
        print(f"Default output: {dout}  {sd.query_devices(dout)['name'] if dout >= 0 else '-'}")
    except Exception as e:
        print(f"(could not query defaults: {e})")
    if sys.platform == "win32":
        try:
            import pyaudiowpatch as pyaudio

            pa = pyaudio.PyAudio()
            print("\nWASAPI loopback devices:")
            for d in pa.get_loopback_device_info_generator():
                print(
                    f"  {d['index']}: {d['name']} ({d['maxInputChannels']} ch, "
                    f"{int(d['defaultSampleRate'])} Hz)"
                )
            pa.terminate()
        except Exception as e:
            print(f"(loopback query failed: {e})")
    return 0


def cmd_autostart(args) -> int:
    from . import autostart

    if args.action == "on":
        autostart.enable()
    elif args.action == "off":
        autostart.disable()
    if args.action in ("on", "off"):
        load_config(CONFIG_PATH)  # creates the file on first run
        save_keys(CONFIG_PATH, {"autostart": args.action == "on"})  # keeps other hand edits
    print(f"Launch at login: {autostart.describe()}")
    return 0


SEMANTIC_CLI_WAIT_S = 120.0  # `search` is one-shot: wait this long for meaning matches


def cmd_search(args) -> int:
    setup_logging(to_file=False, level=logging.WARNING)
    cfg = load_config()
    from .search import search_transcripts
    from .transcribe.output import fmt_hms

    hits = search_transcripts(
        cfg.output_dir,
        args.query,
        args.limit,
        cfg.search.semantic,
        wait_s=SEMANTIC_CLI_WAIT_S,
        notify=lambda m: print(m, file=sys.stderr),
    )
    if not hits:
        print(f"No matches for {args.query!r}.")
        return 1
    for h in hits:
        print(
            f"{h.started_at[:16].replace('T', ' ')}  [{fmt_hms(h.start)}] {h.speaker}: {h.snippet}"
        )
        print(f"    {h.transcript}")
    return 0


def cmd_config(args) -> int:
    from .cloudsync import cloud_sync_provider, warning_text

    cfg = load_config()  # creates the default file on first run
    print(f"# {CONFIG_PATH}\n")
    print(CONFIG_PATH.read_text(encoding="utf-8"))
    print(f"# logs: {LOG_FILE}")
    provider = cloud_sync_provider(cfg.output_dir)
    if provider:
        print(f"# WARNING: {warning_text(cfg.output_dir, provider)}")
    return 0


# ---------------------------------------------------------------------- speaker names
def cmd_speakers(args) -> int:
    from .models import SessionInfo
    from .transcribe import voices
    from .transcribe.intro_hints import transcript_hints

    directory = Path(args.session_dir).expanduser()
    try:
        speakers = voices.session_speakers(directory)
        known = voices.read_session_voices(SessionInfo.load(directory).id)
        hints = transcript_hints(directory)  # reads the transcript only; stores nothing
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"Can't read the transcript: {e}", file=sys.stderr)
        return 1
    for label, lines in speakers:
        print(f"{label}{'' if label in known else '  (no voice data)'}")
        if hint := hints.get(label):
            print(f"    suggested: {hint.name} (said: {hint.line[:80]})")
        for line in lines:
            print(f"    {line[:100]}")
    if hints:
        print("Suggestions are only hints: nothing is renamed until you run `name-speaker`.")
    if not speakers:
        print("No other speakers in this transcript.")
    return 0


def cmd_name_speaker(args) -> int:
    cfg = load_config()
    if not cfg.transcription.speaker_names:
        print("Voice profiles are off: set transcription.speaker_names = true.", file=sys.stderr)
        return 1
    from .transcribe import voices

    try:
        p = voices.name_speaker(Path(args.session_dir).expanduser(), args.label, args.name)
    except (OSError, ValueError, KeyError) as e:
        print(f"Can't name {args.label!r}: {e}", file=sys.stderr)
        return 1
    print(f"{args.label} is now {p.name} (voice profile from {p.count} session(s)).")
    _refresh_summary(Path(args.session_dir).expanduser(), cfg)
    return 0


def _refresh_summary(directory: Path, cfg) -> None:
    """After a speaker rename: mark an existing summary.md stale. The app redoes it while
    idle (never here: that could run the model on the GPU during a recording)."""
    from .transcribe.summary import SUMMARY_NAME, mark_pending

    if not (directory / SUMMARY_NAME).exists():
        return
    mark_pending(directory)
    by_app = "the summary is redone by the app (or run" if cfg.summary.enabled else "(run"
    print(f"{SUMMARY_NAME} still uses the old name: {by_app} `recandcc summarize {directory}`).")


def cmd_voices(args) -> int:
    from .transcribe import voices

    voices.prune_session_voices()  # of sessions whose folder was deleted
    if args.action == "list":
        profiles = voices.load_profiles()
        for p in profiles:
            print(f"{p.name}  ({p.count} session(s), updated {p.updated_at})")
        print(f"{len(profiles)} voice profile(s) in {voices.VOICES_DIR}")
    elif args.all:
        n = voices.delete_all_profiles()
        print(f"Deleted {n} voice profile(s) and all saved session voices.")
    elif not args.name:
        print("Name the profile to delete, or use --all.", file=sys.stderr)
        return 1
    elif not voices.delete_profile(args.name):
        print(f"No voice profile named {args.name!r}.", file=sys.stderr)
        return 1
    else:
        print(f"Deleted the voice profile of {args.name}.")
    return 0


def _add_speaker_commands(sub) -> None:
    s = sub.add_parser("speakers", help="list a session's speakers with their first lines")
    s.add_argument("session_dir")
    s.set_defaults(func=cmd_speakers)
    n = sub.add_parser("name-speaker", help='name a speaker, e.g. "Speaker 2" "Anna"')
    n.add_argument("session_dir")
    n.add_argument("label", help='the label in the transcript, e.g. "Speaker 2"')
    n.add_argument("name")
    n.set_defaults(func=cmd_name_speaker)
    v = sub.add_parser("voices", help="list or delete saved voice profiles")
    v.add_argument("action", choices=["list", "delete"])
    v.add_argument("name", nargs="?")
    v.add_argument("--all", action="store_true", help="with delete: every profile")
    v.set_defaults(func=cmd_voices)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="recandcc", description="Record calls/videos locally and transcribe them."
    )
    sub = p.add_subparsers(dest="command")

    sub.add_parser("run", help="start the menu-bar / tray app").set_defaults(func=cmd_run)

    r = sub.add_parser(
        "record", help="record manually (headless), transcribing live (transcription.live)"
    )
    r.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    r.add_argument("--no-mic", action="store_true", help="system audio only")
    r.add_argument("--no-transcribe", action="store_true", help="just record")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_record)

    t = sub.add_parser("transcribe", help="transcribe a recorded session directory")
    t.add_argument("session_dir")
    t.set_defaults(func=cmd_transcribe)

    u = sub.add_parser("summarize", help="write summary.md (local LLM) for a transcribed session")
    u.add_argument("session_dir")
    u.set_defaults(func=cmd_summarize)

    m = sub.add_parser("monitor", help="print live audio activity + how rules classify it")
    m.add_argument("--interval", type=float, default=1.0)
    m.set_defaults(func=cmd_monitor)

    s = sub.add_parser("search", help="search all transcripts (local full-text index)")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=20, help="show at most N matches")
    s.set_defaults(func=cmd_search)

    sub.add_parser("devices", help="list audio devices").set_defaults(func=cmd_devices)

    a = sub.add_parser("autostart", help="launch at login")
    a.add_argument("action", choices=["on", "off", "status"])
    a.set_defaults(func=cmd_autostart)

    sub.add_parser(
        "config", help="print config path and contents (warns if output_dir is cloud-synced)"
    ).set_defaults(func=cmd_config)
    _add_speaker_commands(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


__all__ = ["main", "build_parser", "LOG_DIR"]
