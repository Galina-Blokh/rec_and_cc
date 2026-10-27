"""User configuration, stored as TOML in the platform config dir."""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import platformdirs
import tomli_w

from .models import Mode

APP_NAME = "RecAndCC"
CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
CONFIG_PATH = CONFIG_DIR / "config.toml"
LOG_DIR = Path(platformdirs.user_log_dir(APP_NAME, appauthor=False))
DEFAULT_OUTPUT_DIR = Path(platformdirs.user_documents_dir()) / "Transcribes"

log = logging.getLogger("recandcc.config")

# Language codes Whisper knows (faster_whisper.tokenizer._LANGUAGE_CODES), for validating
# `transcription.languages` without importing a model library here.
_WHISPER_CODES = (
    "af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu ha "
    "haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml mn mr "
    "ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw ta te tg th "
    "tk tl tr tt uk ur uz vi yi yo zh yue"
)
WHISPER_LANGUAGES = frozenset(_WHISPER_CODES.split())
# Common spellings that are not Whisper codes ("iw" is the old ISO code for Hebrew).
LANGUAGE_ALIASES = {"iw": "he", "hebrew": "he", "english": "en", "russian": "ru"}

# Patterns are case-insensitive substrings matched against AudioProcess.bundle_id and .name.
DEFAULT_CALL_APPS = [
    "us.zoom",
    "zoom.exe",
    "com.microsoft.teams",
    "ms-teams",
    "teams.exe",
    "com.apple.facetime",
    "com.tinyspeck.slackmacgap",
    "slack.exe",
    "com.hnc.discord",
    "discord.exe",
    "ru.keepcoder.telegram",
    "telegram",
    "net.whatsapp",
    "whatsapp",
    "com.skype",
    "skype",
    "com.webex",
    "webex",
]
DEFAULT_MEDIA_APPS = [
    "com.google.chrome",
    "chrome.exe",
    "com.apple.safari",
    "com.apple.webkit",
    "org.mozilla.firefox",
    "firefox.exe",
    "company.thebrowser",
    "arc.exe",
    "com.microsoft.edgemac",
    "msedge.exe",
    "com.brave.browser",
    "brave.exe",
    "com.apple.quicktimeplayerx",
    "org.videolan.vlc",
    "vlc.exe",
    "com.apple.podcasts",
    "com.apple.tv",
]
DEFAULT_IGNORE_APPS = [
    "com.spotify.client",
    "spotify.exe",
    "com.apple.music",
    "music.ui",
    "com.apple.systemsound",
    "coreaudiod",
    "com.apple.notificationcenter",
    "com.apple.siri",
    "com.apple.voicememos",
    "recandcc",
    "afplay",  # command-line player (scripts, tests), not something you watch
    "rcc-audiotap",
]


@dataclass
class DetectionConfig:
    mode: Mode = Mode.ASK
    poll_interval_s: float = 1.0
    min_active_s: float = 3.0  # audio must be active this long before we trigger
    stop_after_inactive_s: float = 20.0  # stop when the trigger app stopped doing audio I/O
    stop_after_silence_s: float = 180.0  # stop when everything has been silent this long
    # Videos/podcasts end crisply (no conversational pauses), so media sessions stop sooner.
    media_stop_after_inactive_s: float = 8.0
    media_stop_after_silence_s: float = 30.0
    silence_dbfs: float = -50.0  # below this level a frame counts as silence
    ask_timeout_s: float = 30.0  # "keep recording?" prompt auto-dismisses after this
    # In ask mode recording starts right away and the prompt asks whether to keep it, so
    # nothing is lost while you decide. Unanswered prompt: "record" keeps it, "ignore" discards.
    ask_timeout_action: str = "record"
    record_mic: str = "calls"  # "always" | "calls" | "never"
    # "all": the Others track records everything the Mac plays. "app": only the app that
    # triggered the session (macOS; falls back to "all" for manual sessions, when the app has
    # no audio process yet, and on Windows).
    system_audio_scope: str = "all"
    # Room listening (opt-in): while idle, keep the mic open and start a "room" session when
    # people talk near the laptop: room_speech_s of speech within room_window_s.
    room_listening: bool = False
    room_speech_s: float = 6.0
    room_window_s: float = 15.0
    room_preroll_s: float = 15.0  # audio before the trigger kept at the session's start
    room_stop_after_silence_s: float = 90.0  # no speech this long ends a room session
    room_vad_threshold: float = 0.5  # Silero speech probability that counts as speech
    # Sensitivity 0-100 (higher = less needed to start), unset = the plain thresholds above.
    # Outside = room/mic (room_speech_s + a noise-floor margin), inside = app audio
    # (min_active_s). See recandcc.sensitivity and the "Sensitivity" menu presets.
    outside_sensitivity: float | None = None
    inside_sensitivity: float | None = None
    call_apps: list[str] = field(default_factory=lambda: list(DEFAULT_CALL_APPS))
    media_apps: list[str] = field(default_factory=lambda: list(DEFAULT_MEDIA_APPS))
    ignore_apps: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_APPS))
    # Per-app override of `mode`, keyed by pattern, e.g. {"us.zoom" = "auto"}.
    app_modes: dict[str, Mode] = field(default_factory=dict)


DEFAULT_LANGUAGES = ("en", "ru")


@dataclass
class TranscriptionConfig:
    model: str = "large-v3-turbo"  # any faster-whisper model name or local path
    # "auto" = MLX on the Apple-silicon GPU when available (~20x faster per pass than the CPU),
    # else faster-whisper. "faster-whisper" | "mlx" force one.
    backend: str = "auto"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    # "float32" uses Apple Accelerate and benchmarked ~20% faster than "int8" on Apple silicon
    # (at ~3 GB RAM instead of ~1 GB); int8 is the better trade-off on typical Windows laptops.
    compute_type: str = "float32" if sys.platform == "darwin" else "int8"
    # Auto-detect among these Whisper codes, e.g. ["en", "ru", "he"]; [] = any language.
    languages: list[str] = field(default_factory=lambda: list(DEFAULT_LANGUAGES))
    workers: int = 0  # parallel transcription workers, 0 = auto
    dedupe_echo: bool = True  # drop mic segments that are just speaker echo of "Others"
    keep_audio: bool = True  # False = delete WAVs after a successful transcription
    # Delete a recording's audio this many days after it was transcribed (0 = keep forever).
    # Transcripts are never deleted. Also settable from the menu: "Delete audio after".
    audio_retention_days: int = 0
    # Split "Others" into "Speaker 1", "Speaker 2", ... in the final transcript (offline
    # speaker diarization). Live captions keep showing "Others".
    diarize: bool = True
    # Split an "Others" segment at the word where the diarized speaker changes. Opt-in: it needs
    # word timestamps (extra time), so it applies to batch transcription only, not to live
    # sessions (on MLX: with mlx-whisper 0.4.3, whose word aligner RecAndCC ports to numpy).
    split_on_speaker_change: bool = False
    # Hints for diarization when you know how many other people are on the call: the number of
    # speakers is picked automatically within [min_speakers, max_speakers]. 0 = no hint.
    min_speakers: int = 0
    max_speakers: int = 0
    # Opt-in: find speaker changes without a pause and overlapped speech with a segmentation
    # model (6 MB, downloaded once when on); off or without the model: fixed ~2 s windows.
    # Off by default until it is measured on real meetings (#13). ADR 0014.
    diarize_overlap: bool = False
    # Remember speakers' voices so that a speaker named once (`recandcc name-speaker`) is named
    # in later sessions. Off by default: voice profiles are biometric data. See voices.py.
    speaker_names: bool = False
    # macOS, opt-in, only with speaker_names: read the other participants' names from the call
    # app's visible participant list (Accessibility permission) and use them for `Speaker N`
    # when that is unambiguous. See participants.py.
    names_from_call_app: bool = False
    # Transcribe while recording: text appears ~1-2 s after each phrase, and the final
    # transcript is ready right when the session stops (no post-session pass).
    live: bool = True
    # Per-language model routing, e.g. {he = "ivrit-ai/whisper-large-v3-turbo-ct2"}: speech
    # regions detected in a listed language are transcribed by its model (loaded on first
    # use), all others by `model`. {} = `model` for every language.
    language_models: dict[str, str] = field(default_factory=dict)
    # Allow downloading Whisper repos without a pinned revision (custom HF repo ids). Off by
    # default: upstream could swap their weights. Already-downloaded copies load either way.
    allow_unpinned_models: bool = False


DEFAULT_SUMMARY_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


@dataclass
class SummaryConfig:
    # Write summary.md (short summary + action items) after each final transcript, with a
    # local LLM (mlx-lm, Apple silicon only). Off by default: it costs ~2 GB of RAM and a
    # one-time ~1.7 GB model download.
    enabled: bool = False
    model: str = DEFAULT_SUMMARY_MODEL  # pinned in transcribe/revisions.py, or a local path
    max_tokens: int = 400  # longest generated answer (summary, action list, chunk notes)


@dataclass
class Config:
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    autostart: bool = False
    # Global start/stop shortcut. ⌃⌥R by default: ⌘⇧R would override Chrome/Safari shortcuts.
    hotkey: str = "<ctrl>+<alt>+r"
    # Floating captions window while recording (needs transcription.live).
    live_captions: bool = True
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    def save(self, path: Path | None = None) -> None:
        # Resolve first: os.replace on a symlinked config.toml (dotfiles) would swap the link
        # for a regular file and leave the real one stale.
        path = (CONFIG_PATH if path is None else path).resolve()
        # A dir we create is private (0700); an existing one keeps its mode. The file itself
        # is 0600 on first write (mkstemp) and keeps the user's mode after that.
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        # Keep the user's own spelling of `languages` ("iw", unknown codes) unless the list
        # was changed since loading: load_config only normalises the in-memory copy.
        tr = self.transcription
        raw = getattr(tr, "_raw_languages", None)
        if raw is not None and normalize_languages(raw) == tr.languages:
            data["transcription"]["languages"] = raw
        # TOML has no null: unset (None) values are omitted. Only these sections have Optional
        # fields today; add any new section with an Optional field here.
        for section in ("detection", "transcription"):
            data[section] = {k: v for k, v in data[section].items() if v is not None}
        with _file_lock(path.with_name(f".{path.name}.lock")):  # shared with save_keys / CLI
            _write_atomic(path, tomli_w.dumps(data).encode())


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def language_code(item) -> str:
    """One `languages` entry as written by the user, lowercased and alias-mapped ("iw" ->
    "he"). The result may still be an unknown code."""
    code = str(item).strip().lower()
    return LANGUAGE_ALIASES.get(code, code)


def normalize_languages(value) -> list[str]:
    """`transcription.languages` as clean Whisper codes: lowercased, aliases mapped
    ("iw" -> "he"), duplicates and unknown codes dropped (with a warning). A single string is
    accepted as a one-item list. `[]` stays `[]` (= any language)."""
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list | tuple):
        items = list(value)
    elif value is None:
        items = []
    else:
        log.warning("transcription.languages must be a list; using the defaults")
        return list(DEFAULT_LANGUAGES)
    out: list[str] = []
    for item in items:
        code = language_code(item)
        if code not in WHISPER_LANGUAGES:
            log.warning("ignoring unknown language %r in transcription.languages", item)
        elif code not in out:
            out.append(code)
    if items and not out:
        log.warning("no valid transcription.languages; using %s", ", ".join(DEFAULT_LANGUAGES))
        return list(DEFAULT_LANGUAGES)
    return out


def normalize_language_models(value) -> dict[str, str]:
    """`transcription.language_models` with clean Whisper codes as keys ("iw" -> "he").
    Unknown language codes and empty or non-string model names are dropped with a warning."""
    if not isinstance(value, dict):
        log.warning("transcription.language_models must be a table; ignoring it")
        return {}
    out: dict[str, str] = {}
    for key, model in value.items():
        code = language_code(key)
        if code not in WHISPER_LANGUAGES:
            log.warning("ignoring unknown language %r in transcription.language_models", key)
        elif not isinstance(model, str) or not model.strip():
            log.warning("ignoring transcription.language_models.%s: not a model name", key)
        else:
            out[code] = model.strip()
    return out


def save_keys(path: Path | None, updates: dict[str, Any]) -> float:
    """Write only the dotted keys in ``updates`` ("transcription.languages": [...]) into
    config.toml, leaving every other key's value exactly as it is on disk (it may have been
    edited by hand since the app loaded it). A value of None removes the key (TOML has no
    null); a table left empty by that is removed too, and removing a key that isn't there
    changes nothing. Like every save, it drops TOML comments and rewrites the file in
    tomli_w's key order. ``path=None`` means CONFIG_PATH, read at call time. Returns the new
    file's mtime. Writers in this process and the CLI are serialized by a lock file; an
    editor saving in the same millisecond is not (it doesn't take it). Raises
    FileNotFoundError if there is no file yet, ValueError if it is not valid TOML or a key's
    parent is not a table."""
    path = (
        CONFIG_PATH if path is None else path
    ).resolve()  # a symlinked config.toml stays a symlink
    if not path.exists():  # don't create the lock file (or the dir) for a missing config
        raise FileNotFoundError(path)
    with _file_lock(path.with_name(f".{path.name}.lock")):  # the CLI is another process
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        changed = False
        for dotted, value in updates.items():
            *parents, leaf = dotted.split(".")
            if value is None:
                changed |= _remove_key(data, parents, leaf)
                continue
            table = data
            for name in parents:
                table = table.setdefault(name, {})
                if not isinstance(table, dict):
                    raise ValueError(f"{name} in {path} is not a table")
            table[leaf] = value
            changed = True
        if not changed:  # only absent keys were removed: leave the file as it is
            return path.stat().st_mtime
        return _write_atomic(path, tomli_w.dumps(data).encode())


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on ``lock_path`` (a sidecar file that is never
    replaced), so read-modify-writes from the app and the CLI don't lose each other's keys."""
    with open(lock_path, "a+b") as f:
        if sys.platform == "win32":
            import msvcrt

            while True:  # LK_LOCK gives up after ~10 s; keep waiting like flock does
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError as e:
                    # LK_LOCK reports "still held after ~10 s" as EDEADLOCK; anything else
                    # (a real permission error) would otherwise retry forever.
                    if e.errno != errno.EDEADLK:
                        raise
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _remove_key(data: dict, parents: list[str], leaf: str) -> bool:
    """Remove ``parents...leaf`` without creating tables; drop tables it leaves empty.
    Returns whether anything was removed."""
    chain = [data]
    for name in parents:
        table = chain[-1].get(name)
        if not isinstance(table, dict):
            return False  # no such table: nothing to remove
        chain.append(table)
    if leaf not in chain[-1]:
        return False
    del chain[-1][leaf]
    for i in reversed(range(len(parents))):  # innermost first
        if chain[i + 1]:
            break
        del chain[i][parents[i]]
    return True


def _write_atomic(path: Path, payload: bytes) -> float:
    """Replace ``path`` in one step, so a crash mid-write never leaves half a config. Returns
    the new file's mtime."""
    # A unique temp name per write: a fixed one let one writer's os.replace consume another's.
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        with contextlib.suppress(OSError):
            os.chmod(tmp, path.stat().st_mode & 0o777)  # keep the user's file mode
        mtime = tmp.stat().st_mtime  # os.replace keeps it
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return mtime


def _replace(src: Path, dst: Path) -> None:
    """os.replace, retried briefly on Windows, where a reader that has the file open (an
    editor, antivirus, the app's own reload) makes it fail with PermissionError."""
    for delay in (0.05, 0.1, 0.2, 0.4):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if sys.platform != "win32":
                raise
            time.sleep(delay)
    os.replace(src, dst)


def save_languages(languages: list, path: Path | None = None) -> None:
    """Write only `transcription.languages` (see `save_keys`)."""
    save_keys(path, {"transcription.languages": list(languages)})


# Keys the native settings window may change: flat name -> dotted config key.
SETTINGS_KEYS = {
    "mode": "detection.mode",
    "output_dir": "output_dir",
    "languages": "transcription.languages",
    "keep_audio": "transcription.keep_audio",
    "speaker_names": "transcription.speaker_names",
    "live_captions": "live_captions",
}


def validate_settings(values) -> dict:
    """Check values sent by the settings window (untrusted JSON) and return them cleaned:
    ``mode`` as a Mode, ``output_dir`` as an expanded Path, ``languages`` normalised
    ([] = any language), the rest as bools. Unknown names are ignored. Raises ValueError on a
    bad value."""
    if not isinstance(values, dict):
        raise ValueError("settings must be an object")
    out: dict = {}
    for name, value in values.items():
        if name not in SETTINGS_KEYS:
            continue
        if name == "mode":
            out[name] = Mode(value)
        elif name == "output_dir":
            if not isinstance(value, str) or not value.strip():
                raise ValueError("output_dir must be a folder path")
            out[name] = Path(value).expanduser()
            if not out[name].is_absolute():
                raise ValueError("output_dir must be an absolute path")
        elif name == "languages":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError("languages must be a list of codes")
            langs = list(dict.fromkeys(language_code(v) for v in value))  # [] = any language
            clean = normalize_languages(langs)
            if len(clean) != len(langs):
                raise ValueError("languages: unknown language code")
            out[name] = clean
        elif isinstance(value, bool):
            out[name] = value
        else:
            raise ValueError(f"{name} must be true or false")
    return out


def save_settings(updates: dict, path: Path | None = None) -> float:
    """Write validated settings (see validate_settings) into config.toml, keeping every other
    key as it is on disk (see `save_keys`). Returns the new file's mtime."""
    return save_keys(
        path,
        {
            SETTINGS_KEYS[name]: (
                value.value
                if isinstance(value, Mode)
                else (str(value) if isinstance(value, Path) else value)
            )
            for name, value in updates.items()
        },
    )


def load_config(path: Path | None = None) -> Config:
    """Load config, creating a default file on first run. Unknown keys are ignored."""
    path = CONFIG_PATH if path is None else path
    if not path.exists():
        cfg = Config()
        cfg.save(path)
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    det = _build(DetectionConfig, data.pop("detection", {}))
    det.mode = Mode(det.mode)
    det.app_modes = {k: Mode(v) for k, v in det.app_modes.items()}
    for key in ("outside_sensitivity", "inside_sensitivity"):
        v = getattr(det, key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, int | float)):
            log.warning("config: [detection] %s must be a number 0-100; ignoring it", key)
            setattr(det, key, None)
    # Could never trigger: say so. Clamped where it is used (the file keeps the value); with
    # an outside sensitivity the window widens instead.
    if det.outside_sensitivity is None and det.room_speech_s > det.room_window_s:
        log.warning(
            "room_speech_s (%s) is longer than room_window_s (%s); using %s",
            det.room_speech_s,
            det.room_window_s,
            det.room_window_s,
        )
    tr = _build(TranscriptionConfig, data.pop("transcription", {}))
    raw = tr.languages
    tr.languages = normalize_languages(raw)
    if tr.languages != raw:
        tr._raw_languages = raw  # type: ignore[attr-defined]  # written back by save()
    tr.language_models = normalize_language_models(tr.language_models)
    for code in tr.language_models:
        if tr.languages and code not in tr.languages:
            log.warning(
                "transcription.language_models.%s is unused: %s is not in transcription.languages",
                code,
                code,
            )
    summary = _build(SummaryConfig, data.pop("summary", {}))
    cfg = _build(Config, data)
    cfg.output_dir = Path(cfg.output_dir).expanduser()
    cfg.detection, cfg.transcription, cfg.summary = det, tr, summary
    return cfg
