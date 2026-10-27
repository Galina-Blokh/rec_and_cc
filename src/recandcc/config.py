"""User configuration, stored as TOML in the platform config dir."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import platformdirs
import tomli_w

from .models import Mode

log = logging.getLogger(__name__)

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
    # Remember speakers' voices so that a speaker named once (`recandcc name-speaker`) is named
    # in later sessions. Off by default: voice profiles are biometric data. See voices.py.
    speaker_names: bool = False
    # Transcribe while recording: text appears ~1-2 s after each phrase, and the final
    # transcript is ready right when the session stops (no post-session pass).
    live: bool = True
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
    output_dir: Path = DEFAULT_OUTPUT_DIR
    autostart: bool = False
    # Global start/stop shortcut. ⌃⌥R by default: ⌘⇧R would override Chrome/Safari shortcuts.
    hotkey: str = "<ctrl>+<alt>+r"
    # Floating captions window while recording (needs transcription.live).
    live_captions: bool = True
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)

    def save(self, path: Path = CONFIG_PATH) -> None:
        """Write the settings. Serialized (the snapshot is taken under the lock, so the last
        save writes the current values) and atomic where the OS allows it."""
        with CONFIG_LOCK:
            data = asdict(self)
            data["output_dir"] = str(self.output_dir)
            # Keep the user's own spelling of `languages` ("iw", unknown codes) unless the list
            # was changed since loading: load_config only normalises the in-memory copy.
            tr = self.transcription
            with LANG_LOCK:  # the pair, as the menu set it (lock order: CONFIG_LOCK first)
                raw = getattr(tr, "_raw_languages", None)
                langs = list(tr.languages)
            data["transcription"]["languages"] = langs
            if raw is not None and normalize_languages(raw) == langs:
                data["transcription"]["languages"] = raw
            for section in ("detection", "transcription"):  # TOML has no null: unset = omitted
                data[section] = {k: v for k, v in data[section].items() if v is not None}
            blob = tomli_w.dumps(data).encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(path, blob)
            _own_writes[str(path)] = (hashlib.sha256(blob).hexdigest(), len(blob))


# Held while saving and while applying a reload, so neither sees the other half-done.
# Re-entrant: load_config() saves the defaults on first run.
CONFIG_LOCK = threading.RLock()
# transcription.languages + its raw spelling change together (menu, reload, save).
LANG_LOCK = threading.Lock()
_own_writes: dict[str, tuple[str, int]] = {}  # path -> signature of what our last save wrote


def file_signature(path: Path) -> tuple[str, int] | None:
    """(sha256, size) of the file's content, None if it can't be read. By content, not
    mtime: a same-size hand edit within a coarse mtime tick is still noticed."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest(), len(data)


def written_by_us(path: Path) -> bool:
    """Whether the file is exactly as our last save left it (not a hand edit)."""
    sig = file_signature(path)
    return sig is not None and _own_writes.get(str(path)) == sig


def _write_atomic(path: Path, blob: bytes) -> None:
    """Temp file + os.replace, so a crash never leaves a torn file. On Windows the replace
    fails while another handle has the file open (a reader, antivirus): retry briefly, then
    write in place rather than lose the setting."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(blob)
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.05 * (attempt + 1))
        log.warning("config: could not replace %s atomically; writing it in place", path.name)
        path.write_bytes(blob)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:  # e.g. still locked on Windows: a stray temp file, not a failed save
            log.debug("config: could not remove %s", tmp.name)


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


def save_languages(languages: list, path: Path = CONFIG_PATH) -> None:
    """Write only `transcription.languages` into config.toml, leaving every other key as it is
    on disk (it may have been edited by hand since the app loaded it). Raises
    FileNotFoundError if there is no file yet, ValueError if it is not valid TOML."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data.setdefault("transcription", {})["languages"] = list(languages)
    path.write_bytes(tomli_w.dumps(data).encode())


def load_config(path: Path = CONFIG_PATH) -> Config:
    """Load config, creating a default file on first run. Unknown keys are ignored."""
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
    tr = _build(TranscriptionConfig, data.pop("transcription", {}))
    raw = tr.languages
    tr.languages = normalize_languages(raw)
    if tr.languages != raw:
        tr._raw_languages = raw  # type: ignore[attr-defined]  # written back by save()
    summary = _build(SummaryConfig, data.pop("summary", {}))
    cfg = _build(Config, data)
    cfg.output_dir = Path(cfg.output_dir).expanduser()
    cfg.detection, cfg.transcription, cfg.summary = det, tr, summary
    return cfg
