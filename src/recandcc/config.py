"""User configuration, stored as TOML in the platform config dir."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import platformdirs
import tomli_w

from .models import Mode

APP_NAME = "RecAndCC"
CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME, appauthor=False))
CONFIG_PATH = CONFIG_DIR / "config.toml"
LOG_DIR = Path(platformdirs.user_log_dir(APP_NAME, appauthor=False))
DEFAULT_OUTPUT_DIR = Path(platformdirs.user_documents_dir()) / "Transcribes"

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
    call_apps: list[str] = field(default_factory=lambda: list(DEFAULT_CALL_APPS))
    media_apps: list[str] = field(default_factory=lambda: list(DEFAULT_MEDIA_APPS))
    ignore_apps: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_APPS))
    # Per-app override of `mode`, keyed by pattern, e.g. {"us.zoom" = "auto"}.
    app_modes: dict[str, Mode] = field(default_factory=dict)
    # Room listening (opt-in, off by default): keep the mic open and record a conversation
    # near the computer when no app is doing audio. See docs/CONFIGURATION.md.
    room_listening: bool = False
    room_min_speech_s: float = 4.0  # this much speech ...
    room_window_s: float = 10.0  # ... within this window starts a room session
    room_preroll_s: float = 15.0  # audio before the trigger kept in the session
    room_stop_after_silence_s: float = 90.0  # no speech this long ends a room session
    room_vad_threshold: float = 0.5  # Silero speech probability that counts as speech


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
    languages: list[str] = field(default_factory=lambda: ["en", "ru"])  # auto-detect among these
    workers: int = 0  # parallel transcription workers, 0 = auto
    dedupe_echo: bool = True  # drop mic segments that are just speaker echo of "Others"
    keep_audio: bool = True  # False = delete WAVs after a successful transcription
    # Delete a recording's audio this many days after it was transcribed (0 = keep forever).
    # Transcripts are never deleted. Also settable from the menu: "Delete audio after".
    audio_retention_days: int = 0
    # Split "Others" into "Speaker 1", "Speaker 2", ... in the final transcript (offline
    # speaker diarization). Live captions keep showing "Others".
    diarize: bool = True
    # Transcribe while recording: text appears ~1-2 s after each phrase, and the final
    # transcript is ready right when the session stops (no post-session pass).
    live: bool = True
    # Allow downloading Whisper repos without a pinned revision (custom HF repo ids). Off by
    # default: upstream could swap their weights. Already-downloaded copies load either way.
    allow_unpinned_models: bool = False


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

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        path.write_bytes(tomli_w.dumps(data).encode())


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


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
    tr = _build(TranscriptionConfig, data.pop("transcription", {}))
    cfg = _build(Config, data)
    cfg.output_dir = Path(cfg.output_dir).expanduser()
    cfg.detection, cfg.transcription = det, tr
    return cfg
