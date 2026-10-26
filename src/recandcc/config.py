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
    "rcc-audiotap",
]


@dataclass
class DetectionConfig:
    mode: Mode = Mode.ASK
    poll_interval_s: float = 1.0
    min_active_s: float = 3.0  # audio must be active this long before we trigger
    stop_after_inactive_s: float = 20.0  # stop when the trigger app stopped doing audio I/O
    stop_after_silence_s: float = 180.0  # stop when everything has been silent this long
    silence_dbfs: float = -50.0  # below this level a frame counts as silence
    ask_timeout_s: float = 30.0  # "record?" prompt auto-dismisses after this
    record_mic: str = "calls"  # "always" | "calls" | "never"
    call_apps: list[str] = field(default_factory=lambda: list(DEFAULT_CALL_APPS))
    media_apps: list[str] = field(default_factory=lambda: list(DEFAULT_MEDIA_APPS))
    ignore_apps: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_APPS))
    # Per-app override of `mode`, keyed by pattern, e.g. {"us.zoom" = "auto"}.
    app_modes: dict[str, Mode] = field(default_factory=dict)


@dataclass
class TranscriptionConfig:
    model: str = "large-v3-turbo"  # any faster-whisper model name or local path
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    # "float32" uses Apple Accelerate and benchmarked ~20% faster than "int8" on Apple silicon
    # (at ~3 GB RAM instead of ~1 GB); int8 is the better trade-off on typical Windows laptops.
    compute_type: str = "float32" if sys.platform == "darwin" else "int8"
    languages: list[str] = field(default_factory=lambda: ["en", "ru"])  # auto-detect among these
    workers: int = 0  # parallel transcription workers, 0 = auto
    dedupe_echo: bool = True  # drop mic segments that are just speaker echo of "Others"
    keep_audio: bool = True  # False = delete WAVs after a successful transcription


@dataclass
class Config:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    autostart: bool = False
    hotkey: str = "<cmd>+<shift>+r"
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
    data = tomllib.loads(path.read_text())
    det = _build(DetectionConfig, data.pop("detection", {}))
    det.mode = Mode(det.mode)
    det.app_modes = {k: Mode(v) for k, v in det.app_modes.items()}
    tr = _build(TranscriptionConfig, data.pop("transcription", {}))
    cfg = _build(Config, data)
    cfg.output_dir = Path(cfg.output_dir).expanduser()
    cfg.detection, cfg.transcription = det, tr
    return cfg
