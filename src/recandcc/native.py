"""Locating the native helper binaries (macOS: rcc-audiotap)."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HELPER_NAME = "rcc-audiotap"
REPO_ROOT = Path(__file__).resolve().parents[2]


class HelperNotFoundError(FileNotFoundError):
    pass


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def find_helper(name: str = HELPER_NAME) -> Path:
    """Return the path to the native helper.

    Search order:
      1. $RCC_AUDIOTAP (explicit override)
      2. <repo>/bin/<name> (output of native/macos/build.sh)
      3. $RCC_APP_BUNDLE/Contents/MacOS/<name> (helper bundled by older builds of the app)
      4. next to sys.executable
      5. $PATH

    The helper deliberately lives outside RecAndCC.app: TCC attributes its audio access to
    the *responsible* app (the launcher that spawned us), wherever the binary is. Keeping it
    out of the bundle means rebuilding the helper doesn't change the app's code signature,
    which would make macOS treat RecAndCC as a new app and drop its permissions.
    """
    candidates: list[Path] = []
    if env := os.environ.get("RCC_AUDIOTAP"):
        override = Path(env).expanduser()
        if not _is_executable(override):
            raise HelperNotFoundError(
                f"RCC_AUDIOTAP points to {override}, which is not an executable file"
            )
        return override
    candidates.append(REPO_ROOT / "bin" / name)
    if bundle := os.environ.get("RCC_APP_BUNDLE"):
        candidates.append(Path(bundle) / "Contents" / "MacOS" / name)
    candidates.append(Path(sys.executable).parent / name)
    for candidate in candidates:
        if _is_executable(candidate):
            return candidate
    if found := shutil.which(name):
        return Path(found)
    raise HelperNotFoundError(
        f"Native helper '{name}' not found (looked in: {', '.join(map(str, candidates))} and $PATH). "  # noqa: E501
        f"Build it with: {REPO_ROOT / 'native' / 'macos' / 'build.sh'}"
    )
