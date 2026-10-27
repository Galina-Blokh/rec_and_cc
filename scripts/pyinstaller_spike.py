"""Measure an unsigned PyInstaller one-folder build of RecAndCC (ADR 0011, follow-up 3).

Measurement only: no signing, no notarization, no model weights, and it never touches
``RecAndCC.app``. Output goes to ``build/pyinstaller-spike/`` (gitignored). Not run in CI.

    uv run python scripts/pyinstaller_spike.py            # build + measure
    uv run python scripts/pyinstaller_spike.py --runs 7 --skip-build
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "build" / "pyinstaller-spike"
NAME = "recandcc"
# The heavy imports run only when RCC_SPIKE_PROBE=1: `--help` alone does not load them.
ENTRY = """\
import os, time
if os.environ.get("RCC_SPIKE_PROBE") == "1":
    t = time.perf_counter()
    for m in ("numpy", "mlx.core", "mlx_whisper", "faster_whisper", "onnxruntime", "rumps"):
        try:
            __import__(m)
        except Exception:
            pass
    print(f"probe-imports {time.perf_counter() - t:.2f}s")
from recandcc import main
main()
"""
HEAVY = ("mlx", "mlx_whisper", "faster_whisper", "ctranslate2", "onnxruntime", "rumps", "numpy")


def build() -> Path:
    if WORK.is_symlink():
        raise SystemExit(f"refusing to delete symlink {WORK}")
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    entry = WORK / "entry.py"
    entry.write_text(ENTRY)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        NAME,
        "--distpath",
        str(WORK / "dist"),
        "--workpath",
        str(WORK / "work"),
        "--specpath",
        str(WORK),
        "--paths",
        str(ROOT / "src"),
        "--collect-submodules",
        "recandcc",
    ]
    for pkg in HEAVY:
        cmd += ["--collect-all", pkg]
    subprocess.run([*cmd, str(entry)], check=True, cwd=ROOT)
    return WORK / "dist" / NAME


def du_mib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def report_size(bundle: Path) -> None:
    print(f"\nbundle total: {du_mib(bundle):.0f} MiB (excludes models)")
    sizes = {
        str(d.relative_to(bundle)): du_mib(d)
        for d in bundle.rglob("*")
        if d.is_dir() and len(d.relative_to(bundle).parts) <= 2
    }
    for name, mib in sorted(sizes.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {mib:8.0f} MiB  {name}")


def timed(cmd: list[str], env: dict[str, str]) -> float:
    t = time.perf_counter()
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, check=True, cwd=tempfile.gettempdir())
    return time.perf_counter() - t


def measure(label: str, cmd: list[str], runs: int, probe: bool) -> None:
    env = {**os.environ, "RCC_SPIKE_PROBE": "1" if probe else "0"}
    first = timed(cmd, env)  # first run after build: the closest to a cold launch without sudo
    warm = [timed(cmd, env) for _ in range(runs)]
    print(
        f"{label:34s} first {first:6.2f}s  warm median {statistics.median(warm):6.2f}s "
        f"(n={runs}, min {min(warm):.2f}, max {max(warm):.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    bundle = WORK / "dist" / NAME if args.skip_build else build()
    report_size(bundle)
    exe = str(bundle / NAME)
    print()
    measure("pyinstaller --help", [exe, "--help"], args.runs, False)
    measure("uv run --help", ["uv", "run", "recandcc", "--help"], args.runs, False)
    measure("pyinstaller + heavy imports", [exe, "--help"], args.runs, True)


if __name__ == "__main__":
    main()
