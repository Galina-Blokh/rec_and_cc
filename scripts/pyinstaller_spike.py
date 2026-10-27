"""Measure an unsigned PyInstaller one-folder build of RecAndCC (ADR 0011, follow-up 3).

macOS only (the collected packages are darwin-only; it exits early elsewhere).
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
import os, sys, time
if os.environ.get("RCC_SPIKE_PROBE") == "1":
    t = time.perf_counter()
    failed = []
    for m in ("numpy", "mlx.core", "mlx_whisper", "faster_whisper", "onnxruntime", "rumps"):
        try:
            if m == "mlx_whisper":  # the app's numba-free import path (transcribe/gpu.py)
                from recandcc.transcribe.gpu import import_mlx_whisper

                import_mlx_whisper()
            else:
                __import__(m)
        except Exception as e:
            failed.append(f"{m}: {e!r}")
    if failed:
        sys.exit("probe: failed imports: " + "; ".join(failed))
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


def timed(cmd: list[str], env: dict[str, str], cwd: str) -> float:
    t = time.perf_counter()
    # check=True: a non-zero exit (e.g. a failed probe import) fails the measurement.
    subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, check=True, cwd=cwd)
    return time.perf_counter() - t


def measure(label: str, cmd: list[str], runs: int, probe: bool, cwd: str) -> None:
    env = {**os.environ, "RCC_SPIKE_PROBE": "1" if probe else "0"}
    first = timed(cmd, env, cwd)  # first run after build: the closest to a cold launch without sudo
    warm = [timed(cmd, env, cwd) for _ in range(runs)]
    print(
        f"{label:34s} first {first:6.2f}s  warm median {statistics.median(warm):6.2f}s "
        f"(n={runs}, min {min(warm):.2f}, max {max(warm):.2f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("pyinstaller_spike.py is macOS-only (mlx, rumps); nothing to do here")
    bundle = WORK / "dist" / NAME if args.skip_build else build()
    report_size(bundle)
    exe = bundle / (NAME + (".exe" if sys.platform == "win32" else ""))
    if not exe.is_file():
        raise SystemExit(f"missing frozen executable: {exe}")
    tmp = tempfile.gettempdir()  # frozen binary: run away from the repo
    print()
    measure("pyinstaller --help", [str(exe), "--help"], args.runs, False, tmp)
    # uv needs the project to resolve the environment, so run it from the repo root.
    measure("uv run --help", ["uv", "run", "recandcc", "--help"], args.runs, False, str(ROOT))
    measure("pyinstaller + heavy imports", [str(exe), "--help"], args.runs, True, tmp)


if __name__ == "__main__":
    main()
