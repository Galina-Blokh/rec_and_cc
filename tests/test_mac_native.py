"""Integration tests for the macOS native helper (rcc-audiotap). Skipped elsewhere.

Stories: US-01, US-08, US-16.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

if sys.platform != "darwin":
    pytest.skip("macOS only", allow_module_level=True)

from recandcc.native import HelperNotFoundError, find_helper  # noqa: E402

try:
    HELPER = find_helper()
except HelperNotFoundError:
    pytest.skip("rcc-audiotap not built (run native/macos/build.sh)", allow_module_level=True)

from recandcc.capture.system_mac import MacSystemAudioSource  # noqa: E402
from recandcc.detect.mac import MacActivityMonitor  # noqa: E402
from recandcc.models import AppTarget  # noqa: E402

# Hosted CI runners have no audio output device and can't grant the System Audio Recording
# permission, so a process tap there never delivers IO. These tests run on real Macs only.
needs_audio_hardware = pytest.mark.skipif(
    os.environ.get("CI") == "true", reason="no audio hardware / TCC grant on CI runners"
)


@pytest.fixture
def speech_file(tmp_path):
    path = tmp_path / "speech.aiff"
    subprocess.run(
        [
            "say",
            "-o",
            str(path),
            "Hello. This is a test of the RecAndCC audio helper. One two three four five.",
        ],
        check=True,
        timeout=30,
    )
    return path


def _play(path):
    return subprocess.Popen(["afplay", str(path)])


def _wait_ready(monitor, timeout=10.0):
    """Block until the helper has produced its first line (Core Audio init can be slow)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if monitor._latest_at > 0:
            return True
        time.sleep(0.05)
    return False


def test_monitor_detects_afplay(tmp_path):
    monitor = MacActivityMonitor(interval=0.25)
    player = None
    try:
        # a live helper prints within a few seconds even on a loaded runner; silence for 20 s
        # means it hangs, which is a real failure, not a reason to skip
        assert _wait_ready(monitor, timeout=20.0), "rcc-audiotap monitor printed nothing in 20 s"
        # Long enough that the clip cannot end before the monitor sees it.
        player = _play(_tone(tmp_path, 440, seconds=60))
        found = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and found is None:
            for proc in monitor.snapshot():
                if proc.pid == player.pid:
                    found = proc
            time.sleep(0.2)
        if found is None and player.poll() is not None:
            pytest.skip("afplay exited before it could be observed")
        assert found is not None, "afplay was not reported by rcc-audiotap monitor"
        assert found.output_active
        assert found.name == "afplay"
    finally:
        if player is not None:
            player.kill()
            player.wait()
        monitor.close()


def test_monitor_cli_outputs_json_lines():
    proc = subprocess.Popen(
        [str(HELPER), "monitor", "--interval", "0.2"], stdout=subprocess.PIPE, text=True
    )
    try:
        import json

        line = proc.stdout.readline()
        assert isinstance(json.loads(line), list)
    finally:
        proc.kill()
        proc.wait()


@needs_audio_hardware
def test_capture_yields_float32_stream(speech_file):
    chunks: list[np.ndarray] = []
    lock = threading.Lock()

    def on_frames(frames: np.ndarray) -> None:
        with lock:
            chunks.append(frames)

    source = MacSystemAudioSource()
    source.start(on_frames)
    time.sleep(0.5)
    player = _play(speech_file)
    try:
        player.wait(timeout=20)
        time.sleep(0.3)
    finally:
        player.kill()
        source.stop()
    source.stop()  # idempotent

    assert chunks, "no audio received from rcc-audiotap capture"
    audio = np.concatenate(chunks)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) > 16_000  # more than a second of 16 kHz audio
    assert np.isfinite(audio).all()
    # Silence is acceptable here (the System Audio Recording permission may be missing),
    # but report what we saw.
    print(f"captured {len(audio) / 16000:.2f}s, peak {np.abs(audio).max():.4f}")


@needs_audio_hardware
def test_capture_exits_when_stdout_closes():
    proc = subprocess.Popen(
        [str(HELPER), "capture"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    proc.stdout.read(6400)
    proc.stdout.close()
    assert proc.wait(timeout=5) == 0


def test_permissions_status_json():
    from recandcc import permissions

    result = permissions.status()
    assert set(result) == {"microphone", "system_audio"}
    assert result["microphone"] in {"granted", "denied", "undetermined", "restricted"}
    assert result["system_audio"] in {"granted", "denied", "undetermined", "unknown"}


def _fake_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_find_helper_prefers_repo_build_over_app_bundle(tmp_path, monkeypatch):
    """Rebuilt helpers in <repo>/bin win, so the app bundle (and its signature) never changes."""
    from recandcc import native

    repo_helper = _fake_exe(tmp_path / "repo" / "bin" / "rcc-audiotap")
    bundle_helper = _fake_exe(tmp_path / "RecAndCC.app" / "Contents" / "MacOS" / "rcc-audiotap")
    monkeypatch.delenv("RCC_AUDIOTAP", raising=False)
    monkeypatch.setenv("RCC_APP_BUNDLE", str(tmp_path / "RecAndCC.app"))
    monkeypatch.setattr(native, "REPO_ROOT", tmp_path / "repo")
    assert find_helper() == repo_helper
    repo_helper.unlink()  # fallback: helper bundled by older builds of the app
    assert find_helper() == bundle_helper


@needs_audio_hardware
def test_request_system_audio_does_not_crash():
    """Regression: the TCC completion block escaped a non-escaping closure → SIGTRAP (exit -5).

    Only runs when the calling app already has the permission, so no dialog appears.
    """
    from recandcc import permissions

    if permissions.status()["system_audio"] != "granted":
        pytest.skip("would show a permission prompt")
    result = subprocess.run(
        [str(HELPER), "request", "system_audio", "--timeout", "5"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert '"system_audio":"granted"' in result.stdout


# --- per-app tap (US-08) ------------------------------------------------------------------
def _tone(tmp_path, freq, seconds=6.0):
    import soundfile as sf

    rate = 48_000
    t = np.arange(int(rate * seconds)) / rate
    path = tmp_path / f"tone{int(freq)}.wav"
    sf.write(path, (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32), rate)
    return path


def _band_share(audio, freq):
    """Share of the signal's energy within 50 Hz of ``freq``."""
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    hz = np.fft.rfftfreq(len(audio), 1 / 16_000)
    return spectrum[np.abs(hz - freq) < 50].sum() / (spectrum.sum() + 1e-12)


def _playing(path):
    """afplay ``path``, returned once Core Audio reports it playing (it has a process object)."""
    monitor = MacActivityMonitor(interval=0.2)
    player = None
    try:
        _wait_ready(monitor)  # start afplay only once the helper is live
        player = _play(path)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if any(p.pid == player.pid and p.output_active for p in monitor.snapshot()):
                return player
            time.sleep(0.1)
    except BaseException:
        if player is not None:
            player.kill()
            player.wait()
        raise
    finally:
        monitor.close()
    player.kill()
    player.wait()
    pytest.skip("afplay never started playing")


def _record(source, during):
    chunks: list[np.ndarray] = []
    source.start(chunks.append)
    try:
        time.sleep(0.5)
        during()
        time.sleep(1.0)
    finally:
        source.stop()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def _say():
    subprocess.run(["say", "A second process that the per app tap must not record."], timeout=30)


def _require_system_audio():
    from recandcc import permissions

    if permissions.status()["system_audio"] != "granted":
        pytest.skip("System Audio Recording permission not granted to this terminal")


@needs_audio_hardware
def test_per_app_tap_records_only_the_target_app(tmp_path):
    _require_system_audio()
    tone = _tone(tmp_path, 1000)
    results = {}
    for name, target in (("all", None), ("app", True)):
        player = _playing(tone)
        try:
            app = AppTarget(pids=(player.pid,)) if target else None
            results[name] = _record(MacSystemAudioSource(target=app), _say)
        finally:
            player.kill()
            player.wait()
    everything, only_app = results["all"], results["app"]
    if np.abs(everything).max() < 1e-4:
        pytest.skip("the global tap got only silence (no permission or no output device)")
    if _band_share(everything, 1000) > 0.98:
        pytest.skip("`say` was not audible to the global tap")
    assert np.abs(only_app).max() > 0.05, "the per-app tap missed afplay"
    assert _band_share(only_app, 1000) > 0.98, "the per-app tap recorded `say` too"


@needs_audio_hardware
def test_per_app_tap_follows_an_added_target(tmp_path):
    """add_target (a call joining a media session) extends the running tap to a new process."""
    _require_system_audio()
    first = _playing(_tone(tmp_path, 1000, seconds=1.5))
    source = MacSystemAudioSource(target=AppTarget(pids=(first.pid,)))
    second = None

    def add_second():
        nonlocal second
        first.wait(timeout=10)
        second = _playing(_tone(tmp_path, 2000, seconds=4.0))
        assert source.add_target(AppTarget(pids=(second.pid,)))
        time.sleep(2.5)  # the helper re-checks the process set every second

    try:
        audio = _record(source, add_second)
    finally:
        for player in (first, second):
            if player is not None:
                player.kill()
                player.wait()
    if np.abs(audio).max() < 1e-4:
        pytest.skip("the tap got only silence (no permission or no output device)")
    assert _band_share(audio, 2000) > 0.2, "the added process was not recorded"


@needs_audio_hardware
def test_per_app_tap_without_a_process_falls_back_to_everything():
    from recandcc.capture import system_mac

    assert system_mac._supports_targets(HELPER, HELPER.stat().st_mtime)
    _require_system_audio()
    source = MacSystemAudioSource(target=AppTarget(pids=(999_999,)))  # no such process
    audio = _record(source, lambda: time.sleep(0.5))
    assert source.scope == "all" and len(audio) > 8_000
    assert source.add_target(AppTarget(pids=(1,))) is False
