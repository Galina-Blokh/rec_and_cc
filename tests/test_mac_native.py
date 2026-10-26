"""Integration tests for the macOS native helper (rcc-audiotap). Skipped elsewhere."""

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


def test_monitor_detects_afplay(speech_file):
    monitor = MacActivityMonitor(interval=0.25)
    player = _play(speech_file)
    try:
        found = None
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and found is None:
            for proc in monitor.snapshot():
                if proc.pid == player.pid:
                    found = proc
            time.sleep(0.2)
        assert found is not None, "afplay was not reported by rcc-audiotap monitor"
        assert found.output_active
        assert found.name == "afplay"
    finally:
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


def test_find_helper_prefers_app_bundle(tmp_path, monkeypatch):
    helper = tmp_path / "RecAndCC.app" / "Contents" / "MacOS" / "rcc-audiotap"
    helper.parent.mkdir(parents=True)
    helper.write_text("#!/bin/sh\n")
    helper.chmod(0o755)
    monkeypatch.delenv("RCC_AUDIOTAP", raising=False)
    monkeypatch.setenv("RCC_APP_BUNDLE", str(tmp_path / "RecAndCC.app"))
    assert find_helper() == helper


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
