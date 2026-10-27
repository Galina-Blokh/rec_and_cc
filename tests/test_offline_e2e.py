"""End to end with the network blocked: record a session, transcribe it, write the transcript.

The fast tests run the real Controller, Engine, Recorder, transcription queue, live
transcriber, VAD, merge, outputs and diarization fallback with fake audio sources and a fake
Whisper model. Blocked, and recorded as an attempt:
- ``socket.connect`` / ``connect_ex`` (loopback included, so a local proxy can't relay
  traffic; AF_UNIX and the loopback pair ``socket.socketpair`` builds on Windows are allowed),
- ``sendto`` / ``sendmsg`` (UDP never calls connect), ``socket.create_connection``,
- DNS: ``getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``,
- launching programs (``subprocess.Popen``, ``os.system``), e.g. a curl; only the app's own
  audio helper (``rcc-audiotap``, permission checks) may run.
Not covered: C extensions that open sockets themselves, and names bound before the test
(``from socket import getaddrinfo``). The only attempt allowed is the one-time speaker-model
download in ``diarize._download``, and only its connection to the model host; it must fail
fast and leave "Others" in the transcript. The slow tests load the real Whisper models from
disk; with the speaker model on disk too they must make no attempt at all.

Stories: US-14.
"""

from __future__ import annotations

import os
import queue
import socket
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from test_live import FakeTranscriber, burst, phrase, silence
from test_recorder import FakeSource

from recandcc.config import Config, TranscriptionConfig
from recandcc.models import SAMPLE_RATE, Mode
from recandcc.native import HELPER_NAME
from recandcc.recorder import Recorder
from recandcc.transcribe import diarize
from recandcc.ui.common import Controller

REAL_MODEL_PATH = diarize.model_path  # before conftest stubs it out for fast tests
AF_UNIX = getattr(socket, "AF_UNIX", None)
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
MODEL_HOST = urlparse(diarize.EMBEDDING_MODEL.url).hostname
LOOPBACK = ("127.0.0.1", "::1", "localhost")


# --------------------------------------------------------------------------- network guard
def _caller() -> str:
    """Innermost app frame (or, failing that, the frame that called into the socket API)."""
    frames = [f for f in traceback.extract_stack()[:-3] if "test_offline_e2e" not in f.filename]
    app = [f for f in frames if "recandcc" in f.filename.replace("\\", "/")]
    f = (app or frames or traceback.extract_stack()[:1])[-1]
    return f"{f.filename}:{f.lineno} in {f.name}()"


@dataclass
class Attempt:
    api: str
    target: str
    host: str | None
    caller: str
    stack: list[str] = field(repr=False)


class NetworkGuard:
    def __init__(self) -> None:
        self.attempts: list[Attempt] = []

    def deny(self, api: str, target) -> OSError:
        stack = [f.name for f in traceback.extract_stack()]
        caller = _caller()
        host = target[0] if isinstance(target, tuple) and target else None
        self.attempts.append(Attempt(api, repr(target), host, caller, stack))
        return OSError(f"network blocked by test: {api}({target!r}) from {caller}")

    def model_downloads(self) -> list[Attempt]:
        """The speaker-model download's first step towards its own host (it fails there)."""
        return [
            a
            for a in self.attempts
            if "_download" in a.stack
            and "model_path" in a.stack
            and a.api in ("socket.create_connection", "socket.getaddrinfo")
            and a.host == MODEL_HOST
        ]

    def unexpected(self) -> list[Attempt]:
        downloads = self.model_downloads()
        return [a for a in self.attempts if a not in downloads]


@pytest.fixture
def net(monkeypatch):
    """Block the network; fail the test if anything but the speaker-model download tried it
    (attempts on worker threads are often caught and logged, so they are checked here too)."""
    guard = NetworkGuard()
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    before = set(threading.enumerate())

    def local(sock, address) -> bool:
        if AF_UNIX is not None and sock.family == AF_UNIX:
            return True
        # Windows builds socket.socketpair() from a loopback connect (asyncio, multiprocessing)
        in_pair = any(f.name == "socketpair" for f in traceback.extract_stack())
        return in_pair and isinstance(address, tuple) and address[0] in LOOPBACK

    def connect(self, address):
        if local(self, address):
            return real_connect(self, address)
        raise guard.deny("socket.connect", address)

    def connect_ex(self, address):
        if local(self, address):
            return real_connect_ex(self, address)
        raise guard.deny("socket.connect_ex", address)

    def sendto(self, data, *args):
        raise guard.deny("socket.sendto", args[-1] if args else None)

    def sendmsg(self, buffers, ancdata=(), flags=0, address=None):
        raise guard.deny("socket.sendmsg", address)

    def resolver(name):
        def deny(host, *a, **k):
            raise guard.deny(name, (host,))

        return deny

    real_popen_init = subprocess.Popen.__init__

    def popen_init(self, args, *a, **k):
        program = args if isinstance(args, str) else str(args[0])
        if Path(program).name == HELPER_NAME:  # our own audio helper: local, no network
            return real_popen_init(self, args, *a, **k)
        raise guard.deny("subprocess.Popen", (program,))

    def system(command):
        raise guard.deny("os.system", (command,))

    def create_connection(address, *a, **k):
        raise guard.deny("socket.create_connection", address)

    def getaddrinfo(host, port, *a, **k):
        raise guard.deny("socket.getaddrinfo", (host, port))

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    if hasattr(socket.socket, "sendmsg"):  # not on Windows
        monkeypatch.setattr(socket.socket, "sendmsg", sendmsg)
    monkeypatch.setattr(socket, "gethostbyname", resolver("socket.gethostbyname"))
    monkeypatch.setattr(socket, "gethostbyname_ex", resolver("socket.gethostbyname_ex"))
    monkeypatch.setattr(subprocess.Popen, "__init__", popen_init)
    monkeypatch.setattr(os, "system", system)
    yield guard
    # let worker threads the test started finish, so a late attempt is still seen
    for th in set(threading.enumerate()) - before:
        th.join(timeout=5)
    if bad := guard.unexpected():
        pytest.fail(
            "network access while offline:\n"
            + "\n".join(f"  {a.api}({a.target}) from {a.caller}" for a in bad)
        )


@pytest.fixture
def no_speaker_model(tmp_path, monkeypatch):
    """The real model lookup, with the model not on disk: diarization tries its one download."""
    monkeypatch.setattr(diarize, "model_path", REAL_MODEL_PATH)
    monkeypatch.setattr(diarize, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(diarize, "_last_failure", {})


def test_guard_blocks_and_names_the_caller(net):
    with pytest.raises(OSError, match="network blocked by test: socket.create_connection"):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(OSError, match="socket.getaddrinfo"):
        socket.getaddrinfo("example.com", 443)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(OSError):
        s.connect(("192.0.2.1", 443))  # TEST-NET-1: never routable
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(OSError):
        s.connect_ex(("127.0.0.1", 8080))  # a local proxy could relay traffic out
    assert [a.api for a in net.attempts] == [
        "socket.create_connection",
        "socket.getaddrinfo",
        "socket.connect",
        "socket.connect_ex",
    ]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s, pytest.raises(OSError):
        s.sendto(b"x", ("192.0.2.1", 53))  # UDP: no connect
    with pytest.raises(OSError, match="gethostbyname"):
        socket.gethostbyname("example.com")
    with pytest.raises(OSError, match="subprocess.Popen"):
        subprocess.run(["curl", "https://example.com"], check=False)
    with pytest.raises(OSError, match="subprocess.Popen"):  # same name elsewhere is not ours
        subprocess.run(["/tmp/x/curl"], check=False)
    with pytest.raises(OSError, match="os.system"):
        os.system("curl https://example.com")  # noqa: S605 - blocked by the guard
    assert [a.api for a in net.attempts][4:] == [
        "socket.sendto",
        "socket.gethostbyname",
        "subprocess.Popen",
        "subprocess.Popen",
        "os.system",
    ]
    assert "test_guard_blocks_and_names_the_caller" in net.attempts[0].stack
    net.attempts.clear()  # the deliberate attempts above are not a failure


def test_guard_allows_only_the_model_hosts_lookup_from_the_download(net):
    def model_path():
        def _download():
            for host in (MODEL_HOST, "elsewhere.example"):
                with pytest.raises(OSError):
                    socket.getaddrinfo(host, 443)

        _download()

    model_path()
    assert [a.host for a in net.model_downloads()] == [MODEL_HOST]
    assert [a.host for a in net.unexpected()] == ["elsewhere.example"]
    net.attempts.clear()


def test_guard_allows_the_loopback_pair_socketpair_builds(net):
    def socketpair():  # same name as the stdlib's Windows fallback
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            with pytest.raises(OSError):  # really attempted: nothing listens there
                s.connect(("127.0.0.1", 9))

    socketpair()
    assert net.attempts == []


# --------------------------------------------------------------------------- sessions
class Monitor:
    def snapshot(self):
        return []

    def close(self):
        pass


AUDIO = np.concatenate([silence(1), burst(1.5), silence(1.5), burst(1)])


def make_ctrl(tmp_path, transcriber_cfg: TranscriptionConfig, transcriber=None):
    cfg = Config(output_dir=tmp_path / "out")
    cfg.detection.mode = Mode.OFF
    cfg.transcription = transcriber_cfg
    sources = {}

    def sysf():
        sources["sys"] = FakeSource(AUDIO)
        return sources["sys"]

    def micf():
        sources["mic"] = FakeSource(silence(len(AUDIO) / SAMPLE_RATE))
        return sources["mic"]

    rec = Recorder(cfg, mic_factory=micf, system_factory=sysf)
    tr = transcriber or FakeTranscriber(cfg.transcription, sync=transcriber_cfg.live)
    ctrl = Controller(cfg, monitor=Monitor(), recorder=rec, transcriber=tr, start_threads=False)
    ctrl.engine._save_config = lambda: None
    return ctrl, sources


def events_until(ctrl, kind, timeout=30.0):
    out, deadline = [], time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = ctrl.events.get(timeout=0.05)
        except queue.Empty:
            continue
        out.append(ev)
        if ev[0] == kind:
            return out
    raise AssertionError(f"no {kind!r} event; got {[e[0] for e in out]}")


def run_session(ctrl, sources, live_wait=0.6, timeout=20.0):
    """Record the fake audio, stop, and wait until the transcript is written. With the fake
    model the default `timeout` also proves a blocked speaker-model download doesn't hang."""
    try:
        ctrl.engine.start_manual()
        ctrl.engine.tick()
        session_dir = ctrl.engine.recorder.session.directory
        sources["sys"].done.wait(5)
        sources["mic"].done.wait(5)
        time.sleep(live_wait)
        ctrl.engine.stop()
        ctrl.engine.tick()
        return session_dir, events_until(ctrl, "transcribed", timeout)
    finally:
        ctrl.shutdown()


def assert_transcript(session_dir, texts):
    md = (session_dir / "transcript.md").read_text(encoding="utf-8")
    for t in texts:
        assert t in md
    assert "Others" in md and "Speaker 1" not in md  # no speaker model: labels stay "Others"
    assert (session_dir / "transcript.json").is_file()
    assert (session_dir / "transcript.srt").is_file()


def test_live_session_offline(tmp_path, net, no_speaker_model):
    from faster_whisper.vad import get_vad_model

    get_vad_model()  # Silero VAD ships with the package: loads with the network blocked
    ctrl, sources = make_ctrl(tmp_path, TranscriptionConfig(model="fake", workers=2, live=True))
    session_dir, evs = run_session(ctrl, sources)  # within 20 s: the blocked download fails fast
    assert any(e[0] == "live" for e in evs)
    assert evs[-1][1].transcribed
    assert_transcript(session_dir, [phrase(1), phrase(4)])
    assert net.model_downloads()  # tried the speaker model once, then gave up quietly


def test_batch_session_offline(tmp_path, net, no_speaker_model):
    from faster_whisper.vad import get_vad_model

    get_vad_model()
    tcfg = TranscriptionConfig(model="fake", workers=2, live=False, languages=["en"])
    ctrl, sources = make_ctrl(tmp_path, tcfg)  # one language: no detection by the fake model
    session_dir, evs = run_session(ctrl, sources, live_wait=0.0)
    assert not [e for e in evs if e[0].startswith("live")]
    _, session, segments = evs[-1]
    assert session.transcribed and segments
    assert_transcript(session_dir, [s.text for s in segments])
    assert net.model_downloads()


def test_failed_speaker_download_is_not_retried_every_session(tmp_path, net, no_speaker_model):
    assert diarize.model_path() is None
    tried = len(net.model_downloads())
    assert tried
    assert diarize.model_path() is None
    assert len(net.model_downloads()) == tried  # within RETRY_AFTER_S: no second attempt
    assert not (tmp_path / "models" / diarize.EMBEDDING_MODEL.name).exists()


# --------------------------------------------------------------------------- real models
def _cached(repo: str, revision: str | None, faster_whisper: bool) -> bool:
    """On disk as the loaders look it up (faster-whisper fetches only its model files)."""
    try:
        if faster_whisper:
            from faster_whisper.utils import download_model

            download_model(repo, local_files_only=True, revision=revision)
        else:
            from huggingface_hub import snapshot_download

            snapshot_download(repo, revision=revision, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - not on disk
        return False


@pytest.mark.slow
@pytest.mark.parametrize("live", [False, True], ids=["batch", "live"])
def test_real_models_session_offline(tmp_path, net, live):
    from recandcc.transcribe import Transcriber
    from recandcc.transcribe.gpu import mlx_repo
    from recandcc.transcribe.revisions import FASTER_WHISPER_REPOS, revision

    tcfg = TranscriptionConfig(workers=2, live=live)
    tr = Transcriber(tcfg)
    repos = [(mlx_repo(tcfg.model), False) if tr.gpu else (FASTER_WHISPER_REPOS[tcfg.model], True)]
    if tr.language_model:
        repos.append((FASTER_WHISPER_REPOS[tr.language_model], True))
    missing = [r for r, fw in repos if not _cached(r, revision(r), fw)]
    if missing:
        pytest.skip(f"models not cached: {missing}")
    ctrl, sources = make_ctrl(tmp_path, tcfg, transcriber=tr)
    session_dir, evs = run_session(ctrl, sources, live_wait=2.0 if live else 0.0, timeout=120)
    assert any(e[0] == "live_start" for e in evs) == live
    assert evs[-1][1].transcribed
    assert (session_dir / "transcript.md").is_file()
    if REAL_MODEL_PATH(download=False) is not None:
        assert not net.model_downloads()  # speaker model on disk: no network at all
