"""CPU summaries (onnxruntime-genai) for machines without MLX, with a fake backend and a fake
hub: no real model, no network. The one slow test runs only when the pinned model is already
cached, and never downloads. Issue #4.

Stories: US-27.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import platformdirs
import pytest
from test_summary import SPOKEN

from recandcc.config import APP_NAME, DEFAULT_SUMMARY_MODEL, PREVIOUS_DEFAULT_SUMMARY_MODEL
from recandcc.transcribe import summary, summary_cpu
from recandcc.transcribe.summary_cpu import CpuGenerator, CpuModel, ModelFile

WEIGHTS, TOKENIZER = b"onnx weights" * 100, b'{"tokenizer": 1}'
ANSWER = "We agreed on the plan."


def _files() -> dict[str, bytes]:
    return {"onnx/model.onnx": WEIGHTS, "tokenizer.json": TOKENIZER}


def _fake_model(files: dict[str, bytes] | None = None, sha: dict[str, str] | None = None):
    files = files or _files()
    sha = sha or {}
    return CpuModel(
        repo="onnx-community/fake",
        revision="a" * 40,
        files=tuple(
            ModelFile(p, sha.get(p, hashlib.sha256(b).hexdigest()), len(b))
            for p, b in files.items()
        ),
        config={"model": {"type": "qwen2"}},
    )


@pytest.fixture
def hub(monkeypatch):
    """A fake Hugging Face: serves _files() for the fake pinned model; records each URL."""
    urls: list[str] = []
    state = {"online": True}

    def _open(url):
        urls.append(url)
        if not state["online"]:
            raise urllib.error.URLError("offline")
        return io.BytesIO(_files()[url.split("/" + "a" * 40 + "/", 1)[1]])

    monkeypatch.setattr(summary_cpu, "_open", _open)
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {"m/qwen": _fake_model()})
    monkeypatch.setattr(summary_cpu, "_verified", set())
    monkeypatch.setattr(summary_cpu, "_mapped_logged", set())
    return types.SimpleNamespace(urls=urls, state=state)


class FakeOg:
    """Fake onnxruntime_genai: echoes a fixed answer token by token and records the search
    options. ``n_ids`` is the prompt length in tokens; ``ort`` is the fake ONNX Runtime core."""

    def __init__(self, answer: str = ANSWER) -> None:
        self.options: list[dict] = []
        self.prompts: list[list[dict]] = []
        self.telemetry_off = 0  # the GenAI switch
        self.ort_telemetry_off = 0  # the ONNX Runtime core switch
        self.n_ids = 3
        og = self

        class Tokenizer:
            def __init__(self, model):
                pass

            def apply_chat_template(self, messages, add_generation_prompt):
                og.prompts.append(json.loads(messages))
                return "prompt"

            def encode(self, prompt):
                return [1] * og.n_ids

            def create_stream(self):
                return types.SimpleNamespace(decode=lambda t: answer[t])

        class GeneratorParams:
            def __init__(self, model):
                pass

            def set_search_options(self, **kw):
                og.options.append(kw)

        class Generator:
            def __init__(self, model, params):
                self.i = -1

            def append_tokens(self, ids):
                pass

            def is_done(self):
                return self.i + 1 >= len(answer)

            def generate_next_token(self):
                self.i += 1

            def get_next_tokens(self):
                return [self.i]

        def disable_telemetry_events():
            og.telemetry_off += 1

        def ort_disable_telemetry_events():
            og.ort_telemetry_off += 1

        self.module = types.SimpleNamespace(
            Model=lambda path: ("model", path),
            Tokenizer=Tokenizer,
            GeneratorParams=GeneratorParams,
            Generator=Generator,
            disable_telemetry_events=disable_telemetry_events,
        )
        self.ort = types.SimpleNamespace(disable_telemetry_events=ort_disable_telemetry_events)


@pytest.fixture
def og(monkeypatch):
    fake = FakeOg()
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", fake.module)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake.ort)
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: True)
    return fake


def test_default_and_previous_default_map_to_the_pinned_1p5b_build():
    build = summary_cpu.QWEN2_5_1_5B
    assert {
        DEFAULT_SUMMARY_MODEL: build,  # the Qwen3.5 4B has no CPU build
        PREVIOUS_DEFAULT_SUMMARY_MODEL: build,
    } == summary_cpu.CPU_MODELS
    assert set(summary_cpu.DOWNLOAD_SIZES) == set(summary_cpu.CPU_MODELS)
    assert set(summary_cpu.DOWNLOAD_SIZES.values()) == {"~1.8 GB"}
    cpu = build
    assert re.fullmatch(r"[0-9a-f]{40}", cpu.revision)
    assert cpu.repo == "onnx-community/Qwen2.5-1.5B-Instruct"
    for f in cpu.files:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
        assert cpu.url(f).startswith(f"https://huggingface.co/{cpu.repo}/resolve/{cpu.revision}/")
    assert cpu.config["model"]["decoder"]["filename"] in {f.path for f in cpu.files}


def test_pinned_download_once_then_offline(hub, og):
    notices = []
    gen = CpuGenerator("m/qwen", on_download=lambda: notices.append(1))
    assert gen([{"role": "user", "content": SPOKEN}], 50) == ANSWER
    folder = summary_cpu.CPU_MODELS["m/qwen"].folder
    assert (folder / "onnx/model.onnx").read_bytes() == WEIGHTS
    assert json.loads((folder / "genai_config.json").read_text("utf-8")) == {
        "model": {"type": "qwen2"}
    }
    assert notices == [1] and len(hub.urls) == 2
    assert all(
        u.startswith("https://huggingface.co/onnx-community/fake/resolve/") for u in hub.urls
    )
    gen.close()
    hub.state["online"] = False  # on disk: loads without the network
    summary_cpu._verified.clear()  # a new process re-checks the sha256s, still offline
    assert CpuGenerator("m/qwen")([{"role": "user", "content": "x"}], 50)
    assert len(hub.urls) == 2 and og.telemetry_off == 2 and og.ort_telemetry_off == 2


def test_checksum_mismatch_keeps_nothing(hub, og, monkeypatch):
    bad = _fake_model(sha={"onnx/model.onnx": "0" * 64})
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {"m/qwen": bad})
    with pytest.raises(summary_cpu.ModelUnavailable):
        CpuGenerator("m/qwen").load()
    assert not [p for p in bad.folder.rglob("*") if p.is_file() and "model" in p.name]


def test_corrupt_file_on_disk_is_downloaded_again(hub, og):
    CpuGenerator("m/qwen").load()
    path = summary_cpu.CPU_MODELS["m/qwen"].folder / "onnx/model.onnx"
    path.write_bytes(b"x" * len(WEIGHTS))  # same size, wrong content
    summary_cpu._verified.clear()
    CpuGenerator("m/qwen").load()
    assert path.read_bytes() == WEIGHTS and len(hub.urls) == 3


def test_oversized_download_is_refused(hub, og, monkeypatch):
    small = _fake_model(files={"onnx/model.onnx": WEIGHTS[:10], "tokenizer.json": TOKENIZER})
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {"m/qwen": small})
    with pytest.raises(summary_cpu.ModelUnavailable):
        CpuGenerator("m/qwen").load()


def test_search_options_and_prompt(hub, og):
    msgs = summary.build_messages("summary", "Me: hi", "en")
    CpuGenerator("m/qwen")(msgs, 77)
    assert og.prompts == [msgs]
    assert og.options == [{"max_length": 3 + 77, "do_sample": False, "repetition_penalty": 1.1}]


def test_mapping_is_logged_once_with_model_names_only(hub, og, caplog):
    with caplog.at_level(logging.INFO, logger="recandcc"):
        CpuGenerator("m/qwen").load()
        CpuGenerator("m/qwen").load()
    assert caplog.text.count("m/qwen runs on the CPU as onnx-community/fake") == 1
    assert "a smaller model" in caplog.text


def test_model_without_cpu_build_or_local_folder(hub, og, tmp_path):
    with pytest.raises(summary_cpu.ModelUnavailable, match="no CPU build"):
        CpuGenerator("mlx-community/gemma-4-e4b-it-4bit").load()
    (tmp_path / "genai_config.json").write_text("{}", "utf-8")
    gen = CpuGenerator(str(tmp_path))
    gen.load()
    assert gen._model == ("model", str(tmp_path)) and hub.urls == []


def test_without_onnxruntime_genai_summaries_are_unavailable(monkeypatch):
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: False)
    with pytest.raises(summary.SummaryUnavailable, match="onnxruntime-genai"):
        CpuGenerator("m/qwen").load()


def test_telemetry_switches_are_called_on_every_load_and_are_optional(hub, og, monkeypatch):
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")  # restored after the test
    CpuGenerator("m/qwen").load()
    assert (og.telemetry_off, og.ort_telemetry_off) == (1, 1)
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    delattr(og.module, "disable_telemetry_events")  # a runtime without the switch still loads
    CpuGenerator("m/qwen").load()
    assert (og.telemetry_off, og.ort_telemetry_off) == (1, 2)


@pytest.mark.parametrize("module", ["onnxruntime", "onnxruntime_genai"])
def test_a_runtime_that_cannot_be_imported_skips_summaries_before_any_download(
    hub, og, monkeypatch, module
):
    monkeypatch.setitem(sys.modules, module, None)  # importing it raises ImportError
    with pytest.raises(summary.ModelUnavailable, match="can't be imported"):
        CpuGenerator("m/qwen").load()
    assert hub.urls == []  # nothing is downloaded for a runtime that can't run


def test_stale_part_files_are_removed_when_a_download_starts(hub, og, monkeypatch):
    folder = summary_cpu.CPU_MODELS["m/qwen"].folder
    (folder / "onnx").mkdir(parents=True)
    gone, running, stalled = 4_000_001, 4_000_002, 4_000_003
    monkeypatch.setattr(summary_cpu, "_pid_alive", lambda pid: pid in {running, stalled})
    dead_part = folder / f".tokenizer.json.part.{gone}"  # its process is gone
    own_part = folder / f".other.bin.part.{os.getpid()}"  # left behind by this process
    running_part = folder / "onnx" / f".model.onnx.part.{running}"  # fresh, process alive
    stalled_part = folder / "onnx" / f".model.onnx.part.{stalled}"  # alive, idle for an hour
    odd_part = folder / ".odd.part.notapid"  # fresh, pid unknown: kept
    for part in (dead_part, own_part, running_part, stalled_part, odd_part):
        part.write_bytes(b"partial")
    old = time.time() - 2 * summary_cpu.STALE_PART_S
    os.utime(stalled_part, (old, old))
    CpuGenerator("m/qwen").load()
    assert not dead_part.exists() and not own_part.exists() and not stalled_part.exists()
    assert running_part.exists() and odd_part.exists()
    assert len(hub.urls) == 2  # the download itself went on


def test_pid_alive_sees_this_process_and_an_exited_child():
    assert summary_cpu._pid_alive(os.getpid())
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not summary_cpu._pid_alive(child.pid)


def test_genai_config_is_replaced_whole_or_not_at_all(tmp_path, monkeypatch):
    path = tmp_path / summary_cpu.CONFIG_NAME
    path.write_text('{"old": true}', "utf-8")

    def replace_fails(src, dst):
        raise OSError("disk full")

    with monkeypatch.context() as mp:
        mp.setattr(summary_cpu.os, "replace", replace_fails)
        with pytest.raises(OSError, match="disk full"):
            summary_cpu._write_atomic(path, '{"new": true}')
    assert path.read_text("utf-8") == '{"old": true}'  # the old config survives
    assert not list(tmp_path.glob(".*.part.*"))  # and no temp file is left behind
    summary_cpu._write_atomic(path, '{"new": true}')
    assert path.read_text("utf-8") == '{"new": true}'
    assert not list(tmp_path.glob(".*.part.*"))


class _Response(io.BytesIO):
    """A fake HTTP response: a status, headers and an empty body."""

    def __init__(self, url, status, headers):
        super().__init__(b"")
        self.url, self.code, self.msg = url, status, "fake"
        self._headers = headers

    def info(self):
        return self._headers


def _fake_transport(routes):
    """An opener that answers https URLs from ``routes`` (url -> (status, headers)) and records
    each request, so redirects are followed without any network."""
    requested: list[str] = []

    class FakeHttps(urllib.request.BaseHandler):
        def https_open(self, req):
            requested.append(req.full_url)
            status, headers = routes[req.full_url]
            return _Response(req.full_url, status, headers)

    opener = urllib.request.OpenerDirector()
    for handler in (
        FakeHttps(),
        summary_cpu._HttpsOnlyRedirect(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener, requested


def test_https_only_redirect_refuses_http_and_follows_https():
    start = "https://huggingface.co/onnx-community/x/resolve/rev/model.onnx"
    refused, requested = _fake_transport(
        {start: (302, {"location": "http://cdn.example.net/model.onnx"})}
    )
    with pytest.raises(urllib.error.URLError, match="non-https"):
        refused.open(start, timeout=5)
    assert requested == [start]  # the http URL is never requested

    target = "https://cdn.example.net/model.onnx"
    allowed, requested = _fake_transport({start: (302, {"location": target}), target: (200, {})})
    assert allowed.open(start, timeout=5).code == 200
    assert requested == [start, target]


def test_open_refuses_a_non_https_url():
    with pytest.raises(ValueError, match="non-https"):
        summary_cpu._open("http://huggingface.co/x")


def test_pause_runs_before_each_token_and_can_stop_the_answer(hub, og):
    waits: list[int] = []
    CpuGenerator("m/qwen", pause=lambda: waits.append(1))([{"role": "user", "content": "x"}], 50)
    assert len(waits) == len(ANSWER)  # once per generated token

    def stop_on_the_fifth():
        waits.append(1)
        if len(waits) == 5:
            raise RuntimeError("recording started")

    waits.clear()
    with pytest.raises(RuntimeError, match="recording started"):
        CpuGenerator("m/qwen", pause=stop_on_the_fifth)([{"role": "user", "content": "x"}], 50)
    assert len(waits) == 5


def test_a_prompt_too_long_for_the_context_is_refused_not_cut(hub, og):
    og.n_ids = summary_cpu.CONTEXT_LENGTH - 50 + 1  # one token more than fits with 50 asked
    with pytest.raises(summary.SummaryUnavailable, match="too long"):
        CpuGenerator("m/qwen")([{"role": "user", "content": "x"}], 50)
    assert og.options == []  # nothing was generated
    og.n_ids = summary_cpu.CONTEXT_LENGTH - 50  # exactly fits
    assert CpuGenerator("m/qwen")([{"role": "user", "content": "x"}], 50) == ANSWER


def _cached(path: Path, f: ModelFile) -> bool:
    """On disk with the pinned size and sha256 (read only: nothing is deleted or downloaded)."""
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest() == f.sha256


@pytest.mark.slow
def test_the_pinned_model_summarizes_when_installed_and_cached(monkeypatch):
    if importlib.util.find_spec("onnxruntime_genai") is None:
        pytest.skip("onnxruntime-genai isn't installed here (it isn't on macOS)")
    real_cache = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False)) / "models"
    monkeypatch.setattr(summary_cpu, "MODELS_DIR", real_cache)  # conftest uses a temp dir
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: True)
    model = summary_cpu.CPU_MODELS[DEFAULT_SUMMARY_MODEL]
    if not all(_cached(model.folder / f.path, f) for f in model.files):
        pytest.skip("the pinned CPU model isn't cached; this test never downloads it")

    def no_network(*args, **kwargs):
        raise AssertionError("the slow test must not download")

    monkeypatch.setattr(summary_cpu, "_open", no_network)
    monkeypatch.setattr(summary_cpu, "_download", no_network)
    gen = CpuGenerator(DEFAULT_SUMMARY_MODEL)
    try:
        answer = gen(
            [{"role": "user", "content": "Summarize: Me: We ship on Friday. Others: Agreed."}], 40
        )
    finally:
        gen.close()
    assert answer.strip()
