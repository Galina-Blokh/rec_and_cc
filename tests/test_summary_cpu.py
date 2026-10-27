"""CPU summaries (onnxruntime-genai) for machines without MLX, with a fake backend and a fake
hub: no real model, no network. Issue #4.

Stories: US-27.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sys
import types
import urllib.error

import pytest
from test_summary import SPOKEN

from recandcc.transcribe import summary, summary_cpu
from recandcc.transcribe.summary_cpu import CpuGenerator, CpuModel, ModelFile

WEIGHTS, TOKENIZER = b"onnx weights" * 100, b'{"tokenizer": 1}'


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
    return types.SimpleNamespace(urls=urls, state=state)


class FakeOg:
    """Fake onnxruntime_genai: echoes a fixed answer token by token and records the search
    options."""

    def __init__(self, answer: str = "We agreed on the plan.") -> None:
        self.options: list[dict] = []
        self.prompts: list[list[dict]] = []
        self.telemetry_off = 0
        og = self

        class Tokenizer:
            def __init__(self, model):
                pass

            def apply_chat_template(self, messages, add_generation_prompt):
                og.prompts.append(json.loads(messages))
                return "prompt"

            def encode(self, prompt):
                return [1, 2, 3]

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

        self.module = types.SimpleNamespace(
            Model=lambda path: ("model", path),
            Tokenizer=Tokenizer,
            GeneratorParams=GeneratorParams,
            Generator=Generator,
            disable_telemetry_events=disable_telemetry_events,
        )


@pytest.fixture
def og(monkeypatch):
    fake = FakeOg()
    monkeypatch.setitem(sys.modules, "onnxruntime_genai", fake.module)
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: True)
    return fake


def test_default_and_previous_default_map_to_the_pinned_1p5b_build(caplog):
    from recandcc.config import DEFAULT_SUMMARY_MODEL, PREVIOUS_DEFAULT_SUMMARY_MODEL

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
    assert gen([{"role": "user", "content": SPOKEN}], 50) == "We agreed on the plan."
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
    assert len(hub.urls) == 2 and og.telemetry_off >= 2


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


def test_mapping_is_logged_with_model_names_only(hub, og, caplog):
    with caplog.at_level(logging.INFO, logger="recandcc"):
        CpuGenerator("m/qwen").load()
    assert "m/qwen runs on the CPU as onnx-community/fake" in caplog.text


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
