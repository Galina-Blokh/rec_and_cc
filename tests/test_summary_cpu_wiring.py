"""Wiring of the CPU summary backend: create_generator, the download notice and the model
fallbacks, with a fake onnxruntime-genai and a fake hub. Issue #4.

Stories: US-27.
"""

# ruff: noqa: F811
from __future__ import annotations

import logging

import pytest
from test_summary import SPOKEN, make_session, seg
from test_summary_cpu import _fake_model, hub, og  # noqa: F401

from recandcc.config import DEFAULT_SUMMARY_MODEL, PREVIOUS_DEFAULT_SUMMARY_MODEL, Config
from recandcc.transcribe import summary, summary_cpu
from recandcc.transcribe.summary_cpu import CpuGenerator


@pytest.mark.parametrize(
    ("mlx", "cpu", "expected"),
    [
        (True, True, summary.MlxGenerator),  # Apple silicon keeps MLX
        (False, True, CpuGenerator),  # Windows, Linux
        (False, False, summary.MlxGenerator),  # neither: its load() says why
    ],
)
def test_generator_is_chosen_per_platform(monkeypatch, mlx, cpu, expected):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: mlx)
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: cpu)
    gen = summary.create_generator("m/qwen")
    assert type(gen) is expected and gen.model_name == "m/qwen"


def test_missing_model_offline_skips_the_summary_without_logging_text(
    hub, og, tmp_path, monkeypatch, caplog
):
    hub.state["online"] = False
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {Config().summary.model: _fake_model()})
    d = make_session(tmp_path, [seg(SPOKEN)])
    notices = []
    q = summary.SummaryQueue(Config(), notify=lambda *n: notices.append(n))
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        q.submit(d)
        q.join()
    assert not (d / "summary.md").exists()
    assert notices[-1] == ("Summary skipped", summary_cpu.NOT_ON_DISK)
    assert notices[0][0] == "Summary model"  # the one-time download notice came first
    assert SPOKEN not in caplog.text


def test_cpu_summary_is_written_without_logging_text(hub, og, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {Config().summary.model: _fake_model()})
    d = make_session(tmp_path, [seg(SPOKEN)])
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        path = summary.summarize_session(d, Config())
    assert "We agreed on the plan." in path.read_text("utf-8")
    assert SPOKEN not in caplog.text and "We agreed" not in caplog.text


@pytest.mark.parametrize(
    ("mlx", "cpu", "size"), [(True, True, "~3.0 GB"), (False, True, "~1.8 GB")]
)
def test_download_notice_names_the_size_of_the_build_downloaded_here(monkeypatch, mlx, cpu, size):
    from recandcc.config import DEFAULT_SUMMARY_MODEL

    monkeypatch.setattr(summary, "mlx_lm_available", lambda: mlx)
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: cpu)
    assert summary.download_notice(DEFAULT_SUMMARY_MODEL) == (
        f"Downloading the summary model ({size}, once)"
    )


@pytest.mark.parametrize("name", [DEFAULT_SUMMARY_MODEL, PREVIOUS_DEFAULT_SUMMARY_MODEL])
def test_default_and_old_1p5b_both_run_the_pinned_1p5b_onnx_build(
    hub,
    og,
    tmp_path,
    monkeypatch,
    caplog,
    name,
):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    fake = _fake_model()  # stands in for the pinned build both names map to
    monkeypatch.setattr(
        summary_cpu,
        "CPU_MODELS",
        {DEFAULT_SUMMARY_MODEL: fake, PREVIOUS_DEFAULT_SUMMARY_MODEL: fake},
    )
    cfg = Config()
    cfg.summary.model = name
    d = make_session(tmp_path, [seg(SPOKEN)])
    with caplog.at_level(logging.INFO, logger="recandcc"):
        path = summary.summarize_session(d, cfg)
    assert path.exists() and hub.urls  # the 1.5B build was fetched, not a 4B one
    assert f"{name} runs on the CPU as onnx-community/fake" in caplog.text
    assert SPOKEN not in caplog.text


def test_other_model_without_cpu_build_falls_back_to_summary_model(
    hub,
    og,
    tmp_path,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {DEFAULT_SUMMARY_MODEL: _fake_model()})
    cfg = Config()
    routed = "mlx-community/Qwen2.5-3B-Instruct-4bit"  # pinned, but no CPU build
    cfg.summary.language_models = {"he": routed}
    from recandcc.models import Segment

    d = make_session(tmp_path, [Segment(0, 30, "כן", "Me", "he")])
    with caplog.at_level(logging.INFO, logger="recandcc"):
        path = summary.summarize_session(d, cfg)
    assert path.exists()
    assert f"summary model {routed} has no CPU build" in caplog.text
    assert f"summary model {routed} unavailable; using {DEFAULT_SUMMARY_MODEL}" in caplog.text
    assert not any("3B" in u for u in hub.urls)


def test_the_pause_hook_reaches_the_cpu_generator_only(monkeypatch):
    waits: list[int] = []
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(summary_cpu, "cpu_available", lambda: True)
    cpu = summary.create_generator("m/qwen", pause=lambda: waits.append(1))
    assert isinstance(cpu, CpuGenerator) and cpu.pause is not None
    cpu.pause()
    assert waits == [1]
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: True)
    mlx = summary.create_generator("m/qwen", pause=lambda: waits.append(1))
    assert type(mlx) is summary.MlxGenerator and not hasattr(mlx, "pause")


def test_summarize_session_hands_its_before_call_to_the_generator(hub, og, tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "mlx_lm_available", lambda: False)
    monkeypatch.setattr(summary_cpu, "CPU_MODELS", {Config().summary.model: _fake_model()})
    d = make_session(tmp_path, [seg(SPOKEN)])
    seen = []
    real = summary.create_generator

    def spy(model, allow_unpinned=False, on_download=None, pause=None):
        seen.append(pause)
        return real(model, allow_unpinned, on_download, pause)

    def wait_idle():
        pass

    monkeypatch.setattr(summary, "create_generator", spy)
    summary.summarize_session(d, Config(), before_call=wait_idle)
    assert seen == [wait_idle]
