"""Cloud summaries: each provider's request (prompt, timestamps, template, effort), keys, retries,
errors, and that a local summary never builds a cloud generator or sends anything.

Stories: US-36.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import threading
import tomllib
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from recandcc.config import DEFAULT_SUMMARY_PROMPT, Config, load_config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import summary, summary_cloud
from recandcc.transcribe.output import PDI, RLI, write_outputs

KEY = "sk-test-do-not-leak"
REAL_OPENER = summary_cloud._OPENER  # the no_network guard replaces it per test
OPENAI_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
SPOKEN = "we ship the beta on Friday"
TEMPLATE = "# Review\n\n## Decisions\nWhat was decided, with the time.\n"
MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """A request a test doesn't expect fails the test instead of reaching a provider."""

    def refuse(req, timeout=None):
        pytest.fail(f"unexpected request to {req.full_url}")

    monkeypatch.setattr(summary_cloud, "_OPENER", SimpleNamespace(open=refuse))


def serve(monkeypatch, *replies) -> list:
    """Answer requests in turn: a dict is a 200 JSON body, an int an HTTP error status, a
    (status, headers) pair an HTTP error with headers, an exception is raised. Returns the
    requests seen."""
    seen, queue = [], list(replies)

    def fake_open(req, timeout=None):
        seen.append(req)
        reply = queue.pop(0)
        if isinstance(reply, tuple):
            status, headers = reply
            raise urllib.error.HTTPError(req.full_url, status, "error", headers, io.BytesIO(b"{}"))
        if isinstance(reply, int):
            raise urllib.error.HTTPError(req.full_url, reply, "error", {}, io.BytesIO(b"{}"))
        if isinstance(reply, Exception):
            raise reply
        return contextlib.nullcontext(io.BytesIO(json.dumps(reply).encode()))

    monkeypatch.setattr(summary_cloud, "_OPENER", SimpleNamespace(open=fake_open))
    return seen


def make_session(tmp_path: Path, segments: list[Segment]) -> Path:
    d = tmp_path / "2026-10-27" / "100000_zoom"
    d.mkdir(parents=True)
    s = SessionInfo("s1", d, datetime(2026, 10, 27, 10), TriggerKind.CALL, "zoom.us")
    s.save()
    write_outputs(s, segments, 60.0, "fake")
    return d


def cloud_cfg(provider: str, **settings) -> Config:
    cfg = Config()
    cfg.summary.provider = provider
    for name, value in settings.items():
        setattr(cfg.summary, name, value)
    return cfg


class Local:
    """A local summary model that records what it is asked."""

    calls: list = []

    def __init__(self, model, allow_unpinned=False, on_download=None):
        self.model_name = model

    def __call__(self, messages, max_tokens):
        Local.calls.append(messages)
        return "- Anna sends the deck"

    def close(self):
        pass


def test_local_summaries_build_no_cloud_generator_and_keep_their_prompts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("a local summary built a cloud generator")

    monkeypatch.setattr(summary_cloud, "CloudGenerator", forbidden)
    monkeypatch.setattr(summary, "MlxGenerator", Local)
    Local.calls = []
    d = make_session(tmp_path, [Segment(5, 9, SPOKEN, "Me", "en")])
    text = summary.summarize_session(d, Config()).read_text("utf-8")
    sent = json.dumps(Local.calls)
    assert "Generated on this computer" in text
    assert DEFAULT_SUMMARY_PROMPT not in sent and "Answer in Markdown" not in sent
    assert "[00:00:05]" not in sent  # the local prompts keep their own transcript lines


def test_cloud_summary_without_a_key_fails_before_anything_is_sent(tmp_path):
    d = make_session(tmp_path, [Segment(5, 9, SPOKEN, "Me", "en")])
    with pytest.raises(summary.SummaryUnavailable, match="OpenAI API key"):
        summary.summarize_session(d, cloud_cfg("openai"))
    assert not (d / "summary.md").exists()


def test_keys_come_from_the_keychain_and_the_environment_is_the_fallback(
    monkeypatch, fake_keychain
):
    summary_cloud.set_key("anthropic", "  sk-ant-test  ")
    assert summary_cloud.get_key("anthropic") == "sk-ant-test"
    summary_cloud.delete_key("anthropic")
    monkeypatch.setenv("RECANDCC_ANTHROPIC_API_KEY", "sk-env")
    assert summary_cloud.get_key("anthropic") == "sk-env"
    fake_keychain[("RecAndCC", "anthropic")] = "sk-kc"
    assert summary_cloud.get_key("anthropic") == "sk-kc"  # the keychain comes first
    with pytest.raises(ValueError, match="spaces"):
        summary_cloud.set_key("openai", "two words")


def test_a_key_that_is_not_printable_ascii_is_refused_without_being_shown(
    fake_keychain, monkeypatch
):
    with pytest.raises(ValueError) as err:
        summary_cloud.set_key("openai", "sk-“test")
    assert "sk-" not in str(err.value)
    fake_keychain[("RecAndCC", "openai")] = "sk-“test"
    assert summary_cloud.get_key("openai") is None
    monkeypatch.setenv("RECANDCC_OPENAI_API_KEY", "sk-env")
    assert summary_cloud.get_key("openai") == "sk-env"  # the next source is still tried


def test_openai_request_carries_prompt_timed_transcript_and_template(
    tmp_path, monkeypatch, fake_keychain
):
    fake_keychain[("RecAndCC", "openai")] = KEY
    seen = serve(monkeypatch, {"output_text": "## Decisions\n- Ship on Friday [00:00:05]."})
    template = tmp_path / "review.md"
    template.write_text(TEMPLATE, encoding="utf-8")
    d = make_session(tmp_path, [Segment(5, 9, SPOKEN, "Me", "en")])
    out = summary.summarize_session(d, cloud_cfg("openai", template=str(template)))
    text = out.read_text("utf-8")
    req = seen[0]
    body = json.loads(req.data)
    assert len(seen) == 1 and req.full_url == OPENAI_URL and req.get_method() == "POST"
    assert req.get_header("Authorization") == f"Bearer {KEY}"
    assert KEY not in req.data.decode() and KEY not in text
    assert body["model"] == "gpt-6.1" and body["reasoning"] == {"effort": "xhigh"}
    assert body["max_output_tokens"] == 32_000  # 8 x the 4000-token answer budget
    assert body["store"] is False
    assert body["instructions"].endswith("Answer in Markdown with ## headings.")
    user = body["input"][-1]["content"]
    assert user.startswith(DEFAULT_SUMMARY_PROMPT)
    assert "Use exactly these sections, in this order" in user
    assert "- Decisions: What was decided, with the time." in user
    assert "[00:00:05] Me: we ship the beta on Friday" in user
    assert text.startswith("# Review\n") and "## Decisions" in text
    assert "_Generated in the cloud by gpt-6.1 (OpenAI)" in text


def test_anthropic_request_has_its_headers_effort_and_timed_transcript(
    tmp_path, monkeypatch, fake_keychain, caplog
):
    fake_keychain[("RecAndCC", "anthropic")] = KEY
    reply = {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": ""},
            {"type": "text", "text": "## Summary\nShip on Friday."},
        ],
    }
    seen = serve(monkeypatch, reply)
    d = make_session(tmp_path, [Segment(3700, 3704, SPOKEN, "Others", "en")])
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary.summarize_session(d, cloud_cfg("anthropic")).read_text("utf-8")
    req = seen[0]
    body = json.loads(req.data)
    assert req.full_url == ANTHROPIC_URL and req.get_header("X-api-key") == KEY
    assert req.get_header("Anthropic-version") == "2026-10-21"
    assert body["model"] == "claude-opus-5-5" and body["max_tokens"] == 32_000
    assert body["output_config"] == {"effort": "xhigh"}
    assert body["thinking"] == {"type": "adaptive"}
    assert "Answer in Markdown with ## headings." in body["system"]
    assert "[01:01:40] Others: we ship the beta on Friday" in body["messages"][0]["content"]
    assert "## Summary\nShip on Friday." in text
    assert "_Generated in the cloud by claude-opus-5-5 (Anthropic)" in text
    assert SPOKEN not in caplog.text and KEY not in caplog.text  # logs hold no text or key


@pytest.mark.parametrize(
    ("model", "effort", "sent"),
    [("claude-opus-5-5", "xhigh", "xhigh"), ("claude-sonnet-4-6", "xhigh", "high")],
)
def test_reasoning_effort_is_mapped_to_claude_levels(model, effort, sent):
    assert summary_cloud._claude_effort(effort, model) == sent
    assert summary_cloud._claude_effort("minimal", model) == "low"


def test_openai_has_no_max_level_so_max_is_xhigh():
    assert summary_cloud._openai_effort("max") == "xhigh"
    assert summary_cloud._openai_effort("low") == "low"


@pytest.mark.parametrize(
    "reply",
    [
        {"output_text": "A"},
        {"output": [{"type": "message", "content": [{"type": "output_text", "text": "A"}]}]},
    ],
)
def test_openai_answer_is_read_from_either_reply_shape(monkeypatch, fake_keychain, reply):
    fake_keychain[("RecAndCC", "openai")] = KEY
    serve(monkeypatch, reply)
    assert summary_cloud.CloudGenerator("openai")(MSGS, 400) == "A"


def test_rate_limits_and_server_errors_are_retried_twice(monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    waits = []
    monkeypatch.setattr(summary_cloud, "_sleep", waits.append)
    seen = serve(monkeypatch, 503, 429, {"output_text": "ok"})
    assert summary_cloud.CloudGenerator("openai")(MSGS, 400) == "ok"
    assert len(seen) == 3 and waits == [2.0, 4.0]
    seen = serve(monkeypatch, 500, 500, 502)  # a third failure is reported, not retried
    with pytest.raises(summary.SummaryUnavailable, match="HTTP 502"):
        summary_cloud.CloudGenerator("openai")(MSGS, 400)
    assert len(seen) == 3


def test_retry_after_is_honoured_up_to_a_minute(monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    waits = []
    monkeypatch.setattr(summary_cloud, "_sleep", waits.append)
    serve(
        monkeypatch,
        (429, {"Retry-After": "7"}),
        (429, {"Retry-After": "900"}),
        {"output_text": "ok"},
    )
    assert summary_cloud.CloudGenerator("openai")(MSGS, 400) == "ok"
    assert waits == [7.0, 60.0]


def test_redirects_are_refused_before_the_location_is_parsed():
    handler = summary_cloud._NoRedirect()
    for code in (301, 302, 303, 307, 308):  # a malformed Location must not raise either
        method = getattr(handler, f"http_error_{code}")
        assert method(None, None, code, "Moved", {"location": "http://[::1"}) is None


def test_a_redirect_from_the_real_opener_is_an_error_and_the_target_gets_nothing(
    monkeypatch, fake_keychain
):
    import http.server

    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if self.path == "/target":
                hits.append(self.headers.get("Authorization", ""))
                self.send_response(200)
                self.end_headers()
                return
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{server.server_address[1]}/target")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        fake_keychain[("RecAndCC", "openai")] = KEY
        port = server.server_address[1]
        monkeypatch.setattr(summary_cloud, "_OPENER", REAL_OPENER)
        monkeypatch.setitem(summary_cloud.URLS, "openai", f"http://127.0.0.1:{port}/start")
        with pytest.raises(summary.SummaryUnavailable, match="HTTP 302"):
            summary_cloud.CloudGenerator("openai")(MSGS, 400)
        assert hits == []  # the key was not sent on to /target
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    ("failure", "match"),
    [(401, "rejected the API key"), (urllib.error.URLError("no route"), "reached")],
)
def test_failures_are_reported_and_the_key_never_appears(
    monkeypatch, fake_keychain, caplog, failure, match
):
    fake_keychain[("RecAndCC", "anthropic")] = KEY
    serve(monkeypatch, failure)
    with (
        caplog.at_level(logging.DEBUG, logger="recandcc"),
        pytest.raises(summary.SummaryUnavailable, match=match) as err,
    ):
        summary_cloud.CloudGenerator("anthropic")(MSGS, 400)
    assert KEY not in repr(err.value) and KEY not in caplog.text


@pytest.mark.parametrize(
    ("provider", "reply", "match"),
    [
        ("openai", {"status": "incomplete", "output_text": "part"}, "did not finish"),
        ("anthropic", {"stop_reason": "max_tokens", "content": []}, "did not finish"),
        ("openai", {"status": "completed", "output_text": "  "}, "returned no answer"),
        ("anthropic", {"content": 5}, "returned no answer"),  # an unexpected shape, no crash
    ],
)
def test_a_cut_off_or_empty_answer_is_never_written(
    monkeypatch, fake_keychain, provider, reply, match
):
    fake_keychain[("RecAndCC", provider)] = KEY
    serve(monkeypatch, reply)
    with pytest.raises(summary.SummaryUnavailable, match=match):
        summary_cloud.CloudGenerator(provider)(MSGS, 400)


def test_a_failed_cloud_summary_writes_nothing_and_never_falls_back_to_local(
    tmp_path, monkeypatch, fake_keychain
):
    fake_keychain[("RecAndCC", "openai")] = KEY
    serve(monkeypatch, 401)
    built = []
    monkeypatch.setattr(summary, "MlxGenerator", lambda *args, **kwargs: built.append(args))
    notices = []
    d = make_session(tmp_path, [Segment(5, 9, SPOKEN, "Me", "en")])
    queue = summary.SummaryQueue(cloud_cfg("openai"), notify=lambda t, m: notices.append((t, m)))
    queue.submit(d)
    queue.join()
    assert built == []  # no local model was even made
    assert not (d / "summary.md").exists() and not (d / summary.PENDING_MARKER).exists()
    assert notices == [("Summary skipped", "OpenAI rejected the API key (HTTP 401)")]


def test_a_transcript_past_the_single_call_limit_goes_in_parts_then_once_more(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(summary, "CLOUD_SINGLE_CALL_CHARS", 60)
    monkeypatch.setattr(summary, "CLOUD_PART_CHARS", 60)
    calls = []

    class Gen:
        def __call__(self, messages, max_tokens):
            calls.append(messages[-1]["content"])
            return f"notes {len(calls)}"

        def close(self):
            pass

    segs = [Segment(i * 10, i * 10 + 5, f"sentence number {i}", "Me", "en") for i in range(12)]
    d = make_session(tmp_path, segs)
    text = summary.summarize_session(d, cloud_cfg("openai"), generate=Gen()).read_text("utf-8")
    assert len(calls) >= 3
    assert "consecutive parts" in calls[-1] and "notes 1" in calls[-1]
    assert "_Generated in the cloud by gpt-6.1 (OpenAI)" in text


def test_an_empty_transcript_sends_nothing(tmp_path, monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    d = make_session(tmp_path, [])  # the no_network guard fails any request
    assert "None." in summary.summarize_session(d, cloud_cfg("openai")).read_text("utf-8")


def test_timed_lines_keep_paragraphs_and_the_transcript_time_format():
    segs = [
        Segment(5, 6, "hi", "Me", "en"),
        Segment(7, 9, "there", "Me", "en"),
        Segment(3700, 3702, "bye", "Others", "en"),
    ]
    assert summary.timed_transcript_lines(segs) == [
        "[00:00:05] Me: hi there",
        "[01:01:40] Others: bye",
    ]


def test_nested_bullets_in_a_hebrew_summary_stay_list_items():
    out = summary.format_cloud("he", "- ראשון\n  - שני", "m", "OpenAI")
    assert f"\n  - {RLI}שני{PDI}" in out


def test_the_default_prompt_is_not_written_to_config_but_a_changed_one_is(tmp_path):
    path = tmp_path / "config.toml"
    Config().save(path)
    assert "prompt" not in tomllib.loads(path.read_text("utf-8"))["summary"]
    cfg = Config()
    cfg.summary.prompt = "Two lines only."
    cfg.save(path)
    assert load_config(path).summary.prompt == "Two lines only."


def test_bad_cloud_settings_fall_back_with_a_warning(tmp_path, caplog):
    path = tmp_path / "config.toml"
    path.write_text(
        '[summary]\nprovider = "ChatGPT"\nreasoning_effort = "extreme"\nprompt = ""\n'
        "cloud_model = 3\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING):
        cfg = load_config(path).summary
    assert (cfg.provider, cfg.reasoning_effort, cfg.cloud_model) == ("local", "xhigh", "")
    assert cfg.prompt == DEFAULT_SUMMARY_PROMPT
    assert "provider must be" in caplog.text and "reasoning_effort must be" in caplog.text


def test_a_negative_retry_after_falls_back_to_the_backoff(monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    waits = []
    monkeypatch.setattr(summary_cloud, "_sleep", waits.append)
    serve(monkeypatch, (429, {"Retry-After": "-5"}), {"output_text": "ok"})
    assert summary_cloud.CloudGenerator("openai")(MSGS, 400) == "ok"
    assert waits == [2.0]


def test_the_request_body_stays_ascii_for_russian_text(tmp_path, monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    seen = serve(monkeypatch, {"output_text": "ok"})
    d = make_session(tmp_path, [Segment(5, 9, "обсуждаем бюджет", "Me", "ru")])
    summary.summarize_session(d, cloud_cfg("openai"))
    assert seen[0].data.isascii() and b"\\u" in seen[0].data  # escaped, still valid JSON


def test_a_reply_that_is_not_an_object_is_unavailable(monkeypatch, fake_keychain):
    fake_keychain[("RecAndCC", "openai")] = KEY
    serve(monkeypatch, [1, 2])
    with pytest.raises(summary.SummaryUnavailable, match="unexpected reply"):
        summary_cloud.CloudGenerator("openai")(MSGS, 400)
