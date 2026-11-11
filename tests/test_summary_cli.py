"""Summaries and speaker naming through a signed-in coding CLI (Claude Code, Codex, Grok via
Cursor) instead of an API key. No CLI is ever run: subprocess.run is faked.

Synthetic transcripts only.
Stories: US-27.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from recandcc.config import KEY_PROVIDERS, Config, load_config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import speaker_names, summary, summary_cli, summary_cloud
from recandcc.transcribe.output import write_outputs
from recandcc.transcribe.summary import NeedsKey, SummaryUnavailable
from recandcc.ui.present import summary_provider_rows

TRANSCRIPT = "transcript text that must never be logged"


class FakeRun:
    """Stands in for summary_cli._call: records each CLI call and answers from ``replies`` in
    turn: (returncode, stdout, stderr)."""

    def __init__(self, *replies):
        self.calls: list[dict] = []
        self.replies = list(replies) or [(0, "an answer", "")]

    def __call__(self, argv, stdin, cwd, env):
        prompt_file = Path(cwd) / summary_cli.PROMPT_FILE
        self.calls.append(
            {
                "argv": argv,
                "input": stdin,
                "cwd": cwd,
                "env": env,
                "file": prompt_file.read_text("utf-8") if prompt_file.exists() else None,
                "profile": (Path(cwd) / "sandbox.sb").read_text("utf-8")
                if (Path(cwd) / "sandbox.sb").exists()
                else None,
            }
        )
        code, out, err = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if "exec" in argv and code == 0:  # codex writes its last message to -o FILE
            Path(argv[argv.index("-o") + 1]).write_text(out, "utf-8")
            out = "progress noise"
        return code, out, err


@pytest.fixture
def cli(monkeypatch):
    monkeypatch.setattr(summary_cli, "find_cli", lambda p: f"/bin/{summary_cli.EXECUTABLES[p]}")
    monkeypatch.setattr(summary_cli.sys, "platform", "darwin")

    def install(*replies):
        fake = FakeRun(*replies)
        monkeypatch.setattr(summary_cli, "_call", fake)
        return fake

    return install


MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": TRANSCRIPT}]


def test_claude_code_runs_print_mode_without_tools_and_reads_stdout(cli):
    fake = cli((0, "the summary\n", ""))
    gen = summary_cli.CliGenerator("claude-code", "", "xhigh")
    assert gen(MESSAGES, 400) == "the summary"
    call = fake.calls[0]
    argv = call["argv"]
    assert argv[:6] == ["/bin/claude", "-p", "--tools", "", "--output-format", "text"]
    assert "--strict-mcp-config" in argv and '{"mcpServers":{}}' in argv  # no MCP servers
    assert argv[argv.index("--setting-sources") + 1] == "local"  # no user hooks or settings
    assert "--disable-slash-commands" in argv and call["profile"] is None
    assert argv[argv.index("--model") + 1] == "opus"
    assert argv[argv.index("--effort") + 1] == "xhigh"
    assert call["input"] == "SYS\n\n" + TRANSCRIPT  # the prompt goes on stdin, not argv
    assert TRANSCRIPT not in " ".join(argv)
    assert Path(call["cwd"]).name.startswith("recandcc-cli-")  # a fresh empty folder
    assert not Path(call["cwd"]).exists()  # and it is gone afterwards
    assert gen.model_name == "opus"


@pytest.mark.skipif(sys.platform == "win32", reason="sandbox-exec profiles are macOS-only")
def test_codex_runs_read_only_and_reads_its_last_message_file(cli):
    fake = cli((0, "codex answer", ""))
    gen = summary_cli.CliGenerator("codex", "", "xhigh")
    assert gen(MESSAGES, 400) == "codex answer"
    argv = fake.calls[0]["argv"]
    assert argv[:3] == [
        "/usr/bin/sandbox-exec",
        "-f",
        str(Path(fake.calls[0]["cwd"]) / "sandbox.sb"),
    ]
    assert argv[3:5] == ["/bin/codex", "exec"] and argv[-1] == "-"
    profile = fake.calls[0]["profile"]
    assert f'(deny file-read* (subpath "{Path.home()}"))' in profile
    assert f'(allow file-read* (subpath "{Path.home()}/.codex"))' in profile
    assert "read-only" in argv and "--skip-git-repo-check" in argv and "--ephemeral" in argv
    assert "gpt-6.1-sol" in argv and "model_reasoning_effort=xhigh" in argv
    assert fake.calls[0]["input"].endswith(TRANSCRIPT)


def test_grok_needs_a_cursor_key_and_falls_back_past_usage_limits(cli, monkeypatch):
    monkeypatch.setattr(summary_cloud, "get_key", lambda provider: None)
    cli((0, "x", ""))
    with pytest.raises(NeedsKey):
        summary_cli.CliGenerator("grok")(MESSAGES, 400)

    monkeypatch.setattr(summary_cloud, "get_key", lambda provider: "cur-key")
    limit = (1, "", "ActionRequiredError: You've hit your usage limit")
    fake = cli(limit, (0, "grok answer", ""))
    assert summary_cli.CliGenerator("grok")(MESSAGES, 400) == "grok answer"
    models = [c["argv"][c["argv"].index("--model") + 1] for c in fake.calls]
    assert models == ["grok-4.7-high", "cursor-grok-4.6-high"]
    assert all(c["env"]["CURSOR_API_KEY"] == "cur-key" for c in fake.calls)
    first = fake.calls[0]
    assert first["input"] is None and first["file"].endswith(TRANSCRIPT)  # prompt in a file
    assert TRANSCRIPT not in " ".join(first["argv"]) and first["argv"][-1] == summary_cli.GROK_ASK
    assert first["argv"][0] == "/usr/bin/sandbox-exec" and "ask" in first["argv"]


@pytest.mark.parametrize(
    "reply, reason",
    [
        ((1, TRANSCRIPT, "boom " + TRANSCRIPT), "failed (exit 1)"),
        ((0, "   ", ""), "returned no answer"),
        ((1, "", "Authentication required. Please run 'agent login'"), "not signed in"),
        ((1, "", "You've hit your usage limit"), "usage limit reached"),
    ],
)
def test_failures_raise_a_fixed_reason_without_any_output(cli, reply, reason):
    cli(reply)
    with pytest.raises(SummaryUnavailable) as e:
        summary_cli.CliGenerator("claude-code")(MESSAGES, 400)
    assert reason in str(e.value) and TRANSCRIPT not in str(e.value)


def test_a_missing_cli_is_unavailable_and_tool_agents_need_macos(monkeypatch):
    monkeypatch.setattr(summary_cli, "find_cli", lambda p: None)
    monkeypatch.setattr(summary_cli.sys, "platform", "darwin")  # on every CI runner
    with pytest.raises(SummaryUnavailable, match="command was not found"):
        summary_cli.CliGenerator("codex")
    monkeypatch.setattr(summary_cli, "find_cli", lambda p: "/bin/x")
    monkeypatch.setattr(summary_cli.sys, "platform", "win32")
    for provider in ("codex", "grok"):
        with pytest.raises(SummaryUnavailable, match="need macOS"):
            summary_cli.CliGenerator(provider)
    assert summary_cli.CliGenerator("claude-code").exe == "/bin/x"  # tool-free: allowed


def test_only_listed_environment_variables_are_passed(cli, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "would-switch-billing")
    monkeypatch.setenv("SOME_TOKEN", "secret")
    fake = cli((0, "ok", ""))
    summary_cli.CliGenerator("claude-code")(MESSAGES, 400)
    env = fake.calls[0]["env"]
    assert "ANTHROPIC_API_KEY" not in env and "SOME_TOKEN" not in env
    assert set(env) <= set(summary_cli.ENV_KEEP)


@pytest.mark.skipif(sys.platform == "win32", reason="sandbox-exec profiles are macOS-only")
def test_the_profile_also_hides_the_transcripts_folder(tmp_path):
    profile = summary_cli.sandbox_profile("codex", "/opt/homebrew/bin/codex", [tmp_path])
    assert f'(deny file-read* (subpath "{tmp_path.resolve()}"))' in profile
    assert "Library/Keychains" in profile  # else macOS reports a lost keychain
    home = Path.home()
    assert f'(deny file-read* (subpath "{home}/.codex/sessions"))' in profile  # past chats
    assert "memories_[0-9]+" in profile
    grok = summary_cli.sandbox_profile("grok", "/opt/homebrew/bin/cursor-agent")
    assert f'(deny file-read* (subpath "{home}/.cursor/chats"))' in grok


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_a_timeout_kills_the_process_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(summary_cli, "TIMEOUT_S", 0.5)
    monkeypatch.setattr(summary_cli, "DRAIN_S", 2.0)
    pid_file = tmp_path / "child.pid"
    script = f"sleep 30 & echo $! > {pid_file}; wait"
    with pytest.raises(SummaryUnavailable, match="did not answer in time"):
        summary_cli._call(["/bin/sh", "-c", script], None, tmp_path, dict(os.environ))
    child = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):  # the grandchild was killed with its group
        os.kill(child, 0)


def test_find_cli_also_looks_where_installers_put_it(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(summary_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(summary_cli, "SEARCH_DIRS", (str(tmp_path),))
    assert summary_cli.find_cli("codex") == str(exe)


def test_config_and_menu_accept_the_cli_providers(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[summary]\nprovider = "claude-code"\n', "utf-8")
    assert load_config(path).summary.provider == "claude-code"
    path.write_text('[summary]\nprovider = "grok-cli"\n', "utf-8")
    assert load_config(path).summary.provider == "local"
    rows = [p for p, _title, _on in summary_provider_rows("codex")]
    assert {"claude-code", "codex", "grok"} <= set(rows)
    assert "grok" in KEY_PROVIDERS and "codex" not in KEY_PROVIDERS


def test_naming_and_summary_through_a_cli_read_the_whole_meeting(tmp_path, cli, caplog):
    long_talk = "We reviewed the release plan in detail. " * 40  # pushes the intro far down
    segments = [Segment(i * 3.0, i * 3.0 + 2, long_talk, "Speaker 1", "en") for i in range(20)]
    segments += [
        Segment(100, 102, "Ana, can you give the client update?", "Me", "en"),
        Segment(103, 105, "Sure, the client is ready.", "Speaker 2", "en"),
    ]
    assert len(" ".join(s.text for s in segments)) > speaker_names.LOCAL_CHARS
    info = SessionInfo("cli", tmp_path, datetime(2026, 11, 11, 10), TriggerKind.CALL, "teams")
    info.save()
    write_outputs(info, segments, 110.0, "fake")
    names = {"Speaker 2": {"name": "Ana", "evidence": "Ana, can you give the client update"}}
    fake = cli((0, json.dumps(names), ""), (0, "## Decisions\n- Ana: client ready.", ""))
    cfg = Config(output_dir=tmp_path)
    cfg.summary.provider = "claude-code"
    with caplog.at_level(logging.DEBUG, logger="recandcc"):
        text = summary.summarize_session(tmp_path, cfg).read_text("utf-8")
    assert len(fake.calls) == 2  # names, then the summary
    assert "Ana, can you give the client update" in fake.calls[0]["input"]  # past the local cut
    data = json.loads((tmp_path / "transcript.json").read_text("utf-8"))
    assert "Ana" in {s["speaker"] for s in data["segments"]}
    assert "Ana: client ready" in text and "Claude Code" in text
    assert TRANSCRIPT not in caplog.text and "Ana" not in caplog.text
