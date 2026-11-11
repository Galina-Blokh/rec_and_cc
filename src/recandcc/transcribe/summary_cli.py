"""Summaries (and speaker naming) through a signed-in coding CLI instead of an API key.

``summary.provider`` = ``"claude-code"`` (Claude Code, ``claude -p``, your Claude plan),
``"codex"`` (Codex CLI, ``codex exec``, your ChatGPT login) or ``"grok"`` (Grok through the
Cursor CLI, ``cursor-agent -p``, a Cursor API key). Each model call runs the CLI once, in a fresh
empty folder, with tools off (Claude Code) or a read-only sandbox (Codex), and passes only
the prompt: the meeting's transcript goes to that provider exactly as with a cloud summary.
Opt-in like every cloud provider; with the default ``local`` nothing here runs.

Nothing is logged but fixed reasons (no prompt, answer, key or CLI output). A missing CLI, a
failed run or a timeout raises SummaryUnavailable, so the queue reports it and never falls
back to the local model.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .summary import Messages, NeedsKey, SummaryUnavailable

log = logging.getLogger(__name__)

CLI_PROVIDERS = ("claude-code", "codex", "grok")
EXECUTABLES = {"claude-code": "claude", "codex": "codex", "grok": "cursor-agent"}
# The app starts from Finder with a short PATH: also look where installers put these
SEARCH_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "~/.npm-global/bin")
GROK_FALLBACKS = ("grok-4.7-high", "cursor-grok-4.6-high", "cursor-grok-4.5-high")
TIMEOUT_S = 900.0
EFFORTS = {  # reasoning effort each CLI accepts; others are not passed
    "claude-code": ("low", "medium", "high", "xhigh", "max"),
    "codex": ("minimal", "low", "medium", "high", "xhigh"),
}
_USAGE_LIMIT = "usage limit"


def find_cli(provider: str) -> str | None:
    """Path of the provider's CLI, or None."""
    name = EXECUTABLES[provider]
    found = shutil.which(name)
    if found:
        return found
    for folder in SEARCH_DIRS:
        path = Path(folder).expanduser() / name
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def prompt_text(messages: Messages) -> str:
    """One prompt from chat messages: the system text first, then the rest in order."""
    parts = [m["content"] for m in messages if m["role"] == "system"]
    parts += [m["content"] for m in messages if m["role"] != "system"]
    return "\n\n".join(parts)


class CliGenerator:
    """A ``ModelGenerator`` that runs a coding CLI once per call."""

    def __init__(self, provider: str, model: str = "", effort: str = "") -> None:
        if provider not in CLI_PROVIDERS:
            raise ValueError(f"not a CLI provider: {provider}")
        from ..config import DEFAULT_CLOUD_MODELS

        self.provider = provider
        # Grok without a model tries GROK_FALLBACKS in turn (a model over its usage limit)
        self.model = model.strip() or ("" if provider == "grok" else DEFAULT_CLOUD_MODELS[provider])
        self.effort = effort.strip() if effort.strip() in EFFORTS.get(provider, ()) else ""
        self.model_name = self.model or DEFAULT_CLOUD_MODELS[provider]
        exe = find_cli(provider)
        if exe is None:
            raise SummaryUnavailable(
                f"the {EXECUTABLES[provider]} command was not found: install it and sign in, "
                "or choose another summary provider"
            )
        self.exe = exe

    def __call__(self, messages: Messages, max_tokens: int) -> str:
        prompt = prompt_text(messages)
        if self.provider == "grok":
            models = [self.model] if self.model else list(GROK_FALLBACKS)
            for i, model in enumerate(models):
                try:
                    return self._run(prompt, model)
                except SummaryUnavailable as e:
                    if _USAGE_LIMIT not in str(e) or i == len(models) - 1:
                        raise
                    log.info("grok model %s is over its usage limit; trying the next one", model)
        return self._run(prompt, self.model)

    def _command(self, folder: Path, model: str) -> tuple[list[str], str | None]:
        """(argv, stdin) for one call in ``folder``."""
        if self.provider == "claude-code":
            argv = [self.exe, "-p", "--tools", "", "--output-format", "text"]
            argv += ["--no-session-persistence"]
            if model:
                argv += ["--model", model]
            if self.effort:
                argv += ["--effort", self.effort]
            return argv, None
        if self.provider == "codex":
            argv = [self.exe, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"]
            argv += ["-C", str(folder), "-o", str(folder / "answer.txt")]
            if model:
                argv += ["-m", model]
            if self.effort:
                argv += ["-c", f"model_reasoning_effort={self.effort}"]
            return [*argv, "-"], None
        argv = [self.exe, "-p", "--trust", "--output-format", "text", "--workspace", str(folder)]
        return [*argv, "--model", model], None

    def _run(self, prompt: str, model: str) -> str:
        env = dict(os.environ)
        if self.provider == "grok":
            from .summary_cloud import get_key

            key = get_key("grok")
            if key is None:
                raise NeedsKey("grok", "no Cursor API key: sign in to Cursor (Grok) first")
            env["CURSOR_API_KEY"] = key
        with tempfile.TemporaryDirectory(prefix="recandcc-cli-") as tmp:
            folder = Path(tmp)
            argv, _ = self._command(folder, model)
            if self.provider == "grok":
                argv = [*argv, prompt]  # cursor-agent reads the prompt from its arguments
                stdin = None
            else:
                stdin = prompt
            try:
                done = subprocess.run(  # noqa: S603 - a fixed argv, no shell; the prompt is data
                    argv,
                    input=stdin,
                    cwd=folder,
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=TIMEOUT_S,
                    check=False,
                )
            except subprocess.TimeoutExpired as e:
                raise SummaryUnavailable(f"{self.provider} did not answer in time") from e
            except OSError as e:
                raise SummaryUnavailable(f"{self.provider} could not be started") from e
            out = done.stdout or ""
            if self.provider == "codex":
                answer_file = folder / "answer.txt"
                out = answer_file.read_text("utf-8") if answer_file.exists() else ""
            if done.returncode != 0 or not out.strip():
                said = f"{done.stdout or ''} {done.stderr or ''}".casefold()
                if _USAGE_LIMIT in said:
                    raise SummaryUnavailable(f"{self.provider}: {_USAGE_LIMIT} reached")
                if "login" in said or "authentication" in said or "not logged in" in said:
                    raise SummaryUnavailable(f"{self.provider} is not signed in")
                raise SummaryUnavailable(
                    f"{self.provider} failed (exit {done.returncode})"
                    if done.returncode
                    else f"{self.provider} returned no answer"
                )
            return out.strip()

    def close(self) -> None:
        pass


__all__ = ["CLI_PROVIDERS", "CliGenerator", "find_cli", "prompt_text"]
