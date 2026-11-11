"""Summaries (and speaker naming) through a signed-in coding CLI instead of an API key.

``summary.provider`` = ``"claude-code"`` (Claude Code, ``claude -p``, your Claude plan),
``"codex"`` (Codex CLI, ``codex exec``, your ChatGPT login) or ``"grok"`` (Grok through the
Cursor CLI, ``cursor-agent -p``, a Cursor API key). Each model call runs the CLI once in a fresh
empty folder. The meeting's transcript goes to that provider exactly as with a cloud summary.
Opt-in like every cloud provider; with the default ``local`` nothing here runs.

The transcript is untrusted input (anyone in the meeting can say "read my files"), and these
CLIs are agents, so each one is contained:

- Claude Code runs without any tool: ``--tools ""``, no MCP server (``--strict-mcp-config``
  with an empty config), no user settings or hooks (``--setting-sources local`` in the empty
  folder) and no skills or commands. The prompt goes in on stdin.
- Codex and Grok keep their tools, so they only run on macOS, inside ``sandbox-exec`` with a
  profile that denies reading the home folder (except the CLI's own install and sign-in
  folders) and the transcripts folder. Codex also runs ``-s read-only``; Grok in ``ask`` mode.
  Codex gets the prompt on stdin; Grok reads it from a file in the temp folder, so no transcript
  text is ever on a command line (Windows batch-shim quoting, argv limits, ``ps``).
- Only a short list of environment variables is passed on (plus Grok's Cursor key).
- A timeout kills the whole process tree, and the pipes are drained with a bound.

Nothing is logged but fixed reasons (no prompt, answer, key or CLI output). A missing CLI, a
failed run or a timeout raises SummaryUnavailable, so the queue reports it and never falls
back to the local model.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Collection
from pathlib import Path

from .summary import Messages, NeedsKey, SummaryUnavailable

log = logging.getLogger(__name__)

CLI_PROVIDERS = ("claude-code", "codex", "grok")
EXECUTABLES = {"claude-code": "claude", "codex": "codex", "grok": "cursor-agent"}
SANDBOXED = ("codex", "grok")  # agents with tools: macOS file sandbox required
# The app starts from Finder with a short PATH: also look where installers put these
SEARCH_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "~/.npm-global/bin")
# What each CLI may still read in the home folder: its install and its own sign-in
OWN_DIRS = {
    "codex": ("~/.codex",),
    "grok": ("~/.cursor", "~/.local/share/cursor-agent", "~/.local/bin"),
}
GROK_FALLBACKS = ("grok-4.7-high", "cursor-grok-4.6-high", "cursor-grok-4.5-high")
TIMEOUT_S = 900.0
DRAIN_S = 5.0  # after a kill, how long the pipes may take to close
EFFORTS = {  # reasoning effort each CLI accepts; others are not passed
    "claude-code": ("low", "medium", "high", "xhigh", "max"),
    "codex": ("minimal", "low", "medium", "high", "xhigh"),
}
ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "SHELL")
ENV_KEEP += ("SYSTEMROOT", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "CODEX_HOME")
PROMPT_FILE = "prompt.txt"
GROK_ASK = (
    f"Read {PROMPT_FILE} in this workspace and do exactly what it asks. Answer with text only."
)
_USAGE_LIMIT = "usage limit"
_TASKKILL = os.path.join(os.environ.get("SYSTEMROOT", r"C:\Windows"), "System32", "taskkill.exe")


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


def _sb(path: str) -> str:
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def sandbox_profile(provider: str, exe: str, private: Collection[Path] = ()) -> str:
    """A sandbox-exec profile: everything allowed but reading the home folder (except the
    CLI's own folders and its install) and ``private`` folders (the transcripts)."""
    home = str(Path.home())
    own = [str(Path(d).expanduser()) for d in OWN_DIRS.get(provider, ())]
    own.append(str(Path(exe).resolve().parent))
    denied = [home, *(str(Path(p).expanduser().resolve()) for p in private)]
    lines = ["(version 1)", "(allow default)"]
    lines += [f"(deny file-read* (subpath {_sb(d)}))" for d in denied]
    lines += [f"(allow file-read* (subpath {_sb(d)}))" for d in own if d.startswith(home)]
    # The keychain files (encrypted; items keep their own access rules): without them macOS
    # tells the owner the login keychain cannot be found
    lines += [f"(allow file-read* (subpath {_sb(home + '/Library/Keychains')}))"]
    lines += [f"(allow file-read-metadata (subpath {_sb(home)}))"]
    return "\n".join(lines) + "\n"


class CliGenerator:
    """A ``ModelGenerator`` that runs a coding CLI once per call."""

    def __init__(
        self,
        provider: str,
        model: str = "",
        effort: str = "",
        private: Collection[Path] = (),
    ) -> None:
        if provider not in CLI_PROVIDERS:
            raise ValueError(f"not a CLI provider: {provider}")
        from ..config import DEFAULT_CLOUD_MODELS

        self.provider = provider
        # Grok without a model tries GROK_FALLBACKS in turn (a model over its usage limit)
        self.model = model.strip() or ("" if provider == "grok" else DEFAULT_CLOUD_MODELS[provider])
        self.effort = effort.strip() if effort.strip() in EFFORTS.get(provider, ()) else ""
        self.model_name = self.model or DEFAULT_CLOUD_MODELS[provider]
        self.private = tuple(private)
        if provider in SANDBOXED and sys.platform != "darwin":
            raise SummaryUnavailable(
                f"{provider} summaries need macOS (its tools are contained by a file sandbox "
                "there); choose Claude Code or another provider"
            )
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

    def _command(self, folder: Path, model: str) -> list[str]:
        """The CLI's argv for one call in ``folder`` (never any transcript text)."""
        if self.provider == "claude-code":
            argv = [self.exe, "-p", "--tools", "", "--output-format", "text"]
            argv += ["--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
            argv += ["--setting-sources", "local", "--disable-slash-commands"]
            argv += ["--no-session-persistence"]
            if model:
                argv += ["--model", model]
            if self.effort:
                argv += ["--effort", self.effort]
            return argv
        if self.provider == "codex":
            argv = [self.exe, "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only"]
            argv += ["-C", str(folder), "-o", str(folder / "answer.txt")]
            if model:
                argv += ["-m", model]
            if self.effort:
                argv += ["-c", f"model_reasoning_effort={self.effort}"]
            return [*argv, "-"]
        argv = [self.exe, "-p", "--trust", "--mode", "ask", "--output-format", "text"]
        return [*argv, "--workspace", str(folder), "--model", model, GROK_ASK]

    def _env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k in ENV_KEEP}
        if self.provider == "grok":
            from .summary_cloud import get_key

            key = get_key("grok")
            if key is None:
                raise NeedsKey("grok", "no Cursor API key: sign in to Cursor (Grok) first")
            env["CURSOR_API_KEY"] = key
        return env

    def _run(self, prompt: str, model: str) -> str:
        env = self._env()
        with tempfile.TemporaryDirectory(prefix="recandcc-cli-") as tmp:
            folder = Path(tmp)
            argv = self._command(folder, model)
            stdin: str | None = prompt
            if self.provider == "grok":
                (folder / PROMPT_FILE).write_text(prompt, "utf-8")
                stdin = None
            if self.provider in SANDBOXED:
                profile = folder / "sandbox.sb"
                profile.write_text(sandbox_profile(self.provider, self.exe, self.private), "utf-8")
                argv = ["/usr/bin/sandbox-exec", "-f", str(profile), *argv]
            code, out, err = _call(argv, stdin, folder, env)
            if self.provider == "codex":
                answer_file = folder / "answer.txt"
                out = answer_file.read_text("utf-8") if answer_file.exists() else ""
            if code != 0 or not out.strip():
                said = f"{out} {err}".casefold()
                if _USAGE_LIMIT in said:
                    raise SummaryUnavailable(f"{self.provider}: {_USAGE_LIMIT} reached")
                if "login" in said or "authentication" in said or "not logged in" in said:
                    raise SummaryUnavailable(f"{self.provider} is not signed in")
                raise SummaryUnavailable(
                    f"{self.provider} failed (exit {code})"
                    if code
                    else f"{self.provider} returned no answer"
                )
            return out.strip()

    def close(self) -> None:
        pass


def _call(argv: list[str], stdin: str | None, cwd: Path, env: dict) -> tuple[int, str, str]:
    """Run ``argv`` once; on timeout kill its whole process tree. (returncode, stdout, stderr)."""
    windows = sys.platform == "win32"
    try:
        proc = subprocess.Popen(  # noqa: S603 - a fixed argv, no shell; the prompt is data
            argv,
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=not windows,  # its own process group: killed as one
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if windows else 0,
        )
    except OSError as e:
        raise SummaryUnavailable(f"{Path(argv[0]).name} could not be started") from e
    try:
        out, err = proc.communicate(stdin, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        _kill_tree(proc)
        with contextlib.suppress(subprocess.TimeoutExpired):  # a grandchild holds a pipe:
            proc.communicate(timeout=DRAIN_S)  # give up on it, never block the queue
        raise SummaryUnavailable(f"{Path(argv[0]).name} did not answer in time") from e
    if not windows:  # helpers it left behind in its group (e.g. a background worker)
        _kill_tree(proc)
    return proc.returncode, out or "", err or ""


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(  # noqa: S603 - a fixed command
                [_TASKKILL, "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=DRAIN_S,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        if proc.poll() is None:
            proc.kill()


__all__ = ["CLI_PROVIDERS", "CliGenerator", "find_cli", "prompt_text", "sandbox_profile"]
