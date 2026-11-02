"""Cloud summaries (opt-in): one HTTPS request per summary to ChatGPT's API (OpenAI) or Claude's
API (Anthropic), picked by ``summary.provider``. "local" never reaches this module.

summary.py builds the prompt and the timestamped transcript; this module sends them and reads the
answer back, with no SDK (``urllib.request``, default certificate checks). The key is in the OS
keychain (``keyring``: macOS Keychain, Windows Credential Manager; service "RecAndCC", account =
provider), or else in ``RECANDCC_OPENAI_API_KEY`` / ``RECANDCC_ANTHROPIC_API_KEY``. Errors raise
SummaryUnavailable with a fixed reason: never the key, the request or the reply.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from ..config import resolve_cloud_model
from .summary import CLOUD_LABELS, Messages, SummaryUnavailable

log = logging.getLogger("recandcc.transcribe")

SERVICE = "RecAndCC"
ENV_VARS = {"openai": "RECANDCC_OPENAI_API_KEY", "anthropic": "RECANDCC_ANTHROPIC_API_KEY"}
URLS = {
    "openai": "https://api.openai.com/v1/responses",
    "anthropic": "https://api.anthropic.com/v1/messages",
}
# The current Messages API version (claude-api skill); nothing this request sends has changed.
ANTHROPIC_VERSION = "2026-10-21"
TIMEOUT_S = 300.0
RETRIES = 2  # extra attempts after a 429 or a 5xx, 2 s and then 4 s apart
MAX_OUTPUT_TOKENS = 128_000  # the output limit of the current models
TRUNCATED = "the cloud model did not finish its answer"
_OLD_CLAUDE = re.compile(r"claude-(?:opus|sonnet)-4-6")  # effort up to max, no xhigh


def _keyring():
    import keyring  # lazily: only a cloud summary or a key change needs the keychain

    return keyring


def get_key(provider: str) -> str | None:
    """The API key of ``provider``: the keychain entry, else its environment variable."""
    key = ""
    try:
        key = (_keyring().get_password(SERVICE, provider) or "").strip()
    except Exception as e:  # noqa: BLE001 - a locked or unavailable keychain: the type only
        log.warning("could not read the %s key from the keychain (%s)", provider, type(e).__name__)
    return key or os.environ.get(ENV_VARS[provider], "").strip() or None


def set_key(provider: str, key: str) -> None:
    """Store ``key`` for ``provider`` in the keychain. Raises ValueError (the key is not in the
    message) if it is empty or has spaces."""
    key = key.strip()
    if not key or any(c.isspace() for c in key):
        raise ValueError("the API key is empty or has spaces in it")
    _keyring().set_password(SERVICE, provider, key)


def delete_key(provider: str) -> None:
    """Remove the keychain entry of ``provider``, if there is one."""
    keyring = _keyring()
    if keyring.get_password(SERVICE, provider):
        keyring.delete_password(SERVICE, provider)


def _budget(max_tokens: int) -> int:
    """Output limit of one call. Reasoning is billed as output before the answer, so it gets
    8 times the answer's size (at least 4000), or the answer can come back empty."""
    return min(max(max_tokens * 8, 4000), MAX_OUTPUT_TOKENS)


def _claude_effort(effort: str, model: str) -> str:
    """Claude's ``output_config.effort``: Claude has no level below low (Opus 5.5 always thinks),
    and Opus/Sonnet 4.6 have no xhigh (high is the nearest)."""
    if effort in ("none", "minimal"):
        return "low"
    if effort == "xhigh" and _OLD_CLAUDE.match(model):
        return "high"
    return effort


def _openai_text(reply: dict[str, Any]) -> str:
    if isinstance(reply.get("output_text"), str):
        return reply["output_text"]
    texts = []
    for item in reply.get("output") or []:
        if isinstance(item, dict):
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    texts.append(str(part.get("text", "")))
    return "".join(texts)


def _claude_text(reply: dict[str, Any]) -> str:
    return "".join(
        str(block.get("text", ""))
        for block in reply.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _http_reason(label: str, status: int) -> str:
    if status in (401, 403):
        return f"{label} rejected the API key (HTTP {status})"
    if status == 429:
        return f"{label} is rate-limiting requests (HTTP 429); try again later"
    if status >= 500:
        return f"{label} failed (HTTP {status}); try again later"
    return f"{label} refused the request (HTTP {status}): check cloud_model and reasoning_effort"


class CloudGenerator:
    """A summary model in the cloud (summary.ModelGenerator): one HTTPS request per call. The key
    is read here, when the generator is made, so a missing key fails before any transcript is
    sent. ``close()`` drops the key from memory."""

    def __init__(self, provider: str, model: str = "", effort: str = "xhigh") -> None:
        if provider not in URLS:
            raise ValueError(f"not a cloud provider: {provider!r}")
        self.provider = provider
        self.model_name = resolve_cloud_model(provider, model)
        self.effort = effort
        self._key = get_key(provider) or ""
        if not self._key:
            raise SummaryUnavailable(
                f"no {CLOUD_LABELS[provider]} API key: set {ENV_VARS[provider]} or store one in "
                "the keychain"
            )

    def __call__(self, messages: Messages, max_tokens: int) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        budget = _budget(max_tokens)
        body: dict[str, Any]
        if self.provider == "openai":
            body = {
                "model": self.model_name,
                "instructions": system,
                "input": turns,
                "reasoning": {"effort": self.effort},
                "max_output_tokens": budget,
            }
        else:
            body = {
                "model": self.model_name,
                "system": system,
                "messages": turns,
                "max_tokens": budget,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _claude_effort(self.effort, self.model_name)},
            }
        reply = self._post(body)
        if self.provider == "openai":
            text, cut = _openai_text(reply), reply.get("status") == "incomplete"
        else:
            text = _claude_text(reply)
            cut = reply.get("stop_reason") in ("max_tokens", "refusal")
        if cut:
            raise SummaryUnavailable(TRUNCATED)
        if not text.strip():
            raise SummaryUnavailable("the cloud model returned no answer")
        return text.strip()

    def close(self) -> None:
        self._key = ""

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        label = CLOUD_LABELS[self.provider]
        if self.provider == "openai":
            headers = {"Authorization": f"Bearer {self._key}"}
        else:
            headers = {"x-api-key": self._key, "anthropic-version": ANTHROPIC_VERSION}
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        retries = 0
        while True:
            # https only: the URL is one of the constants above
            req = urllib.request.Request(  # noqa: S310
                URLS[self.provider], data=data, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:  # noqa: S310
                    raw = resp.read()
            except urllib.error.HTTPError as e:
                status = e.code
                e.close()
                if (status == 429 or status >= 500) and retries < RETRIES:
                    retries += 1
                    log.info("%s answered HTTP %d; trying again", label, status)
                    time.sleep(2.0 * 2 ** (retries - 1))
                    continue
                raise SummaryUnavailable(_http_reason(label, status)) from None
            except (OSError, http.client.HTTPException):
                raise SummaryUnavailable(
                    f"{label} could not be reached (network or timeout)"
                ) from None
            try:
                reply = json.loads(raw)
            except ValueError:
                raise SummaryUnavailable(f"{label} sent a reply that is not JSON") from None
            if not isinstance(reply, dict):
                raise SummaryUnavailable(f"{label} sent an unexpected reply")
            return reply
