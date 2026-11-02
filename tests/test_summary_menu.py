"""Summary menus: the "Summaries with" rows, the sign-in prompt, keys kept in the keychain, and a
cloud summary that waits for its key (sign-in once, then the summary is queued again).

Stories: US-36.
"""

import threading
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from recandcc.config import Config, load_config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import summary
from recandcc.transcribe.output import write_outputs
from recandcc.ui import common, present
from recandcc.ui.common import Controller


class Monitor:
    def snapshot(self):
        return []


class Q:
    def submit(self, *_a, **_k):
        pass

    def pending(self):
        return 0


def _ctrl(tmp_path: Path, monkeypatch) -> tuple[Controller, Path]:
    path = tmp_path / "config.toml"
    path.write_text('[detection]\nmode = "off"\n')
    monkeypatch.setattr(common, "CONFIG_PATH", path)
    monkeypatch.setattr(common, "load_config", lambda: load_config(path))
    ctrl = Controller(
        load_config(path), monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False
    )
    return ctrl, path


def _join(name: str) -> None:
    for t in threading.enumerate():
        if t.name == name:
            t.join(5)


def _events(ctrl: Controller) -> list:
    out = []
    while not ctrl.events.empty():
        out.append(ctrl.events.get_nowait())
    return out


def _opened(monkeypatch) -> list:
    opened: list[str] = []
    monkeypatch.setattr(common, "open_path", lambda p, text_editor=False: opened.append(str(p)))
    return opened


def make_session(tmp_path: Path, segments: list[Segment]) -> Path:
    d = tmp_path / "2026-10-27" / "100000_zoom"
    d.mkdir(parents=True)
    s = SessionInfo("s1", d, datetime(2026, 10, 27, 10), TriggerKind.CALL, "zoom.us")
    s.save()
    write_outputs(s, segments, 60.0, "fake")
    return d


def test_the_submenu_checks_only_the_current_provider():
    rows = present.summary_provider_rows("openai")
    assert [p for p, _, _ in rows] == ["local", "openai", "anthropic"]
    assert [title for _, title, on in rows if on] == ["ChatGPT (OpenAI)"]


def test_the_sign_in_prompt_hides_the_answer_and_says_what_a_key_costs():
    script = present.signin_script("openai")
    assert "with hidden answer" in script and "billed per use" in script
    assert "spending limit" in script and "platform.openai.com" not in script  # the page opens
    assert present.KEY_PAGES["openai"] == "https://platform.openai.com/api-keys"
    assert present.signin_answer("button returned:Save, text returned:sk-x") == "sk-x"
    assert present.signin_answer("") == ""  # cancelled


def test_choosing_a_provider_with_a_key_saves_only_that_setting(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    fake_keychain[("RecAndCC", "openai")] = "sk-stored"
    ctrl.set_summary_provider("openai", lambda p: pytest.fail("no prompt: a key is stored"))
    _join("summary-provider")
    assert ctrl.cfg.summary.provider == "openai"
    assert tomllib.loads(path.read_text())["summary"]["provider"] == "openai"
    assert ("summaries_changed",) in _events(ctrl)


def test_a_cloud_provider_without_a_key_opens_its_page_and_stays_put_when_cancelled(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    opened = _opened(monkeypatch)
    asked = []
    ctrl.set_summary_provider("anthropic", lambda p: asked.append(p) or "")
    _join("summary-provider")
    assert asked == ["anthropic"]
    assert opened == [present.KEY_PAGES["anthropic"]]
    assert ctrl.cfg.summary.provider == "local"


def test_a_typed_key_is_kept_and_one_with_spaces_is_refused(tmp_path, monkeypatch, fake_keychain):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    _opened(monkeypatch)
    notes: list[tuple[str, str]] = []
    ctrl.post_notice = lambda title, message: notes.append((title, message))
    assert ctrl.sign_in("openai", lambda p: "sk-new") is True
    assert fake_keychain[("RecAndCC", "openai")] == "sk-new"
    assert ctrl.sign_in("openai", lambda p: "two words") is False
    assert fake_keychain[("RecAndCC", "openai")] == "sk-new"  # the good key stays
    assert notes and "not an API key" in notes[0][1] and "two" not in notes[0][1]


def test_a_summary_without_a_key_opens_the_sign_in_once_and_is_queued_again(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    opened = _opened(monkeypatch)
    notes: list[tuple[str, str]] = []
    ctrl.post_notice = lambda title, message: notes.append((title, message))
    queued: list[Path] = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)
    gate = threading.Event()  # the prompt waits until both summaries are waiting
    asked: list[str] = []

    def ask(provider: str) -> str:
        asked.append(provider)
        gate.wait(5)
        return "sk-saved"

    a, b = tmp_path / "a", tmp_path / "b"
    ctrl.on_summary_needs_key("openai", a, ask)
    ctrl.on_summary_needs_key("openai", b, ask)  # same provider: waits, no second prompt
    gate.set()
    _join("summary-sign-in")
    assert asked == ["openai"] and opened == [present.KEY_PAGES["openai"]]
    assert len(notes) == 1 and "API key" in notes[0][1]
    assert queued == [a, b]
    assert fake_keychain[("RecAndCC", "openai")] == "sk-saved"


def test_a_cloud_summary_with_no_key_waits_instead_of_failing(tmp_path, fake_keychain):
    waited: list[tuple[str, Path]] = []
    cfg = Config()
    cfg.summary.provider = "openai"
    queue = summary.SummaryQueue(cfg, needs_key=lambda p, d: waited.append((p, d)))
    d = make_session(tmp_path, [Segment(5, 9, "we ship the beta", "Me", "en")])
    queue.submit(d)
    queue.join()
    assert waited == [("openai", d)]
    assert (d / summary.PENDING_MARKER).exists()  # still owed: queued again after the sign-in
    assert not (d / "summary.md").exists()
