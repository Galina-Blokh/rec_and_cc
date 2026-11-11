"""Summary menus: the "Summaries with" rows, the sign-in prompt, keys kept in the keychain, and a
cloud summary that waits for its key (sign-in once, then the summary is queued again).

Stories: US-36.
"""

import threading
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from recandcc.config import Config, load_config, save_keys
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
    expected = ["local", "openai", "anthropic", "claude-code", "codex", "grok"]
    assert [p for p, _, _ in rows] == expected
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


def test_an_old_cloud_dialog_cannot_override_a_later_local_selection(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    _opened(monkeypatch)
    opened, release = threading.Event(), threading.Event()

    def ask(provider):
        opened.set()
        assert release.wait(5)
        return "sk-older-dialog"

    ctrl.set_summary_provider("openai", ask)
    assert opened.wait(5)
    ctrl.set_summary_provider("local", lambda p: pytest.fail("Local needs no key"))
    assert ctrl.cfg.summary.provider == "local"  # immediate, before a worker saves it
    release.set()
    _join("summary-provider")
    _join("save-config")
    assert ctrl.cfg.summary.provider == "local"
    assert tomllib.loads(path.read_text())["summary"]["provider"] == "local"


def test_a_failed_local_selection_save_cannot_restore_cloud(tmp_path, monkeypatch, fake_keychain):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    save_keys(path, {"summary.provider": "openai"})
    ctrl.cfg.summary.provider = "openai"

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(common, "save_keys", refuse)
    ctrl.set_summary_provider("local", lambda p: pytest.fail("Local needs no key"))
    _join("save-config")
    assert ctrl.cfg.summary.provider == "local"
    assert "summary.provider" in ctrl._unsaved


def test_selecting_a_ready_provider_resumes_waiting_manual_summaries(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    fake_keychain[("RecAndCC", "openai")] = "sk-stored"
    d = tmp_path / "waiting"
    ctrl._waiting_summaries["openai"] = [d]
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)
    ctrl.set_summary_provider("openai", lambda p: pytest.fail("a key is stored"))
    _join("summary-provider")
    assert queued == [d] and not ctrl._waiting_summaries


def test_waiting_summaries_are_not_retried_on_every_idle_refresh(tmp_path, monkeypatch):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl.cfg.summary.enabled = True
    ctrl.cfg.output_dir = tmp_path
    d = make_session(tmp_path, [Segment(0, 5, "Release on Friday.", "Me", "en")])
    summary.mark_pending(d)
    ctrl._waiting_summaries["openai"] = [d]
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)
    ctrl._resubmit_summaries()
    ctrl._resubmit_summaries()
    assert queued == []


def test_cancelled_sign_in_explains_how_to_resume_without_reopening_the_dialog(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl.cfg.summary.provider = "openai"
    opened = _opened(monkeypatch)
    asked = []

    def ask(provider):
        asked.append(provider)
        return ""

    ctrl.on_summary_needs_key("openai", tmp_path / "a", ask)
    _join("summary-sign-in")
    _events(ctrl)
    ctrl.on_summary_needs_key("openai", tmp_path / "a", ask)
    _join("summary-key-check")
    assert asked == ["openai"] and len(opened) == 1
    assert any(e[1] == "Summary waiting for API key" and "Sign in" in e[2] for e in _events(ctrl))


def test_a_missing_key_event_arriving_after_sign_in_requeues_immediately(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl.cfg.summary.provider = "openai"
    ctrl._signin_started.add("openai")
    fake_keychain[("RecAndCC", "openai")] = "sk-saved"
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)
    d = tmp_path / "late"
    ctrl.on_summary_needs_key("openai", d, lambda p: pytest.fail("already signed in"))
    _join("summary-key-check")
    assert queued == [d]


@pytest.mark.parametrize("current", ["local", "anthropic"])
def test_a_late_missing_key_event_uses_the_new_provider_without_an_old_prompt(
    tmp_path, monkeypatch, fake_keychain, current
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl.cfg.summary.provider = "openai"
    fake_keychain[("RecAndCC", "anthropic")] = "sk-stored"
    opened = _opened(monkeypatch)
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", lambda d: queued.append((d, ctrl.cfg.summary.provider)))
    ctrl.set_summary_provider(current, lambda p: pytest.fail("no key prompt needed"))
    _join("summary-provider")
    d = tmp_path / "late"
    ctrl.on_summary_needs_key("openai", d, lambda p: pytest.fail("stale provider"))
    _join("summary-requeue")
    assert queued == [(d, current)] and opened == []
    assert ctrl._waiting_summaries == {}


def test_config_provider_change_resumes_waiting_summaries_with_automatic_summaries_off(
    tmp_path, monkeypatch
):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    save_keys(path, {"summary.provider": "openai"})
    ctrl.cfg.summary.provider = "openai"
    assert not ctrl.cfg.summary.enabled
    d = tmp_path / "waiting"
    ctrl._waiting_summaries["openai"] = [d]
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)
    save_keys(path, {"summary.provider": "local"})
    ctrl._config_mtime = -1
    ctrl._maybe_reload_config()
    assert ctrl.cfg.summary.provider == "local" and queued == [d]
    assert not ctrl._waiting_summaries


def test_config_local_choice_also_supersedes_an_older_cloud_dialog(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, path = _ctrl(tmp_path, monkeypatch)
    save_keys(path, {"summary.provider": "openai"})
    ctrl.cfg.summary.provider = "openai"
    _opened(monkeypatch)
    opened, release = threading.Event(), threading.Event()

    def ask(provider):
        opened.set()
        assert release.wait(5)
        return "sk-older-dialog"

    ctrl.set_summary_provider("anthropic", ask)
    assert opened.wait(5)
    save_keys(path, {"summary.provider": "local"})
    ctrl._config_mtime = -1
    ctrl._maybe_reload_config()
    release.set()
    _join("summary-provider")
    assert ctrl.cfg.summary.provider == "local"
    assert tomllib.loads(path.read_text())["summary"]["provider"] == "local"


def test_cancelling_an_old_prompt_after_local_selection_does_not_claim_a_key_is_needed(
    tmp_path, monkeypatch, fake_keychain
):
    ctrl, _ = _ctrl(tmp_path, monkeypatch)
    ctrl.cfg.summary.provider = "openai"
    _opened(monkeypatch)
    opened, release = threading.Event(), threading.Event()
    queued = []
    monkeypatch.setattr(ctrl, "_summarize", queued.append)

    def ask(provider):
        opened.set()
        assert release.wait(5)
        return ""

    d = tmp_path / "waiting"
    ctrl.on_summary_needs_key("openai", d, ask)
    assert opened.wait(5)
    ctrl.set_summary_provider("local", lambda p: pytest.fail("Local needs no key"))
    _join("summary-requeue")
    release.set()
    _join("summary-sign-in")
    assert queued == [d]
    assert not any(
        e[0] == "notice" and e[1] == "Summary waiting for API key" for e in _events(ctrl)
    )


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
    ctrl.cfg.summary.provider = "openai"
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


@pytest.mark.parametrize("announce", [False, True])
def test_waiting_for_a_key_preserves_manual_announcement_and_allows_immediate_retry(
    tmp_path, monkeypatch, announce
):
    calls, waiting, completed = [], [], []
    d = make_session(tmp_path, [Segment(0, 5, "Release on Friday.", "Me", "en")])

    def write(directory, *_args, **_kwargs):
        calls.append(directory)
        if len(calls) == 1:
            raise summary.NeedsKey("openai", "An API key is required")
        (directory / "summary.md").write_text("# Meeting summary\n", "utf-8")
        return directory / "summary.md"

    def saved_key(provider, directory):
        waiting.append((provider, directory))
        queue.submit(directory)  # retry can arrive as soon as the key callback runs

    monkeypatch.setattr(summary, "summarize_session", write)
    queue = summary.SummaryQueue(
        Config(),
        needs_key=saved_key,
        on_done=lambda directory, written, requested: completed.append(
            (directory, written, requested)
        ),
    )
    queue.submit(d, announce=announce)
    queue.join()
    assert calls == [d, d] and waiting == [("openai", d)]
    assert completed == [(d, True, announce)]
    assert not (d / summary.PENDING_MARKER).exists()
