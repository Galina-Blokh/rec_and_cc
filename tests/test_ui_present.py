"""Menu-bar presenter logic shared by the macOS and Windows shells: menu titles and check
marks, notification texts, the ask-mode dialog script and its answers, toast placement.

Stories: US-03, US-04, US-05, US-11, US-15, US-16, US-17.
"""

from datetime import datetime
from pathlib import Path

from recandcc.models import Mode, SessionInfo, TriggerKind
from recandcc.retention import RETENTION_CHOICES
from recandcc.ui import present as p
from recandcc.ui.common import PROMPT_BUTTONS


def _session(tmp_path: Path, app: str | None = "Zoom") -> SessionInfo:
    return SessionInfo("s1", tmp_path / "2026-10-27_10-00", datetime.now(), TriggerKind.CALL, app)


def test_mode_labels_and_check_marks():
    assert [label for _, label in p.MODE_LABELS] == ["Off", "Ask", "Auto"]
    assert {m for m, _ in p.MODE_LABELS} == set(Mode)
    assert p.check(True) == 1
    assert p.check(False) == 0


def test_record_item_title_shows_state_and_shortcut():
    assert p.record_item_title(False) == "Start recording"
    assert p.record_item_title(True) == "Stop recording"
    assert p.record_item_title(True, "⌃⌥R") == "Stop recording  ⌃⌥R"
    assert p.record_item_title(False, "Ctrl+Alt+R", sep="\t") == "Start recording\tCtrl+Alt+R"


def test_retention_labels():
    assert p.retention_label(0) == "Never (keep audio)"
    assert p.retention_label(1) == "1 day"
    assert p.retention_label(7) == "7 days"
    assert len({p.retention_label(d) for d in RETENTION_CHOICES}) == len(RETENTION_CHOICES)


def test_permission_titles():
    assert p.permission_title("microphone", "granted").startswith("Microphone: ")
    assert p.permission_title("microphone", "granted") != "Microphone: granted"
    assert p.permission_title("system_audio", "weird") == "System audio: weird"


def test_prompt_and_transcribed_notices(tmp_path):
    assert p.prompt_notice("Zoom", TriggerKind.CALL) == (
        "Recording — keep it?",
        "Zoom: a call detected",
    )
    assert p.prompt_notice("Safari", TriggerKind.MEDIA)[1] == "Safari: audio detected"
    s = _session(tmp_path)
    assert p.transcribed_notice(s, [object()]) == ("Transcript ready", "Zoom")
    assert p.transcribed_notice(_session(tmp_path, None), None)[1] == "2026-10-27_10-00"
    title, msg = p.transcribed_notice(s, RuntimeError("boom"))
    assert (title, msg) == ("Transcription failed", "2026-10-27_10-00: boom")


def test_live_transcript_target_prefers_the_transcript(tmp_path):
    assert p.live_transcript_target(str(tmp_path)) == tmp_path
    (tmp_path / "transcript.md").write_text("x", "utf-8")
    assert p.live_transcript_target(tmp_path) == tmp_path / "transcript.md"


def test_prompt_dialog_script_and_timeout():
    assert p.dialog_timeout(1.5) == 5
    assert p.dialog_timeout(30.9) == 30
    script = p.prompt_dialog_script('He said "keep"', 30)
    assert script.startswith('display dialog "He said \\"keep\\""')
    assert 'buttons {"Discard", "Always", "Keep"} default button "Keep"' in script
    assert script.endswith("giving up after 30")


def test_prompt_dialog_choice_maps_buttons_and_ignores_the_rest():
    for label, choice in PROMPT_BUTTONS:
        assert p.prompt_dialog_choice(f"button returned:{label}, gave up:false\n") == choice
    assert p.prompt_dialog_choice("button returned:, gave up:true") is None
    assert p.prompt_dialog_choice("") is None
    assert p.prompt_dialog_choice(None) is None


def test_onboarding_dialog():
    script = p.onboarding_script()
    assert 'with title "Welcome to RecAndCC"' in script
    assert 'buttons {"Later", "Grant now"} default button "Grant now"' in script
    assert p.onboarding_accepted("button returned:Grant now\n")
    assert not p.onboarding_accepted("button returned:Later\n")
    assert not p.onboarding_accepted(None)


def test_toast_goes_top_right_and_expires():
    w, h = p.TOAST_SIZE
    x, y = p.toast_origin((0.0, 25.0, 1440.0, 875.0))
    assert (x, y) == (1440.0 - w - p.TOAST_MARGIN, 25.0 + 875.0 - h - p.TOAST_MARGIN)
    assert p.toast_origin((-1920.0, 0.0, 1920.0, 1080.0))[0] < 0  # a screen left of main
    assert not p.toast_expired(None, 100.0)
    assert not p.toast_expired(106.0, 105.9)
    assert p.toast_expired(106.0, 106.0)
