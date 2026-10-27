"""Menu-bar presenter logic shared by the macOS and Windows shells: menu titles and check
marks, notification texts, the ask-mode dialog script and its answers, toast placement,
search results.

Stories: US-03, US-04, US-05, US-11, US-15, US-16, US-17, US-19, US-20.
"""

from datetime import datetime
from pathlib import Path

from recandcc.models import Mode, SessionInfo, TriggerKind
from recandcc.retention import RETENTION_CHOICES
from recandcc.search import Hit
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


def test_language_menu_reflects_the_setting():
    assert p.language_menu(["en", "ru"]) == [
        ("en", "English", True),
        ("ru", "Russian", True),
        ("he", "Hebrew", False),
        ("", "Any language", False),
    ]
    assert [on for _, _, on in p.language_menu([])] == [False, False, False, True]
    rows = p.language_menu(["he", "de", "fr"])  # codes added by hand stay visible
    assert rows[2] == ("he", "Hebrew", True)
    assert rows[3] == (None, "Also from settings: de, fr", True)


def test_toggle_languages_keeps_one_and_the_users_spelling():
    assert p.toggle_languages(["en", "ru"], None, "he") == ["en", "ru", "he"]
    assert p.toggle_languages(["en", "ru"], None, "ru") == ["en"]
    assert p.toggle_languages(["en"], None, "en") is None  # last one: refused
    assert p.toggle_languages(["en", "ru"], None, p.ANY_LANGUAGE) == []
    assert p.toggle_languages([], None, "he") == ["he"]
    # hand-written entries ("iw", codes not in the menu, unknown ones) are kept as written
    raw = ["EN", "iw", "de", "xx"]
    assert p.toggle_languages(["en", "he", "de"], raw, "ru") == [*raw, "ru"]
    assert p.toggle_languages(["en", "he", "de"], raw, "he") == ["EN", "de", "xx"]
    # raw that falls back to the defaults on load: write clean codes instead
    assert p.toggle_languages(["en", "ru"], ["xx"], "en") == ["ru"]


def test_retention_labels():
    assert p.retention_label(0) == "Never (keep audio)"
    assert p.retention_label(1) == "1 day"
    assert p.retention_label(7) == "7 days"
    assert len({p.retention_label(d) for d in RETENTION_CHOICES}) == len(RETENTION_CHOICES)


def test_permission_titles():
    assert p.permission_title("microphone", "granted") == "Microphone: ✅ granted"
    assert p.permission_title("microphone", "denied") == "Microphone: ❌ denied"
    assert p.permission_title("system_audio", "undetermined") == "System audio: ⚠️ not asked"
    assert p.permission_title("system_audio", "restricted") == "System audio: ⛔ restricted"
    assert p.permission_title("system_audio", "unknown") == "System audio: ❔ unknown"
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
    assert PROMPT_BUTTONS == (("Discard", "ignore"), ("Always", "always"), ("Keep", "record"))
    for label, choice in PROMPT_BUTTONS:
        assert p.prompt_dialog_choice(f"button returned:{label}, gave up:false\n") == choice
    # an answer that matches no button leaves it to the engine's timeout, as before
    assert p.prompt_dialog_choice("button returned:Later, gave up:false\n") is None
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


def _hit(folder: Path, start: float = 725.4, snippet: str = "…the [budget]\n for…") -> Hit:
    return Hit(folder, "s1", "2026-10-26T14:03:12", start, "Me", "en", snippet)


def test_search_hit_row_and_notices(tmp_path):
    hit = _hit(tmp_path / "2026-10-26" / "140312_zoom-us")
    row = ("2026-10-26 14:03 zoom-us", "12:05", "Me", "…the [budget] for…")
    assert p.search_hit_row(hit) == row
    assert p.search_notice([hit]) is None
    assert p.search_notice([]) == ("Search", "No matches")  # never the query: may be logged
    assert p.search_notice(RuntimeError("disk")) == ("Search failed", "disk")
    assert p.search_status("budget", [hit]) == "1 match for “budget”"
    assert p.search_status("budget", [hit, hit]) == "2 matches for “budget”"
    assert p.search_status("budget", []) == "No matches for “budget”"
    assert p.search_status("budget", RuntimeError("disk")) == "Search failed: disk"


def test_search_open_target_falls_back_to_the_folder(tmp_path):
    hit = _hit(tmp_path)
    assert p.search_open_target(hit) == tmp_path
    hit.transcript.write_text("# t", encoding="utf-8")
    assert p.search_open_target(hit) == hit.transcript
