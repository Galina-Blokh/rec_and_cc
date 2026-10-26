from recandcc.config import DetectionConfig
from recandcc.engine import classify, effective_mode, group_activity, match_pattern
from recandcc.models import AudioProcess, Mode, TriggerKind


def P(bundle, name=None, inp=False, out=True, pid=100):
    return AudioProcess(
        pid=pid, bundle_id=bundle, name=name or bundle, input_active=inp, output_active=out
    )


def test_match_case_insensitive_on_bundle_or_name():
    assert match_pattern(["us.zoom"], P("US.Zoom.xos")) == "us.zoom"
    assert match_pattern(["zoom.exe"], P("x", name="Zoom.exe")) == "zoom.exe"
    assert match_pattern(["teams"], P("com.foo")) is None
    assert match_pattern([""], P("com.foo")) is None


def test_classify_known_apps():
    det = DetectionConfig()
    assert classify(P("us.zoom.xos", inp=True), det) == ("us.zoom", TriggerKind.CALL)
    assert classify(P("us.zoom.xos", out=True), det) == ("us.zoom", TriggerKind.CALL)
    assert classify(P("com.google.Chrome"), det) == ("com.google.chrome", TriggerKind.MEDIA)


def test_browser_with_mic_is_call():
    det = DetectionConfig()
    assert classify(P("com.google.Chrome.helper", inp=True), det)[1] == TriggerKind.CALL


def test_unknown_apps():
    det = DetectionConfig()
    assert classify(P("com.acme.Thing", inp=True), det) == ("com.acme.thing", TriggerKind.CALL)
    assert classify(P("com.acme.Thing"), det) == ("com.acme.thing", TriggerKind.MEDIA)


def test_ignored_and_inactive_and_own_pid():
    det = DetectionConfig()
    assert classify(P("com.spotify.client"), det) is None
    assert classify(P("org.python.python", name="recandcc"), det) is None
    assert classify(P("com.acme", inp=False, out=False), det) is None
    assert classify(P("com.acme", pid=42), det, own_pids=[42]) is None


def test_effective_mode_override_and_master_off():
    det = DetectionConfig(mode=Mode.ASK, app_modes={"us.zoom": Mode.AUTO})
    assert effective_mode(det, "us.zoom", [P("us.zoom.xos")]) == Mode.AUTO
    assert effective_mode(det, "com.google.chrome", [P("com.google.Chrome")]) == Mode.ASK
    det.app_modes["com.google.chrome"] = Mode.OFF
    assert effective_mode(det, "com.google.chrome", [P("com.google.Chrome")]) == Mode.OFF
    det.mode = Mode.OFF
    assert effective_mode(det, "us.zoom", [P("us.zoom.xos")]) == Mode.OFF


def test_group_merges_helpers():
    det = DetectionConfig()
    apps = group_activity(
        [
            P("com.google.Chrome.helper", name="Google Chrome Helper", pid=1),
            P("com.google.Chrome", name="Google Chrome", pid=2, inp=True),
            P("us.zoom.xos", name="zoom.us", pid=3),
        ],
        det,
    )
    assert set(apps) == {"com.google.chrome", "us.zoom"}
    chrome = apps["com.google.chrome"]
    assert chrome.name == "Google Chrome"
    assert chrome.kind == TriggerKind.CALL
    assert len(chrome.procs) == 2


def test_display_name_strips_helper():
    det = DetectionConfig()
    apps = group_activity(
        [P("com.google.Chrome.helper", name="Google Chrome Helper (Aperitif)")], det
    )
    assert apps["com.google.chrome"].name == "Google Chrome"


def test_missing_permissions_and_labels(tmp_path):
    from recandcc.ui.common import missing_permissions, recent_transcripts, transcript_label

    assert missing_permissions({"microphone": "granted", "system_audio": "granted"}) == []
    assert missing_permissions({"microphone": "undetermined", "system_audio": "denied"}) == [
        "system_audio",
        "microphone",
    ]
    assert missing_permissions({"microphone": "denied"}, record_mic="never") == []
    assert missing_permissions({}) == []  # unknown is not "missing"
    d = tmp_path / "2026-10-26" / "140312_zoom-us"
    d.mkdir(parents=True)
    (d / "transcript.md").write_text("x")
    assert recent_transcripts(tmp_path) == [d / "transcript.md"]
    assert transcript_label(d / "transcript.md") == "2026-10-26 14:03 zoom-us"


def test_autostart_prefers_app_bundle(tmp_path, monkeypatch):
    import sys

    from recandcc import autostart

    app = tmp_path / "RecAndCC.app"
    app.mkdir()
    monkeypatch.setenv("RCC_APP_BUNDLE", str(app))
    assert autostart.mac_program_arguments() == ["/usr/bin/open", "-a", str(app)]
    monkeypatch.setenv("RCC_APP_BUNDLE", str(tmp_path / "missing.app"))
    monkeypatch.setattr(autostart.Path, "home", lambda: tmp_path)
    if not (autostart.Path("/Applications") / "RecAndCC.app").exists():
        assert autostart.mac_program_arguments() == [sys.executable, "-m", "recandcc", "run"]
