"""Stories: US-01, US-02, US-03, US-04, US-05, US-06, US-07, US-16, US-17."""

import threading
import time
from pathlib import Path

import pytest

from recandcc.config import Config, DetectionConfig
from recandcc.engine import IDLE, RECORDING, Engine
from recandcc.models import AudioProcess, Mode, SessionInfo, Track, TriggerKind


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeMonitor:
    def __init__(self):
        self.procs: list[AudioProcess] = []
        self.closed = False
        self.threads = {"snapshot": set(), "close": []}

    def snapshot(self):
        self.threads["snapshot"].add(threading.current_thread().name)
        return list(self.procs)

    def close(self):
        self.closed = True
        self.threads["close"].append(threading.current_thread().name)


class FakeRecorder:
    def __init__(self):
        self.is_recording = False
        self.starts: list[tuple] = []
        self.stops = 0
        self.silence = 0.0
        self.fail = False
        self.silent = []

    def start(self, kind, app=None):
        if self.fail:
            raise RuntimeError("boom")
        self.is_recording = True
        self.starts.append((kind, app))
        from datetime import datetime

        return SessionInfo(
            id="s",
            directory=Path("/tmp/x"),
            started_at=datetime.now(),
            trigger_kind=kind,
            trigger_app=app,
        )

    def stop(self):
        self.is_recording = False
        self.stops += 1
        from datetime import datetime

        return SessionInfo(
            id=f"s{self.stops}",
            directory=Path("/tmp/x"),
            started_at=datetime.now(),
            trigger_kind=TriggerKind.CALL,
        )

    def seconds_since_sound(self):
        return self.silence

    def silent_tracks(self):
        return self.silent


class FakeQueue:
    def __init__(self):
        self.submitted = []

    def submit(self, s):
        self.submitted.append(s)


ZOOM = AudioProcess(
    pid=10, bundle_id="us.zoom.xos", name="zoom.us", input_active=True, output_active=True
)
CHROME = AudioProcess(
    pid=11,
    bundle_id="com.google.Chrome",
    name="Google Chrome",
    input_active=False,
    output_active=True,
)
SPOTIFY = AudioProcess(
    pid=12, bundle_id="com.spotify.client", name="Spotify", input_active=False, output_active=True
)


class Harness:
    def __init__(self, mode=Mode.AUTO, **det_kw):
        det = DetectionConfig(
            mode=mode,
            min_active_s=3,
            stop_after_inactive_s=20,
            stop_after_silence_s=180,
            ask_timeout_s=30,
            **det_kw,
        )
        self.cfg = Config(detection=det)
        self.clock = FakeClock()
        self.mon = FakeMonitor()
        self.rec = FakeRecorder()
        self.q = FakeQueue()
        self.prompts = []
        self.states = []
        self.notices = []
        self.saves = 0
        self.perm = {"microphone": "granted", "system_audio": "granted"}

        def save():
            self.saves += 1

        self.eng = Engine(
            self.cfg,
            self.mon,
            self.rec,
            self.q,
            clock=self.clock,
            on_prompt=lambda a, k: self.prompts.append((a, k)),
            on_state_change=self.states.append,
            save_config=save,
            own_pids=[999],
            on_notice=lambda t, m: self.notices.append(t),
            permission_status=lambda: self.perm,
        )

    def run(self, seconds, step=1.0):
        end = self.clock.t + seconds
        while self.clock.t < end:
            self.eng.tick()
            self.clock.t += step
        self.eng.tick()

    @property
    def status(self):
        return self.eng.state.status


def test_auto_start_and_stop_on_inactivity():
    h = Harness(Mode.AUTO)
    h.mon.procs = [ZOOM]
    h.run(1)
    assert h.status == IDLE  # debounce
    h.run(3)
    assert h.status == RECORDING
    assert h.rec.starts == [(TriggerKind.CALL, "zoom.us")]
    assert h.eng.state.app == "zoom.us"
    h.run(10)
    h.mon.procs = []
    h.run(15)
    assert h.status == RECORDING  # still within stop_after_inactive_s
    h.run(6)
    assert h.status == IDLE
    assert h.rec.stops == 1
    assert len(h.q.submitted) == 1
    assert [s.status for s in h.states] == [RECORDING, IDLE]


def test_min_active_debounce_resets_on_gap():
    h = Harness(Mode.AUTO)
    for _ in range(5):
        h.mon.procs = [CHROME]
        h.run(2)
        h.mon.procs = []
        h.run(1)
    assert h.rec.starts == []


def test_off_mode_does_nothing():
    h = Harness(Mode.OFF, app_modes={"us.zoom": Mode.AUTO})
    h.mon.procs = [ZOOM, CHROME]
    h.run(30)
    assert h.status == IDLE and h.rec.starts == [] and h.prompts == []


def test_ignore_list_and_own_pid():
    h = Harness(Mode.AUTO)
    me = AudioProcess(
        pid=999,
        bundle_id="org.python.python",
        name="Python",
        input_active=True,
        output_active=False,
    )
    h.mon.procs = [SPOTIFY, me]
    h.run(30)
    assert h.rec.starts == []


def test_app_mode_override_auto_in_ask():
    h = Harness(Mode.ASK, app_modes={"us.zoom": Mode.AUTO})
    h.mon.procs = [ZOOM]
    h.run(4)
    assert h.status == RECORDING and h.prompts == []


def test_app_mode_override_off():
    h = Harness(Mode.AUTO, app_modes={"chrome": Mode.OFF})
    h.mon.procs = [CHROME]
    h.run(10)
    assert h.rec.starts == []


def test_call_preferred_over_media():
    h = Harness(Mode.AUTO)
    h.mon.procs = [CHROME, ZOOM]
    h.run(4)
    assert h.rec.starts == [(TriggerKind.CALL, "zoom.us")]


class DiscardingRecorder(FakeRecorder):
    """FakeRecorder whose sessions live in a real directory, so discards can be checked."""

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.dirs = []

    def stop(self):
        s = super().stop()
        s.directory = self.root / f"s{self.stops}"
        s.directory.mkdir(parents=True)
        self.dirs.append(s.directory)
        return s


def test_ask_records_immediately_and_record_keeps_it():
    h = Harness(Mode.ASK)
    h.mon.procs = [CHROME]
    h.run(4)
    # Recording starts with the prompt, so nothing is lost while the user decides.
    assert h.prompts == [("Google Chrome", TriggerKind.MEDIA)]
    assert h.rec.starts == [(TriggerKind.MEDIA, "Google Chrome")]
    assert h.status == RECORDING and h.eng.state.prompt_id is not None
    h.run(5)
    assert len(h.prompts) == 1  # no re-prompt while pending
    h.eng.respond("record")
    h.eng.tick()
    assert h.status == RECORDING and h.eng.state.prompt_id is None
    assert len(h.rec.starts) == 1 and h.rec.stops == 0
    h.mon.procs = []
    h.run(25)
    assert h.rec.stops == 1 and len(h.q.submitted) == 1


def test_ask_ignore_discards_and_suppresses_until_audio_stops(tmp_path):
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = DiscardingRecorder(tmp_path)
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.eng.tick()
    assert h.status == IDLE and h.rec.stops == 1
    assert not h.rec.dirs[0].exists()  # discarded, not transcribed
    assert h.q.submitted == []
    h.run(60)
    assert h.status == IDLE and len(h.prompts) == 1
    h.mon.procs = []
    h.run(25)
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.status == RECORDING and len(h.prompts) == 2


def test_ask_always_sets_auto_and_saves():
    h = Harness(Mode.ASK)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.eng.respond("always")
    h.eng.tick()
    assert h.status == RECORDING and len(h.rec.starts) == 1
    assert h.cfg.detection.app_modes == {"us.zoom": Mode.AUTO}
    assert h.saves == 1
    # next time: no prompt
    h.mon.procs = []
    h.run(25)
    assert h.status == IDLE
    h.mon.procs = [ZOOM]
    h.run(4)
    assert h.status == RECORDING and len(h.prompts) == 1


def test_unanswered_prompt_keeps_recording_by_default():
    h = Harness(Mode.ASK)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.run(31)  # nobody clicked
    assert h.status == RECORDING and h.eng.state.prompt_id is None
    assert h.rec.stops == 0
    h.run(120)
    assert len(h.prompts) == 1  # no re-prompt for the same call
    h.mon.procs = []
    h.run(25)
    assert h.rec.stops == 1 and len(h.q.submitted) == 1


def test_unanswered_prompt_discards_when_configured(tmp_path):
    h = Harness(Mode.ASK, ask_timeout_action="ignore")
    h.rec = h.eng.recorder = DiscardingRecorder(tmp_path)
    h.mon.procs = [CHROME]
    h.run(4)
    h.run(31)
    assert h.status == IDLE and h.q.submitted == []
    assert not h.rec.dirs[0].exists()
    h.run(30)
    assert len(h.prompts) == 1


def test_audio_ending_before_answer_keeps_recording():
    h = Harness(Mode.ASK)
    h.cfg.detection.ask_timeout_s = 300
    h.mon.procs = [CHROME]
    h.run(4)
    h.mon.procs = []
    h.run(25)  # short clip ended while the prompt was still open
    assert h.status == IDLE and h.rec.stops == 1 and len(h.q.submitted) == 1
    assert h.eng.state.prompt_id is None


def test_manual_stop_while_prompt_pending_keeps_recording():
    h = Harness(Mode.ASK)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.eng.stop()
    h.eng.tick()
    assert h.status == IDLE and len(h.q.submitted) == 1


def test_stale_prompt_id_ignored(tmp_path):
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = DiscardingRecorder(tmp_path)
    h.mon.procs = [CHROME]
    h.run(4)
    pid = h.eng.state.prompt_id
    h.eng.respond("ignore", prompt_id=pid + 100)
    h.eng.tick()
    assert h.status == RECORDING and h.eng.state.prompt_id == pid
    h.eng.respond("ignore", prompt_id=pid)
    h.eng.tick()
    assert h.status == IDLE and h.rec.stops == 1


def test_silence_stop():
    h = Harness(Mode.AUTO)
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.status == RECORDING
    h.rec.silence = 200
    h.eng.tick()
    assert h.status == IDLE and h.rec.stops == 1
    h.rec.silence = 0
    h.run(10)  # app still "active" but silent: must not immediately restart
    assert len(h.rec.starts) == 1


def test_manual_session_not_auto_stopped():
    h = Harness(Mode.AUTO)
    h.eng.start_manual()
    h.eng.tick()
    assert h.status == RECORDING and h.eng.state.manual
    assert h.rec.starts == [(TriggerKind.MANUAL, None)]
    h.rec.silence = 10_000
    h.mon.procs = [ZOOM]
    h.run(100)
    assert h.status == RECORDING and len(h.rec.starts) == 1
    h.clock.t += 5
    assert h.eng.state.elapsed == pytest.approx(105, abs=2)
    h.eng.stop()
    h.eng.tick()
    assert h.status == IDLE and len(h.q.submitted) == 1


def test_manual_stop_of_auto_session_suppresses_restart():
    h = Harness(Mode.AUTO)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.eng.stop()
    h.run(10)
    assert h.status == IDLE and len(h.rec.starts) == 1


def test_set_mode_off_stops_auto_recording():
    h = Harness(Mode.AUTO)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.eng.set_mode(Mode.OFF)
    h.eng.tick()
    assert h.status == IDLE and h.cfg.detection.mode == Mode.OFF and h.saves == 1
    assert h.states[-1].mode == Mode.OFF


def test_start_failure_returns_to_idle_without_retry_loop():
    h = Harness(Mode.AUTO)
    h.rec.fail = True
    h.mon.procs = [ZOOM]
    h.run(10)
    assert h.status == IDLE


def test_permission_notices():
    h = Harness(Mode.AUTO)
    h.perm = {"microphone": "granted", "system_audio": "denied"}
    h.rec.silent = [Track.OTHERS]
    h.eng.start_manual()
    h.eng.tick()
    assert h.notices == ["Recording may be silent"]
    h.eng.stop()
    h.eng.tick()
    assert h.notices[-1] == "Recording was completely silent"


def test_thread_and_shutdown():
    import time

    h = Harness(Mode.AUTO)
    h.cfg.detection.poll_interval_s = 0.01
    h.eng.clock = time.monotonic
    h.eng.start()
    h.eng.start_manual()
    deadline = time.time() + 2
    while h.status != RECORDING and time.time() < deadline:
        time.sleep(0.01)
    assert h.status == RECORDING
    h.eng.shutdown()
    assert h.status == IDLE and h.rec.stops == 1 and len(h.q.submitted) == 1
    assert h.mon.closed


def test_monitor_closed_on_the_engine_thread():
    """Per-thread monitor resources (Windows COM) are released by the thread that polled."""
    h = Harness(Mode.AUTO)  # fake clock: no recording starts, only polling
    h.cfg.detection.poll_interval_s = 0.01
    h.eng.start()
    deadline = time.monotonic() + 2
    while not h.mon.threads["snapshot"] and time.monotonic() < deadline:
        time.sleep(0.01)
    h.eng.shutdown()
    assert h.mon.threads["snapshot"] == {"engine"}
    assert h.mon.threads["close"][0] == "engine"  # then once more from shutdown()'s caller
    h.eng.shutdown()  # a second shutdown is harmless


# --- a call starting during a media recording -------------------------------------------
def _upgrade_harness():
    h = Harness(mode=Mode.AUTO)
    h.rec.upgrades = []
    h.rec.upgrade_to_call = h.rec.upgrades.append
    return h


def test_call_during_media_recording_upgrades_session():
    h = _upgrade_harness()
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.rec.starts == [(TriggerKind.MEDIA, "Google Chrome")]
    h.mon.procs = [CHROME, ZOOM]
    h.run(4)
    assert h.rec.upgrades == ["zoom.us"]
    assert h.eng.state.kind == TriggerKind.CALL and h.eng.state.app == "zoom.us"
    assert h.rec.stops == 0  # same session continues
    # The video stops but the call goes on: the session must keep recording.
    h.mon.procs = [ZOOM]
    h.run(60)
    assert h.rec.stops == 0 and h.eng.state.status == RECORDING
    # Call ends → session stops after stop_after_inactive_s.
    h.mon.procs = []
    h.run(25)
    assert h.rec.stops == 1
    assert len(h.rec.starts) == 1


def test_same_browser_starting_mic_upgrades_immediately():
    h = _upgrade_harness()
    h.mon.procs = [CHROME]
    h.run(4)
    meet = AudioProcess(
        pid=11,
        bundle_id="com.google.Chrome",
        name="Google Chrome",
        input_active=True,
        output_active=True,
    )
    h.mon.procs = [meet]
    h.eng.tick()
    assert h.rec.upgrades == ["Google Chrome"]
    assert h.eng.state.kind == TriggerKind.CALL


def test_call_upgrade_respects_off_override_and_debounce():
    h = _upgrade_harness()
    h.cfg.detection.app_modes["us.zoom"] = Mode.OFF
    h.mon.procs = [CHROME]
    h.run(4)
    h.mon.procs = [CHROME, ZOOM]
    h.run(10)
    assert h.rec.upgrades == []
    h2 = _upgrade_harness()
    h2.mon.procs = [CHROME]
    h2.run(4)
    h2.mon.procs = [CHROME, ZOOM]
    h2.run(1)  # call active < min_active_s
    assert h2.rec.upgrades == []


def test_media_stops_sooner_than_calls():
    h = Harness(Mode.AUTO)
    h.mon.procs = [CHROME]
    h.run(4)
    h.mon.procs = []
    h.run(9)  # media_stop_after_inactive_s = 8
    assert h.rec.stops == 1
    h2 = Harness(Mode.AUTO)
    h2.mon.procs = [ZOOM]
    h2.run(4)
    h2.mon.procs = []
    h2.run(9)
    assert h2.rec.stops == 0  # calls wait stop_after_inactive_s = 20
    h2.run(12)
    assert h2.rec.stops == 1


def test_media_silence_stop_is_shorter():
    h = Harness(Mode.AUTO)
    h.mon.procs = [CHROME]  # e.g. a paused tab that keeps its audio stream open
    h.run(4)
    h.rec.silence = 31.0
    h.eng.tick()
    assert h.rec.stops == 1


def test_silent_track_warning_only_without_permission():
    h = Harness(Mode.AUTO)
    h.rec.silent = [Track.OTHERS]  # nothing played during the session
    h.mon.procs = [ZOOM]
    h.run(4)
    h.mon.procs = []
    h.run(25)
    assert "Recording was completely silent" not in h.notices  # permission is granted
    h.perm = {"microphone": "granted", "system_audio": "denied"}
    h.mon.procs = [CHROME]
    h.run(4)
    h.mon.procs = []
    h.run(10)
    assert "Recording was completely silent" in h.notices


# --- session start/end callbacks (live transcription) -----------------------------------
def _hooked(h):
    calls = []
    h.eng.on_session_start = lambda s: calls.append(("start", s.trigger_app))

    def end(s, kept):
        # discarded sessions are reported before their directory is deleted
        exists = s is not None and s.directory.exists()
        calls.append(("end", kept, exists, len(h.q.submitted)))

    h.eng.on_session_end = end
    return calls


def test_session_callbacks_on_stop():
    h = Harness(Mode.AUTO)
    calls = _hooked(h)
    h.mon.procs = [ZOOM]
    h.run(4)
    assert calls == [("start", "zoom.us")]
    assert h.eng.state.session_dir  # populated before/while the live session runs
    h.mon.procs = []
    h.run(25)
    assert calls[1][:2] == ("end", True) and calls[1][3] == 0  # before the queue submit
    assert len(h.q.submitted) == 1


def test_session_callbacks_on_discard(tmp_path):
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = DiscardingRecorder(tmp_path)
    calls = _hooked(h)
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.eng.tick()
    assert calls == [("start", "Google Chrome"), ("end", False, True, 0)]
    assert not h.rec.dirs[0].exists() and h.q.submitted == []


def test_session_end_reported_when_recorder_drops_session():
    h = Harness(Mode.AUTO)
    calls = _hooked(h)
    h.rec.stop = lambda: None  # too short: the recorder deleted it
    h.eng.start_manual()
    h.eng.tick()
    h.eng.stop()
    h.eng.tick()
    assert calls == [("start", None), ("end", False, False, 0)]


def test_failing_session_callbacks_do_not_break_engine():
    h = Harness(Mode.AUTO)

    def boom(*_):
        raise RuntimeError("x")

    h.eng.on_session_start = h.eng.on_session_end = boom
    h.eng.start_manual()
    h.eng.tick()
    assert h.status == RECORDING
    h.eng.stop()
    h.eng.tick()
    assert h.status == IDLE and len(h.q.submitted) == 1
