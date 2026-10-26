from pathlib import Path

import pytest

from recandcc.config import Config, DetectionConfig
from recandcc.engine import IDLE, PROMPTING, RECORDING, Engine
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

    def snapshot(self):
        return list(self.procs)

    def close(self):
        self.closed = True


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


def test_ask_record():
    h = Harness(Mode.ASK)
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.status == PROMPTING
    assert h.prompts == [("Google Chrome", TriggerKind.MEDIA)]
    h.run(5)
    assert len(h.prompts) == 1  # no re-prompt while pending
    h.eng.respond("record")
    h.eng.tick()
    assert h.status == RECORDING
    assert h.rec.starts == [(TriggerKind.MEDIA, "Google Chrome")]


def test_ask_ignore_suppresses_until_audio_stops():
    h = Harness(Mode.ASK)
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.run(60)
    assert h.status == IDLE and len(h.prompts) == 1
    h.mon.procs = []
    h.run(25)
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.status == PROMPTING and len(h.prompts) == 2


def test_ask_always_sets_auto_and_saves():
    h = Harness(Mode.ASK)
    h.mon.procs = [ZOOM]
    h.run(4)
    h.eng.respond("always")
    h.eng.tick()
    assert h.status == RECORDING
    assert h.cfg.detection.app_modes == {"us.zoom": Mode.AUTO}
    assert h.saves == 1
    # next time: no prompt
    h.mon.procs = []
    h.run(25)
    assert h.status == IDLE
    h.mon.procs = [ZOOM]
    h.run(4)
    assert h.status == RECORDING and len(h.prompts) == 1


def test_ask_timeout_expires_and_suppresses():
    h = Harness(Mode.ASK)
    h.mon.procs = [CHROME]
    h.run(4)
    assert h.status == PROMPTING
    h.run(31)
    assert h.status == IDLE
    h.run(30)
    assert len(h.prompts) == 1 and h.rec.starts == []


def test_stale_prompt_id_ignored():
    h = Harness(Mode.ASK)
    h.mon.procs = [CHROME]
    h.run(4)
    pid = h.eng.state.prompt_id
    h.eng.respond("record", prompt_id=pid + 100)
    h.eng.tick()
    assert h.status == PROMPTING
    h.eng.respond("record", prompt_id=pid)
    h.eng.tick()
    assert h.status == RECORDING


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
