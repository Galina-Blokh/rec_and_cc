"""Stories: US-01, US-02, US-03, US-04, US-05, US-06, US-07, US-16, US-17, US-22."""

import shutil
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
            run_in_background=lambda fn: fn(),
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


# --- discard must stick, even when Windows still locks a file (US-03)

DAY = "2026-10-27"


def _fill(s: SessionInfo) -> SessionInfo:
    """Give a session what the real Recorder + live transcript leave on disk."""
    from datetime import datetime

    s.directory.mkdir(parents=True, exist_ok=True)
    s.ended_at = datetime.now()
    s.tracks = {Track.OTHERS: "others.wav", Track.ME: "me.wav"}
    for name in ("others.wav", "me.wav", "transcript.md"):
        (s.directory / name).write_bytes(b"x")
    s.save()
    return s


class LockedWavRecorder(DiscardingRecorder):
    def stop(self):
        return _fill(super().stop())


def _lock(monkeypatch, *names):
    """Make deleting these files fail the way Windows does while they are still open."""
    real_unlink = Path.unlink

    def unlink(self, *a, **kw):
        if self.name in names:
            raise PermissionError(13, "file in use", str(self))
        return real_unlink(self, *a, **kw)

    monkeypatch.setattr(Path, "unlink", unlink)


def _discard_one(tmp_path):
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = LockedWavRecorder(tmp_path / DAY)
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.eng.tick()
    return h


def test_failed_discard_is_marked_and_never_transcribed(tmp_path, monkeypatch):
    from recandcc.ui.common import recent_transcripts, untranscribed_sessions

    _lock(monkeypatch, "others.wav", "transcript.md")
    h = _discard_one(tmp_path)
    d = h.rec.dirs[0]
    assert (d / "others.wav").exists() and h.q.submitted == []
    assert not (d / "me.wav").exists()  # one locked file doesn't keep the rest
    assert (d / SessionInfo.DISCARD_MARKER).exists()
    assert h.notices == ["Recording not fully deleted"]
    assert untranscribed_sessions(tmp_path) == []  # startup recovery skips it
    assert recent_transcripts(tmp_path) == []  # and the menu doesn't list it


def test_discard_sticks_when_no_marker_can_be_written(tmp_path, monkeypatch):
    from recandcc.ui.common import purge_discarded_sessions, untranscribed_sessions

    real_touch = Path.touch

    def touch(self, *a, **kw):
        if self.name == SessionInfo.DISCARD_MARKER:
            raise PermissionError(13, "denied", str(self))
        return real_touch(self, *a, **kw)

    with monkeypatch.context() as m:
        _lock(m, "others.wav")
        m.setattr(Path, "touch", touch)
        h = _discard_one(tmp_path)
        d = h.rec.dirs[0]
        assert (d / "others.wav").exists() and untranscribed_sessions(tmp_path) == []
    assert purge_discarded_sessions(tmp_path) == 0  # session.json became the marker
    assert not d.exists()


def test_startup_retry_deletes_leftover_discarded_session(tmp_path, monkeypatch):
    from recandcc.ui.common import purge_discarded_sessions, untranscribed_sessions

    with monkeypatch.context() as m:
        _lock(m, "others.wav")
        h = _discard_one(tmp_path)
        assert purge_discarded_sessions(tmp_path) == 1  # still locked: left for next time
    kept = LockedWavRecorder(tmp_path / DAY)
    kept.stops = 5
    kept.stop()  # an ordinary unfinished session must survive the purge
    truncated = tmp_path / DAY / "crashed"  # discarded, with a cut-short session.json
    truncated.mkdir()
    (truncated / SessionInfo.FILENAME).write_text("{", encoding="utf-8")
    (truncated / SessionInfo.DISCARD_MARKER).touch()
    stray = tmp_path / DAY / "unreadable"  # unreadable but not discarded: left alone
    stray.mkdir()
    (stray / SessionInfo.FILENAME).write_text("{", encoding="utf-8")
    assert purge_discarded_sessions(tmp_path) == 0
    assert not h.rec.dirs[0].exists() and not truncated.exists() and stray.exists()
    assert [s.id for s in untranscribed_sessions(tmp_path)] == ["s6"]


class RealDirRecorder(FakeRecorder):
    """Session on disk from start(); stop() behaves as `outcome` says."""

    def __init__(self, root, outcome):
        super().__init__()
        self.dir, self.outcome, self.marked_at_stop = root / DAY / "s", outcome, None

    def start(self, kind, app=None):
        self.session = super().start(kind, app)
        self.session.directory = self.dir
        return _fill(self.session)

    def stop(self):
        self.marked_at_stop = (self.dir / SessionInfo.DISCARD_MARKER).exists()
        if self.outcome == "raise":
            raise RuntimeError("boom")
        if self.outcome == "too_short":  # recorder's own rmtree(ignore_errors) left files
            (self.dir / "me.wav").unlink()
            return None
        self.session.save()  # what Recorder.stop() does last
        return self.session


@pytest.mark.parametrize("outcome", ["ok", "raise", "too_short"])
def test_discard_marks_before_stop_and_always_deletes(tmp_path, outcome):
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = RealDirRecorder(tmp_path, outcome)
    h.mon.procs = [CHROME]
    h.run(4)
    h.eng.respond("ignore")
    h.eng.tick()
    assert h.rec.marked_at_stop  # a crash during stop() can't leave it recoverable
    assert h.status == IDLE and not h.rec.dir.exists() and h.q.submitted == []


def test_discard_of_vanished_too_short_session_schedules_no_delete(tmp_path):
    """stop() already removed it; a late delete could hit a new session reusing the name."""
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = RealDirRecorder(tmp_path, "too_short")
    scheduled = []
    h.eng.run_in_background = scheduled.append
    h.mon.procs = [CHROME]
    h.run(4)
    real_stop = h.rec.stop

    def stop_and_remove():
        real_stop()
        shutil.rmtree(h.rec.dir)  # Recorder.stop() deleted the whole folder
        return None

    h.rec.stop = stop_and_remove
    h.eng.respond("ignore")
    h.eng.tick()
    assert h.status == IDLE and scheduled == [] and not h.rec.dir.exists()


def test_background_discard_error_is_logged_not_raised(tmp_path, monkeypatch, caplog):
    import recandcc.engine as engine

    def boom(directory):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "discard_session", boom)
    h = Harness(Mode.ASK)
    h.rec = h.eng.recorder = RealDirRecorder(tmp_path, "ok")
    h.mon.procs = [CHROME]
    h.run(4)
    with caplog.at_level("ERROR", logger="recandcc.engine"):
        h.eng.respond("ignore")
        h.eng.tick()  # inline run_in_background: would raise here if not caught
    assert h.status == IDLE
    assert any("could not delete discarded" in r.message and r.exc_info for r in caplog.records)


def test_discard_deletes_off_the_engine_thread(tmp_path, monkeypatch):
    import threading
    import time

    from recandcc.engine import _run_daemon

    release, real_unlink = threading.Event(), Path.unlink

    def slow_unlink(self, *a, **kw):
        if self.suffix == ".wav":
            release.wait(5)
        return real_unlink(self, *a, **kw)

    h = Harness(Mode.ASK)
    h.eng.run_in_background = _run_daemon
    h.rec = h.eng.recorder = RealDirRecorder(tmp_path, "ok")
    h.mon.procs = [CHROME]
    h.run(4)
    monkeypatch.setattr(Path, "unlink", slow_unlink)
    h.eng.respond("ignore")
    h.eng.tick()  # returns although the WAV delete is still blocked
    assert h.status == IDLE and h.rec.dir.exists()
    release.set()
    deadline = time.monotonic() + 5
    while h.rec.dir.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not h.rec.dir.exists() and h.notices == []


def test_discard_tolerates_vanished_files_and_save_is_atomic(tmp_path):
    from datetime import datetime

    from recandcc.models import discard_session

    assert discard_session(tmp_path / "gone")  # already deleted (e.g. by a racing purge)
    s = _fill(SessionInfo("s", tmp_path / "d", datetime.now(), TriggerKind.CALL))
    names = sorted(p.name for p in s.directory.iterdir())
    assert names == ["me.wav", "others.wav", SessionInfo.FILENAME, "transcript.md"]  # no temp
    assert discard_session(s.directory) and not s.directory.exists()


def _saved(tmp_path):
    from datetime import datetime

    s = SessionInfo("s", tmp_path, datetime.now(), TriggerKind.CALL)
    s.save()
    s.trigger_app = "zoom.us"
    return s


def _no_temp(tmp_path):
    return [p.name for p in tmp_path.iterdir()] == [SessionInfo.FILENAME]


@pytest.mark.parametrize("failures", [2, 99])  # retried until it works / falls back in place
def test_save_survives_a_session_json_held_open(tmp_path, monkeypatch, failures):
    import recandcc.models as models

    real_replace, calls = models.os.replace, []

    def replace(src, dst):
        calls.append(dst)
        if len(calls) <= failures:
            raise PermissionError(13, "in use", str(dst))
        return real_replace(src, dst)

    s = _saved(tmp_path)
    monkeypatch.setattr(models.os, "replace", replace)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    s.save()
    assert len(calls) == min(failures + 1, models._REPLACE_TRIES)
    assert SessionInfo.load(tmp_path).trigger_app == "zoom.us" and _no_temp(tmp_path)


def test_save_cut_short_keeps_the_previous_session_json(tmp_path, monkeypatch):
    real_write = Path.write_text

    def write_text(self, data, *a, **kw):
        if self.name.endswith(".tmp"):
            real_write(self, data[:10], *a, **kw)  # crash / disk full mid-write
            raise OSError(28, "no space left")
        return real_write(self, data, *a, **kw)

    s = _saved(tmp_path)
    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(OSError):
        s.save()
    assert SessionInfo.load(tmp_path).trigger_app is None and _no_temp(tmp_path)


# --- room listening ----------------------------------------------------------------------
VOICE_MEMOS = AudioProcess(
    pid=13,
    bundle_id="com.apple.VoiceMemos",
    name="VoiceMemos",
    input_active=True,
    output_active=False,
)
ROOMS_APP = AudioProcess(
    pid=14,
    bundle_id="com.example.chatrooms",
    name="Chatrooms",
    input_active=False,
    output_active=True,
)


class FakeRoom:
    def __init__(self):
        self.running = self.speech = self.fail = False
        self.quiet = 0.0  # seconds_since_speech()
        self.opens = self.closes = 0

    def start(self):
        if self.fail:
            return False
        self.opens += not self.running
        self.running = True
        return True

    def close_mic(self):
        self.closes += self.running
        self.running = self.speech = False

    close = close_mic

    def triggered(self):
        return self.running and self.speech

    def seconds_since_speech(self):
        return self.quiet

    def session_source(self):  # only its identity is checked: the fake recorder opens nothing
        raise AssertionError("not opened by the fake recorder")


def _room_harness(mode=Mode.AUTO, **det_kw):
    h = _upgrade_harness() if mode == Mode.AUTO else Harness(mode)
    for k, v in {"room_listening": True, **det_kw}.items():
        setattr(h.cfg.detection, k, v)
    h.room = h.eng.room = FakeRoom()
    h.mic_sources = []
    start = h.rec.start

    def start_checked(kind, app=None, mic_source=None):
        if mic_source is None:  # the recorder opens the mic itself: the room mic is closed
            assert not h.room.running
        else:  # a room session records the monitor's own stream (pre-roll + live)
            assert kind == TriggerKind.ROOM and mic_source == h.room.session_source
        h.mic_sources.append(mic_source)
        return start(kind, app)

    h.rec.start = start_checked
    return h


def _start_room_session(h):
    h.run(1)
    h.room.speech = True
    h.eng.tick()
    assert h.rec.starts[-1] == (TriggerKind.ROOM, "Room") and h.mic_sources[-1] is not None


def test_room_listening_off_never_opens_the_mic():
    h = _room_harness(room_listening=False)
    h.room.speech = True
    h.run(30)
    assert h.room.opens == 0 and h.rec.starts == []
    h.room.speech = False
    h.cfg.detection.room_listening = True
    h.run(2)
    assert h.room.opens == 1 and h.room.running
    h.cfg.detection.room_listening = False  # switched off (menu = pause): the mic closes
    h.run(2)
    assert not h.room.running and h.room.closes == 1


def test_room_session_records_the_monitor_stream_and_stops_on_no_speech():
    h = _room_harness()
    _start_room_session(h)
    assert h.eng.state.kind == TriggerKind.ROOM
    h.run(300)  # people keep talking: no app to go inactive, no stop
    assert h.status == RECORDING and h.room.running  # the session's mic stays open
    h.room.quiet = 90  # room_stop_after_silence_s without speech
    h.eng.tick()
    assert h.status == IDLE and h.rec.stops == 1 and len(h.q.submitted) == 1
    assert h.room.running  # listening again
    h.room.quiet = 0  # a new conversation right after: not suppressed
    h.eng.tick()
    assert len(h.rec.starts) == 2


def test_apps_take_precedence_over_room_speech():
    h = _room_harness()
    h.mon.procs = [ZOOM]
    h.run(4)
    assert h.rec.starts == [(TriggerKind.CALL, "zoom.us")] and h.mic_sources == [None]
    assert not h.room.running  # closed before the recorder opened its mic
    h.mon.procs = []
    h.run(25)  # the call ended; the room monitor reopened
    assert h.status == IDLE and h.room.running
    _start_room_session(h)
    h.mon.procs = [ZOOM]  # a call starts during the room session: follow it, no new session
    h.run(4)
    assert h.rec.upgrades == ["zoom.us"] and h.eng.state.kind == TriggerKind.CALL
    assert h.rec.stops == 1 and len(h.rec.starts) == 2 and h.room.running
    h.mon.procs, h.room.speech = [], False
    h.run(25)  # now the call's inactivity rule applies
    assert h.status == IDLE and h.rec.stops == 2


def test_room_never_records_what_an_app_switched_off_or_stopped_is_doing():
    h = _room_harness(app_modes={"us.zoom": Mode.OFF})
    h.mon.procs = [ZOOM]  # a call the user excluded: the mic hears it
    h.run(10)
    h.room.speech = True
    h.run(10)
    assert h.rec.starts == []
    h.mon.procs = [VOICE_MEMOS]  # an ignored app using the mic
    h.run(10)
    assert h.rec.starts == []
    h.mon.procs = [AudioProcess(999, "python", "python3", True, False)]  # our own room mic
    h.eng.tick()
    assert h.rec.starts == [(TriggerKind.ROOM, "Room")]


def test_stopped_room_session_stays_stopped_while_the_talk_goes_on():
    h = _room_harness()
    _start_room_session(h)
    h.eng.stop()
    h.eng.tick()
    assert h.status == IDLE and h.room.running
    for _ in range(40):
        h.room.speech = True
        h.run(1)
    assert len(h.rec.starts) == 1
    h.room.speech = False
    h.run(25)  # the room was quiet for stop_after_inactive_s
    h.room.speech = True
    h.eng.tick()
    assert len(h.rec.starts) == 2


def test_room_listening_follows_modes_and_mic_policy():
    h = _room_harness(Mode.ASK)
    _start_room_session(h)
    assert h.prompts == [("Room", TriggerKind.ROOM)] and h.status == RECORDING
    h.eng.respond("always")
    h.eng.tick()
    assert h.cfg.detection.app_modes == {"@room": Mode.AUTO}
    h.eng.stop()
    h.room.speech = False
    h.run(25)
    h.mon.procs = [ROOMS_APP]  # "always" for the room must not auto-record "Chatrooms"
    h.run(4)
    assert h.prompts[-1] == ("Chatrooms", TriggerKind.MEDIA)
    h.eng.respond("ignore")
    h.mon.procs = []
    h.eng.set_mode(Mode.OFF)
    h.run(2)
    assert not h.room.running
    h.eng.set_mode(Mode.AUTO)
    h.cfg.detection.record_mic = "never"  # a room session without the mic records nothing
    h.run(2)
    assert not h.room.running


def test_room_mic_failure_backs_off_and_shutdown_closes_it():
    h = _room_harness()
    h.room.fail = True
    h.run(30)
    assert h.room.opens == 0 and not h.room.running
    h.room.fail = False
    h.run(31)
    assert h.room.opens == 1 and h.room.running
    h.eng.shutdown()
    assert not h.room.running
