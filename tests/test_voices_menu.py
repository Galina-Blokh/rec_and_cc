"""Menu access to speaker names: the Controller's worker functions with fakes (no AppKit).

Stories: US-26.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import time

import pytest
from test_transcribe import make_session
from test_voices import ANNA, BOB, CARL, _final

from recandcc.config import Config, TranscriptionConfig
from recandcc.transcribe import voices
from recandcc.ui.common import Controller
from recandcc.ui.present import (
    forgot_notice,
    named_notice,
    speaker_choice_label,
    speakers_notice,
)


class SpyLock:
    """A lock that tells when a second thread starts waiting for it."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.waiting = threading.Event()

    def __enter__(self):
        if not self._lock.acquire(blocking=False):
            self.waiting.set()
            self._lock.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


def controller(speaker_names: bool = True) -> Controller:
    ctrl = Controller.__new__(Controller)
    ctrl.events = queue.Queue()
    ctrl.cfg = Config(transcription=TranscriptionConfig(speaker_names=speaker_names))
    return ctrl


def test_menu_naming_updates_profile_and_transcript_like_the_cli(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    ctrl.speakers_async(tmp_path)
    kind, directory, speakers = ctrl.events.get(timeout=5)
    assert (kind, directory) == ("speakers", tmp_path)
    assert [label for label, _ in speakers] == ["Speaker 1", "Speaker 2"]
    assert speakers_notice(speakers) is None and "hi" in speaker_choice_label(2, *speakers[1])
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert ctrl.events.get(timeout=5) == ("named", "Speaker 2", "Carl", None)
    assert "Carl:** hi" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert "Carl" in {s["speaker"] for s in data["segments"]}
    (profile,) = voices.load_profiles()
    assert profile.name == "Carl" and float(profile.centroid @ CARL) > 0.9
    assert named_notice(None)[0] == "Speaker named"


def test_naming_is_refused_when_speaker_names_is_off(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller(speaker_names=False)
    assert not ctrl.speaker_names_enabled
    ctrl.name_speaker_async(tmp_path, "Speaker 1", "Bob")
    _, _, _, error = ctrl.events.get(timeout=5)
    assert error and voices.load_profiles() == []
    assert "Speaker 1:** hello" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert named_notice(error)[0] == "Can't name the speaker"


def test_session_without_voice_data_has_no_speakers_to_name(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=False)
    ctrl = controller()
    ctrl.speakers_async(tmp_path)
    _, _, speakers = ctrl.events.get(timeout=5)
    assert speakers == [] and speakers_notice(speakers)[0] == "No speakers to name"


def test_forget_all_voices_removes_profiles_and_session_voices(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    voices.save_profile("Anna", ANNA)
    voices.save_profile("Bob", BOB)
    ctrl = controller()
    ctrl.forget_voices_async()
    kind, count = ctrl.events.get(timeout=5)
    assert (kind, count) == ("forgot", 2)
    assert voices.load_profiles() == [] and voices.read_session_voices("s1") == {}
    assert "2 voice profile" in forgot_notice(count)[1]
    assert forgot_notice(OSError())[0] == "Can't delete voices"


def test_names_are_never_logged(tmp_path, monkeypatch, caplog):
    _final(tmp_path, monkeypatch, speaker_names=True)
    caplog.set_level(logging.INFO)
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Zebulon")
    ctrl.events.get(timeout=5)
    ctrl.name_speaker_async(tmp_path, "Speaker 9", "Zebulon")  # fails
    ctrl.events.get(timeout=5)
    ctrl.forget_voices_async()
    ctrl.events.get(timeout=5)
    assert "Zebulon" not in caplog.text and "hello" not in caplog.text


def test_unexpected_error_still_posts_a_name_free_outcome(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    monkeypatch.setattr(voices, "name_speaker", lambda *a: (_ for _ in ()).throw(TypeError("x")))
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Zebulon")
    _, _, _, error = ctrl.events.get(timeout=5)
    assert error and "Zebulon" not in error and "Speaker 2" not in error
    assert "Zebulon" not in " ".join(named_notice(None) + named_notice(error))


def test_forget_waits_for_a_running_naming(tmp_path, monkeypatch):
    import threading

    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    started, release = threading.Event(), threading.Event()
    real_write = voices.write_outputs

    def slow_write(*args, **kwargs):  # the real naming, stalled on its transcript rewrite
        started.set()
        assert release.wait(5)
        return real_write(*args, **kwargs)

    spy = SpyLock()
    monkeypatch.setattr(voices, "NAMING_LOCK", spy)
    monkeypatch.setattr(voices, "write_outputs", slow_write)
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert started.wait(5)
    ctrl.forget_voices_async()
    assert spy.waiting.wait(5)  # forget is queued on NAMING_LOCK behind the naming
    assert ctrl.events.empty() and voices.load_profiles()  # ...and has deleted nothing yet
    release.set()
    assert ctrl.events.get(timeout=5)[0] == "named"
    assert ctrl.events.get(timeout=5) == ("forgot", 1)
    assert voices.load_profiles() == []  # the forget came last: nothing came back


def test_choice_label_flattens_multiline_hints():
    label = speaker_choice_label(1, "Speaker 1", ["hello\nthere", "a\r\n b"])
    assert "\n" not in label and "\r" not in label


def test_two_namings_of_one_session_both_survive(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    ctrl = controller()
    ctrl.name_speaker_async(tmp_path, "Speaker 1", "Anna")
    ctrl.name_speaker_async(tmp_path, "Speaker 2", "Carl")
    assert [ctrl.events.get(timeout=5)[3] for _ in range(2)] == [None, None]
    data = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert {"Anna", "Carl"} <= {s["speaker"] for s in data["segments"]}


def test_diarization_started_before_a_forget_cannot_bring_voices_back(tmp_path, monkeypatch):
    from test_transcribe import make_session

    session = make_session(tmp_path)
    started = voices.forget_generation()
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=started)
    assert voices.read_session_voices(session.id)
    voices.delete_all_profiles()
    assert voices.read_session_voices(session.id) == {}
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=started)  # began before
    assert voices.read_session_voices(session.id) == {}
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=voices.forget_generation())
    assert voices.read_session_voices(session.id)


def test_atomic_write_survives_concurrent_writers_and_a_busy_file(tmp_path, monkeypatch):
    import threading
    from pathlib import Path

    from recandcc.transcribe.output import atomic_write

    target = tmp_path / "t.md"
    errors = []

    def write(i):
        try:
            for _ in range(20):
                atomic_write(target, f"text {i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and target.read_text(encoding="utf-8").startswith("text ")
    assert [p.name for p in tmp_path.iterdir()] == ["t.md"]  # no temp left behind

    real, calls = Path.replace, []

    def busy(self, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("held")
        return real(self, dst)

    monkeypatch.setattr(Path, "replace", busy)
    atomic_write(target, "after retries")
    assert target.read_text(encoding="utf-8") == "after retries" and len(calls) == 3


def test_names_from_deleted_profiles_are_not_matched_after_a_forget(tmp_path, monkeypatch):
    from test_voices import SEGS, TURNS

    from recandcc.transcribe.engine import Transcriber

    _final(tmp_path, monkeypatch, speaker_names=True)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    stale = voices.forget_generation()
    voices.delete_all_profiles()
    voices.save_profile("Anna", ANNA)  # a profile that appeared after the forget
    session = make_session(tmp_path)
    out = t._labelled(session, list(SEGS), TURNS, stale)
    assert "Anna" not in {s.speaker for s in out}  # the stale run gets no names
    out = t._labelled(session, list(SEGS), TURNS, voices.forget_generation())
    assert "Anna" in {s.speaker for s in out}


def test_a_forget_in_another_process_drops_a_stale_diarization_write(tmp_path, monkeypatch):
    """The generation lives in a file, so the CLI's `voices delete --all` counts too."""
    import threading

    session = make_session(tmp_path)
    started = voices.forget_generation()
    # "another process": its own lock objects, nothing shared with this one but the folder
    monkeypatch.setattr(voices, "LOCK", threading.RLock())
    voices.delete_all_profiles()
    assert (voices.VOICES_DIR / voices.GENERATION_FILE).is_file()
    monkeypatch.setattr(voices, "LOCK", threading.RLock())  # and back in the app process
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=started)
    assert voices.read_session_voices(session.id) == {}
    assert voices.profiles_if_current(started) == []
    voices.write_session_voices(session, {"Speaker 1": ANNA}, since=voices.forget_generation())
    assert voices.read_session_voices(session.id)


def test_forget_between_voice_write_and_transcript_rewrite_wins(tmp_path, monkeypatch):
    _final(tmp_path, monkeypatch, speaker_names=True)
    real = voices.write_outputs

    def forget_first(*args, **kwargs):
        voices.delete_all_profiles()  # lands after the voices were saved, before the transcript
        return real(*args, **kwargs)

    monkeypatch.setattr(voices, "write_outputs", forget_first)
    voices.name_speaker(tmp_path, "Speaker 2", "Carl")
    assert "Carl:** hi" in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert voices.load_profiles() == [] and voices.read_session_voices("s1") == {}


def test_slow_relabel_keeps_a_name_and_holds_the_naming_lock(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from test_voices import SEGS, TURNS

    from recandcc.models import SessionInfo
    from recandcc.transcribe import engine as engine_mod
    from recandcc.transcribe.engine import Transcriber

    _final(tmp_path, monkeypatch, speaker_names=True)
    voices.name_speaker(tmp_path, "Speaker 2", "Carl")
    before = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    done: Future = Future()
    done.set_result(TURNS)
    t._relabel_later(SessionInfo.load(tmp_path), list(SEGS), 60.0, done)
    assert (tmp_path / "transcript.md").read_text(encoding="utf-8") == before  # still Carl

    held = []
    real = engine_mod.write_outputs
    monkeypatch.setattr(
        engine_mod,
        "write_outputs",
        lambda *a, **k: (held.append(voices.NAMING_LOCK.locked()), real(*a, **k))[1],
    )
    fresh = tmp_path / "fresh"  # a session nobody named: the relabel writes
    fresh.mkdir()
    voices.delete_all_profiles()  # or its voices would match Carl
    _final(fresh, monkeypatch, speaker_names=True)
    held.clear()
    t._relabel_later(SessionInfo.load(fresh), list(SEGS), 60.0, done)
    assert held and all(held)


def test_forget_retries_a_busy_file_and_keeps_going_after_a_failure(tmp_path, monkeypatch):
    from pathlib import Path

    voices.save_profile("Anna", ANNA)
    voices.save_profile("Bob", BOB)
    real, seen = Path.unlink, []

    def flaky(self, missing_ok=False):
        if self.suffix == ".json" and self not in seen:
            seen.append(self)
            raise PermissionError("held")  # busy once: the retry succeeds
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    assert voices.delete_all_profiles() == 2 and voices.load_profiles() == []

    voices.save_profile("Anna", ANNA)
    voices.save_profile("Bob", BOB)
    first = sorted(voices.VOICES_DIR.glob("*.json"))[0]

    def stuck(self, missing_ok=False):
        if self == first:
            raise PermissionError("held for good")
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", stuck)
    with pytest.raises(OSError, match="1 voice file"):
        voices.delete_all_profiles()
    assert list(voices.VOICES_DIR.glob("*.json")) == [first]  # the rest is gone


def test_dialog_scripts_quote_names_and_hints():
    from recandcc.ui.common import chooser_script, name_dialog_script

    label = speaker_choice_label(1, 'Speaker "1"', ['say "hi" \\ there', "x"])
    script = chooser_script([label, "2. B"], 'Who is "this"?', "Name…")
    assert 'say \\"hi\\" \\\\ there' in script
    assert 'with prompt "Who is \\"this\\"?"' in script
    assert "\n" not in script
    assert 'Name for Speaker \\"1\\":' in name_dialog_script('Speaker "1"')


def test_every_voice_mutation_takes_the_file_lock_reentrantly(tmp_path, monkeypatch):
    taken = []
    real = voices._file_lock

    def spy(*args, **kwargs):
        taken.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(voices, "_file_lock", spy)
    session = make_session(tmp_path)
    voices.save_profile("Anna", ANNA)  # also nests inside name_speaker and delete_profile
    voices.write_session_voices(session, {"Speaker 1": ANNA})
    voices.delete_session_voices(session.id)
    assert voices.delete_profile("Anna") and voices.delete_all_profiles() == 0
    assert len(taken) >= 5


def test_fast_final_write_holds_the_naming_lock(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from test_voices import SEGS, TURNS

    from recandcc.models import SessionInfo
    from recandcc.transcribe import engine as engine_mod
    from recandcc.transcribe.engine import Transcriber

    _final(tmp_path, monkeypatch, speaker_names=True)
    held = []
    real = engine_mod.write_outputs
    monkeypatch.setattr(
        engine_mod,
        "write_outputs",
        lambda *a, **k: (held.append(voices.NAMING_LOCK.locked()), real(*a, **k))[1],
    )
    done: Future = Future()
    done.set_result(TURNS)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    t.write_final(SessionInfo.load(tmp_path), list(SEGS), 60.0, done)
    assert held == [True]


def test_relabel_never_overwrites_names_even_after_a_forget(tmp_path, monkeypatch):
    from concurrent.futures import Future

    from test_voices import SEGS, TURNS

    from recandcc.models import SessionInfo
    from recandcc.transcribe.engine import Transcriber

    _final(tmp_path, monkeypatch, speaker_names=True)
    voices.name_speaker(tmp_path, "Speaker 2", "Carl")
    voices.delete_all_profiles()  # the session voices are gone: only the transcript tells
    assert voices.has_user_names(tmp_path)
    before = (tmp_path / "transcript.md").read_text(encoding="utf-8")
    done: Future = Future()
    done.set_result(TURNS)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    t._relabel_later(SessionInfo.load(tmp_path), list(SEGS), 60.0, done)
    assert (tmp_path / "transcript.md").read_text(encoding="utf-8") == before


def test_prune_does_not_hold_the_file_lock_while_checking_folders(tmp_path, monkeypatch):
    held = []
    monkeypatch.setattr(
        voices, "_deleted", lambda d: held.append(voices._depth(voices.VOICES_DIR)) or True
    )
    voices.write_session_voices(make_session(tmp_path), {"Speaker 1": ANNA})
    assert voices.prune_session_voices() == 1
    assert held == [0]


def test_stale_run_neither_names_nor_deletes_after_a_forget(tmp_path, monkeypatch):
    from test_voices import SEGS, TURNS

    from recandcc.transcribe.engine import Transcriber

    _final(tmp_path, monkeypatch, speaker_names=True)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    session = make_session(tmp_path)
    voices.save_profile("Anna", ANNA)
    stale = voices.forget_generation()
    real = voices.profiles_if_current

    def forget_after_loading(since, voices_dir=None):
        profiles = real(since, voices_dir)
        voices.delete_all_profiles()  # lands between the match and the write
        return profiles

    monkeypatch.setattr(voices, "profiles_if_current", forget_after_loading)
    out = t._labelled(session, list(SEGS), TURNS, stale)
    assert "Anna" not in {s.speaker for s in out}
    monkeypatch.setattr(voices, "profiles_if_current", real)
    voices.write_session_voices(session, {"Speaker 1": ANNA})  # stored after the forget
    t._labelled(session, list(SEGS), None, stale)  # an old run with no voices to store
    assert voices.read_session_voices(session.id)  # ...must not delete the newer ones


def test_file_lock_depth_is_kept_per_folder(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    with voices._file_lock(a):
        assert voices._depth(a) == 1 and voices._depth(b) == 0
        with voices._file_lock(b):  # another folder: a real lock of its own
            assert voices._depth(b) == 1
        with voices._file_lock(a):
            assert voices._depth(a) == 2
    assert voices._depth(a) == 0 and voices._depth(b) == 0


def test_windows_lock_polls_and_gives_up_cleanly(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0)

    def locking(fd, mode, n):
        calls.append(mode)
        raise OSError("busy")

    fake.locking = locking
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(voices, "LOCK_WAIT_S", 0.1)

    class F:
        def seek(self, n): ...
        def fileno(self):
            return 0

    with pytest.raises(OSError, match="in use by another process"):
        voices._lock_windows(F())
    assert len(calls) > 1  # it polled instead of failing on the first try


def test_osascript_timeout_returns_empty(monkeypatch):
    import subprocess

    from recandcc.ui import common

    def slow(*a, **k):
        assert k["timeout"] == common.DIALOG_TIMEOUT_S
        raise subprocess.TimeoutExpired("osascript", k["timeout"])

    monkeypatch.setattr(common.subprocess, "run", slow)
    assert common.osascript("x", common.DIALOG_TIMEOUT_S) == ""


def test_posix_lock_gives_up_instead_of_hanging(monkeypatch):
    import sys

    if sys.platform == "win32":
        pytest.skip("POSIX flock")
    import fcntl

    monkeypatch.setattr(voices, "LOCK_WAIT_S", 0.1)

    def busy(f, op):
        raise BlockingIOError("held")

    monkeypatch.setattr(fcntl, "flock", busy)
    with pytest.raises(OSError, match="in use by another process"), voices.store_lock():
        pass
    with voices.store_lock_best_effort():  # finalize carries on
        pass


def test_forget_while_a_session_is_queued_keeps_no_voices_or_names(tmp_path, monkeypatch):
    """The forget generation is taken when recording starts and carried with the session: a
    "Forget all voices" while it waits in the queue leaves nothing to name for it."""
    from concurrent.futures import Future

    from test_voices import SEGS, TURNS

    from recandcc import participants
    from recandcc.models import SessionInfo
    from recandcc.transcribe.engine import Transcriber

    session = make_session(tmp_path)
    session.voices_generation = voices.forget_generation()  # as Recorder.start sets it
    session.save()
    voices.save_profile("Anna", ANNA)
    ctrl = controller()
    ctrl.forget_voices_async()  # while the session is queued
    assert ctrl.events.get(timeout=5)[0] == "forgot"
    participants.store(session, ["Anna Smith", "Bob Jones"])  # gathered before the forget
    assert voices.read_session_names(session.id) == []
    loaded = SessionInfo.load(tmp_path)  # the generation survives a reload from disk
    assert loaded.voices_generation == session.voices_generation
    done: Future = Future()
    done.set_result(TURNS)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    assert t.write_final(loaded, list(SEGS), 60.0, done)
    assert voices.read_session_voices(session.id) == {}
    ctrl.speakers_async(tmp_path)
    _, _, speakers = ctrl.events.get(timeout=5)
    assert speakers == []  # "Name speakers…" finds nothing for it


def test_session_recorded_after_a_forget_keeps_voices_and_names(tmp_path, monkeypatch):
    from recandcc import participants

    voices.delete_all_profiles()
    session = make_session(tmp_path)
    session.voices_generation = voices.forget_generation()
    participants.store(session, ["Anna Smith"])
    assert voices.read_session_names(session.id) == ["Anna Smith"]
    _final(tmp_path, monkeypatch, speaker_names=True)
    assert voices.read_session_voices(session.id)


def test_queue_stamps_an_older_session_without_a_generation(tmp_path):
    from recandcc.transcribe.queue import TranscriptionQueue

    class _Fake:
        def transcribe_session(self, session):
            return []

    session = make_session(tmp_path)
    assert session.voices_generation is None
    q = TranscriptionQueue(_Fake())  # type: ignore[arg-type]
    try:
        q.submit(session)
    finally:
        q.shutdown()
    assert session.voices_generation == voices.forget_generation()


def test_forget_dialog_has_a_timeout_and_a_timeout_cancels(monkeypatch):
    import threading

    pytest.importorskip("rumps")
    from recandcc.ui import mac

    seen: list = []
    done = threading.Event()

    def fake_osascript(script, timeout=None):
        seen.append(timeout)
        done.set()
        return ""  # what osascript returns when the dialog times out

    monkeypatch.setattr(mac, "osascript", fake_osascript)
    app = mac.MacApp.__new__(mac.MacApp)
    app.ctrl = type("C", (), {"forget_voices_async": lambda self: seen.append("forgot")})()
    app._forget_voices(None)
    assert done.wait(5)
    time.sleep(0.05)
    assert seen == [mac.DIALOG_TIMEOUT_S]  # no forget after the timeout


def test_a_forget_by_another_process_after_matching_leaves_no_name_in_the_text(
    tmp_path, monkeypatch
):
    """The names were matched, then a forget (another process: no NAMING_LOCK) landed before
    the transcript was written: it is written with numbers, and no voices are kept."""
    from concurrent.futures import Future

    from test_voices import SEGS, TURNS

    from recandcc.transcribe.engine import Transcriber

    voices.save_profile("Anna", ANNA)
    real = Transcriber._labelled
    calls = []

    def labelled_then_forget(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        if not calls:
            assert "Anna" in {s.speaker for s in out}  # matched before the forget
            voices.delete_all_profiles()
        calls.append(1)
        return out

    monkeypatch.setattr(Transcriber, "_labelled", labelled_then_forget)
    session = make_session(tmp_path)
    done: Future = Future()
    done.set_result(TURNS)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    out = t.write_final(session, list(SEGS), 60.0, done)
    assert "Anna" not in {s.speaker for s in out}
    assert "Anna" not in (tmp_path / "transcript.md").read_text(encoding="utf-8")
    assert voices.read_session_voices(session.id) == {}


def test_busy_store_makes_finalize_wait_once_and_skip_voice_work(tmp_path, monkeypatch):
    from test_voices import SEGS, TURNS

    from recandcc.transcribe.engine import Transcriber

    voices.save_profile("Anna", ANNA)
    waits = []

    def busy(f):
        waits.append(1)
        raise OSError("the voice data is in use by another process; try again")

    real_posix, real_windows = voices._lock_posix, voices._lock_windows
    monkeypatch.setattr(voices, "_lock_posix", busy)
    monkeypatch.setattr(voices, "_lock_windows", busy)
    session = make_session(tmp_path)
    t = Transcriber(TranscriptionConfig(model="fake", speaker_names=True))
    out = t._labelled(session, list(SEGS), TURNS, voices.forget_generation())
    assert waits == [1]  # one LOCK_WAIT_S at most, not one per voice step
    assert "Anna" not in {s.speaker for s in out} and "Speaker 1" in {s.speaker for s in out}
    # restore only the lock stubs (never monkeypatch.undo(): it would also drop the private
    # voices folder and read the user's real one)
    monkeypatch.setattr(voices, "_lock_posix", real_posix)
    monkeypatch.setattr(voices, "_lock_windows", real_windows)
    assert voices.read_session_voices(session.id) == {}


def test_waiting_for_another_process_does_not_hold_the_in_process_lock(tmp_path, monkeypatch):
    import threading

    polling, release = threading.Event(), threading.Event()
    real = voices._lock_windows if os.name == "nt" else voices._lock_posix

    def slow_os_lock(f):  # another process holds the voices lock until `release`
        polling.set()
        assert release.wait(5)
        real(f)

    monkeypatch.setattr(voices, "_lock_posix", slow_os_lock)
    monkeypatch.setattr(voices, "_lock_windows", slow_os_lock)
    worker = threading.Thread(target=lambda: voices.save_profile("Anna", ANNA))
    worker.start()
    try:
        assert polling.wait(5)
        assert voices.LOCK.acquire(timeout=1)  # free while that thread waits for the OS lock
        voices.LOCK.release()
    finally:
        release.set()
        worker.join(5)
    assert [p.name for p in voices.load_profiles()] == ["Anna"]


def test_profile_left_by_a_failed_forget_never_matches(tmp_path, monkeypatch):
    from test_transcribe import make_session as session_in

    voices.save_profile("Anna", ANNA)
    voices.save_profile("Bob", BOB)
    session = session_in(tmp_path)
    voices.write_session_voices(session, {"Speaker 1": ANNA})
    voices.write_session_names(session, ["Anna Smith"])
    anna = voices._hashed(voices.VOICES_DIR, voices._key("Anna"))
    real = voices._retry

    def stuck(op):
        if getattr(op, "__defaults__", None) and op.__defaults__[0] == anna:
            raise PermissionError("held by a scanner")
        return real(op)

    monkeypatch.setattr(voices, "_retry", stuck)
    with pytest.raises(OSError, match="could not be deleted"):
        voices.delete_all_profiles()
    assert anna.exists()  # left behind...
    assert voices.load_profiles() == []  # ...but it never matches again
    voices.save_profile("Anna", CARL)  # a new profile of that name starts afresh
    (profile,) = voices.load_profiles()
    assert profile.count == 1 and float(profile.centroid @ CARL) > 0.99


def test_naming_guard_blocks_a_second_dialog_until_released():
    pytest.importorskip("rumps")
    import threading

    from recandcc.ui import mac

    app = mac.MacApp.__new__(mac.MacApp)
    app._naming_open, app._naming_lock = False, threading.Lock()
    assert app._begin_naming()
    assert not app._begin_naming()  # a second chooser while one is up
    app._end_naming()
    assert app._begin_naming()


def test_live_markdown_never_sleeps_on_a_busy_file(tmp_path, monkeypatch):
    from pathlib import Path

    from recandcc.transcribe import output

    slept = []
    monkeypatch.setattr(output.time, "sleep", lambda s: slept.append(s))

    def busy(self, target):
        raise PermissionError("held by a reader")

    monkeypatch.setattr(Path, "replace", busy)
    with pytest.raises(PermissionError):
        output.atomic_write(tmp_path / "transcript.md", "x", tries=1)
    assert slept == [] and not list(tmp_path.glob(".*.tmp"))
    tries = []
    monkeypatch.setattr(output, "atomic_write", lambda p, t, **k: tries.append(k))
    output.write_live_markdown(make_session(tmp_path), [], 1.0)
    assert tries == [{"tries": 1}]
