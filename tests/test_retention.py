"""Audio retention and keep_audio = false cleanup of transcribed sessions.

Stories: US-15.
"""

from datetime import datetime, timedelta

import pytest

from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.retention import purge_old_audio, purge_transcribed_audio

NOW = datetime(2026, 10, 26, 12, 0).astimezone()


def make_session(root, name, ended_days_ago, transcribed=True):
    d = root / "2026-10-01" / name
    d.mkdir(parents=True)
    for f in ("me.wav", "others.wav", "transcript.md"):
        (d / f).write_bytes(b"x")
    ended = None if ended_days_ago is None else NOW - timedelta(days=ended_days_ago)
    SessionInfo(
        id=name,
        directory=d,
        started_at=NOW - timedelta(days=ended_days_ago or 0, hours=1),
        trigger_kind=TriggerKind.CALL,
        ended_at=ended,
        tracks={Track.ME: "me.wav", Track.OTHERS: "others.wav"},
        transcribed=transcribed,
    ).save()
    return d


def test_deletes_only_old_transcribed_audio(tmp_path):
    old = make_session(tmp_path, "old", 10)
    recent = make_session(tmp_path, "recent", 2)
    pending = make_session(tmp_path, "pending", 10, transcribed=False)
    live = make_session(tmp_path, "live", None, transcribed=False)

    deleted = purge_old_audio(tmp_path, 7, now=NOW)

    assert sorted(p.name for p in deleted) == ["me.wav", "others.wav"]
    assert not list(old.glob("*.wav"))
    assert (old / "transcript.md").exists() and (old / "session.json").exists()
    for d in (recent, pending, live):
        assert len(list(d.glob("*.wav"))) == 2


def test_disabled_and_missing_dir(tmp_path):
    make_session(tmp_path, "old", 100)
    assert purge_old_audio(tmp_path, 0, now=NOW) == []
    assert purge_old_audio(tmp_path / "nope", 7, now=NOW) == []


def test_ignores_unreadable_sessions(tmp_path):
    bad = tmp_path / "2026-10-01" / "bad"
    bad.mkdir(parents=True)
    (bad / "session.json").write_text("{not json", encoding="utf-8")
    (bad / "me.wav").write_bytes(b"x")
    assert purge_old_audio(tmp_path, 1, now=NOW) == []
    assert (bad / "me.wav").exists()


def test_keep_audio_false_purges_only_transcribed_sessions(tmp_path):
    done = make_session(tmp_path, "done", 0)  # left behind by a slow diarization
    pending = make_session(tmp_path, "pending", 0, transcribed=False)
    live = make_session(tmp_path, "live", None, transcribed=False)

    deleted = purge_transcribed_audio(tmp_path, ended_before=NOW + timedelta(minutes=1))

    assert sorted(p.name for p in deleted) == ["me.wav", "others.wav"]
    assert not list(done.glob("*.wav"))
    assert (done / "transcript.md").exists() and (done / "session.json").exists()
    for d in (pending, live):
        assert len(list(d.glob("*.wav"))) == 2
    assert purge_transcribed_audio(tmp_path) == []  # idempotent
    assert purge_transcribed_audio(tmp_path / "nope") == []


def test_startup_purge_spares_sessions_of_this_run(tmp_path):
    before = make_session(tmp_path, "before", 1)
    now = make_session(tmp_path, "now", 0)  # ended at NOW: may still be being diarized
    purge_transcribed_audio(tmp_path, ended_before=NOW - timedelta(hours=1))
    assert not list(before.glob("*.wav"))
    assert len(list(now.glob("*.wav"))) == 2


def test_startup_purge_keeps_untracked_wavs(tmp_path):
    d = make_session(tmp_path, "done", 1)
    (d / "notes.wav").write_bytes(b"x")  # not a session track
    purge_transcribed_audio(tmp_path)
    assert [p.name for p in d.glob("*.wav")] == ["notes.wav"]


@pytest.mark.parametrize("keep_audio", [True, False])
def test_controller_purges_leftovers_at_startup_only_without_keep_audio(tmp_path, keep_audio):
    from recandcc.config import Config
    from recandcc.ui.common import Controller

    class Monitor:
        def snapshot(self):
            return []

    class Q:
        def submit(self, *_a, **_k):
            pass

        def pending(self):
            return 0

    done = make_session(tmp_path, "done", 1)
    pending = make_session(tmp_path, "pending", 1, transcribed=False)
    cfg = Config(output_dir=tmp_path)
    cfg.transcription.keep_audio = keep_audio
    ctrl = Controller(cfg, monitor=Monitor(), recorder=object(), tq=Q(), start_threads=False)
    ctrl._resubmit_unfinished()  # startup: purge leftovers, then re-queue the rest

    assert len(list(done.glob("*.wav"))) == (2 if keep_audio else 0)
    assert len(list(pending.glob("*.wav"))) == 2


def test_failed_unlink_is_logged_and_retried_on_next_start(tmp_path, monkeypatch, caplog):
    from pathlib import Path

    d = make_session(tmp_path, "done", 1)
    real_unlink = Path.unlink

    def locked(self, *a, **k):
        if self.name == "others.wav":
            raise PermissionError("file in use")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked)
    with caplog.at_level("WARNING", logger="recandcc.retention"):
        deleted = purge_transcribed_audio(tmp_path)
    assert [p.name for p in deleted] == ["me.wav"]
    assert (d / "others.wav").exists()
    assert any("could not delete" in r.message for r in caplog.records)

    monkeypatch.setattr(Path, "unlink", real_unlink)  # lock released: next start
    assert [p.name for p in purge_transcribed_audio(tmp_path)] == ["others.wav"]
    assert not list(d.glob("*.wav"))


def test_purge_never_leaves_the_session_folder(tmp_path):
    outside = tmp_path / "precious.wav"
    outside.write_bytes(b"x")
    d = make_session(tmp_path, "evil", 1)
    (d / "notes.txt").write_bytes(b"x")
    s = SessionInfo.load(d)
    s.tracks = {Track.ME: "../../precious.wav", Track.OTHERS: str(outside)}
    s.save()
    assert purge_transcribed_audio(tmp_path) == []
    s.tracks = {Track.ME: "notes.txt"}
    s.save()
    assert purge_transcribed_audio(tmp_path) == []
    try:
        (d / "link.wav").symlink_to(outside)
    except OSError:
        pass  # Windows without symlink rights
    else:
        s.tracks = {Track.ME: "link.wav"}
        s.save()
        assert purge_transcribed_audio(tmp_path) == []
    assert outside.exists() and (d / "notes.txt").exists()
