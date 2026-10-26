from datetime import datetime, timedelta

from recandcc.models import SessionInfo, Track, TriggerKind
from recandcc.retention import purge_old_audio

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
