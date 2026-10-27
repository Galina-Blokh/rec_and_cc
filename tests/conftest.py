import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running tests that load real models")


@pytest.fixture(autouse=True)
def _no_diarization_model(request, monkeypatch):
    """Fast tests never download or load the speaker model: diarization runs but finds no
    model (as when offline), so transcripts keep "Others". Slow tests use the real one."""
    if request.node.get_closest_marker("slow") is None:
        from recandcc.transcribe import diarize

        monkeypatch.setattr(diarize, "model_path", lambda *a, **k: None)


@pytest.fixture(autouse=True)
def _private_voices_dir(tmp_path_factory, monkeypatch):
    """No test reads or writes the user's real voice profiles."""
    from recandcc.transcribe import voices

    monkeypatch.setattr(voices, "VOICES_DIR", tmp_path_factory.mktemp("voices"))
