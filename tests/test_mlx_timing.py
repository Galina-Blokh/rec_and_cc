"""MLX word timestamps on the numpy aligner (vendored ``mlx_timing``), with fakes.

The registration and output-shape tests run everywhere; the ``find_alignment`` test needs
mlx. The real-model check is in test_no_torch.py (slow).

Stories: US-12.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import types

import numpy as np
import pytest

from recandcc.transcribe import gpu, mlx_timing
from recandcc.transcribe.mlx_timing import WordTiming

KEY = "mlx_whisper.timing"


@pytest.fixture
def clean_timing(monkeypatch):
    """No ``mlx_whisper.timing`` registered, numba/scipy absent; restored afterwards."""
    saved = sys.modules.pop(KEY, None)
    monkeypatch.delitem(sys.modules, "mlx_whisper.transcribe", raising=False)
    real_find_spec = gpu.importlib.util.find_spec
    monkeypatch.setattr(
        gpu.importlib.util,
        "find_spec",
        lambda name, *a: None if name in ("numba", "scipy") else real_find_spec(name, *a),
    )
    monkeypatch.setattr(gpu, "_version_warned", False)

    def fake_import():  # as import_mlx_whisper: register, then "import" transcribe
        gpu._install_timing()
        timing = sys.modules[KEY]
        sys.modules["mlx_whisper.transcribe"] = types.SimpleNamespace(
            add_word_timestamps=timing.add_word_timestamps
        )

    monkeypatch.setattr(gpu, "import_mlx_whisper", fake_import)
    yield
    sys.modules.pop(KEY, None)
    if saved is not None:
        sys.modules[KEY] = saved


def test_numpy_aligner_is_registered_instead_of_the_stub(clean_timing, monkeypatch):
    monkeypatch.setattr(gpu, "_mlx_whisper_version", lambda: "0.4.3")
    assert gpu.mlx_word_timestamps()
    assert sys.modules[KEY] is mlx_timing


def test_other_mlx_whisper_version_keeps_words_off_and_warns_once(
    clean_timing, monkeypatch, caplog
):
    monkeypatch.setattr(gpu, "_mlx_whisper_version", lambda: "0.5.0")
    with caplog.at_level(logging.INFO, "recandcc.transcribe"):
        gpu._install_timing()
        assert not caplog.records  # importing alone (split off) only logs at DEBUG
        assert not gpu.mlx_word_timestamps()
        assert not gpu.mlx_word_timestamps()
    assert len([r for r in caplog.records if "0.5.0" in r.getMessage()]) == 1
    with pytest.raises(RuntimeError, match="word timestamps"):
        sys.modules[KEY].add_word_timestamps()


def test_checking_words_never_registers_anything_itself(clean_timing, monkeypatch):
    monkeypatch.setattr(gpu, "_install_timing", lambda: pytest.fail("not read-only"))
    monkeypatch.setitem(
        sys.modules, "mlx_whisper.transcribe", types.SimpleNamespace(add_word_timestamps=print)
    )
    assert gpu.mlx_word_timestamps() and KEY not in sys.modules


def test_words_follow_what_mlx_whisper_transcribe_imported(clean_timing, monkeypatch):
    monkeypatch.setattr(gpu, "_mlx_whisper_version", lambda: "0.4.3")
    loaded = types.SimpleNamespace(add_word_timestamps=gpu._stub_word_timestamps)
    monkeypatch.setitem(sys.modules, "mlx_whisper.transcribe", loaded)
    assert not gpu.mlx_word_timestamps()  # registered too late: transcribe holds the stub
    loaded.add_word_timestamps = mlx_timing.add_word_timestamps
    assert gpu.mlx_word_timestamps()


def _fake_spec(monkeypatch, folder):
    spec = types.SimpleNamespace(submodule_search_locations=[str(folder)])
    real = gpu.importlib.util.find_spec
    monkeypatch.setattr(
        gpu.importlib.util, "find_spec", lambda n, *a: spec if n == "mlx_whisper" else real(n, *a)
    )


def test_version_is_read_from_the_package_file_without_dist_info(tmp_path, monkeypatch):
    (tmp_path / "_version.py").write_text('# c\n\n__version__ = "0.4.3"\n')
    _fake_spec(monkeypatch, tmp_path)

    def no_dist_info(_name):
        raise gpu.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(gpu.importlib.metadata, "version", no_dist_info)  # frozen bundle
    assert gpu._mlx_whisper_version() == "0.4.3"


def test_version_falls_back_to_importlib_metadata(tmp_path, monkeypatch):
    _fake_spec(monkeypatch, tmp_path)  # no _version.py
    monkeypatch.setattr(gpu.importlib.metadata, "version", lambda _name: "0.4.2")
    assert gpu._mlx_whisper_version() == "0.4.2"


def test_alignment_heads_are_capped_but_turbo_is_unchanged():
    turbo = [[layer, h] for layer in (2, 3) for h in range(20)]
    assert mlx_timing.capped_heads(turbo) == turbo
    large = [[layer, h] for layer in range(16, 32) for h in range(20)]  # 320
    picked = mlx_timing.capped_heads(large)
    assert len(picked) == mlx_timing.MAX_ALIGNMENT_HEADS == 64
    assert picked == mlx_timing.capped_heads(list(reversed(large)))  # deterministic
    assert picked[0] == [31, 0] and {layer for layer, _ in picked} == set(range(16, 32))
    assert len({tuple(p) for p in picked}) == 64


def test_installed_upstream_timing_is_the_vendored_version():
    """If this fails, mlx-whisper's timing.py changed: review mlx_timing.py against it."""
    import hashlib

    spec = gpu.importlib.util.find_spec("mlx_whisper")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("mlx-whisper is not installed")
    path = gpu.Path(next(iter(spec.submodule_search_locations))) / "timing.py"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert gpu._mlx_whisper_version() == gpu.VENDORED_TIMING_VERSION
    assert digest == "2736c2bf19f2ebbaa521dae9deb3748c6b5d5f319af729f4078dc2b0769822cd"


@pytest.fixture
def fake_mlx(monkeypatch):
    calls = []

    def transcribe(audio, **kw):
        calls.append(kw)
        words = [
            {"word": " a", "start": 0.0, "end": 0.5, "probability": 0.9},
            {"word": " b.", "start": 0.5, "end": 1.0, "probability": 0.9},
        ]
        return {"segments": [{"start": 0.0, "end": 1.0, "text": " a b.", "words": words}]}

    monkeypatch.setattr(
        gpu, "import_mlx_whisper", lambda: types.SimpleNamespace(transcribe=transcribe)
    )
    w = gpu.MlxWhisper("large-v3-turbo")
    monkeypatch.setattr(w, "_get_model", lambda: None)
    yield w, calls
    w.close()


def test_transcribe_without_words_keeps_the_three_tuples(fake_mlx):
    w, calls = fake_mlx
    assert w.transcribe(np.zeros(16000, np.float32), "en") == [(0.0, 1.0, " a b.")]
    assert "word_timestamps" not in calls[0]


def test_transcribe_with_words_adds_word_tuples(fake_mlx):
    w, calls = fake_mlx
    out = w.transcribe(np.zeros(16000, np.float32), "en", words=True)
    assert out == [(0.0, 1.0, " a b.", [(0.0, 0.5, " a"), (0.5, 1.0, " b.")])]
    assert calls[0]["word_timestamps"] is True


def test_add_word_timestamps_splits_words_per_segment(monkeypatch):
    alignment = [
        WordTiming(" Hello", [1], 0.0, 0.4, 0.9),
        WordTiming(" world", [2], 0.4, 0.8, 0.9),
        WordTiming(".", [3], 0.8, 0.9, 0.9),
        WordTiming(" Bye", [4], 1.0, 1.3, 0.9),
        WordTiming("!", [5], 1.3, 1.4, 0.9),
    ]
    monkeypatch.setattr(mlx_timing, "find_alignment", lambda *a, **k: alignment)
    segments = [
        {"seek": 100, "start": 1.0, "end": 1.8, "tokens": [1, 2, 3, 100]},  # 100 = eot
        {"seek": 100, "start": 2.0, "end": 2.3, "tokens": [4, 5]},
    ]
    tokenizer = types.SimpleNamespace(eot=100)
    mlx_timing.add_word_timestamps(
        segments=segments,
        model=None,
        tokenizer=tokenizer,
        mel=None,
        num_frames=300,
        last_speech_timestamp=0.0,
    )
    got = [[(w["word"], w["start"], w["end"]) for w in s["words"]] for s in segments]
    assert got == [[(" Hello", 1.0, 1.4), (" world.", 1.4, 1.8)], [(" Bye!", 2.0, 2.3)]]
    assert (segments[0]["start"], segments[0]["end"]) == (1.0, 1.8)


def test_find_alignment_runs_on_mlx_with_numpy_kernels():
    mx = pytest.importorskip("mlx.core")
    sot, no_ts, eot, vocab, frames = [50258], 50363, 50257, 50364, 100
    text_tokens = [10, 11]
    n = len(sot) + 1 + len(text_tokens) + 1
    qk = np.full((n, frames), -5.0, np.float32)
    for row, col in ((1, 5), (2, 30), (3, 70), (4, 95)):  # a rising diagonal
        qk[row, col - 3 : col + 3] = 5.0

    class Model:
        alignment_heads = mx.array([[0, 0]])

        def forward_with_cross_qk(self, mel, tokens):
            return mx.zeros((1, n, vocab)), [mx.array(qk[None, None])]

    tokenizer = types.SimpleNamespace(
        sot_sequence=sot,
        no_timestamps=no_ts,
        eot=eot,
        split_to_word_tokens=lambda toks: ([" a", " b", ""], [[10], [11], [eot]]),
    )
    out = mlx_timing.find_alignment(Model(), tokenizer, text_tokens, mx.zeros((3000, 128)), 200)
    assert [w.word for w in out] == [" a", " b"]
    assert 0 <= out[0].start <= out[0].end <= out[1].start <= out[1].end <= frames / 50


def test_numba_and_scipy_are_never_imported():
    code = (
        "import sys\n"
        "from recandcc.transcribe import gpu, mlx_timing\n"
        "gpu.mlx_word_timestamps()\n"
        "print([m for m in ('numba', 'scipy') if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
