"""Acceptance timing of the real summary pipeline on a synthetic 1-hour en/ru meeting.

Stories: US-27.

Apple silicon only, and only if the pinned summary model is already in the local cache (a
test never downloads). The transcript is generated text, not a real recording.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from recandcc.config import Config
from recandcc.models import Segment, SessionInfo, TriggerKind
from recandcc.transcribe import summary
from recandcc.transcribe.output import write_outputs

HARD_BOUND_S = 180.0  # the story says ~1-2 min; this is the failure line
TARGET_WORDS = 9_000

EN_TOPICS = [
    "the quarterly roadmap",
    "the pricing change for enterprise customers",
    "the migration of the billing service",
    "hiring for the support team",
    "the security review of the new API",
]
RU_TOPICS = [
    "план разработки на квартал",
    "изменение цен для корпоративных клиентов",
    "перенос сервиса оплаты на новую платформу",
    "найм в команду поддержки",
    "проверка безопасности нового интерфейса",
]
EN_TEMPLATES = [
    "I think {t} needs more time because the numbers from last month were not final yet.",
    "Let us go back to {t}: what did the customers say when we showed them the draft?",
    "On {t} the main risk is that two teams depend on the same release, so we should agree on "
    "an order before we promise any date to anybody outside this call.",
    "Fair point about {t}, but the budget is fixed for this quarter and we cannot add people.",
]
RU_TEMPLATES = [
    "Я думаю, что {t} требует больше времени, потому что цифры за прошлый месяц ещё не "
    "окончательные.",
    "Давайте вернёмся к теме: {t}. Что сказали клиенты, когда мы показали им черновик?",
    "По теме «{t}» главный риск в том, что две команды зависят от одного релиза, поэтому нужно "
    "договориться о порядке, прежде чем обещать кому-то сроки.",
    "Справедливое замечание про {t}, но бюджет на этот квартал уже утверждён, и добавить людей "
    "мы не можем.",
]
ACTIONS = {
    40: "Anna, please send the updated pricing deck to the whole team by Friday.",
    95: "I will book the review room and invite the security group next Tuesday.",
    150: "Анна, пожалуйста, отправь обновлённую презентацию с ценами всей команде до пятницы.",
    210: "Я забронирую комнату для обсуждения и приглашу группу безопасности во вторник.",
    270: "Boris, can you write down the billing migration plan before the next meeting?",
}
SPEAKERS = ["Me", "Anna", "Boris", "Speaker 3"]


def synthetic_meeting(target_words: int = TARGET_WORDS) -> list[Segment]:
    """Deterministic ~1 h transcript: alternating en / ru blocks of 12 turns, four speakers
    (changing every turn, so turns never merge into one paragraph), a few clear action items."""
    segs: list[Segment] = []
    words, i, t = 0, 0, 0.0
    step = 3600.0 / (target_words / 25)  # ~25 words per turn, ~1 h overall
    while words < target_words:
        lang = "en" if (i // 12) % 2 == 0 else "ru"
        topics, templates = (EN_TOPICS, EN_TEMPLATES) if lang == "en" else (RU_TOPICS, RU_TEMPLATES)
        text = ACTIONS.get(i) or templates[i % len(templates)].format(t=topics[(i // 3) % 5])
        words += len(text.split())
        segs.append(Segment(t, t + step * 0.8, text, SPEAKERS[i % len(SPEAKERS)], lang))
        t += step
        i += 1
    return segs


def test_synthetic_meeting_is_deterministic_and_big_enough():
    a, b = synthetic_meeting(), synthetic_meeting()
    assert a == b
    assert sum(len(s.text.split()) for s in a) >= TARGET_WORDS
    assert {s.language for s in a} == {"en", "ru"}
    assert a[-1].end > 3300
    assert len(summary.chunk_lines(summary.transcript_lines(a))) > 1


def _model_cached(repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        from recandcc.transcribe.revisions import revision

        snapshot_download(repo, revision=revision(repo), local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 - not cached / hub missing: skip
        return False


@pytest.mark.slow
@pytest.mark.skipif(
    not summary.mlx_lm_available(), reason="summaries need Apple silicon with mlx-lm"
)
def test_summary_timing_one_hour_meeting_offline(tmp_path: Path, monkeypatch, capsys):
    cfg = Config()
    if not _model_cached(cfg.summary.model):
        pytest.skip("pinned summary model is not in the local cache (tests never download)")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    segs = synthetic_meeting()
    d = tmp_path / "2026-10-27" / "100000_zoom"
    d.mkdir(parents=True)
    s = SessionInfo("s1", d, datetime(2026, 10, 27, 10), TriggerKind.CALL, "zoom.us")
    s.save()
    write_outputs(s, segs, 3600.0, "fake")
    saved = json.loads((d / "transcript.json").read_text(encoding="utf-8"))["segments"]
    assert len(saved) > 100

    chunk_counts: list[int] = []
    real_chunk = summary.chunk_lines

    def spy(lines, max_chars=0):
        chunks = real_chunk(lines, max_chars)
        chunk_counts.append(len(chunks))
        return chunks

    monkeypatch.setattr(summary, "chunk_lines", spy)

    t0 = time.monotonic()
    path = summary.summarize_session(d, cfg)
    elapsed = time.monotonic() - t0
    with capsys.disabled():
        print(f"\nsummary_timing: {elapsed:.1f}s, {len(segs)} turns, {chunk_counts[0]} chunks")

    text = path.read_text(encoding="utf-8")
    assert path.name == "summary.md"
    assert "## Action items" in text or "## Задачи" in text
    body = text.split("\n## ", 1)[0].split("\n", 1)[1]
    assert body.strip(), "empty summary"
    assert chunk_counts and chunk_counts[0] > 1, "map-reduce path did not run"
    assert elapsed < HARD_BOUND_S, f"summary took {elapsed:.0f}s (story: ~1-2 min)"
