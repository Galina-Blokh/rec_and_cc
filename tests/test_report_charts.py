"""Build-report SVG charts: axis ticks, bar maths, empty inputs and escaping. No network.

Stories: none.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts" / "report"
_spec = importlib.util.spec_from_file_location("charts", ROOT / "charts.py")
charts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(charts)

PLOT_H = charts.H - charts.PAD_T - charts.PAD_B


def _rects(svg):
    return re.findall(r'<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"', svg)


def test_ticks_cover_top_with_round_steps():
    assert charts._ticks(0) == [0, 1]
    assert charts._ticks(-3) == [0, 1]
    assert charts._ticks(100) == [0, 25, 50, 75, 100]
    assert charts._ticks(7) == [0, 2, 4, 6, 8]
    assert charts._ticks(0.5) == [0, 1]


def test_stacked_bars_sit_on_each_other_and_scale_to_axis():
    svg = charts.bars(
        ["a"],
        [
            {"name": "x", "color": "--s1", "values": [30]},
            {"name": "y", "color": "--s2", "values": [70]},
        ],
        stacked=True,
    )
    (x1, y1, _, h1), (x2, y2, _, h2) = _rects(svg)
    assert x1 == x2
    assert float(h1) == pytest.approx(0.3 * PLOT_H, abs=0.1)
    assert float(h2) == pytest.approx(0.7 * PLOT_H, abs=0.1)
    assert float(y2) + float(h2) == pytest.approx(float(y1), abs=0.2)
    assert 'data-tip="a · y: 70"' in svg


def test_grouped_bars_side_by_side_and_zero_skipped():
    svg = charts.bars(
        ["a", "b"],
        [
            {"name": "x", "color": "--s1", "values": [5, 0]},
            {"name": "y", "color": "--s2", "values": [10, 10]},
        ],
    )
    rects = _rects(svg)
    assert len(rects) == 3
    assert float(rects[0][0]) < float(rects[1][0])


def test_all_zero_and_empty_inputs_do_not_crash():
    assert "<svg" in charts.bars(["a"], [{"name": "x", "color": "--s1", "values": [0]}])
    assert "<svg" in charts.bars([], [])
    assert "<svg" in charts.hbars([], [])
    assert "<svg" in charts.steps([], [])
    assert "<svg" in charts.steps([{"name": "x", "color": "--s1", "points": []}], [])
    assert "<svg" in charts.heatstrips([])
    assert "<svg" in charts.swimlanes([], [], 0, 0, [])


def test_hbars_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        charts.hbars(["a", "b"], [1])


def test_labels_and_tips_are_escaped():
    evil = '<b>"x"</b>'
    out = "".join(
        [
            charts.bars([evil], [{"name": evil, "color": "--s1", "values": [1]}]),
            charts.hbars([evil], [1]),
            charts.steps([{"name": evil, "color": "--s1", "points": [(0, 1)]}], [(0, evil)]),
            charts.swimlanes(
                [evil],
                [{"lane": evil, "start": 0, "end": 1, "color": "--s1", "tip": evil}],
                0,
                1,
                [],
            ),
            charts.heatstrips([(evil, [1] * 24)]),
        ]
    )
    assert "<b>" not in out
    assert '"x"' not in out
    assert "&lt;b&gt;" in out
