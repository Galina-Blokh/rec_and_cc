"""Hand-rolled SVG charts for the report: no libraries, no network.

Every mark carries a `data-tip` the page shows on hover, and every chart has a table view
built from the same rows, so the picture and the numbers cannot disagree.
"""

from __future__ import annotations

from html import escape

W, H = 720, 260
PAD_L, PAD_R, PAD_T, PAD_B = 56, 12, 12, 58


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 10 or v == int(v) else f"{v:,.1f}"


def _ticks(top: float, n: int = 4) -> list[float]:
    """Round axis ticks from 0 to the first tick at or above `top`."""
    if top <= 0:
        return [0, 1]
    raw = top / n
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    k = int(-(-top // step))
    return [i * step for i in range(k + 1)]


def _frame(inner: str, ticks: list[float], height: int = H) -> str:
    top = ticks[-1]
    plot_h = height - PAD_T - PAD_B
    grid = "".join(
        f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{PAD_T + plot_h - t / top * plot_h:.1f}" '
        f'y2="{PAD_T + plot_h - t / top * plot_h:.1f}"/>'
        f'<text class="tick" x="{PAD_L - 6}" y="{PAD_T + plot_h - t / top * plot_h + 4:.1f}" '
        f'text-anchor="end">{_fmt(t)}</text>'
        for t in ticks
    )
    return (
        f'<svg viewBox="0 0 {W} {height}" role="img" preserveAspectRatio="xMinYMin meet">'
        f"{grid}{inner}</svg>"
    )


def _xlabels(labels: list[str], x0: float, step: float, height: int, every: int) -> str:
    out = []
    for i, lab in enumerate(labels):
        if i % every:
            continue
        x = x0 + i * step + step / 2
        out.append(
            f'<text class="tick" x="{x:.1f}" y="{height - PAD_B + 14}" text-anchor="end" '
            f'transform="rotate(-35 {x:.1f} {height - PAD_B + 14})">{escape(lab)}</text>'
        )
    return "".join(out)


def bars(
    labels: list[str],
    series: list[dict],
    stacked: bool = False,
    height: int = H,
    unit: str = "",
) -> str:
    """Vertical bars. `series` = [{"name", "color" (css var), "values"}]."""
    n = max(len(labels), 1)
    plot_w, plot_h = W - PAD_L - PAD_R, height - PAD_T - PAD_B
    step = plot_w / n
    if stacked:
        top = max((sum(s["values"][i] for s in series) for i in range(len(labels))), default=0)
    else:
        top = max((v for s in series for v in s["values"]), default=0)
    ticks = _ticks(top)
    top = ticks[-1]
    out = []
    for i, lab in enumerate(labels):
        base = 0.0
        k = len(series) if not stacked else 1
        bw = max(step * 0.78 / k, 1)
        for j, s in enumerate(series):
            v = s["values"][i]
            if not v:
                continue
            h = v / top * plot_h
            x = PAD_L + i * step + step * 0.11 + (0 if stacked else j * bw)
            y = PAD_T + plot_h - h - (base / top * plot_h if stacked else 0)
            tip = f"{lab} · {s['name']}: {_fmt(v)}{unit}"
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                f'fill="var({s["color"]})" data-tip="{escape(tip)}"/>'
            )
            if stacked:
                base += v
    every = max(1, n // 24)
    return _frame("".join(out) + _xlabels(labels, PAD_L, step, height, every), ticks, height)


def hbars(
    labels: list[str], values: list[float], colors: list[str] | None = None, unit: str = ""
) -> str:
    """Horizontal bars, one series, labels on the left."""
    row = 26
    height = max(len(labels), 1) * row + 24
    left = 210
    top = max(values, default=0) or 1
    out = []
    for i, (lab, v) in enumerate(zip(labels, values, strict=True)):
        w = (W - left - 70) * (v / top)
        y = 8 + i * row
        color = (colors or [])[i] if colors else "--s1"
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 15}" text-anchor="end">{escape(lab[:34])}</text>'
            f'<rect x="{left}" y="{y + 3}" width="{max(w, 0.5):.1f}" height="{row - 9}" '
            f'fill="var({color})" data-tip="{escape(lab)}: {_fmt(v)}{unit}"/>'
            f'<text class="tick" x="{left + w + 6:.1f}" y="{y + 15}">{_fmt(v)}{unit}</text>'
        )
    return f'<svg viewBox="0 0 {W} {height}" role="img">{"".join(out)}</svg>'


def steps(series: list[dict], x_labels: list[tuple[float, str]], area: bool = False) -> str:
    """Step lines over a numeric x axis (epoch seconds). `series` = [{"name","color","points"}]."""
    xs = [p[0] for s in series for p in s["points"]] or [0, 1]
    x0, x1 = min(xs), max(max(xs), min(xs) + 1)
    ticks = _ticks(max((p[1] for s in series for p in s["points"]), default=0))
    top = ticks[-1]
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def sx(x: float) -> float:
        return PAD_L + (x - x0) / (x1 - x0) * plot_w

    def sy(y: float) -> float:
        return PAD_T + plot_h - y / top * plot_h

    out = []
    for s in series:
        pts = sorted(s["points"])
        if not pts:
            continue
        d = f"M{sx(pts[0][0]):.1f},{sy(0):.1f}"
        prev = 0.0
        for x, y in pts:
            d += f" L{sx(x):.1f},{sy(prev):.1f} L{sx(x):.1f},{sy(y):.1f}"
            prev = y
        d += f" L{sx(x1):.1f},{sy(prev):.1f}"
        if area:
            out.append(
                f'<path d="{d} L{sx(x1):.1f},{sy(0):.1f} Z" fill="var({s["color"]})" opacity=".18"/>'
            )
        out.append(f'<path d="{d}" fill="none" stroke="var({s["color"]})" stroke-width="2"/>')
        for x, y in pts[:: max(1, len(pts) // 60)]:
            out.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="transparent" '
                f'data-tip="{escape(s["name"])}: {_fmt(y)}"/>'
            )
    for x, lab in x_labels:
        out.append(
            f'<text class="tick" x="{sx(x):.1f}" y="{H - PAD_B + 16}" text-anchor="middle">{escape(lab)}</text>'
        )
    return _frame("".join(out), ticks)


def swimlanes(
    lanes: list[str], bars_: list[dict], x0: float, x1: float, x_labels: list[tuple[float, str]]
) -> str:
    """One lane per person; each bar = {lane, start, end, color, tip}."""
    row = 30
    height = len(lanes) * row + 40
    left = 150
    span = max(x1 - x0, 1)

    def sx(x: float) -> float:
        return left + (x - x0) / span * (W - left - PAD_R)

    out = []
    for i, lane in enumerate(lanes):
        y = 8 + i * row
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 18}" text-anchor="end">{escape(lane[:22])}</text>'
        )
        out.append(
            f'<line class="grid" x1="{left}" x2="{W - PAD_R}" y1="{y + row - 2}" y2="{y + row - 2}"/>'
        )
    per_lane: dict[str, int] = {}
    for b in bars_:
        i = lanes.index(b["lane"])
        k = per_lane.get(b["lane"], 0)
        per_lane[b["lane"]] = k + 1
        y = 8 + i * row + 4 + (k % 4) * 5
        x = sx(b["start"])
        w = max(sx(b["end"]) - x, 2)
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="5" rx="1" '
            f'fill="var({b["color"]})" data-tip="{escape(b["tip"])}"/>'
        )
    for x, lab in x_labels:
        out.append(
            f'<text class="tick" x="{sx(x):.1f}" y="{height - 8}" text-anchor="middle">{escape(lab)}</text>'
        )
    return f'<svg viewBox="0 0 {W} {height}" role="img">{"".join(out)}</svg>'


def heatstrips(rows: list[tuple[str, list[int]]]) -> str:
    """24 cells per row (UK hour of day), shaded by count."""
    cell, left = 22, 150
    height = len(rows) * 28 + 30
    top = max((v for _, vals in rows for v in vals), default=0) or 1
    out = []
    for i, (name, vals) in enumerate(rows):
        y = 6 + i * 28
        out.append(
            f'<text class="tick" x="{left - 8}" y="{y + 15}" text-anchor="end">{escape(name[:22])}</text>'
        )
        for h, v in enumerate(vals):
            shade = 1 + min(6, int(v / top * 6)) if v else 0
            fill = f"var(--seq{shade})" if shade else "var(--grid)"
            out.append(
                f'<rect x="{left + h * cell}" y="{y}" width="{cell - 2}" height="20" rx="2" fill="{fill}" '
                f'data-tip="{escape(name)} · {h:02d}:00: {v} commits"/>'
            )
    for h in range(0, 24, 3):
        out.append(
            f'<text class="tick" x="{left + h * cell + 10}" y="{height - 6}" text-anchor="middle">{h:02d}</text>'
        )
    return f'<svg viewBox="0 0 {left + 24 * cell + 10} {height}" role="img">{"".join(out)}</svg>'
