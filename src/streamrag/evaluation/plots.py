"""Dependency-free SVG charts for the Phase 10 report and dashboard (experiments/plots/).

Design rules (reference data-viz palette, validated light + dark): single-series panels use one hue with the
category on the axis and the value at the bar tip (identity never by colour alone); bars <= 24 px thick with a 4 px
rounded data end and a square baseline; 2 px lines; >= 8 px markers with a 2 px surface ring; hairline solid grid;
text in ink tokens, never the series colour; every mark carries a <title> (hover tooltip); dark mode via
prefers-color-scheme. Charts only draw measured values handed to them - a missing value is drawn as "not measured".
"""

from __future__ import annotations

from html import escape

STYLE = """<style>
.s{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--base:#c3c2b7;
--c1:#2a78d6;--c2:#eb6834;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){.s{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--base:#383835;--c1:#3987e5;--c2:#d95926}}
.s .bg{fill:var(--surface)} .s .t{fill:var(--ink);font-size:14px;font-weight:600}
.s .sub{fill:var(--ink2);font-size:11px} .s .lab{fill:var(--ink2);font-size:11px}
.s .val{fill:var(--ink);font-size:11px;font-variant-numeric:tabular-nums} .s .ax{fill:var(--muted);font-size:10px}
.s .g{stroke:var(--grid);stroke-width:1} .s .b{stroke:var(--base);stroke-width:1}
.s .m1{fill:var(--c1)} .s .m2{fill:var(--c2)} .s .l1{stroke:var(--c1);fill:none;stroke-width:2;stroke-linejoin:round;
stroke-linecap:round} .s .l2{stroke:var(--c2);fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.s .ring{stroke:var(--surface);stroke-width:2}
</style>"""


def _svg(w: int, h: int, body: str, title: str, subtitle: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="s" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{escape(title)}">{STYLE}<rect class="bg" width="{w}" height="{h}"/>'
            f'<text class="t" x="16" y="24">{escape(title)}</text>'
            + (f'<text class="sub" x="16" y="42">{escape(subtitle)}</text>' if subtitle else "") + body + "</svg>")


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    import math
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= v:
            return m * e
    return 10 * e


def _bar(x0: float, y: float, length: float, thick: float, cls: str, tip: str) -> str:
    """Horizontal bar from baseline x0: square at the baseline, 4 px rounded data end."""
    r = min(4.0, thick / 2, max(length, 0) / 2)
    x1 = x0 + max(length, 0)
    d = (f"M{x0},{y} H{x1 - r} Q{x1},{y} {x1},{y + r} V{y + thick - r} Q{x1},{y + thick} {x1 - r},{y + thick} "
         f"H{x0} Z")
    return f'<path class="{cls}" d="{d}"><title>{escape(tip)}</title></path>'


def hbar(title: str, items: list[tuple[str, float | None]], fmt: str = "{:.2f}", subtitle: str = "",
         vmax: float | None = None, unit: str = "") -> str:
    """One measure across categories (single series): label left, bar, value at the tip."""
    lab_w, w = 230, 720
    row, thick = 30, 18
    top = 60
    h = top + row * len(items) + 34
    plot_w = w - lab_w - 90
    vals = [v for _, v in items if v is not None]
    mx = vmax if vmax is not None else _nice_max(max(vals) if vals else 1.0)
    body = []
    for k in range(5):
        x = lab_w + plot_w * k / 4
        body.append(f'<line class="g" x1="{x}" y1="{top - 6}" x2="{x}" y2="{h - 28}"/>'
                    f'<text class="ax" x="{x}" y="{h - 14}" text-anchor="middle">{fmt.format(mx * k / 4)}{unit}</text>')
    body.append(f'<line class="b" x1="{lab_w}" y1="{top - 6}" x2="{lab_w}" y2="{h - 28}"/>')
    for i, (label, v) in enumerate(items):
        y = top + i * row
        body.append(f'<text class="lab" x="{lab_w - 8}" y="{y + thick - 5}" text-anchor="end">{escape(label)}</text>')
        if v is None:
            body.append(f'<text class="ax" x="{lab_w + 6}" y="{y + thick - 5}">not measured</text>')
            continue
        L = plot_w * min(v, mx) / mx if mx else 0
        body.append(_bar(lab_w, y, L, thick, "m1", f"{label}: {fmt.format(v)}{unit}"))
        body.append(f'<text class="val" x="{lab_w + L + 6}" y="{y + thick - 5}">{fmt.format(v)}{unit}</text>')
    return _svg(w, h, "".join(body), title, subtitle)


def strip(title: str, groups: list[tuple[str, list[float]]], unit: str = " ms", subtitle: str = "") -> str:
    """Distribution per group: one dot per turn, p50 and p95 ticks (percentile strip)."""
    lab_w, w, row = 200, 760, 40
    top = 62
    h = top + row * len(groups) + 36
    plot_w = w - lab_w - 40
    allv = [v for _, vs in groups for v in vs]
    mx = _nice_max(max(allv) if allv else 1.0)
    body = []
    for k in range(5):
        x = lab_w + plot_w * k / 4
        body.append(f'<line class="g" x1="{x}" y1="{top - 8}" x2="{x}" y2="{h - 28}"/>'
                    f'<text class="ax" x="{x}" y="{h - 14}" text-anchor="middle">{mx * k / 4:,.0f}{unit}</text>')
    for i, (label, vs) in enumerate(groups):
        y = top + i * row + 10
        body.append(f'<text class="lab" x="{lab_w - 8}" y="{y + 4}" text-anchor="end">{escape(label)}</text>')
        s = sorted(vs)
        for v in s:
            x = lab_w + plot_w * v / mx
            body.append(f'<circle class="m1 ring" cx="{x:.1f}" cy="{y}" r="4" fill-opacity="0.55">'
                        f'<title>{escape(label)}: {v:,.0f}{unit}</title></circle>')
        if s:
            for p, name in ((0.5, "p50"), (0.95, "p95")):
                if name == "p95" and len(s) < 20:
                    continue
                q = s[min(len(s) - 1, int(round((len(s) - 1) * p)))]
                x = lab_w + plot_w * q / mx
                body.append(f'<line class="b" x1="{x}" y1="{y - 11}" x2="{x}" y2="{y + 11}" stroke-width="2"/>'
                            f'<text class="ax" x="{x}" y="{y - 13}" text-anchor="middle">{name} {q:,.0f}</text>')
    return _svg(w, h, "".join(body), title, subtitle)


def scatter(title: str, points: list[tuple[str, float, float]], xlab: str, ylab: str, subtitle: str = "",
            highlight: set[str] | None = None) -> str:
    """Labelled points (identity by direct label; the proposed system(s) in the accent hue)."""
    w, h = 760, 440
    L, R, T, B = 70, 170, 60, 60
    pw, ph = w - L - R, h - T - B
    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    xmx = _nice_max(max(xs) if xs else 1)
    ymn = max(0.0, (min(ys) if ys else 0) - 0.1)
    ymx = min(1.0, (max(ys) if ys else 1) + 0.05) if ys and max(ys) <= 1 else _nice_max(max(ys) if ys else 1)
    body = []
    for k in range(5):
        y = T + ph * k / 4
        yv = ymx - (ymx - ymn) * k / 4
        x = L + pw * k / 4
        body.append(f'<line class="g" x1="{L}" y1="{y}" x2="{L + pw}" y2="{y}"/>'
                    f'<text class="ax" x="{L - 6}" y="{y + 3}" text-anchor="end">{yv:.2f}</text>'
                    f'<text class="ax" x="{x}" y="{T + ph + 16}" text-anchor="middle">{xmx * k / 4:,.0f}</text>')
    body.append(f'<line class="b" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>'
                f'<text class="lab" x="{L + pw / 2}" y="{h - 14}" text-anchor="middle">{escape(xlab)}</text>'
                f'<text class="lab" x="16" y="{T - 10}">{escape(ylab)}</text>')
    placed: list[tuple[float, float]] = []
    for name, xv, yv in sorted(points, key=lambda p: p[1]):
        x = L + pw * xv / xmx
        y = T + ph * (ymx - yv) / (ymx - ymn) if ymx > ymn else T
        cls = "m2" if highlight and name in highlight else "m1"
        body.append(f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="5"><title>{escape(name)}: '
                    f'{xlab} {xv:,.0f}, {ylab} {yv:.3f}</title></circle>')
        ly = y + 4
        while any(abs(ly - py) < 13 and abs(x - px) < 160 for px, py in placed):
            ly += 13
        placed.append((x, ly))
        body.append(f'<text class="val" x="{x + 9:.1f}" y="{ly:.1f}">{escape(name)}</text>')
    return _svg(w, h, "".join(body), title, subtitle)


def steps(title: str, series: list[tuple[str, list[tuple[float, float]]]], xlab: str, ylab: str,
          subtitle: str = "") -> str:
    """Up to two step lines (legend + direct end labels)."""
    w, h = 760, 380
    L, R, T, B = 60, 150, 70, 56
    pw, ph = w - L - R, h - T - B
    xmx = _nice_max(max((x for _, pts in series for x, _ in pts), default=1))
    ymx = max(1.0, max((y for _, pts in series for _, y in pts), default=1))
    body = []
    for k in range(5):
        y = T + ph * k / 4
        x = L + pw * k / 4
        body.append(f'<line class="g" x1="{L}" y1="{y}" x2="{L + pw}" y2="{y}"/>'
                    f'<text class="ax" x="{L - 6}" y="{y + 3}" text-anchor="end">{ymx * (1 - k / 4):.2f}</text>'
                    f'<text class="ax" x="{x}" y="{T + ph + 16}" text-anchor="middle">{xmx * k / 4:,.0f}</text>')
    body.append(f'<text class="lab" x="{L + pw / 2}" y="{h - 12}" text-anchor="middle">{escape(xlab)}</text>'
                f'<text class="lab" x="16" y="{T - 22}">{escape(ylab)}</text>')
    for i, (name, pts) in enumerate(series[:2]):
        cls = f"l{i + 1}"
        d = ""
        prev_y = None
        for x, y in sorted(pts):
            px, py = L + pw * x / xmx, T + ph * (1 - y / ymx)
            d += f"M{px:.1f},{py:.1f}" if not d else (f" H{px:.1f} V{py:.1f}" if prev_y is not None else "")
            prev_y = py
        if pts:
            ex, ey = L + pw, T + ph * (1 - sorted(pts)[-1][1] / ymx)
            d += f" H{ex:.1f}"
            body.append(f'<path class="{cls}" d="{d}"><title>{escape(name)}</title></path>')
            body.append(f'<text class="val" x="{ex + 6}" y="{ey + 4 + i * 12:.1f}">{escape(name)}</text>')
        lx = L + i * 230
        body.append(f'<line class="{cls}" x1="{lx}" y1="{T - 40}" x2="{lx + 18}" y2="{T - 40}"/>'
                    f'<text class="lab" x="{lx + 24}" y="{T - 36}">{escape(name)}</text>')
    return _svg(w, h, "".join(body), title, subtitle)
