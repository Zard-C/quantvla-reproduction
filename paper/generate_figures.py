"""Generate compact academic SVG figures for the paper.

The figures use only aggregate values already documented in the phase reports.
They are designed to be converted to vector PDF with `rsvg-convert`.
"""

from pathlib import Path


OUT = Path(__file__).resolve().parent / "figures"


C = {
    "ink": "#222222",
    "muted": "#666666",
    "grid": "#d6d6d6",
    "light": "#f7f7f7",
    "blue": "#4c78a8",
    "orange": "#f58518",
    "green": "#54a24b",
    "red": "#e45756",
    "purple": "#7f6dba",
    "gray": "#9d9d9d",
}


def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<style>'
            'text{font-family:Helvetica,Arial,sans-serif;fill:#222222}'
            '.label{font-size:11px}'
            '.small{font-size:9.5px;fill:#666666}'
            '.axis{font-size:9px;fill:#666666}'
            '.panel{font-size:12px;font-weight:700}'
            '</style>',
        ]

    def text(self, x, y, t, cls="label", anchor="start", fill=None, weight=None):
        attrs = [f'x="{x}"', f'y="{y}"', f'class="{cls}"', f'text-anchor="{anchor}"']
        if fill:
            attrs.append(f'style="fill:{fill}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        self.parts.append(f'<text {" ".join(attrs)}>{esc(t)}</text>')

    def line(self, x1, y1, x2, y2, stroke=None, width=1, dash=None, marker=None):
        attrs = [
            f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
            f'stroke="{stroke or C["ink"]}"', f'stroke-width="{width}"', 'fill="none"',
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker:
            attrs.append(f'marker-end="url(#{marker})"')
        self.parts.append(f'<line {" ".join(attrs)}/>')

    def rect(self, x, y, w, h, fill, stroke=None, width=1, rx=0):
        attrs = [
            f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"',
            f'fill="{fill}"', f'rx="{rx}"',
        ]
        if stroke:
            attrs += [f'stroke="{stroke}"', f'stroke-width="{width}"']
        self.parts.append(f'<rect {" ".join(attrs)}/>')

    def path(self, d, stroke=None, width=1.2, fill="none", dash=None, marker=None):
        attrs = [f'd="{d}"', f'stroke="{stroke or C["ink"]}"', f'stroke-width="{width}"', f'fill="{fill}"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        if marker:
            attrs.append(f'marker-end="url(#{marker})"')
        self.parts.append(f'<path {" ".join(attrs)}/>')

    def arrow_def(self):
        self.parts.append(
            '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" '
            'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L7,3 z" '
            'fill="#222222"/></marker></defs>'
        )

    def done(self):
        self.parts.append("</svg>")
        return "\n".join(self.parts)


def save(name, svg):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(svg.done(), encoding="utf-8")


def axes(s, x, y, w, h, ymax, ticks):
    s.line(x, y + h, x + w, y + h, C["ink"], 0.8)
    s.line(x, y, x, y + h, C["ink"], 0.8)
    for t in ticks:
        yy = y + h - h * t / ymax
        s.line(x, yy, x + w, yy, C["grid"], 0.5)
        s.text(x - 5, yy + 3, t, "axis", anchor="end")


def fig2():
    s = SVG(720, 250)
    modes = ["id", "ATM", "OHB", "A+O"]
    worse = [0, 37, 24, 34]
    rmse_reduction = [0.0, 0.00772748, 0.01275, 0.0142674]
    colors = [C["gray"], C["orange"], C["green"], C["purple"]]

    s.text(54, 24, "(a) mean rel. RMSE reduction", "panel")
    x, y, w, h = 60, 48, 260, 145
    axes(s, x, y, w, h, 0.016, [0, 0.004, 0.008, 0.012, 0.016])
    for i, (m, v, col) in enumerate(zip(modes, rmse_reduction, colors)):
        bx = x + 28 + i * 56
        bh = h * v / 0.016
        s.rect(bx, y + h - bh, 28, bh, col)
        s.text(bx + 14, y + h + 15, m, "axis", anchor="middle")
        s.text(bx + 14, y + h - bh - 5, f"{v:.3f}", "axis", anchor="middle")
    s.text(x + w / 2, 226, "higher is better", "small", anchor="middle")

    s.text(412, 24, "(b) observations worse by NMSE", "panel")
    x2, y2, w2, h2 = 420, 48, 240, 145
    axes(s, x2, y2, w2, h2, 40, [0, 10, 20, 30, 40])
    for i, (m, v, col) in enumerate(zip(modes, worse, colors)):
        bx = x2 + 26 + i * 52
        bh = h2 * v / 40
        s.rect(bx, y2 + h2 - bh, 28, bh, col)
        s.text(bx + 14, y2 + h2 + 15, m, "axis", anchor="middle")
        s.text(bx + 14, y2 + h2 - bh - 5, f"{v}", "axis", anchor="middle")
    s.text(x2 + w2 / 2, 226, "lower is better", "small", anchor="middle")
    save("fig2_offline_drift_regressions.svg", s)


def fig3():
    s = SVG(720, 210)
    rows = [
        ("ATM vs none", [14, 13, 100, 23], "+1"),
        ("OHB vs none", [16, 13, 100, 21], "+3"),
        ("OHB vs ATM", [15, 13, 101, 21], "+2"),
    ]
    colors = [C["green"], C["red"], C["blue"], C["gray"]]
    names = ["repair", "regress", "same success", "same fail"]
    x, y, w, bar_h = 128, 42, 455, 28
    scale = w / 150
    for i, (label, vals, net) in enumerate(rows):
        yy = y + i * 46
        s.text(20, yy + 19, label, "label")
        xx = x
        for val, col in zip(vals, colors):
            ww = val * scale
            s.rect(xx, yy, ww, bar_h, col)
            if ww > 34:
                s.text(xx + ww / 2, yy + 19, val, "axis", anchor="middle", fill="white" if col != C["gray"] else C["ink"], weight="700")
            xx += ww
        s.text(x + w + 18, yy + 19, f"net {net}", "label", weight="700")
    for t in [0, 50, 100, 150]:
        xx = x + t * scale
        s.line(xx, y - 8, xx, y + 2 * 46 + bar_h + 8, C["grid"], 0.6)
        s.text(xx, y - 14, t, "axis", anchor="middle")
    lx, ly = 122, 184
    for i, (name, col) in enumerate(zip(names, colors)):
        xx = lx + i * 128
        s.rect(xx, ly - 9, 11, 11, col)
        s.text(xx + 17, ly, name, "small")
    save("fig3_paired_rollout_flips.svg", s)


def fig4():
    s = SVG(720, 260)
    policies = ["FP16", "none", "ATM", "OHB", "A+O"]
    colors = [C["blue"], C["gray"], C["orange"], C["green"], C["purple"]]
    data = {
        "FP16": [13, 11, 12, 15, 8, 15, 6, 11, 3, 14],
        "none": [10, 13, 13, 15, 8, 14, 9, 8, 9, 14],
        "ATM": [11, 13, 14, 13, 13, 14, 10, 8, 4, 14],
        "OHB": [13, 14, 15, 13, 12, 15, 8, 8, 6, 12],
        "A+O": [13, 14, 13, 15, 12, 15, 7, 10, 3, 12],
    }
    x, y, w, h = 48, 34, 620, 160
    axes(s, x, y, w, h, 15, [0, 5, 10, 15])
    group = w / 10
    bw = 7
    for task in range(10):
        gx = x + task * group + 9
        if task in [4, 8]:
            s.rect(x + task * group + 2, y - 4, group - 4, h + 8, "none", "#b35c00", 0.8)
        for j, (p, col) in enumerate(zip(policies, colors)):
            v = data[p][task]
            bh = h * v / 15
            s.rect(gx + j * (bw + 3), y + h - bh, bw, bh, col)
        s.text(x + task * group + group / 2, y + h + 15, task, "axis", anchor="middle")
    s.text(x + w / 2, 224, "task id", "small", anchor="middle")
    for i, (p, col) in enumerate(zip(policies, colors)):
        xx = 145 + i * 88
        s.rect(xx, 242 - 9, 11, 11, col)
        s.text(xx + 16, 242, p, "small")
    save("fig4_task_redistribution.svg", s)


def main():
    # Figure 1 is rendered natively in LaTeX/TikZ so mathematical notation is
    # typeset by LaTeX. This script generates the data figures only.
    fig2()
    fig3()
    fig4()
    print(f"Wrote SVG figures to {OUT}")


if __name__ == "__main__":
    main()
