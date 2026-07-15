"""Generate compact academic SVG figures for the paper.

The figures use only aggregate values already documented in the phase reports.
They are designed to be converted to vector PDF with `rsvg-convert`.
"""

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent / "figures"
ROOT = Path(__file__).resolve().parents[1]


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

    def circle(self, cx, cy, r, fill, stroke=None, width=1):
        attrs = [f'cx="{cx}"', f'cy="{cy}"', f'r="{r}"', f'fill="{fill}"']
        if stroke:
            attrs += [f'stroke="{stroke}"', f'stroke-width="{width}"']
        self.parts.append(f'<circle {" ".join(attrs)}/>')

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


def fig5():
    data_path = ROOT / "toy_quantvla" / "results" / "phase34_multifold_tactic_selection.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        tactics = data.get("tactics", [])
    else:
        tactics = [
            {
                "tactic": "speed_only",
                "mean_speedup": 2.01,
                "worst_speedup": 1.75,
                "total_regressions": 7,
                "worst_success_rate": 0.667,
            },
            {
                "tactic": "window_0_120",
                "mean_speedup": 1.84,
                "worst_speedup": 1.71,
                "total_regressions": 3,
                "worst_success_rate": 0.733,
            },
            {
                "tactic": "combo_blocks0_3_window_0_120",
                "mean_speedup": 1.41,
                "worst_speedup": 1.07,
                "total_regressions": 1,
                "worst_success_rate": 0.733,
            },
        ]

    label = {
        "speed_only": "speed-only",
        "window_0_120": "0-120",
        "combo_blocks0_3_window_0_120": "blocks0-3 + 0-120",
    }
    color = {
        "speed_only": C["red"],
        "window_0_120": C["green"],
        "combo_blocks0_3_window_0_120": C["blue"],
    }
    points_left = sorted(
        [
            {
                "name": item["tactic"],
                "x": float(item["mean_speedup"]),
                "y": float(item["total_regressions"]),
                "worst_speedup": float(item["worst_speedup"]),
                "worst_success": float(item["worst_success_rate"]),
            }
            for item in tactics
            if item.get("tactic") in label
        ],
        key=lambda row: row["x"],
    )

    n17_json = ROOT / "toy_quantvla" / "results" / "phase42_n17_bo_heldout_validation_15case_v1_summary.json"
    if n17_json.exists():
        data = json.loads(n17_json.read_text(encoding="utf-8"))
        pair_by_tactic = {
            row["right"]: row
            for row in data.get("pairs", [])
            if row.get("left") == "fp16"
        }
        n17_rows = []
        for run in data.get("runs", []):
            tactic = run.get("tactic")
            if tactic == "fp16":
                continue
            pair = pair_by_tactic.get(tactic, {})
            n17_rows.append(
                {
                    "deployment_point": tactic,
                    "speedup_vs_fp16_p50": str(run.get("speedup_vs_fp16_avg_p50") or 1.0),
                    "regressions_vs_fp16": str(len(pair.get("regressed") or [])),
                    "success_rate": str(run.get("success_rate") or 0.0),
                }
            )
    else:
        n17_rows = [
            {"deployment_point": "speed_only", "speedup_vs_fp16_p50": "1.33", "regressions_vs_fp16": "1", "success_rate": "0.867"},
            {"deployment_point": "window_0_20", "speedup_vs_fp16_p50": "1.05", "regressions_vs_fp16": "0", "success_rate": "1.000"},
            {"deployment_point": "window_2_12", "speedup_vs_fp16_p50": "1.30", "regressions_vs_fp16": "0", "success_rate": "0.933"},
            {"deployment_point": "window_4_9", "speedup_vs_fp16_p50": "1.37", "regressions_vs_fp16": "2", "success_rate": "0.800"},
            {"deployment_point": "window_6_11", "speedup_vs_fp16_p50": "1.35", "regressions_vs_fp16": "2", "success_rate": "0.867"},
        ]
    n17_label = {
        "speed_only": "speed-only",
        "window_0_20": "0-20",
        "window_2_12": "2-12",
        "window_4_9": "4-9",
        "window_6_11": "6-11",
    }
    n17_color = {
        "speed_only": C["red"],
        "window_0_20": C["blue"],
        "window_2_12": C["green"],
        "window_4_9": C["orange"],
        "window_6_11": C["purple"],
    }
    points_right = sorted(
        [
            {
                "name": row["deployment_point"],
                "x": float(row["speedup_vs_fp16_p50"]),
                "y": float(row["regressions_vs_fp16"]),
                "success": float(row["success_rate"]),
            }
            for row in n17_rows
            if row.get("deployment_point") in n17_label
        ],
        key=lambda row: row["x"],
    )

    s = SVG(760, 300)

    def draw_panel(points, x0, y0, w, h, xmin, xmax, ymax, xticks, yticks, title, labels, colors, subtitle=None, offsets=None):
        def sx(v):
            return x0 + w * (v - xmin) / (xmax - xmin)

        def sy(v):
            return y0 + h - h * v / ymax

        s.text(x0, y0 - 18, title, "panel")
        if subtitle:
            s.text(x0, y0 - 5, subtitle, "small")
        s.line(x0, y0 + h, x0 + w, y0 + h, C["ink"], 0.9)
        s.line(x0, y0, x0, y0 + h, C["ink"], 0.9)
        for tick in xticks:
            xx = sx(tick)
            s.line(xx, y0, xx, y0 + h, C["grid"], 0.5)
            s.text(xx, y0 + h + 16, f"{tick:.2f}x", "axis", anchor="middle")
        for tick in yticks:
            yy = sy(tick)
            s.line(x0, yy, x0 + w, yy, C["grid"], 0.5)
            s.text(x0 - 8, yy + 3, tick, "axis", anchor="end")

        if len(points) >= 2:
            d = " ".join(
                ("M" if idx == 0 else "L") + f"{sx(p['x']):.1f},{sy(p['y']):.1f}"
                for idx, p in enumerate(points)
            )
            s.path(d, C["muted"], 1.1, "none", "4 3")

        for idx, p in enumerate(points):
            xx, yy = sx(p["x"]), sy(p["y"])
            s.circle(xx, yy, 7.5, colors[p["name"]], "white", 1.5)
            dx, dy = (offsets or {}).get(p["name"], (9, -8 if idx % 2 == 0 else 16))
            s.text(xx + dx, yy + dy, labels[p["name"]], "label", fill=colors[p["name"]], weight="700")
            if "worst_speedup" in p:
                s.text(xx + dx, yy + dy + 13, f"worst {p['worst_speedup']:.2f}x", "small")
            else:
                s.text(xx + dx, yy + dy + 13, f"succ {p['success']:.3f}", "small")

    draw_panel(
        points_left,
        58,
        54,
        280,
        170,
        1.0,
        2.15,
        8,
        [1.0, 1.25, 1.5, 1.75, 2.0],
        [0, 2, 4, 6, 8],
        "(a) N1.5 multi-fold validation",
        label,
        color,
        "mean speedup vs total regressions",
        offsets={
            "combo_blocks0_3_window_0_120": (10, -16),
            "window_0_120": (10, 26),
            "speed_only": (10, -18),
        },
    )
    draw_panel(
        points_right,
        430,
        54,
        250,
        170,
        1.0,
        1.45,
        3,
        [1.0, 1.2, 1.4],
        [0, 1, 2, 3],
        "(b) N1.7 CLSG-BO held-out",
        n17_label,
        n17_color,
        "p50 speedup vs paired regressions",
        offsets={
            "window_0_20": (10, -24),
            "window_2_12": (12, -28),
            "window_4_9": (10, -12),
            "window_6_11": (16, 18),
            "speed_only": (12, -10),
        },
    )

    s.text(210, 270, "server-p50 speedup", "small", anchor="middle")
    s.text(550, 270, "server-p50 speedup", "small", anchor="middle")
    save("fig5_tactic_pareto.svg", s)


def fig6():
    data_path = ROOT / "toy_quantvla" / "results" / "phase39_threshold_pilot_summary.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        rows = {
            item["window"]: item
            for item in data.get("threshold_rows", [])
            if item.get("case") == "4:9" and item.get("direction") == "y"
        }
    else:
        rows = {
            "full": {
                "outcomes": [
                    {"epsilon": 1e-6, "success": True},
                    {"epsilon": 3e-6, "success": True},
                    {"epsilon": 1e-5, "success": True},
                    {"epsilon": 3e-5, "success": False},
                    {"epsilon": 1e-4, "success": False},
                    {"epsilon": 3e-4, "success": False},
                    {"epsilon": 5e-4, "success": False},
                ],
            },
            "early": {
                "outcomes": [
                    {"epsilon": 1e-6, "success": True},
                    {"epsilon": 3e-6, "success": True},
                    {"epsilon": 1e-5, "success": True},
                    {"epsilon": 3e-5, "success": False},
                    {"epsilon": 1e-4, "success": False},
                    {"epsilon": 3e-4, "success": False},
                    {"epsilon": 5e-4, "success": False},
                ],
            },
        }

    s = SVG(720, 250)
    x0, y0, w, h = 72, 36, 560, 150
    xmin, xmax = -6.0, -3.3

    def sx(eps):
        import math

        return x0 + w * (math.log10(float(eps)) - xmin) / (xmax - xmin)

    def sy(success, offset=0):
        return y0 + (35 if success else 118) + offset

    s.text(48, 24, "Closed-loop perturbation budget on task 4 init 9", "panel")
    s.line(x0, y0 + h, x0 + w, y0 + h, C["ink"], 0.9)
    s.line(x0, y0, x0, y0 + h, C["ink"], 0.9)
    for tick, label in [(1e-6, "1e-6"), (1e-5, "1e-5"), (1e-4, "1e-4"), (1e-3, "1e-3")]:
        xx = sx(tick)
        s.line(xx, y0, xx, y0 + h, C["grid"], 0.5)
        s.text(xx, y0 + h + 16, label, "axis", anchor="middle")
    for yy, label in [(sy(True), "success"), (sy(False), "failure")]:
        s.line(x0, yy, x0 + w, yy, C["grid"], 0.6)
        s.text(x0 - 8, yy + 3, label, "axis", anchor="end")

    left, right = sx(1e-5), sx(3e-5)
    s.rect(left, y0 + 4, right - left, h - 8, "#fff2cc", "#d6a800", 0.7)
    s.text((left + right) / 2, y0 + 18, "(1e-5, 3e-5]", "small", anchor="middle", fill="#8a6200", weight="700")

    specs = [
        ("full", C["red"], -5, "full horizon"),
        ("early", C["blue"], 5, "early 0-75"),
    ]
    for window, color, offset, label in specs:
        outcomes = sorted(rows.get(window, {}).get("outcomes", []), key=lambda row: float(row["epsilon"]))
        if not outcomes:
            continue
        points = [(sx(row["epsilon"]), sy(bool(row["success"]), offset)) for row in outcomes]
        d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}" for i, (x, y) in enumerate(points))
        s.path(d, color, 1.6)
        for row, (x, y) in zip(outcomes, points):
            s.circle(x, y, 5.5, color, "white", 1.2)
            if float(row["epsilon"]) in {1e-5, 3e-5}:
                s.text(x, y - 11 if row["success"] else y + 18, "S" if row["success"] else "F", "axis", anchor="middle", fill=color, weight="700")
        lx = 500 if window == "full" else 500
        ly = 54 if window == "full" else 73
        s.rect(lx, ly - 9, 12, 12, color)
        s.text(lx + 18, ly, label, "small")

    s.text(x0 + w / 2, 232, "injected y-action perturbation magnitude (log scale)", "small", anchor="middle")
    s.text(360, 210, "Both full and early perturbations flip between 1e-5 and 3e-5.", "small", anchor="middle")
    save("fig6_perturbation_budget.svg", s)


def main():
    # Figure 1 is rendered natively in LaTeX/TikZ so mathematical notation is
    # typeset by LaTeX. This script generates the data figures only.
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"Wrote SVG figures to {OUT}")


if __name__ == "__main__":
    main()
