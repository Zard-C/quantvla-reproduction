"""Analyze Phase 28D duration-window fallback outcomes.

This is intentionally separate from the Phase 28D summary generator. The
summary script reports what ran; this script interprets repair/regression
patterns, step counts, and what the result implies for a finer duration proxy.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

SOURCE = RESULTS / "phase28D_duration_window_33case_v1_summary.json"
OUT_JSON = RESULTS / "phase28D_duration_window_33case_v1_analysis.json"
OUT_MD = DOCS / "phase28d_duration_window_analysis_zh.md"

D_WINDOWS = ["window_80_220", "window_120_320", "window_180_420", "window_0_250"]
DISPLAY_RUNS = ["baseline", "speed_only", *D_WINDOWS]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def episode_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {ep["case"]: ep for ep in run.get("episodes", [])}


def outcome(ep: dict[str, Any]) -> str:
    return ("S" if ep["success"] else "F") + str(ep["steps"])


def mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 2) if values else None


def median(values: list[float]) -> float | None:
    return round(statistics.median(values), 2) if values else None


def step_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    def pack(items: list[int]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
        return {
            "count": len(items),
            "mean": mean(items),
            "median": median(items),
            "min": min(items),
            "max": max(items),
        }

    all_steps = [int(ep["steps"]) for ep in episodes]
    success_steps = [int(ep["steps"]) for ep in episodes if ep["success"]]
    fail_steps = [int(ep["steps"]) for ep in episodes if not ep["success"]]
    return {
        "all": pack(all_steps),
        "success": pack(success_steps),
        "failure": pack(fail_steps),
    }


def window_bounds(name: str) -> tuple[int, int] | None:
    if not name.startswith("window_"):
        return None
    _, start, end = name.split("_")
    return int(start), int(end)


def overlap_fraction(ep: dict[str, Any], bounds: tuple[int, int]) -> float:
    start, end = bounds
    steps = int(ep["steps"])
    if steps <= 0:
        return 0.0
    overlap = max(0, min(steps, end) - start)
    return overlap / steps


def window_coverage(run: dict[str, Any]) -> dict[str, Any] | None:
    bounds = window_bounds(run["name"])
    if bounds is None:
        return None
    episodes = run["episodes"]
    fractions = [overlap_fraction(ep, bounds) for ep in episodes]
    success_fractions = [overlap_fraction(ep, bounds) for ep in episodes if ep["success"]]
    fail_fractions = [overlap_fraction(ep, bounds) for ep in episodes if not ep["success"]]
    fallback = run.get("fallback") or {}
    compiled = int(fallback.get("compiled_requests") or 0)
    eager = int(fallback.get("eager_requests") or 0)
    total = compiled + eager
    return {
        "window": {"start": bounds[0], "end": bounds[1]},
        "server_eager_fraction": round(eager / total, 4) if total else None,
        "approx_step_eager_fraction_all": mean(fractions),
        "approx_step_eager_fraction_success": mean(success_fractions),
        "approx_step_eager_fraction_failure": mean(fail_fractions),
    }


def pair_detail(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    lmap = episode_map(left)
    rmap = episode_map(right)
    cases = sorted(set(lmap) & set(rmap), key=case_key)
    repaired = []
    regressed = []
    for case in cases:
        l_ep = lmap[case]
        r_ep = rmap[case]
        if not l_ep["success"] and r_ep["success"]:
            repaired.append(
                {
                    "case": case,
                    "left": outcome(l_ep),
                    "right": outcome(r_ep),
                    "step_delta": int(r_ep["steps"]) - int(l_ep["steps"]),
                }
            )
        if l_ep["success"] and not r_ep["success"]:
            regressed.append(
                {
                    "case": case,
                    "left": outcome(l_ep),
                    "right": outcome(r_ep),
                    "step_delta": int(r_ep["steps"]) - int(l_ep["steps"]),
                }
            )
    return {
        "left": left["name"],
        "right": right["name"],
        "common": len(cases),
        "repaired": repaired,
        "regressed": regressed,
        "net": len(repaired) - len(regressed),
    }


def case_category(base_ep: dict[str, Any], speed_ep: dict[str, Any]) -> str:
    b = bool(base_ep["success"])
    s = bool(speed_ep["success"])
    if b and not s:
        return "harmful_speed_regression"
    if not b and s:
        return "beneficial_speed_branch"
    if b and s:
        return "both_success"
    return "both_failure"


def collect_key_cases(by_run: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = episode_map(by_run["baseline"])
    speed = episode_map(by_run["speed_only"])
    run_maps = {name: episode_map(by_run[name]) for name in DISPLAY_RUNS}
    cases = sorted(base, key=case_key)
    out = []
    for case in cases:
        outcomes = {name: outcome(run_maps[name][case]) for name in DISPLAY_RUNS}
        success_values = {name: bool(run_maps[name][case]["success"]) for name in DISPLAY_RUNS}
        d_values = {name: success_values[name] for name in D_WINDOWS}
        interesting = (
            success_values["baseline"] != success_values["speed_only"]
            or len(set(d_values.values())) > 1
            or any(value != success_values["speed_only"] for value in d_values.values())
        )
        if not interesting:
            continue
        out.append(
            {
                "case": case,
                "category": case_category(base[case], speed[case]),
                "outcomes": outcomes,
            }
        )
    return out


def summarize(data: dict[str, Any]) -> dict[str, Any]:
    by_run = {run["name"]: run for run in data["runs"]}
    overview = []
    for name in DISPLAY_RUNS:
        run = by_run[name]
        coverage = window_coverage(run)
        fallback = run.get("fallback") or {}
        compiled = int(fallback.get("compiled_requests") or 0)
        eager = int(fallback.get("eager_requests") or 0)
        overview.append(
            {
                "name": name,
                "label": run.get("label"),
                "successes": run.get("total_successes"),
                "episodes": run.get("total_episodes"),
                "success_rate": run.get("success_rate"),
                "server_p50_ms": run.get("server_p50_ms"),
                "speedup_vs_baseline": run.get("speedup_vs_baseline_server_p50"),
                "compiled_requests": compiled or None,
                "eager_requests": eager or None,
                "server_eager_fraction": coverage.get("server_eager_fraction") if coverage else None,
                "step_stats": step_stats(run.get("episodes", [])),
                "window_coverage": coverage,
            }
        )

    pairs = []
    for base_name in ["baseline", "speed_only"]:
        for window in D_WINDOWS:
            pairs.append(pair_detail(by_run[base_name], by_run[window]))

    key_cases = collect_key_cases(by_run)
    return {
        "source": str(SOURCE.relative_to(ROOT)),
        "overview": overview,
        "pairs": pairs,
        "key_cases": key_cases,
        "candidate_proxy": {
            "harmful_set": [
                item["case"] for item in key_cases if item["category"] == "harmful_speed_regression"
            ],
            "beneficial_set": [
                item["case"] for item in key_cases if item["category"] == "beneficial_speed_branch"
            ],
            "stable_success_risk_set": [
                item["case"] for item in key_cases if item["category"] == "both_success"
            ],
        },
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def fmt_cases(items: list[dict[str, Any]]) -> str:
    if not items:
        return "-"
    return ", ".join(f"{it['case']} ({it['left']}->{it['right']})" for it in items)


def build_report(analysis: dict[str, Any]) -> str:
    overview_rows = []
    for item in analysis["overview"]:
        stats = item["step_stats"]
        overview_rows.append(
            [
                item["name"],
                f"{item['successes']}/{item['episodes']}",
                fmt(item["server_p50_ms"]),
                f"{fmt(item['speedup_vs_baseline'])}x" if item["speedup_vs_baseline"] else "-",
                fmt(item["server_eager_fraction"]),
                fmt(stats["all"]["mean"]),
                fmt(stats["success"]["median"]),
                fmt(stats["failure"]["median"]),
            ]
        )

    pair_rows = [
        [
            f"{pair['right']} vs {pair['left']}",
            len(pair["repaired"]),
            len(pair["regressed"]),
            pair["net"],
            fmt_cases(pair["repaired"]),
            fmt_cases(pair["regressed"]),
        ]
        for pair in analysis["pairs"]
    ]

    key_rows = []
    for item in analysis["key_cases"]:
        outcomes = item["outcomes"]
        key_rows.append(
            [
                item["case"],
                item["category"],
                outcomes["baseline"],
                outcomes["speed_only"],
                outcomes["window_80_220"],
                outcomes["window_120_320"],
                outcomes["window_180_420"],
                outcomes["window_0_250"],
            ]
        )

    coverage_rows = []
    for item in analysis["overview"]:
        coverage = item.get("window_coverage")
        if not coverage:
            continue
        coverage_rows.append(
            [
                item["name"],
                fmt(coverage["server_eager_fraction"]),
                fmt(coverage["approx_step_eager_fraction_all"]),
                fmt(coverage["approx_step_eager_fraction_success"]),
                fmt(coverage["approx_step_eager_fraction_failure"]),
            ]
        )

    proxy = analysis["candidate_proxy"]
    lines = [
        "# Phase28D Duration-Window 结果分析",
        "",
        f"数据源: `{analysis['source']}`。",
        "",
        "这份报告只分析 Phase28D 的 duration-window fallback 结果，不启动新的 rollout。它的目的有两个：第一，把 `repair/regress` 和执行步数差异说清楚；第二，为下一轮 finer duration proxy 独立出候选设计。",
        "",
        "## 总览",
        "",
        md_table(
            ["run", "success", "server p50 ms", "speedup", "server eager frac", "mean steps", "median success steps", "median failure steps"],
            overview_rows,
        ),
        "",
        "失败 episode 基本都是 `991` 步 horizon failure，所以成功率变化会直接改变平均执行步数。`window_0_250` 的平均步数接近 baseline，是因为它把几个 speed-only horizon failure 拉回了成功；但它的 eager 请求比例也最高，server p50 从 speed-only 的约 `70 ms` 升到约 `89 ms`。",
        "",
        "## Repair / Regress",
        "",
        md_table(["pair", "repair", "regress", "net", "repair cases", "regress cases"], pair_rows),
        "",
        "最重要的对照是 `speed_only`。`window_0_250` 相比 speed-only 修复 `4:6, 6:0, 6:6, 8:7`，同时打坏 `6:7, 6:9`，净收益 `+2`。这说明 early duration fallback 有信息量，但它不是单调安全操作。",
        "",
        "一个合理解释是：`0-250` 覆盖了接近物体、对齐、接触、夹爪闭合和初始搬起这些高敏感阶段。对 manipulation policy 来说，这些步骤决定了后续轨迹是否进入正确 basin；一旦物体已经稳定抓起，后续很多 step 更接近带约束的搬运/放置，误差敏感性可能下降。因此 `0-250` 的收益不一定来自“早期”本身，而可能来自它刚好覆盖了 grasp-critical duration。",
        "",
        "相比 FP16 baseline，`window_0_250` 只差 `-1`：它保留了 speed-only 在 `8:9` 上的 beneficial branch，但仍然打坏 `8:3, 8:10`。这提醒我们，目标不应该只是“复制 baseline”，而应该区分 harmful regression 和 beneficial branch。",
        "",
        "## 执行步数视角",
        "",
        md_table(
            ["case", "category", "baseline", "speed", "80-220", "120-320", "180-420", "0-250"],
            key_rows,
        ),
        "",
        "几个关键模式：",
        "",
        "- `4:6, 6:0, 6:6, 8:7` 是 speed-only 的 harmful regression。`window_0_250` 能修复它们，说明早期控制段对这些失败 basin 很关键。",
        "- `8:3` 很反直觉：`80-220` 和 `120-320` 能救，`0-250` 反而失败。这说明不是越早、越宽越好；过早 eager 会改变它进入的 branch。",
        "- `8:10` 只被 `180-420` 修复，说明 late manipulation / settling 段也有独立敏感性。",
        "- `6:7, 6:9, 8:9` 是 speed-only 的 beneficial branch。一个只追求 baseline mimicry 的 proxy 会倾向于把它们拉回 baseline failure，这是不对的。",
        "",
        "## Window 命中比例",
        "",
        md_table(
            ["window", "server eager frac", "approx step frac all", "approx step frac success", "approx step frac failure"],
            coverage_rows,
        ),
        "",
        "`window_0_250` 对成功 episode 的平均覆盖比例接近 `0.79`，所以它更像“早期大段 eager policy”，不是轻量 guard。`80-220/120-320/180-420` 的 server eager fraction 约 `0.20-0.26`，更便宜，但目前 repair/regress 平衡不够好。",
        "",
        "## 对 finer duration proxy 的启发",
        "",
        f"- Harmful speed regression set: `{', '.join(proxy['harmful_set'])}`",
        f"- Beneficial speed branch set: `{', '.join(proxy['beneficial_set'])}`",
        f"- Stable-success risk set: `{', '.join(proxy['stable_success_risk_set'])}`",
        "",
        "下一轮不应该只问“哪个窗口更像 FP16”，而应该问“哪个窗口能修复 harmful regressions，同时保留 beneficial branches”。一个更合适的 proxy score 可以写成：",
        "",
        "$$",
        "Score(w)=|H_{repair}(w)|-|B_{regress}(w)|-\\alpha |S_{regress}(w)|-\\lambda \\Delta latency(w)",
        "$$",
        "",
        "其中 `H` 是 baseline 成功但 speed-only 失败的 harmful set，`B` 是 baseline 失败但 speed-only 成功的 beneficial set，`S` 是 baseline 和 speed-only 都成功的稳定成功集合。`Delta latency` 用 server p50 或 eager fraction 近似。",
        "",
        "建议把 finer duration proxy 独立成 Phase29：先跑少量 atomic windows 定位敏感时间桶，再跑组合窗口。优先级如下：",
        "",
        "1. Atomic buckets: `0:80,80:160,160:240,240:320,320:500`。用来判断敏感性到底来自 initial approach、pre-grasp/contact、grasp/lift，还是 late transport/settling。",
        "2. Grasp-centered prefixes/combinations: `0:120,0:180,0:220,0:250,80:240,120:280`。用来验证 broad `0:250` 的收益是否能用更窄、更贴近抓取阶段的窗口保留。",
        "3. 如果 Phase29 仍然存在强烈 case-specific 交叉，就进入 layer × duration 的二阶 proxy，而不是继续只调一维 duration。",
        "",
        "对应独立 runner: `toy_quantvla/run_phase29_finer_duration_proxy.sh`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = load_json(SOURCE)
    analysis = summarize(data)
    OUT_JSON.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(analysis), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
