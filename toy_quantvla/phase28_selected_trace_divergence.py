"""First-divergence analysis for selected Phase 28A rollout traces."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE_ROOT = ROOT / "toy_quantvla" / "results" / "phase28A_proxy_guided_15case_v1_selected_trace"
DOCS = ROOT / "docs"
OUT_JSON = ROOT / "toy_quantvla" / "results" / "phase28A_proxy_guided_15case_v1_selected_trace_divergence.json"
OUT_MD = DOCS / "phase28_proxy_guided_trace_divergence_zh.md"

CASES = ["4:6", "6:0", "8:10", "8:9", "4:9"]
RUNS = {
    "baseline": "baseline",
    "speed_only": "speed_only",
    "proxy_blocks8_15": "proxy_blocks8_15",
}
PAIRS = [
    ("baseline", "speed_only"),
    ("speed_only", "proxy_blocks8_15"),
    ("baseline", "proxy_blocks8_15"),
]


def l2(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def first_above(values: list[float], threshold: float) -> int | None:
    for idx, value in enumerate(values):
        if value > threshold:
            return idx
    return None


def case_glob(case: str) -> str:
    task, init = case.split(":")
    return f"task={int(task):02d}--init={int(init):02d}*.json"


def load_trace(run: str, case: str) -> dict[str, Any]:
    matches = sorted((TRACE_ROOT / run).glob(case_glob(case)))
    if not matches:
        raise FileNotFoundError(f"missing trace for {run} {case}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def outcome(trace: dict[str, Any]) -> str:
    return ("S" if trace["success"] else "F") + str(trace["num_steps"])


def fmt_step(value: int | None) -> str:
    return "-" if value is None else str(value)


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    n = min(len(left["steps"]), len(right["steps"]))
    eef_pos = []
    gripper = []
    action = []
    for idx in range(n):
        lstep = left["steps"][idx]
        rstep = right["steps"][idx]
        eef_pos.append(l2(lstep["pre_robot0_eef_pos"], rstep["pre_robot0_eef_pos"]))
        gripper.append(l2(lstep["pre_robot0_gripper_qpos"], rstep["pre_robot0_gripper_qpos"]))
        action.append(l2(lstep["libero_action"], rstep["libero_action"]))

    return {
        "common_steps": n,
        "left_outcome": outcome(left),
        "right_outcome": outcome(right),
        "first_action_l2_gt_0p05": first_above(action, 0.05),
        "first_eef_pos_gt_1cm": first_above(eef_pos, 0.01),
        "first_eef_pos_gt_5cm": first_above(eef_pos, 0.05),
        "max_eef_pos_m": round(max(eef_pos) if eef_pos else 0.0, 5),
        "max_gripper_qpos": round(max(gripper) if gripper else 0.0, 5),
        "action_l2_p50": round(quantile(action, 0.50), 5),
        "action_l2_p95": round(quantile(action, 0.95), 5),
        "action_l2_max": round(max(action) if action else 0.0, 5),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_data() -> dict[str, Any]:
    traces = {run: {case: load_trace(dirname, case) for case in CASES} for run, dirname in RUNS.items()}
    comparisons = []
    for left_name, right_name in PAIRS:
        for case in CASES:
            row = compare(traces[left_name][case], traces[right_name][case])
            row.update({"case": case, "left": left_name, "right": right_name})
            comparisons.append(row)
    return {
        "phase": "phase28A_selected_trace_divergence",
        "cases": CASES,
        "runs": sorted(RUNS),
        "thresholds": {
            "action_l2": 0.05,
            "eef_pos_1cm_m": 0.01,
            "eef_pos_5cm_m": 0.05,
        },
        "comparisons": comparisons,
    }


def case_note(case: str) -> str:
    notes = {
        "4:6": "speed-only regression; proxy_blocks8_15 restores a short successful branch close to baseline.",
        "6:0": "speed-only regression; proxy_blocks8_15 restores the baseline branch with small EEF deviation.",
        "8:10": "speed-only regression; proxy_blocks8_15 repairs outcome but follows a longer branch than baseline.",
        "8:9": "speed-only repair; proxy_blocks8_15 preserves the beneficial success branch.",
        "4:9": "proxy_blocks8_15 regression; protection changes a baseline/speed-only success into horizon failure.",
    }
    return notes[case]


def build_report(data: dict[str, Any]) -> str:
    by_pair_case = {
        (row["left"], row["right"], row["case"]): row for row in data["comparisons"]
    }

    speed_proxy_rows = []
    for case in CASES:
        row = by_pair_case[("speed_only", "proxy_blocks8_15", case)]
        speed_proxy_rows.append(
            [
                case,
                row["left_outcome"],
                row["right_outcome"],
                fmt_step(row["first_action_l2_gt_0p05"]),
                fmt_step(row["first_eef_pos_gt_1cm"]),
                fmt_step(row["first_eef_pos_gt_5cm"]),
                row["max_eef_pos_m"],
                row["action_l2_p50"],
                row["action_l2_p95"],
                case_note(case),
            ]
        )

    baseline_proxy_rows = []
    for case in CASES:
        row = by_pair_case[("baseline", "proxy_blocks8_15", case)]
        baseline_proxy_rows.append(
            [
                case,
                row["left_outcome"],
                row["right_outcome"],
                fmt_step(row["first_eef_pos_gt_1cm"]),
                fmt_step(row["first_eef_pos_gt_5cm"]),
                row["max_eef_pos_m"],
                row["action_l2_p50"],
                row["action_l2_p95"],
            ]
        )

    lines = [
        "# Phase 28A Selected Trace Divergence",
        "",
        "这份报告使用从 5090 `/tmp` 拉回的 selected rollout traces，分析 `baseline`、`speed_only`、`proxy_blocks8_15` 在关键病例上的 first-divergence。阈值是经验诊断阈值：`libero_action` L2 大于 `0.05` 记为动作开始分叉，EEF 位置差大于 `1cm` / `5cm` 记为状态分叉。",
        "",
        "注意：这里的 `S/F + steps` 来自 rollout trace 里的 env step 数；失败样本在 client latency summary 里可能显示为 `F991`，因为官方 eval loop 在 terminated episode 后还会多发一次 action request。",
        "",
        "## speed-only vs proxy_blocks8_15",
        "",
        md_table(
            [
                "case",
                "speed",
                "proxy",
                "first action >0.05",
                "first EEF >1cm",
                "first EEF >5cm",
                "max EEF m",
                "action p50",
                "action p95",
                "note",
            ],
            speed_proxy_rows,
        ),
        "",
        "这张表说明两个事实。第一，动作差异通常早于可见状态分叉：例如 `4:6` 在 step `35` 已经出现 action L2 > 0.05，但 EEF 到 step `83` 才超过 1cm，到 step `192` 才超过 5cm。第二，成功/失败翻转不要求一开始就有巨大状态误差；小的动作分叉会先进入闭环，再经过几十到上百步放大成不同轨迹分支。",
        "",
        "## baseline vs proxy_blocks8_15",
        "",
        md_table(
            [
                "case",
                "baseline",
                "proxy",
                "first EEF >1cm",
                "first EEF >5cm",
                "max EEF m",
                "action p50",
                "action p95",
            ],
            baseline_proxy_rows,
        ),
        "",
        "`4:6` 和 `6:0` 是最干净的 repair 证据：proxy_blocks8_15 与 baseline 都成功，而且最大 EEF 差分别只有约 `1.8cm` 和 `4.9cm`，没有超过 5cm 的大分叉。这说明保护 `blocks8-15` 不只是随机改变结果，而是在这些 case 上把 speed-only 拉回了接近 baseline 的成功 basin。",
        "",
        "`8:10` 则更复杂：baseline 与 proxy 都成功，但 proxy 轨迹明显偏离 baseline，并且成功步数从 `385` 变成 `604`。这更像是 proxy 找到另一个可行 branch，而不是严格复刻 FP16。",
        "",
        "`4:9` 是风险样本：baseline 和 speed-only 都成功，但 proxy_blocks8_15 失败，并且 EEF 在 step `115` 超过 5cm。这提醒我们，保护敏感层不是单调安全操作；它本身也会重分配闭环轨迹。",
        "",
        "## 结论",
        "",
        "Phase 28A 的 trace 证据支持一个更具体的工程判断：closed-loop proxy 不应该只输出一个“保护哪些层”的静态答案，还需要用 paired traces 检查它到底是在恢复 baseline basin、进入另一条可行 branch，还是制造新的失败 branch。",
        "",
        "对当前配置来说，`proxy_blocks8_15` 是更好的 speed-quality 折中，但还不是最终方案。它修复了 speed-only 在 `4:6, 6:0, 8:10` 的退化，并保住 `8:9` 的 beneficial success；代价是 `4:9` 出现新 regression，且 server p50 加速从 `1.68x` 降到 `1.26x`。",
        "",
        "## 产物",
        "",
        f"- Selected traces: `{TRACE_ROOT.relative_to(ROOT)}`",
        f"- Divergence JSON: `{OUT_JSON.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build_data()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
