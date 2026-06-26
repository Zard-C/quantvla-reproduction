"""Analyze action-channel sensitivity from existing oracle traces.

This is an offline analysis. It reads Phase 15 action-channel oracle summaries,
Phase 15 oracle traces, and Phase 17 first-divergence artifacts. It does not
run inference or require a GPU.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs" / "phase17_channel"

ACTION_SUMMARY_PATH = RESULTS / "phase15_action_channel_oracle_summary.json"
BRANCHING_PATH = RESULTS / "phase17_first_divergence_branching.json"
GRIPPER_TRACE_PATHS = [
    RESULTS / "phase15_gripper_oracle_dual_server_task04_init09_trace.json",
    RESULTS / "phase15_gripper_oracle_dual_server_task06_init08_trace.json",
]
CONTINUOUS_TRACE_DIR = RESULTS / "phase15_continuous_oracle_traces"

OUT_JSON = RESULTS / "phase17_action_channel_sensitivity.json"
OUT_MD = DOCS / "action_channel_sensitivity_zh.md"

CONTINUOUS_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
TRANSLATION_KEYS = ["x", "y", "z"]
ROTATION_KEYS = ["roll", "pitch", "yaw"]
ACTION_GROUPS = {
    "translation": TRANSLATION_KEYS,
    "rotation": ROTATION_KEYS,
    "continuous": CONTINUOUS_KEYS,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize_values(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": median(values),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def round_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def seconds_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) * 1000.0, 1)


def l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def case_label(trace: dict[str, Any]) -> str:
    return f"task{trace['task_id']}:init{trace['init_index']}"


def diff_for_key(raw_action: dict[str, Any], key: str) -> float | None:
    primary_key = f"primary.action.{key}"
    oracle_key = f"oracle.action.{key}"
    if primary_key not in raw_action or oracle_key not in raw_action:
        return None
    return float(raw_action[primary_key]) - float(raw_action[oracle_key])


def gripper_bin(raw_value: float, threshold: float = 0.5) -> int:
    return -1 if float(raw_value) > threshold else 1


def summarize_oracle_outcomes() -> dict[str, Any]:
    data = read_json(ACTION_SUMMARY_PATH)
    runs = []
    for run in data["runs"]:
        label = run["label"]
        keys = run["oracle_action_keys"]
        runs.append(
            {
                "label": label,
                "oracle_action_keys": keys,
                "successes": int(run["total_successes"]),
                "episodes": int(run["total_episodes"]),
                "success_rate": float(run["success_rate"]),
                "steps_by_case": run["steps_by_case"],
                "client_policy_p50_ms": seconds_to_ms(run.get("client_policy_p50_seconds")),
                "primary_remote_p50_ms": seconds_to_ms(run.get("primary_remote_p50_seconds")),
                "oracle_remote_p50_ms": seconds_to_ms(run.get("oracle_remote_p50_seconds")),
            }
        )

    by_label = {run["label"]: run for run in runs}
    synergy = {
        "continuous_vs_translation": by_label["continuous_action"]["success_rate"]
        - by_label["translation_only"]["success_rate"],
        "continuous_vs_rotation": by_label["continuous_action"]["success_rate"]
        - by_label["rotation_only"]["success_rate"],
        "continuous_vs_best_single_group": by_label["continuous_action"]["success_rate"]
        - max(
            by_label["translation_only"]["success_rate"],
            by_label["rotation_only"]["success_rate"],
            by_label["gripper_only"]["success_rate"],
        ),
        "full_action_minus_continuous": by_label["full_action"]["success_rate"]
        - by_label["continuous_action"]["success_rate"],
    }

    return {
        "source": str(ACTION_SUMMARY_PATH.relative_to(ROOT)),
        "runs": runs,
        "synergy": synergy,
        "interpretation": data.get("interpretation"),
        "summary_abs_diff_by_channel": data["continuous_same_observation_abs_diff"],
    }


def summarize_continuous_traces() -> dict[str, Any]:
    traces = []
    all_dim_abs: dict[str, list[float]] = {key: [] for key in CONTINUOUS_KEYS}
    all_group_l2: dict[str, list[float]] = {group: [] for group in ACTION_GROUPS}

    for path in sorted(CONTINUOUS_TRACE_DIR.glob("*.json")):
        trace = read_json(path)
        label = case_label(trace)
        dim_abs: dict[str, list[float]] = {key: [] for key in CONTINUOUS_KEYS}
        dim_signed: dict[str, list[float]] = {key: [] for key in CONTINUOUS_KEYS}
        group_l2: dict[str, list[float]] = {group: [] for group in ACTION_GROUPS}
        group_top: dict[str, dict[str, Any] | None] = {group: None for group in ACTION_GROUPS}
        per_step_top: list[dict[str, Any]] = []

        for step in trace["steps"]:
            raw = step["raw_action"]
            diffs: dict[str, float] = {}
            for key in CONTINUOUS_KEYS:
                diff = diff_for_key(raw, key)
                if diff is None:
                    continue
                diffs[key] = diff
                dim_signed[key].append(diff)
                dim_abs[key].append(abs(diff))
                all_dim_abs[key].append(abs(diff))

            if len(diffs) != len(CONTINUOUS_KEYS):
                continue

            step_record = {
                "policy_step": int(step["policy_step"]),
                "diff_by_channel": {key: round_float(diffs[key], 6) for key in CONTINUOUS_KEYS},
            }
            for group, keys in ACTION_GROUPS.items():
                group_value = l2([diffs[key] for key in keys])
                group_l2[group].append(group_value)
                all_group_l2[group].append(group_value)
                step_record[f"{group}_l2"] = round_float(group_value, 6)
                current_top = group_top[group]
                if current_top is None or group_value > current_top["l2"]:
                    group_top[group] = {
                        "policy_step": int(step["policy_step"]),
                        "l2": group_value,
                        "max_abs": max(abs(diffs[key]) for key in keys),
                        "argmax_dim": max(keys, key=lambda key: abs(diffs[key])),
                        "diff_by_channel": {key: round_float(diffs[key], 6) for key in keys},
                    }
            per_step_top.append(step_record)

        top_continuous_steps = sorted(per_step_top, key=lambda row: row["continuous_l2"], reverse=True)[:5]
        traces.append(
            {
                "source": str(path.relative_to(ROOT)),
                "case": label,
                "success": bool(trace["success"]),
                "num_steps": int(trace["num_steps"]),
                "dim_abs_stats": {
                    key: {stat: round_float(value, 6) for stat, value in summarize_values(values).items()}
                    for key, values in dim_abs.items()
                },
                "dim_signed_mean": {
                    key: round_float(sum(values) / len(values), 6) if values else None
                    for key, values in dim_signed.items()
                },
                "group_l2_stats": {
                    group: {stat: round_float(value, 6) for stat, value in summarize_values(values).items()}
                    for group, values in group_l2.items()
                },
                "group_top": {
                    group: {
                        **top,
                        "l2": round_float(top["l2"], 6),
                        "max_abs": round_float(top["max_abs"], 6),
                    }
                    if top is not None
                    else None
                    for group, top in group_top.items()
                },
                "top_continuous_steps": top_continuous_steps,
            }
        )

    return {
        "source_dir": str(CONTINUOUS_TRACE_DIR.relative_to(ROOT)),
        "traces": traces,
        "aggregate_dim_abs_stats": {
            key: {stat: round_float(value, 6) for stat, value in summarize_values(values).items()}
            for key, values in all_dim_abs.items()
        },
        "aggregate_group_l2_stats": {
            group: {stat: round_float(value, 6) for stat, value in summarize_values(values).items()}
            for group, values in all_group_l2.items()
        },
    }


def summarize_gripper_oracle_traces() -> dict[str, Any]:
    traces = []
    all_abs: list[float] = []
    mismatch_count = 0
    step_count = 0
    for path in GRIPPER_TRACE_PATHS:
        trace = read_json(path)
        diffs: list[float] = []
        mismatches = 0
        top = None
        for step in trace["steps"]:
            raw = step["raw_action"]
            primary = raw.get("primary.action.gripper")
            oracle = raw.get("gripper_oracle.action.gripper")
            if primary is None or oracle is None:
                continue
            diff = float(primary) - float(oracle)
            abs_diff = abs(diff)
            diffs.append(abs_diff)
            all_abs.append(abs_diff)
            step_count += 1
            mismatch = gripper_bin(primary) != gripper_bin(oracle)
            if mismatch:
                mismatches += 1
                mismatch_count += 1
            if top is None or abs_diff > top["abs_diff"]:
                top = {
                    "policy_step": int(step["policy_step"]),
                    "primary": float(primary),
                    "oracle": float(oracle),
                    "diff": diff,
                    "abs_diff": abs_diff,
                    "primary_bin": gripper_bin(primary),
                    "oracle_bin": gripper_bin(oracle),
                    "bin_mismatch": mismatch,
                }
        traces.append(
            {
                "source": str(path.relative_to(ROOT)),
                "case": case_label(trace),
                "success": bool(trace["success"]),
                "num_steps": int(trace["num_steps"]),
                "bin_mismatch_count": mismatches,
                "abs_diff_stats": {
                    stat: round_float(value, 6) for stat, value in summarize_values(diffs).items()
                },
                "top_abs_diff": {
                    **top,
                    "diff": round_float(top["diff"], 6),
                    "abs_diff": round_float(top["abs_diff"], 6),
                }
                if top is not None
                else None,
            }
        )
    return {
        "traces": traces,
        "aggregate": {
            "steps": step_count,
            "bin_mismatch_count": mismatch_count,
            "abs_diff_stats": {
                stat: round_float(value, 6) for stat, value in summarize_values(all_abs).items()
            },
        },
    }


def summarize_branching_channels() -> dict[str, Any]:
    data = read_json(BRANCHING_PATH)
    cases = []
    for item in data["closed_loop_branching"]:
        events = item["events"]
        cases.append(
            {
                "case": item["label"].replace("_", ":"),
                "first_continuous_action_ge_0p005": events.get("first_continuous_action_ge_0p005"),
                "first_continuous_action_ge_0p01": events.get("first_continuous_action_ge_0p01"),
                "first_continuous_action_ge_0p05": events.get("first_continuous_action_ge_0p05"),
                "first_gripper_binary_mismatch": events.get("first_gripper_binary_mismatch"),
                "max_continuous_action_before_gripper_mismatch": item.get(
                    "max_continuous_action_before_gripper_mismatch"
                ),
            }
        )
    return {
        "source": str(BRANCHING_PATH.relative_to(ROOT)),
        "cases": cases,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def event_cell(event: dict[str, Any] | None) -> str:
    if event is None:
        return "-"
    return "{step} / {dim} / {value}".format(
        step=event["policy_step"],
        dim=event.get("continuous_action_argmax_dim") or event.get("full_action_argmax_dim"),
        value=fmt(event.get("continuous_action_max_abs"), 6),
    )


def build_report(artifact: dict[str, Any]) -> str:
    outcomes = artifact["oracle_outcomes"]
    continuous = artifact["continuous_oracle_trace_stats"]
    gripper = artifact["gripper_oracle_trace_stats"]
    branching = artifact["branching_channel_events"]

    outcome_rows = []
    for run in outcomes["runs"]:
        keys = ",".join(run["oracle_action_keys"])
        outcome_rows.append(
            [
                run["label"],
                f"`{keys}`",
                f'{run["successes"]}/{run["episodes"]}',
                run["steps_by_case"].get("task4:init9"),
                run["steps_by_case"].get("task6:init8"),
                f'{fmt(run["client_policy_p50_ms"], 1)} ms',
            ]
        )

    dim_rows = []
    for key, stats in continuous["aggregate_dim_abs_stats"].items():
        dim_rows.append(
            [
                key,
                fmt(stats["mean"], 6),
                fmt(stats["p50"], 6),
                fmt(stats["p99"], 6),
                fmt(stats["max"], 6),
            ]
        )

    group_rows = []
    for group, stats in continuous["aggregate_group_l2_stats"].items():
        group_rows.append(
            [
                group,
                fmt(stats["mean"], 6),
                fmt(stats["p50"], 6),
                fmt(stats["p99"], 6),
                fmt(stats["max"], 6),
            ]
        )

    per_case_rows = []
    for trace in continuous["traces"]:
        top = trace["group_top"]["continuous"]
        trans = trace["group_top"]["translation"]
        rot = trace["group_top"]["rotation"]
        per_case_rows.append(
            [
                trace["case"],
                trace["success"],
                trace["num_steps"],
                f'{top["policy_step"]} / {top["argmax_dim"]} / {fmt(top["l2"], 6)}',
                f'{trans["policy_step"]} / {trans["argmax_dim"]} / {fmt(trans["l2"], 6)}',
                f'{rot["policy_step"]} / {rot["argmax_dim"]} / {fmt(rot["l2"], 6)}',
            ]
        )

    branching_rows = []
    for case in branching["cases"]:
        branching_rows.append(
            [
                case["case"],
                event_cell(case["first_continuous_action_ge_0p005"]),
                event_cell(case["first_continuous_action_ge_0p01"]),
                event_cell(case["first_continuous_action_ge_0p05"]),
                event_cell(case["max_continuous_action_before_gripper_mismatch"]),
                event_cell(case["first_gripper_binary_mismatch"]),
            ]
        )

    gripper_rows = []
    for trace in gripper["traces"]:
        top = trace["top_abs_diff"]
        gripper_rows.append(
            [
                trace["case"],
                trace["success"],
                trace["num_steps"],
                trace["bin_mismatch_count"],
                fmt(trace["abs_diff_stats"]["mean"], 6),
                fmt(trace["abs_diff_stats"]["max"], 6),
                f'{top["policy_step"]} / {fmt(top["diff"], 6)} / {top["bin_mismatch"]}',
            ]
        )

    lines = [
        "# Phase 17: action channel 敏感性分析",
        "",
        "## 结论摘要",
        "",
        "这轮分析回答的是：闭环敏感扰动到底落在哪些动作通道上。结论不是某一个单独通道有决定性问题，而是 `x/y/z/roll/pitch/yaw` 这 6 个连续动作维度的联合小偏移改变了闭环轨迹。",
        "",
        "关键证据：",
        "",
        "- `gripper-only` oracle 是 `0/2`，说明只修夹爪救不回失败。",
        "- `translation-only` 和 `rotation-only` oracle 都是 `0/2`，说明单独修位置或姿态也救不回。",
        "- `continuous-action` oracle 是 `2/2`，而且不替换 gripper，说明连续 6D 联合修正足以恢复两个病例。",
        "- `full-action` oracle 也是 `2/2`，但相比 continuous-action 没有额外收益，说明 gripper 不是必要修正项。",
        "",
        "因此这里的 channel sensitivity 更像多维协同敏感性，而不是单轴敏感性。",
        "",
        "## Oracle 成败矩阵",
        "",
        md_table(
            ["setting", "oracle keys", "success", "task4:init9 steps", "task6:init8 steps", "client p50"],
            outcome_rows,
        ),
        "",
        "这个矩阵的逻辑很强：如果根因是夹爪通道，`gripper-only` 应该救回来；如果根因是纯位置或纯姿态，`translation-only` 或 `rotation-only` 应该至少救回一部分。但实际只有连续 6D 一起替换才成功。",
        "",
        "## 同观测逐维差异",
        "",
        md_table(["channel", "mean abs", "p50 abs", "p99 abs", "max abs"], dim_rows),
        "",
        "逐维看，`x/z/pitch` 的 max 明显大，来自 `task6:init8` 的 step253 局部 outlier；`y` 的 p99 也偏高。`roll/yaw` 较小，但它们仍属于连续动作组合的一部分。这里不能把结论简化成只修 `x` 或只修 `z`，因为 oracle 结果已经显示 translation-only 不够。",
        "",
        "## 动作组 L2 敏感性",
        "",
        md_table(["group", "mean L2", "p50 L2", "p99 L2", "max L2"], group_rows),
        "",
        "组级别看，translation L2 通常更大，rotation L2 较小但在 outlier 步与 translation 同向出现。闭环系统吃到的是完整 6D action，不是独立通道；因此较小的姿态误差也可能改变末端接触法向、物体受力和后续视觉状态。",
        "",
        "## 每个病例的 top channel event",
        "",
        md_table(
            ["case", "success", "steps", "top continuous", "top translation", "top rotation"],
            per_case_rows,
        ),
        "",
        "`task6:init8` 的 top continuous event 正是之前定位的 step253，主导维度是 `x`，但同时带着 `z/pitch` 分量。`task4:init9` 没有同等大的单点 spike，更像较早、多步的小偏移累积。",
        "",
        "## 闭环 first-divergence 里的主导维度",
        "",
        md_table(
            [
                "case",
                "first cont >=0.005",
                "first cont >=0.01",
                "first cont >=0.05",
                "max cont before gripper split",
                "first gripper split",
            ],
            branching_rows,
        ),
        "",
        "这张表连接了 open-loop channel diff 与 closed-loop 分叉。`task4:init9` 早期先由 `y` 触发连续动作阈值，随后 `z/x` 在位姿分叉和夹爪相位分裂附近占主导；`task6:init8` 早期是 `x/z` 主导。也就是说，敏感扰动主要集中在末端位姿相关的连续维度，而不是原始 gripper channel。",
        "",
        "## Gripper 通道排除证据",
        "",
        md_table(
            ["case", "oracle success", "steps", "bin mismatch", "mean abs diff", "max abs diff", "top step / diff / bin mismatch"],
            gripper_rows,
        ),
        "",
        f"两个 gripper-only oracle trace 合计 {gripper['aggregate']['steps']} 步，二值 mismatch 总数是 {gripper['aggregate']['bin_mismatch_count']}。这说明在 compiled 已经走到的 observation 上，primary gripper 与 eager oracle gripper 的二值判断一致。gripper mismatch 更像轨迹分叉后的表征，而不是最初触发源。",
        "",
        "## 工程含义",
        "",
        "后续如果要做闭环误差修正，优先级应该是：",
        "",
        "1. 监控连续 6D action 的联合 drift，尤其是 translation L2 与 pitch 同步异常。",
        "2. 不要只调 gripper threshold / hysteresis；这会修表征，不修触发源。",
        "3. 局部 fallback 或校准目标应该围绕完整连续动作向量做，而不是单独修某一轴。",
        "4. 对 compile / FP4 后端的验收指标应加入 `continuous L2 p99/max` 和 `first-divergence`，不能只看平均 action MSE。",
        "",
        "## 限制",
        "",
        "当前 oracle 样本只有两个重点病例，因此结论是机制性证据，不是总体统计结论。它足以指导下一轮工程验证：先保护连续 6D，再看闭环成功率是否恢复；但不能宣称所有 LIBERO 失败都由同样通道组合导致。",
        "",
        "## 产物",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Report: `{OUT_MD.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    artifact = {
        "sources": [
            str(ACTION_SUMMARY_PATH.relative_to(ROOT)),
            str(CONTINUOUS_TRACE_DIR.relative_to(ROOT)),
            *(str(path.relative_to(ROOT)) for path in GRIPPER_TRACE_PATHS),
            str(BRANCHING_PATH.relative_to(ROOT)),
        ],
        "oracle_outcomes": summarize_oracle_outcomes(),
        "continuous_oracle_trace_stats": summarize_continuous_traces(),
        "gripper_oracle_trace_stats": summarize_gripper_oracle_traces(),
        "branching_channel_events": summarize_branching_channels(),
    }
    write_json(OUT_JSON, artifact)
    write_text(OUT_MD, build_report(artifact))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
