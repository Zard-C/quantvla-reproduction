"""Mine closed-loop-sensitive perturbation evidence from existing artifacts.

This script is intentionally offline: it reads prior rollout summaries, traces,
oracle diagnostics, and latency reports that already exist in the repository.
It does not run inference or require a GPU.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_table_after(text: str, marker: str) -> list[dict[str, str]]:
    marker_idx = text.index(marker)
    after = text[marker_idx:]
    lines = after.splitlines()
    table_start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("|"):
            table_start = idx
            break
    if table_start is None:
        return []

    table_lines: list[str] = []
    for line in lines[table_start:]:
        if not line.strip().startswith("|"):
            break
        table_lines.append(line.strip())
    if len(table_lines) < 2:
        return []

    headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def fraction_success(value: str) -> tuple[int, int]:
    match = re.search(r"(-?\d+)\s*/\s*(\d+)", value)
    if not match:
        raise ValueError(f"Expected fraction like '8/15', got {value!r}")
    return int(match.group(1)), int(match.group(2))


def signed_int(value: str) -> int:
    match = re.search(r"[-+]?\d+", value)
    if not match:
        raise ValueError(f"Expected integer, got {value!r}")
    return int(match.group(0))


def exact_mcnemar_p(repairs: int, regressions: int) -> float:
    """Two-sided exact McNemar/binomial sign-test p-value."""
    n = int(repairs) + int(regressions)
    if n == 0:
        return 1.0
    k = min(int(repairs), int(regressions))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def parse_phase5_ablation() -> dict[str, Any]:
    path = DOCS / "phase5_llm_dit_mlp_ablation_init0_14.md"
    text = read_text(path)

    per_task_rows = extract_table_after(text, "## Per-Task Result")
    task_rows: list[dict[str, Any]] = []
    for row in per_task_rows:
        none, total = fraction_success(row["none"])
        atm, _ = fraction_success(row["atm"])
        ohb, _ = fraction_success(row["ohb"])
        values = {"none": none, "atm": atm, "ohb": ohb}
        task_rows.append(
            {
                "task_id": int(row["task id"]),
                "task": row["task"],
                "total": total,
                "none": none,
                "atm": atm,
                "ohb": ohb,
                "atm_delta": signed_int(row["atm - none"]),
                "ohb_delta": signed_int(row["ohb - none"]),
                "volatility": max(values.values()) - min(values.values()),
                "best_mode": max(values, key=values.get),
                "worst_mode": min(values, key=values.get),
            }
        )

    per_init_rows = extract_table_after(text, "## Per-Init Result")
    init_rows: list[dict[str, Any]] = []
    for row in per_init_rows:
        none, total = fraction_success(row["none"])
        atm, _ = fraction_success(row["atm"])
        ohb, _ = fraction_success(row["ohb"])
        values = {"none": none, "atm": atm, "ohb": ohb}
        init_rows.append(
            {
                "init_index": int(row["init index"]),
                "total": total,
                "none": none,
                "atm": atm,
                "ohb": ohb,
                "volatility": max(values.values()) - min(values.values()),
                "best_mode": max(values, key=values.get),
                "worst_mode": min(values, key=values.get),
            }
        )

    paired_rows = extract_table_after(text, "## Paired Outcomes")
    paired = []
    for row in paired_rows:
        repairs = signed_int(row["repaired failures"])
        regressions = signed_int(row["new regressions"])
        paired.append(
            {
                "comparison": row["comparison"],
                "repairs": repairs,
                "regressions": regressions,
                "same_success": signed_int(row["same success"]),
                "same_failure": signed_int(row["same failure"]),
                "net": signed_int(row["net"]),
                "churn": repairs + regressions,
                "churn_rate": (repairs + regressions) / 150.0,
                "exact_mcnemar_p": exact_mcnemar_p(repairs, regressions),
            }
        )

    return {
        "source": str(path.relative_to(ROOT)),
        "task_rows": sorted(task_rows, key=lambda item: (-item["volatility"], item["task_id"])),
        "init_rows": sorted(init_rows, key=lambda item: (-item["volatility"], item["init_index"])),
        "paired_outcomes": paired,
    }


def parse_phase5_fp16_vs_quant() -> dict[str, Any]:
    path = DOCS / "phase5_init5_14_generalization.md"
    text = read_text(path)
    per_task_rows = extract_table_after(text, "Per-task comparison:")
    task_rows = []
    for row in per_task_rows:
        fp16, total = fraction_success(row["FP16"])
        quant, _ = fraction_success(row["quant"])
        task_rows.append(
            {
                "task_id": int(row["task id"]),
                "task": row["task"],
                "total": total,
                "fp16": fp16,
                "quant_atm_ohb": quant,
                "delta": signed_int(row["delta"]),
            }
        )

    paired_rows = extract_table_after(text, "Paired outcome changes")
    paired = {row["transition"]: signed_int(row["count"]) for row in paired_rows}
    repairs = paired.get("FP16 failure, quant success", 0)
    regressions = paired.get("FP16 success, quant failure", 0)
    return {
        "source": str(path.relative_to(ROOT)),
        "task_rows": sorted(task_rows, key=lambda item: (item["delta"], -item["task_id"])),
        "paired_outcomes": paired,
        "paired_summary": {
            "repairs": repairs,
            "regressions": regressions,
            "net": repairs - regressions,
            "churn": repairs + regressions,
            "churn_rate": (repairs + regressions) / 100.0,
            "exact_mcnemar_p": exact_mcnemar_p(repairs, regressions),
        },
    }


def first_row(rows: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return row
    return None


def summarize_first_divergence() -> dict[str, Any]:
    path = RESULTS / "phase14_first_divergence_blocks_8_15_4_9_analysis.json"
    data = read_json(path)
    rows = data["per_step"]

    events = []
    thresholds = [
        ("first_action_max_ge_0.005", lambda row: float(row["action_max"]) >= 0.005),
        ("first_pre_pos_l2_ge_0.5mm", lambda row: float(row["pre_pos_l2"]) >= 0.0005),
        ("first_pre_pos_l2_ge_1mm", lambda row: float(row["pre_pos_l2"]) >= 0.001),
        ("first_pre_pos_l2_ge_5mm", lambda row: float(row["pre_pos_l2"]) >= 0.005),
        ("first_pre_pos_l2_ge_10mm", lambda row: float(row["pre_pos_l2"]) >= 0.010),
        ("first_gripper_binary_mismatch", lambda row: row["baseline_action"][6] != row["compiled_action"][6]),
    ]
    for label, predicate in thresholds:
        row = first_row(rows, predicate)
        if row is None:
            continue
        events.append(
            {
                "event": label,
                "policy_step": int(row["policy_step"]),
                "action_dim": row.get("action_dim"),
                "action_max": float(row["action_max"]),
                "pre_pos_l2_m": float(row["pre_pos_l2"]),
                "baseline_gripper": float(row["baseline_action"][6]),
                "compiled_gripper": float(row["compiled_action"][6]),
            }
        )

    max_pre = max(rows, key=lambda row: float(row["pre_pos_l2"]))
    return {
        "source": str(path.relative_to(ROOT)),
        "baseline_success": data["baseline_success"],
        "compiled_success": data["compiled_success"],
        "baseline_steps": data["baseline_steps"],
        "compiled_steps": data["compiled_steps"],
        "events": events,
        "max_pre_pos_l2": {
            "policy_step": int(max_pre["policy_step"]),
            "pre_pos_l2_m": float(max_pre["pre_pos_l2"]),
        },
    }


def summarize_action_channel_oracle() -> dict[str, Any]:
    path = RESULTS / "phase15_action_channel_oracle_summary.json"
    data = read_json(path)
    runs = []
    for run in data["runs"]:
        runs.append(
            {
                "label": run["label"],
                "oracle_action_keys": run["oracle_action_keys"],
                "successes": run["total_successes"],
                "episodes": run["total_episodes"],
                "success_rate": run["success_rate"],
                "steps_by_case": run["steps_by_case"],
                "client_policy_p50_ms": 1000.0 * run["client_policy_p50_seconds"],
            }
        )
    return {
        "source": str(path.relative_to(ROOT)),
        "runs": runs,
        "continuous_same_observation_abs_diff": data["continuous_same_observation_abs_diff"],
    }


def summarize_compile_scope_continuous() -> dict[str, Any]:
    path = RESULTS / "phase16_compile_scope_continuous_4_9_6_8_s260_v1_summary.json"
    data = read_json(path)
    runs = []
    for run in data["runs"]:
        aggregate = run.get("aggregate", {})
        episode_summaries = run.get("episode_summaries", [])
        exceptions = [episode for episode in episode_summaries if episode.get("exception")]
        successes = sum(1 for episode in episode_summaries if episode.get("success"))
        row = {
            "target": run["target"],
            "episodes": run["episodes"],
            "successes": successes,
            "valid_drift_steps": run.get("valid_drift_steps", 0),
            "compile_wrap_seconds": run.get("compile_wrap_seconds"),
            "exception_count": len(exceptions),
        }
        for key in [
            "continuous_l2",
            "continuous_max_abs",
            "translation_l2",
            "rotation_l2",
            "gripper_abs",
            "compiled_over_eager_time",
        ]:
            stats = aggregate.get(key, {})
            row[f"{key}_mean"] = stats.get("mean", 0.0)
            row[f"{key}_p50"] = stats.get("p50", 0.0)
            row[f"{key}_p99"] = stats.get("p99", 0.0)
            row[f"{key}_max"] = stats.get("max", 0.0)
        runs.append(row)
    return {"source": str(path.relative_to(ROOT)), "runs": runs}


def parse_phase13_speed_behavior() -> dict[str, Any]:
    path = DOCS / "phase13_block_island_30case_mem_v1.md"
    text = read_text(path)
    summary_rows = extract_table_after(text, "## 总表")
    flip_rows = extract_table_after(text, "逐 case 翻转")
    return {
        "source": str(path.relative_to(ROOT)),
        "summary_rows": summary_rows,
        "flip_rows": flip_rows,
    }


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def build_report(data: dict[str, Any]) -> str:
    ablation = data["phase5_ablation"]
    fp16_quant = data["phase5_fp16_vs_quant"]
    first_div = data["first_divergence"]
    oracle = data["action_channel_oracle"]
    compile_scope = data["compile_scope_continuous"]
    speed = data["phase13_speed_behavior"]

    lines: list[str] = []
    lines.append("# Phase 17 闭环敏感扰动离线挖掘")
    lines.append("")
    lines.append("本报告只使用仓库中已有的 rollout、trace、oracle、compile-scope 和 latency 结果；没有重新跑推理，也不需要 5090/GPU。")
    lines.append("")
    lines.append("## 核心结论")
    lines.append("")
    lines.append("当前数据已经支持把问题改写成：先识别哪些扰动是闭环敏感的，再决定把校准、残差修正、mixed precision 或 fallback 预算花在哪里。")
    lines.append("")
    lines.append("已有证据指向四类敏感性：")
    lines.append("")
    lines.append("1. task/init 敏感性：task4 和 task8 在不同补偿模式下波动最大。")
    lines.append("2. 时刻敏感性：同观测误差很小，但闭环里早期小差异会在接触/夹爪阶段放大。")
    lines.append("3. 动作子空间敏感性：gripper 单独不是根因，translation 或 rotation 单独也不够，连续 6D 联合修正能恢复两个病例。")
    lines.append("4. 模块/后端敏感性：blocks 8-15 eager-island 后端有明显 p50 加速，但 matched rollout 有净回归。")
    lines.append("")

    lines.append("## 1. Task/Init 敏感性")
    lines.append("")
    top_tasks = ablation["task_rows"][:6]
    lines.append(md_table(
        ["task", "none", "atm", "ohb", "volatility", "best", "worst"],
        [
            [
                item["task_id"],
                f'{item["none"]}/{item["total"]}',
                f'{item["atm"]}/{item["total"]}',
                f'{item["ohb"]}/{item["total"]}',
                item["volatility"],
                item["best_mode"],
                item["worst_mode"],
            ]
            for item in top_tasks
        ],
    ))
    lines.append("")
    lines.append("最敏感的是 task4 和 task8：二者在 `none/atm/ohb` 三种模式之间都有 5/15 的成功率摆动。task4 被 ATM/OHB 明显修复，task8 被 ATM/OHB 明显打坏。这是典型的轨迹重分配，而不是单调提升。")
    lines.append("")
    lines.append(md_table(
        ["comparison", "repairs", "regressions", "same success", "same failure", "net"],
        [
            [
                row["comparison"],
                row["repairs"],
                row["regressions"],
                row["same_success"],
                row["same_failure"],
                row["net"],
            ]
            for row in ablation["paired_outcomes"]
        ],
    ))
    lines.append("")
    lines.append("同一批 150 个 task-init pair 里，ATM/OHB 的净收益很小，但 repairs/regressions 都有两位数。这说明聚合成功率低估了行为 churn。")
    lines.append("")
    lines.append(md_table(
        ["comparison", "churn", "churn rate", "exact McNemar p"],
        [
            [
                row["comparison"],
                row["churn"],
                f'{100.0 * row["churn_rate"]:.1f}%',
                fmt_num(row["exact_mcnemar_p"], 3),
            ]
            for row in ablation["paired_outcomes"]
        ],
    ))
    lines.append("")
    lines.append("这些 p-value 都不支持把小净增益解释为稳定 dominance；但 churn rate 接近 18%-19%，说明闭环轨迹重分配本身非常明显。")
    lines.append("")
    top_inits = ablation["init_rows"][:5]
    lines.append("最敏感的 init slice：")
    lines.append("")
    lines.append(md_table(
        ["init", "none", "atm", "ohb", "volatility", "best", "worst"],
        [
            [
                item["init_index"],
                f'{item["none"]}/{item["total"]}',
                f'{item["atm"]}/{item["total"]}',
                f'{item["ohb"]}/{item["total"]}',
                item["volatility"],
                item["best_mode"],
                item["worst_mode"],
            ]
            for item in top_inits
        ],
    ))
    lines.append("")
    lines.append("FP16 vs `llm_dit_mlp + atm_ohb` 在 init5-14 上也呈现同样结构：")
    lines.append("")
    lines.append(md_table(
        ["task", "FP16", "quant", "delta"],
        [
            [row["task_id"], f'{row["fp16"]}/{row["total"]}', f'{row["quant_atm_ohb"]}/{row["total"]}', row["delta"]]
            for row in sorted(fp16_quant["task_rows"], key=lambda item: -abs(item["delta"]))[:6]
        ],
    ))
    lines.append("")
    paired = fp16_quant["paired_outcomes"]
    paired_summary = fp16_quant["paired_summary"]
    lines.append(
        f"init5-14 的净 `+6/100` 来自 `{paired_summary['repairs']}` 个 repair 减去 `{paired_summary['regressions']}` 个 regression。"
        f" churn rate 为 `{100.0 * paired_summary['churn_rate']:.1f}%`，exact McNemar p 为 `{fmt_num(paired_summary['exact_mcnemar_p'], 3)}`。"
    )
    lines.append("")

    lines.append("## 2. First-Divergence 敏感时刻")
    lines.append("")
    lines.append(f"病例：`task4:init9`，baseline success `{first_div['baseline_steps']}` steps，compiled fail `{first_div['compiled_steps']}` steps。")
    lines.append("")
    lines.append(md_table(
        ["event", "step", "dim", "action max", "EEF pre diff"],
        [
            [
                event["event"],
                event["policy_step"],
                event["action_dim"],
                fmt_num(event["action_max"], 4),
                f'{fmt_num(1000.0 * event["pre_pos_l2_m"], 3)} mm',
            ]
            for event in first_div["events"]
        ],
    ))
    lines.append("")
    lines.append(f"最大 EEF pre-position 分离出现在 step {first_div['max_pre_pos_l2']['policy_step']}，约 `{fmt_num(1000.0 * first_div['max_pre_pos_l2']['pre_pos_l2_m'], 1)} mm`。关键点是：第一次明显 action diff 出现在 step 8，而 gripper 二值化相反出现在 step 46；这更像闭环放大链条，而不是单步数值爆炸。")
    lines.append("")

    lines.append("## 3. Action-Channel Oracle")
    lines.append("")
    lines.append(md_table(
        ["setting", "oracle keys", "success", "task4:init9", "task6:init8", "client p50"],
        [
            [
                run["label"],
                ",".join(run["oracle_action_keys"]),
                f'{run["successes"]}/{run["episodes"]}',
                run["steps_by_case"].get("task4:init9"),
                run["steps_by_case"].get("task6:init8"),
                f'{fmt_num(run["client_policy_p50_ms"], 1)} ms',
            ]
            for run in oracle["runs"]
        ],
    ))
    lines.append("")
    lines.append("这张表把 gripper 从“疑似根因”降级成“分叉表征”：gripper-only oracle 仍 0/2，continuous 6D oracle 2/2 成功且不需要替换 gripper。闭环敏感扰动更像 translation+rotation 的联合小偏移。")
    lines.append("")
    diff_rows = []
    for key, stats in oracle["continuous_same_observation_abs_diff"].items():
        diff_rows.append([key, fmt_num(stats["mean"], 6), fmt_num(stats["p99"], 6), fmt_num(stats["max"], 6)])
    lines.append(md_table(["channel", "mean abs", "p99 abs", "max abs"], diff_rows))
    lines.append("")

    lines.append("## 4. Module/Backend 敏感性")
    lines.append("")
    if speed["summary_rows"]:
        lines.append(md_table(list(speed["summary_rows"][0].keys()), [list(row.values()) for row in speed["summary_rows"]]))
        lines.append("")
    if speed["flip_rows"]:
        lines.append("30-case matched set 的逐 case 翻转：")
        lines.append("")
        lines.append(md_table(list(speed["flip_rows"][0].keys()), [list(row.values()) for row in speed["flip_rows"]]))
        lines.append("")
    lines.append("这条后端路线有接近 2x 的 p50 速度收益，但 30-case 成功率从 16/30 到 13/30，说明不能只看 latency。")
    lines.append("")

    valid_scope_rows = [
        row for row in compile_scope["runs"] if row["valid_drift_steps"] > 0
    ]
    valid_scope_rows = sorted(
        valid_scope_rows,
        key=lambda row: (
            -row["successes"],
            row["continuous_l2_p99"],
            row["compiled_over_eager_time_p50"],
        ),
    )
    lines.append("Phase16 continuous diff sweep 中可用于比较的 scope：")
    lines.append("")
    lines.append(md_table(
        ["target", "success", "drift steps", "cont l2 p99", "trans l2 p99", "rot l2 p99", "speed ratio p50"],
        [
            [
                row["target"],
                f'{row["successes"]}/{row["episodes"]}',
                row["valid_drift_steps"],
                fmt_num(row["continuous_l2_p99"], 6),
                fmt_num(row["translation_l2_p99"], 6),
                fmt_num(row["rotation_l2_p99"], 6),
                fmt_num(row["compiled_over_eager_time_p50"], 3),
            ]
            for row in valid_scope_rows
        ],
    ))
    lines.append("")

    lines.append("## 5. 现在能形成的 Risk Map")
    lines.append("")
    lines.append(md_table(
        ["axis", "high-risk evidence", "engineering implication"],
        [
            ["task", "task4/task8 volatility = 5/15", "校准/评估必须覆盖这些边界任务"],
            ["init", "init-level 成功率非单调，ATM init8 从 9/10 到 5/10", "不能用少量 init smoke test 代表整体"],
            ["time", "step 8 小 action diff，step 46 gripper 分叉，后续 EEF 分离到厘米级", "trace 应记录 first divergence 和接触窗口"],
            ["action", "continuous 6D oracle 2/2，单独 gripper/translation/rotation 0/2", "residual correction 应优先修连续动作联合方向"],
            ["module/backend", "blocks 8-15 eager island p50 加速约 2x，但 16/30 -> 13/30", "低精度/compile scope 必须同时过闭环行为门槛"],
        ],
    ))
    lines.append("")
    lines.append("## 后续不跑 GPU 也能继续做的事")
    lines.append("")
    lines.append("1. 从 request trace 中抽更多 step-level action diff，扩展 first-divergence 样本数。")
    lines.append("2. 把 keyframe 病例按接触失败、抓取时机、目标边界、多物体干扰分类。")
    lines.append("3. 对 150-case paired outcomes 做 bootstrap/McNemar，量化 aggregate gap 与 churn 的不确定性。")
    lines.append("4. 把 `state × action direction × module` 的 risk map 接入论文 discussion，作为 closed-loop correction 的数据依据。")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for source in data["source_files"]:
        lines.append(f"- `{source}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = {
        "phase": "phase17_closed_loop_sensitivity_mining",
        "source_files": [
            "docs/phase5_llm_dit_mlp_ablation_init0_14.md",
            "docs/phase5_init5_14_generalization.md",
            "docs/phase13_block_island_30case_mem_v1.md",
            "toy_quantvla/results/phase14_first_divergence_blocks_8_15_4_9_analysis.json",
            "toy_quantvla/results/phase15_action_channel_oracle_summary.json",
            "toy_quantvla/results/phase16_compile_scope_continuous_4_9_6_8_s260_v1_summary.json",
        ],
        "phase5_ablation": parse_phase5_ablation(),
        "phase5_fp16_vs_quant": parse_phase5_fp16_vs_quant(),
        "first_divergence": summarize_first_divergence(),
        "action_channel_oracle": summarize_action_channel_oracle(),
        "compile_scope_continuous": summarize_compile_scope_continuous(),
        "phase13_speed_behavior": parse_phase13_speed_behavior(),
    }

    write_json(RESULTS / "phase17_closed_loop_sensitivity_mining.json", data)
    report = build_report(data)
    (DOCS / "phase17_closed_loop_sensitivity_mining_zh.md").write_text(report, encoding="utf-8")
    print("Wrote toy_quantvla/results/phase17_closed_loop_sensitivity_mining.json")
    print("Wrote docs/phase17_closed_loop_sensitivity_mining_zh.md")


if __name__ == "__main__":
    main()
