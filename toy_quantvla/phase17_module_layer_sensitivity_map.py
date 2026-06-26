"""Build a module/layer sensitivity map from existing experiment artifacts.

This script is offline-only. It combines DiT module structure, compile-scope
drift sweeps, step-focused bisects, and matched rollout summaries into a
single JSON artifact plus a Chinese Markdown report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs" / "phase17_module_map"

MODULES_PATH = RESULTS / "phase13_action_head_modules_v1.json"
COMPILE_SCOPE_PATH = RESULTS / "phase16_compile_scope_continuous_4_9_6_8_s260_v1_summary.json"
STEP253_COMPILE_BISECT_PATH = RESULTS / "phase16_step253_compile_bisect_v1_summary.json"
STEP253_BLOCK03_BISECT_PATH = RESULTS / "phase16_step253_block03_bisect_v1_summary.json"

BLOCK_ISLAND_30_BASELINE = RESULTS / "phase13_block_island_30case_mem_v1_baseline_client_latency.json"
BLOCK_ISLAND_30_COMPILED = RESULTS / "phase13_block_island_30case_mem_v1_compiled_client_latency.json"
BLOCK_ISLAND_6_15_BASELINE = RESULTS / "phase14_block_island_6_15_5flip_v1_baseline_client_latency.json"
BLOCK_ISLAND_6_15_COMPILED = RESULTS / "phase14_block_island_6_15_5flip_v1_compiled_client_latency.json"

FP16_15CASE = RESULTS / "phase9_seeded_matched_t4_6_8_15_v1_fp16_client_latency.json"
FP4_UP_PROJ_15CASE = RESULTS / "phase9_seeded_matched_t4_6_8_15_v1_fp4_up_proj_warmdesc_client_latency.json"
FP4_DIT_MLP_15CASE = RESULTS / "phase10_seeded_matched_dit_mlp_only_v1_fp4_dit_mlp_only_warmdesc_client_latency.json"
RESIDUAL_CACHE_15CASE = RESULTS / "phase11_seeded_matched_residual_cache_v1_fp16_dit_residual_cache_b1215_s1to6_client_latency.json"

ONLINE_DRIFT_PATHS = [
    RESULTS / "phase7_online_drift_llm_mlp_20.json",
    RESULTS / "phase7_online_drift_dit_mlp_40.json",
    RESULTS / "phase7_online_drift_llm_dit_none_10_v2.json",
    RESULTS / "phase7_online_drift_llm_dit_atm_ohb_10_v2.json",
]

OUT_JSON = RESULTS / "phase17_module_layer_sensitivity_map.json"
OUT_MD = DOCS / "module_layer_sensitivity_map_zh.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def round_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def seconds_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) * 1000.0, 1)


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


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{fmt(value, 1)} ms"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def safe_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def summarize_action_head_structure() -> dict[str, Any]:
    data = read_json(MODULES_PATH)
    rows = data["rows"]
    block_params = {idx: 0 for idx in range(16)}
    block_linear_count = {idx: 0 for idx in range(16)}
    class_counts: dict[str, int] = {}

    for row in rows:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
        match = re.match(r"transformer_blocks\.(\d+)(?:\.|$)", row["name"])
        if not match:
            continue
        idx = int(match.group(1))
        if row["children"] == 0:
            block_params[idx] += int(row["parameters"])
            if row["class"] == "Linear":
                block_linear_count[idx] += 1

    return {
        "source": str(MODULES_PATH.relative_to(ROOT)),
        "action_head_class": data["action_head_class"],
        "action_head_model_class": data["action_head_model_class"],
        "module_rows": len(rows),
        "transformer_blocks": 16,
        "block_params": block_params,
        "block_linear_count": block_linear_count,
        "total_block_params": sum(block_params.values()),
        "class_counts": class_counts,
    }


def summarize_compile_scope_drift() -> list[dict[str, Any]]:
    data = read_json(COMPILE_SCOPE_PATH)
    rows = []
    for run in data["runs"]:
        aggregate = run["aggregate"]
        cmax = aggregate["continuous_max_abs"]
        cl2 = aggregate["continuous_l2"]
        compiled = aggregate.get("compiled_get_action_seconds", {})
        eager = aggregate.get("eager_get_action_seconds", {})
        compiled_p50 = seconds_to_ms(compiled.get("p50"))
        eager_p50 = seconds_to_ms(eager.get("p50"))
        speedup = None
        if compiled_p50 and eager_p50 and compiled_p50 > 0:
            speedup = round(eager_p50 / compiled_p50, 2)

        valid_steps = int(run["valid_drift_steps"])
        if valid_steps == 0:
            sensitivity = "invalid"
            reason = "no valid replay steps; CUDAGraph overwrite path"
            cmax_p99 = None
            cmax_max = None
            cl2_p99 = None
            compiled_p50 = None
            eager_p50 = None
            speedup = None
        elif cmax["max"] >= 0.05:
            sensitivity = "high"
            reason = "contains deterministic continuous-action spike"
            cmax_p99 = round_float(cmax["p99"], 6)
            cmax_max = round_float(cmax["max"], 6)
            cl2_p99 = round_float(cl2["p99"], 6)
        elif cmax["p99"] <= 0.0032 and (speedup is None or speedup <= 1.0):
            sensitivity = "low-drift-slow"
            reason = "low same-observation drift but no useful speedup"
            cmax_p99 = round_float(cmax["p99"], 6)
            cmax_max = round_float(cmax["max"], 6)
            cl2_p99 = round_float(cl2["p99"], 6)
        else:
            sensitivity = "medium"
            reason = "moderate drift/speed tradeoff"
            cmax_p99 = round_float(cmax["p99"], 6)
            cmax_max = round_float(cmax["max"], 6)
            cl2_p99 = round_float(cl2["p99"], 6)

        rows.append(
            {
                "target": run["target"],
                "valid_steps": valid_steps,
                "continuous_max_p99": cmax_p99,
                "continuous_max_max": cmax_max,
                "continuous_l2_p99": cl2_p99,
                "compiled_p50_ms": compiled_p50,
                "eager_p50_ms": eager_p50,
                "p50_speedup": speedup,
                "sensitivity": sensitivity,
                "reason": reason,
            }
        )
    return rows


def summarize_step253_bisect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (STEP253_COMPILE_BISECT_PATH, STEP253_BLOCK03_BISECT_PATH):
        data = read_json(path)
        for run in data["runs"]:
            target = run["target"]
            if target in seen:
                continue
            seen.add(target)
            top = run["top_step"]
            cmax = float(top["continuous_max_abs_max"])
            compiled_ms = seconds_to_ms(top["compiled_p50_seconds"])
            eager_ms = seconds_to_ms(top["eager_p50_seconds"])
            speedup = None
            if compiled_ms and eager_ms and compiled_ms > 0:
                speedup = round(eager_ms / compiled_ms, 2)
            rows.append(
                {
                    "target": target,
                    "top_step": int(top["policy_step"]),
                    "top_dim": top["top_dim"],
                    "continuous_max_abs": round_float(cmax, 6),
                    "continuous_l2_p99": round_float(top["continuous_l2_p99"], 6),
                    "repeat_span": round_float(top["same_seed_repeat_span_max"], 6),
                    "compiled_p50_ms": compiled_ms,
                    "eager_p50_ms": eager_ms,
                    "p50_speedup": speedup,
                    "spike_removed_at_0p005": cmax < 0.005,
                    "raw_diff": {key: round_float(value, 6) for key, value in top["raw_diff"].items()},
                }
            )

    order = [
        "action_head_model",
        "action_head_model_blocks_0_15_eager",
        "action_head_model_blocks_0_7_eager",
        "action_head_model_blocks_0_3_eager",
        "action_head_model_blocks_0_1_eager",
        "action_head_model_blocks_0_0_eager",
        "action_head_model_blocks_2_3_eager",
        "action_head_model_blocks_1_1_eager",
        "action_head_model_blocks_2_2_eager",
        "action_head_model_blocks_3_3_eager",
        "action_head_model_blocks_4_7_eager",
        "action_head_model_blocks_8_15_eager",
        "action_head_model_attn_all_eager",
        "action_head_model_ff_all_eager",
        "action_head_dit_attn_all",
    ]
    rank = {target: idx for idx, target in enumerate(order)}
    return sorted(rows, key=lambda item: rank.get(item["target"], 999))


def rollout_summary(path: Path, label: str) -> dict[str, Any]:
    data = read_json(path)
    p50_ms = seconds_to_ms(safe_get(data, "policy_latency_seconds", "p50"))
    p90_ms = seconds_to_ms(safe_get(data, "policy_latency_seconds", "p90"))
    task_counts: dict[str, dict[str, int]] = {}
    for episode in data.get("episodes", []):
        task = f"task{episode['task_id']}"
        if task not in task_counts:
            task_counts[task] = {"successes": 0, "episodes": 0}
        task_counts[task]["episodes"] += 1
        task_counts[task]["successes"] += int(bool(episode["success"]))
    return {
        "source": str(path.relative_to(ROOT)),
        "label": label,
        "successes": int(data["total_successes"]),
        "episodes": int(data["total_episodes"]),
        "success_rate": round_float(data["success_rate"], 4),
        "client_p50_ms": p50_ms,
        "client_p90_ms": p90_ms,
        "task_counts": task_counts,
    }


def summarize_rollouts() -> dict[str, Any]:
    baseline_30 = rollout_summary(BLOCK_ISLAND_30_BASELINE, "fp16_baseline_30")
    blocks_8_15 = rollout_summary(BLOCK_ISLAND_30_COMPILED, "compile_blocks_8_15_eager_30")
    baseline_5 = rollout_summary(BLOCK_ISLAND_6_15_BASELINE, "fp16_baseline_5_flip")
    blocks_6_15 = rollout_summary(BLOCK_ISLAND_6_15_COMPILED, "compile_blocks_6_15_eager_5_flip")
    fp16_15 = rollout_summary(FP16_15CASE, "fp16_baseline_15_seeded")
    fp4_up = rollout_summary(FP4_UP_PROJ_15CASE, "fp4_llm_up_proj_15_seeded")
    fp4_dit = rollout_summary(FP4_DIT_MLP_15CASE, "fp4_dit_mlp_15_seeded")
    residual_cache = rollout_summary(RESIDUAL_CACHE_15CASE, "fp16_dit_residual_cache_b12_15_s1_6")

    comparisons = [
        {
            "name": "blocks_8_15_eager_vs_fp16_30",
            "baseline": baseline_30,
            "candidate": blocks_8_15,
            "delta_successes": blocks_8_15["successes"] - baseline_30["successes"],
            "p50_speedup": round(baseline_30["client_p50_ms"] / blocks_8_15["client_p50_ms"], 2),
        },
        {
            "name": "blocks_6_15_eager_vs_fp16_5_flip",
            "baseline": baseline_5,
            "candidate": blocks_6_15,
            "delta_successes": blocks_6_15["successes"] - baseline_5["successes"],
            "p50_speedup": round(baseline_5["client_p50_ms"] / blocks_6_15["client_p50_ms"], 2),
        },
        {
            "name": "fp4_llm_up_proj_vs_fp16_15",
            "baseline": fp16_15,
            "candidate": fp4_up,
            "delta_successes": fp4_up["successes"] - fp16_15["successes"],
            "p50_speedup": round(fp16_15["client_p50_ms"] / fp4_up["client_p50_ms"], 2),
        },
        {
            "name": "fp4_dit_mlp_vs_fp16_15",
            "baseline": fp16_15,
            "candidate": fp4_dit,
            "delta_successes": fp4_dit["successes"] - fp16_15["successes"],
            "p50_speedup": round(fp16_15["client_p50_ms"] / fp4_dit["client_p50_ms"], 2),
        },
        {
            "name": "residual_cache_b12_15_vs_fp16_15",
            "baseline": fp16_15,
            "candidate": residual_cache,
            "delta_successes": residual_cache["successes"] - fp16_15["successes"],
            "p50_speedup": round(fp16_15["client_p50_ms"] / residual_cache["client_p50_ms"], 2),
        },
    ]
    return {
        "summaries": [
            baseline_30,
            blocks_8_15,
            baseline_5,
            blocks_6_15,
            fp16_15,
            fp4_up,
            fp4_dit,
            residual_cache,
        ],
        "comparisons": comparisons,
    }


def summarize_online_quant_drift() -> list[dict[str, Any]]:
    rows = []
    for path in ONLINE_DRIFT_PATHS:
        if not path.exists():
            continue
        data = read_json(path)
        summaries = data["episode_summaries"]
        raw_mean = mean(ep["drift_summary"]["raw_relative_rmse"]["mean"] for ep in summaries)
        raw_max = max(ep["drift_summary"]["raw_relative_rmse"]["max"] for ep in summaries)
        libero_max = max(ep["drift_summary"]["libero_action_max_abs_diff"]["max"] for ep in summaries)
        student_over_teacher = []
        for ep in summaries:
            timing = ep["drift_summary"].get("student_over_teacher_time")
            if timing:
                student_over_teacher.append(timing["mean"])
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "scope": data.get("scope"),
                "mode": data.get("mode", "none"),
                "quant_format": data.get("quant_format"),
                "patched_modules": int(data.get("patched_modules", 0)),
                "episodes": int(data.get("episodes", len(summaries))),
                "raw_relative_rmse_mean": round_float(raw_mean, 6),
                "raw_relative_rmse_max": round_float(raw_max, 6),
                "libero_action_max_abs_diff_max": round_float(libero_max, 6),
                "student_over_teacher_time_mean": round_float(mean(student_over_teacher), 3)
                if student_over_teacher
                else None,
            }
        )
    return rows


def build_layer_heatmap(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_target = {row["target"]: row for row in step_rows}

    def row_for(block_range: str, evidence_target: str, note: str) -> dict[str, Any]:
        evidence = by_target.get(evidence_target)
        return {
            "block_or_scope": block_range,
            "evidence_target": evidence_target,
            "continuous_max_abs": evidence["continuous_max_abs"] if evidence else None,
            "spike_removed": evidence["spike_removed_at_0p005"] if evidence else None,
            "p50_speedup": evidence["p50_speedup"] if evidence else None,
            "risk": "low for step253" if evidence and evidence["spike_removed_at_0p005"] else "high for step253",
            "note": note,
        }

    return [
        row_for(
            "block0",
            "action_head_model_blocks_0_0_eager",
            "只保留 block0 eager 即可消除 step253 spike；当前最强层级定位。",
        ),
        row_for(
            "block1",
            "action_head_model_blocks_1_1_eager",
            "单独 block1 eager 不能消除 step253 spike。",
        ),
        row_for(
            "block2",
            "action_head_model_blocks_2_2_eager",
            "单独 block2 eager 不能消除 step253 spike。",
        ),
        row_for(
            "block3",
            "action_head_model_blocks_3_3_eager",
            "单独 block3 eager 不能消除 step253 spike。",
        ),
        row_for(
            "blocks2-3",
            "action_head_model_blocks_2_3_eager",
            "block2-3 eager 不能消除 step253 spike；问题不在这个后续小段。",
        ),
        row_for(
            "blocks4-7",
            "action_head_model_blocks_4_7_eager",
            "block4-7 eager 不能消除 step253 spike。",
        ),
        row_for(
            "blocks8-15",
            "action_head_model_blocks_8_15_eager",
            "历史上改善部分 15-case，但 30-case 净回归，且不消除 task6 step253 spike。",
        ),
    ]


def build_report(artifact: dict[str, Any]) -> str:
    structure = artifact["action_head_structure"]
    compile_scope = artifact["compile_scope_drift"]
    step253 = artifact["step253_bisect"]
    layer_heatmap = artifact["layer_heatmap"]
    rollouts = artifact["matched_rollout_comparisons"]["comparisons"]
    quant_drift = artifact["online_quant_drift"]

    block_params = structure["block_params"]
    block_linear = structure["block_linear_count"]
    block_rows = [
        [idx, block_linear[str(idx)] if isinstance(block_linear, dict) and str(idx) in block_linear else block_linear[idx], f"{block_params[str(idx)] if isinstance(block_params, dict) and str(idx) in block_params else block_params[idx]:,}"]
        for idx in range(16)
    ]

    compile_rows = [
        [
            row["target"],
            row["valid_steps"],
            fmt(row["continuous_max_p99"], 6),
            fmt(row["continuous_max_max"], 6),
            fmt(row["continuous_l2_p99"], 6),
            fmt_ms(row["compiled_p50_ms"]),
            f'{fmt(row["p50_speedup"], 2)}x' if row["p50_speedup"] else "-",
            row["sensitivity"],
        ]
        for row in compile_scope
    ]

    selected_step_targets = {
        "action_head_model",
        "action_head_model_blocks_0_0_eager",
        "action_head_model_blocks_1_1_eager",
        "action_head_model_blocks_2_2_eager",
        "action_head_model_blocks_3_3_eager",
        "action_head_model_blocks_4_7_eager",
        "action_head_model_blocks_8_15_eager",
        "action_head_dit_attn_all",
    }
    step_rows = [
        [
            row["target"],
            row["top_step"],
            row["top_dim"],
            fmt(row["continuous_max_abs"], 6),
            fmt(row["continuous_l2_p99"], 6),
            row["spike_removed_at_0p005"],
            fmt_ms(row["compiled_p50_ms"]),
            f'{fmt(row["p50_speedup"], 2)}x' if row["p50_speedup"] else "-",
        ]
        for row in step253
        if row["target"] in selected_step_targets
    ]

    heat_rows = [
        [
            row["block_or_scope"],
            row["evidence_target"],
            fmt(row["continuous_max_abs"], 6),
            row["spike_removed"],
            f'{fmt(row["p50_speedup"], 2)}x' if row["p50_speedup"] else "-",
            row["note"],
        ]
        for row in layer_heatmap
    ]

    rollout_rows = [
        [
            row["name"],
            f'{row["baseline"]["successes"]}/{row["baseline"]["episodes"]}',
            f'{row["candidate"]["successes"]}/{row["candidate"]["episodes"]}',
            row["delta_successes"],
            fmt_ms(row["baseline"]["client_p50_ms"]),
            fmt_ms(row["candidate"]["client_p50_ms"]),
            f'{fmt(row["p50_speedup"], 2)}x',
        ]
        for row in rollouts
    ]

    quant_rows = [
        [
            row["scope"],
            row["mode"],
            row["patched_modules"],
            row["episodes"],
            fmt(row["raw_relative_rmse_mean"], 6),
            fmt(row["raw_relative_rmse_max"], 6),
            fmt(row["libero_action_max_abs_diff_max"], 6),
            fmt(row["student_over_teacher_time_mean"], 3),
        ]
        for row in quant_drift
    ]

    design_rows = [
        [
            "DiT block0 compiled path",
            "high",
            "step253 spike can be removed by block0 eager only",
            "优先做 block0 eager closed-loop 或 block0 内部 hook。",
        ],
        [
            "DiT blocks8-15 eager island",
            "medium/high",
            "keeps ~2x p50 but 30-case 13/30 vs FP16 16/30",
            "可作为速度候选，但不是行为透明 backend。",
        ],
        [
            "DiT attention/FF small scopes",
            "low drift / low utility",
            "continuous max <=0.0064 but compiled p50 not faster",
            "适合诊断，不适合部署加速。",
        ],
        [
            "LLM up_proj FP4",
            "behavior-changing",
            "15-case seeded 9/15 vs FP16 7/15, but speed not improved",
            "可作为轨迹重分配研究对象，不是稳定加速结论。",
        ],
        [
            "DiT MLP FP4",
            "medium",
            "15-case 7/15 vs FP16 7/15 but p50 slower",
            "kernel/packing 成本没解决前不适合 rollout 扩大。",
        ],
        [
            "DiT residual cache b12,b15",
            "offline-stable but closed-loop risky",
            "offline drift low; 15-case 5/15 vs FP16 7/15",
            "缓存类方法也要通过闭环门槛，不能只看 action drift。",
        ],
    ]

    lines = [
        "# Phase 17: 模块/层级敏感性地图",
        "",
        "## 结论摘要",
        "",
        "这张地图把目前的证据收敛成一个判断：闭环敏感性主要集中在 action head 的 DiT 路径，尤其是大图编译后的前段 block0 compiled path；LLM/DiT MLP 的低比特替换也会改变轨迹分布，但当前更像行为扰动源，不是已经可用的加速 backend。",
        "",
        "最重要的更新是：早期 Phase 13 看到 blocks 8-15 eager island 能救回部分 case，曾让后半 blocks 看起来像主要风险区；但 Phase 16.6 的 step253 bisect 进一步把一个稳定 outlier 缩到 `block0 compiled path`。所以现在的地图不是“后半 blocks 单点敏感”，而是“action_head.model 大图路径存在多处敏感边界，block0 是已定位的确定性尖峰源”。",
        "",
        "## Action Head 结构",
        "",
        f"- action head: `{structure['action_head_class']}`",
        f"- action head model: `{structure['action_head_model_class']}`",
        f"- DiT transformer blocks: `{structure['transformer_blocks']}`",
        f"- transformer block total parameters: `{structure['total_block_params']:,}`",
        "",
        md_table(["block", "linear layers", "leaf params"], block_rows),
        "",
        "每个 transformer block 的参数规模基本一致，因此敏感性差异不能简单归因于某个 block 参数更多；更可能来自 block 所处阶段、residual/norm/attention/FFN 组合路径，以及编译边界造成的数值路径变化。",
        "",
        "## Same-Observation Compile Scope 地图",
        "",
        md_table(
            [
                "target",
                "steps",
                "cont max p99",
                "cont max max",
                "cont L2 p99",
                "compiled p50",
                "speedup",
                "sensitivity",
            ],
            compile_rows,
        ),
        "",
        "读法：小 scope 的 `attn_all` / `ff_all` 漂移最低，但速度没有收益；大图 `action_head_model` 快，但存在 `0.054932` 连续动作尖峰；后半 eager island 没有消掉这个尖峰。",
        "",
        "## Step253 层级 Bisect",
        "",
        md_table(
            [
                "target",
                "top step",
                "dim",
                "cont max",
                "L2 p99",
                "spike removed",
                "compiled p50",
                "speedup",
            ],
            step_rows,
        ),
        "",
        "关键结论：`blocks_0_0_eager` 能消除 step253 尖峰；`blocks_1_1_eager`、`blocks_2_2_eager`、`blocks_3_3_eager`、`blocks_4_7_eager`、`blocks_8_15_eager` 都不能。当前最小已知责任范围是 `DiT block0 compiled path`。",
        "",
        "## Block-Level Heatmap",
        "",
        md_table(
            ["block/scope", "evidence target", "cont max", "spike removed", "speedup", "note"],
            heat_rows,
        ),
        "",
        "这张 heatmap 只对 step253 这个已定位 outlier 有效，不等于所有任务所有阶段的全局敏感性。它的价值在于给出了一个可以工程化验证的最小保护候选：`action_head_model_blocks_0_0_eager`。",
        "",
        "## Matched Rollout 工程地图",
        "",
        md_table(
            [
                "comparison",
                "baseline",
                "candidate",
                "delta",
                "baseline p50",
                "candidate p50",
                "p50 speedup",
            ],
            rollout_rows,
        ),
        "",
        "这里能看到几个工程事实：`blocks_8_15_eager` 有接近 2x p50，但 30-case 成功率从 16/30 到 13/30；`blocks_6_15_eager` 更保守但速度掉到约 1.2x，行为也没有改善；DiT MLP FP4 15-case 行为持平但更慢；residual cache 离线漂移低，但闭环 15-case 回归。",
        "",
        "## Quant Scope Drift 补充",
        "",
        md_table(
            [
                "scope",
                "mode",
                "modules",
                "episodes",
                "raw RMSE mean",
                "raw RMSE max",
                "LIBERO max diff",
                "student/teacher time",
            ],
            quant_rows,
        ),
        "",
        "这组是 teacher-controlled online drift，不是闭环成功率。它说明量化 scope 越大，动作漂移整体会变大；ATM/OHB 能缓解 full selected scope 的漂移，但不能把问题变成单调安全。",
        "",
        "## 当前模块优先级",
        "",
        md_table(["module/scope", "sensitivity", "evidence", "next action"], design_rows),
        "",
        "## 对后续工程的含义",
        "",
        "1. 如果继续 torch.compile 路线，下一步最值得验证的是 `action_head_model_blocks_0_0_eager` 的小规模 closed-loop，而不是继续盲目扩大后半 eager island。",
        "2. 如果继续 FP4/CUTLASS 路线，不能只按模块数量扩大 scope；要先用 same-observation continuous drift 和 first-divergence 过滤。",
        "3. LLM up_proj、DiT MLP、residual cache 都能改变轨迹分布，但目前都没有同时满足“速度收益 + 闭环透明”。",
        "4. 真正有价值的敏感性指标应该绑定模块边界：`module scope -> continuous drift -> first divergence -> outcome flip`，而不是只看 module-level average MSE。",
        "",
        "## 限制",
        "",
        "这张地图汇总的是已有实验，不是新的 GPU benchmark。某些结论是 case-specific：例如 block0 定位来自 `task6:init8 step253`，后半 blocks 结论来自 Phase 13/14 的 matched subsets。它适合指导下一步实验优先级，不应被解读成所有 VLA 模型的普适层级定律。",
        "",
        "## 产物",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Report: `{OUT_MD.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    step253 = summarize_step253_bisect()
    artifact = {
        "sources": [
            str(MODULES_PATH.relative_to(ROOT)),
            str(COMPILE_SCOPE_PATH.relative_to(ROOT)),
            str(STEP253_COMPILE_BISECT_PATH.relative_to(ROOT)),
            str(STEP253_BLOCK03_BISECT_PATH.relative_to(ROOT)),
            str(BLOCK_ISLAND_30_BASELINE.relative_to(ROOT)),
            str(BLOCK_ISLAND_30_COMPILED.relative_to(ROOT)),
            str(BLOCK_ISLAND_6_15_BASELINE.relative_to(ROOT)),
            str(BLOCK_ISLAND_6_15_COMPILED.relative_to(ROOT)),
            str(FP16_15CASE.relative_to(ROOT)),
            str(FP4_UP_PROJ_15CASE.relative_to(ROOT)),
            str(FP4_DIT_MLP_15CASE.relative_to(ROOT)),
            str(RESIDUAL_CACHE_15CASE.relative_to(ROOT)),
            *(str(path.relative_to(ROOT)) for path in ONLINE_DRIFT_PATHS if path.exists()),
        ],
        "action_head_structure": summarize_action_head_structure(),
        "compile_scope_drift": summarize_compile_scope_drift(),
        "step253_bisect": step253,
        "layer_heatmap": build_layer_heatmap(step253),
        "matched_rollout_comparisons": summarize_rollouts(),
        "online_quant_drift": summarize_online_quant_drift(),
    }
    write_json(OUT_JSON, artifact)
    write_text(OUT_MD, build_report(artifact))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
