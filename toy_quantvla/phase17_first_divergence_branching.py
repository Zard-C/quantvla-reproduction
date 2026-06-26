"""Analyze first-divergence and trajectory branching from existing traces.

The script is offline-only. It reads prior closed-loop paired traces,
same-observation replay summaries, and focused compile-bisect summaries, then
emits a compact JSON artifact plus a Chinese Markdown report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs" / "phase17_branching"

DIM_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

GRIPPER_MARGIN_PATH = RESULTS / "phase14_gripper_margin_blocks_8_15_4_9_6_8_analysis.json"
FOCUSED_REPLAY_PATH = RESULTS / "phase16_step253_focused_replay_v1_summary.json"
COMPILE_BISECT_PATH = RESULTS / "phase16_step253_compile_bisect_v1_summary.json"
BLOCK03_BISECT_PATH = RESULTS / "phase16_step253_block03_bisect_v1_summary.json"

OUT_JSON = RESULTS / "phase17_first_divergence_branching.json"
OUT_MD = DOCS / "first_divergence_trajectory_branching_zh.md"


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


def mm(value_m: Any) -> float | None:
    if value_m is None:
        return None
    return round(float(value_m) * 1000.0, 3)


def pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


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


def continuous_max_abs(row: dict[str, Any]) -> float:
    diffs = row.get("action_diff") or []
    if len(diffs) < 6:
        return 0.0
    return max(abs(float(value)) for value in diffs[:6])


def continuous_argmax_dim(row: dict[str, Any]) -> str | None:
    diffs = row.get("action_diff") or []
    if len(diffs) < 6:
        return None
    idx = max(range(6), key=lambda i: abs(float(diffs[i])))
    return DIM_NAMES[idx]


def first_event(
    rows: list[dict[str, Any]],
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return {
                "event": name,
                "policy_step": int(row["policy_step"]),
                "continuous_action_max_abs": round_float(continuous_max_abs(row), 6),
                "continuous_action_argmax_dim": continuous_argmax_dim(row),
                "full_action_max_abs": round_float(row.get("action_max_abs_diff"), 6),
                "full_action_argmax_dim": row.get("action_argmax_dim"),
                "pre_pos_l2_mm": mm(row.get("pre_pos_l2")),
                "post_pos_l2_mm": mm(row.get("post_pos_l2")),
                "gripper_bin_mismatch": bool(row.get("gripper_bin_mismatch", False)),
                "baseline_libero_gripper": row.get("baseline_libero_gripper"),
                "compiled_libero_gripper": row.get("compiled_libero_gripper"),
                "raw_gripper_diff": round_float(row.get("raw_gripper_diff"), 6),
            }
    return None


def max_event(rows: list[dict[str, Any]], name: str, key: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    row = max(rows, key=key)
    return {
        "event": name,
        "policy_step": int(row["policy_step"]),
        "continuous_action_max_abs": round_float(continuous_max_abs(row), 6),
        "continuous_action_argmax_dim": continuous_argmax_dim(row),
        "full_action_max_abs": round_float(row.get("action_max_abs_diff"), 6),
        "full_action_argmax_dim": row.get("action_argmax_dim"),
        "pre_pos_l2_mm": mm(row.get("pre_pos_l2")),
        "post_pos_l2_mm": mm(row.get("post_pos_l2")),
        "gripper_bin_mismatch": bool(row.get("gripper_bin_mismatch", False)),
        "baseline_libero_gripper": row.get("baseline_libero_gripper"),
        "compiled_libero_gripper": row.get("compiled_libero_gripper"),
        "raw_gripper_diff": round_float(row.get("raw_gripper_diff"), 6),
    }


def top_online_event(rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not rows:
        return None
    row = rows[0]
    return {
        "policy_step": int(row["policy_step"]),
        "libero_action_max_abs_diff": round_float(row.get("libero_action_max_abs_diff"), 6),
        "libero_action_argmax_dim": row.get("libero_action_argmax_dim"),
        "raw_gripper_diff": round_float(row.get("raw_gripper_diff"), 6),
        "gripper_bin_mismatch": bool(row.get("gripper_bin_mismatch", False)),
        "eager_libero_gripper": row.get("eager_libero_gripper"),
        "compiled_libero_gripper": row.get("compiled_libero_gripper"),
        "eager_margin_to_threshold": round_float(row.get("eager_margin_to_threshold"), 6),
        "compiled_margin_to_threshold": round_float(row.get("compiled_margin_to_threshold"), 6),
    }


def summarize_closed_loop() -> list[dict[str, Any]]:
    data = read_json(GRIPPER_MARGIN_PATH)
    online_by_label = {item["label"]: item for item in data["online_traces"]}
    summaries: list[dict[str, Any]] = []

    for pair in data["closed_loop_pairs"]:
        rows = pair["per_step"]
        online = online_by_label.get(pair["label"])
        first_mismatch_step = pair.get("first_gripper_bin_mismatch_step")

        prefix_before_mismatch = [
            row for row in rows if first_mismatch_step is None or int(row["policy_step"]) < int(first_mismatch_step)
        ]
        prefix_before_mismatch = prefix_before_mismatch or rows

        events = {
            "first_continuous_action_ge_0p005": first_event(
                rows, "first_continuous_action_ge_0p005", lambda row: continuous_max_abs(row) >= 0.005
            ),
            "first_continuous_action_ge_0p01": first_event(
                rows, "first_continuous_action_ge_0p01", lambda row: continuous_max_abs(row) >= 0.01
            ),
            "first_continuous_action_ge_0p05": first_event(
                rows, "first_continuous_action_ge_0p05", lambda row: continuous_max_abs(row) >= 0.05
            ),
            "first_pre_pos_l2_ge_0p5mm": first_event(
                rows, "first_pre_pos_l2_ge_0p5mm", lambda row: float(row.get("pre_pos_l2", 0.0)) >= 0.0005
            ),
            "first_pre_pos_l2_ge_1mm": first_event(
                rows, "first_pre_pos_l2_ge_1mm", lambda row: float(row.get("pre_pos_l2", 0.0)) >= 0.001
            ),
            "first_pre_pos_l2_ge_5mm": first_event(
                rows, "first_pre_pos_l2_ge_5mm", lambda row: float(row.get("pre_pos_l2", 0.0)) >= 0.005
            ),
            "first_pre_pos_l2_ge_10mm": first_event(
                rows, "first_pre_pos_l2_ge_10mm", lambda row: float(row.get("pre_pos_l2", 0.0)) >= 0.010
            ),
            "first_gripper_binary_mismatch": first_event(
                rows, "first_gripper_binary_mismatch", lambda row: bool(row.get("gripper_bin_mismatch", False))
            ),
        }

        summaries.append(
            {
                "label": pair["label"],
                "baseline_success": bool(pair["baseline_success"]),
                "compiled_success": bool(pair["compiled_success"]),
                "baseline_steps": int(pair["baseline_steps"]),
                "compiled_steps": int(pair["compiled_steps"]),
                "common_steps": int(pair["common_steps"]),
                "closed_loop_gripper_mismatch_count": int(pair["gripper_bin_mismatch_count"]),
                "closed_loop_first_gripper_mismatch_step": first_mismatch_step,
                "same_observation": {
                    "success": bool(online["success"]) if online else None,
                    "steps": int(online["steps"]) if online else None,
                    "gripper_mismatch_count": int(online["gripper_bin_mismatch_count"]) if online else None,
                    "first_gripper_mismatch_step": online.get("first_gripper_bin_mismatch_step") if online else None,
                    "top_action_diff": top_online_event(online.get("top_libero_action_diff") if online else None),
                    "top_raw_gripper_diff": top_online_event(online.get("top_raw_gripper_diff") if online else None),
                },
                "events": events,
                "max_pre_pos_l2": max_event(
                    rows, "max_pre_pos_l2", lambda row: float(row.get("pre_pos_l2", 0.0))
                ),
                "max_continuous_action_before_gripper_mismatch": max_event(
                    prefix_before_mismatch,
                    "max_continuous_action_before_gripper_mismatch",
                    continuous_max_abs,
                ),
                "max_pre_pos_l2_before_gripper_mismatch": max_event(
                    prefix_before_mismatch,
                    "max_pre_pos_l2_before_gripper_mismatch",
                    lambda row: float(row.get("pre_pos_l2", 0.0)),
                ),
            }
        )

    return summaries


def summarize_focused_replay() -> dict[str, Any]:
    data = read_json(FOCUSED_REPLAY_PATH)
    selected_targets = {
        "action_head_model",
        "action_head_model_blocks_8_15_eager",
        "action_head_dit_attn_all",
    }
    run_rows = []
    for run in data["runs"]:
        if run["target"] not in selected_targets:
            continue
        top = run["top_step"]
        run_rows.append(
            {
                "target": run["target"],
                "top_policy_step": int(top["policy_step"]),
                "top_dim": top["top_dim"],
                "continuous_max_abs": round_float(top["continuous_max_abs_max"], 6),
                "continuous_l2_p99": round_float(top["continuous_l2_p99"], 6),
                "same_seed_repeat_span_max": round_float(top["same_seed_repeat_span_max"], 6),
                "compiled_p50_ms": seconds_to_ms(top["compiled_p50_seconds"]),
                "eager_p50_ms": seconds_to_ms(top["eager_p50_seconds"]),
                "raw_diff": {k: round_float(v, 6) for k, v in top["raw_diff"].items()},
            }
        )

    action_head_model_profile = [
        {
            "policy_step": int(row["policy_step"]),
            "continuous_max_abs": round_float(row["continuous_max_abs_max"], 6),
            "continuous_l2_p99": round_float(row["continuous_l2_p99"], 6),
            "top_dim": row["top_dim"],
        }
        for row in data["step_rows"]
        if row["target"] == "action_head_model"
    ]

    return {
        "source": str(FOCUSED_REPLAY_PATH.relative_to(ROOT)),
        "runs": run_rows,
        "action_head_model_window_profile": action_head_model_profile,
    }


def summarize_compile_bisect() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in (COMPILE_BISECT_PATH, BLOCK03_BISECT_PATH):
        data = read_json(path)
        for run in data["runs"]:
            target = run["target"]
            if target in seen:
                continue
            seen.add(target)
            top = run["top_step"]
            compiled_ms = seconds_to_ms(top["compiled_p50_seconds"])
            eager_ms = seconds_to_ms(top["eager_p50_seconds"])
            speedup = None
            if compiled_ms and compiled_ms > 0 and eager_ms:
                speedup = round(eager_ms / compiled_ms, 2)
            rows.append(
                {
                    "target": target,
                    "top_policy_step": int(top["policy_step"]),
                    "continuous_max_abs": round_float(top["continuous_max_abs_max"], 6),
                    "continuous_l2_p99": round_float(top["continuous_l2_p99"], 6),
                    "same_seed_repeat_span_max": round_float(top["same_seed_repeat_span_max"], 6),
                    "top_dim": top["top_dim"],
                    "compiled_p50_ms": compiled_ms,
                    "eager_p50_ms": eager_ms,
                    "eager_over_compiled_speedup": speedup,
                    "spike_eliminated_at_0p005": float(top["continuous_max_abs_max"]) < 0.005,
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
    order_idx = {target: idx for idx, target in enumerate(order)}
    return sorted(rows, key=lambda row: order_idx.get(row["target"], 999))


def event_step(summary: dict[str, Any], event_name: str) -> str:
    event = summary["events"].get(event_name)
    if event is None:
        return "-"
    return str(event["policy_step"])


def event_step_with_mm(summary: dict[str, Any], event_name: str) -> str:
    event = summary["events"].get(event_name)
    if event is None:
        return "-"
    return f'{event["policy_step"]} ({fmt(event["pre_pos_l2_mm"], 3)} mm)'


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def build_report(artifact: dict[str, Any]) -> str:
    closed_loop = artifact["closed_loop_branching"]
    focused = artifact["focused_step253_replay"]
    bisect = artifact["compile_boundary_bisect"]

    closed_loop_table = markdown_table(
        [
            "case",
            "FP16",
            "compiled",
            "same-obs gripper flips",
            "first cont >=0.005",
            "first pose >=0.5mm",
            "first pose >=1mm",
            "first gripper phase split",
            "max pose split",
        ],
        [
            [
                item["label"],
                "success" if item["baseline_success"] else "fail",
                "success" if item["compiled_success"] else "fail",
                item["same_observation"]["gripper_mismatch_count"],
                event_step(item, "first_continuous_action_ge_0p005"),
                event_step_with_mm(item, "first_pre_pos_l2_ge_0p5mm"),
                event_step_with_mm(item, "first_pre_pos_l2_ge_1mm"),
                event_step_with_mm(item, "first_gripper_binary_mismatch"),
                f'{fmt(item["max_pre_pos_l2"]["pre_pos_l2_mm"], 1)} mm @ {item["max_pre_pos_l2"]["policy_step"]}',
            ]
            for item in closed_loop
        ],
    )

    same_obs_table = markdown_table(
        [
            "case",
            "same-obs success",
            "max action diff",
            "argmax dim",
            "gripper flips",
            "max raw gripper diff",
            "compiled margin at max gripper diff",
        ],
        [
            [
                item["label"],
                item["same_observation"]["success"],
                fmt(item["same_observation"]["top_action_diff"]["libero_action_max_abs_diff"], 6),
                item["same_observation"]["top_action_diff"]["libero_action_argmax_dim"],
                item["same_observation"]["gripper_mismatch_count"],
                fmt(item["same_observation"]["top_raw_gripper_diff"]["raw_gripper_diff"], 6),
                fmt(item["same_observation"]["top_raw_gripper_diff"]["compiled_margin_to_threshold"], 4),
            ]
            for item in closed_loop
        ],
    )

    focused_table = markdown_table(
        [
            "target",
            "top step",
            "dim",
            "cont max",
            "L2 p99",
            "repeat span",
            "compiled p50",
            "eager p50",
        ],
        [
            [
                row["target"],
                row["top_policy_step"],
                row["top_dim"],
                fmt(row["continuous_max_abs"], 6),
                fmt(row["continuous_l2_p99"], 6),
                fmt(row["same_seed_repeat_span_max"], 6),
                f'{fmt(row["compiled_p50_ms"], 1)} ms',
                f'{fmt(row["eager_p50_ms"], 1)} ms',
            ]
            for row in focused["runs"]
        ],
    )

    profile_table = markdown_table(
        ["step", "cont max", "L2 p99", "dim"],
        [
            [
                row["policy_step"],
                fmt(row["continuous_max_abs"], 6),
                fmt(row["continuous_l2_p99"], 6),
                row["top_dim"],
            ]
            for row in focused["action_head_model_window_profile"]
        ],
    )

    selected_targets = {
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
        "action_head_dit_attn_all",
    }
    bisect_table = markdown_table(
        [
            "target",
            "cont max",
            "L2 p99",
            "repeat span",
            "compiled p50",
            "speedup",
            "spike removed",
        ],
        [
            [
                row["target"],
                fmt(row["continuous_max_abs"], 6),
                fmt(row["continuous_l2_p99"], 6),
                fmt(row["same_seed_repeat_span_max"], 6),
                f'{fmt(row["compiled_p50_ms"], 1)} ms',
                f'{fmt(row["eager_over_compiled_speedup"], 2)}x',
                row["spike_eliminated_at_0p005"],
            ]
            for row in bisect
            if row["target"] in selected_targets
        ],
    )

    task4, task6 = closed_loop
    lines = [
        "# Phase 17: first-divergence / trajectory branching 分析",
        "",
        "## 结论摘要",
        "",
        "这轮分析不重新跑推理，只复用已经落盘的 trace。核心结论是：当前看到的失败不是简单的同一 observation 上夹爪阈值抖动，而是闭环轨迹被早期连续动作误差逐步推到不同状态区域，随后在接触或夹爪相位附近发生分叉。",
        "",
        "最重要的证据有四条：",
        "",
        f"- `task4:init9` 在闭环里第 {event_step(task4, 'first_gripper_binary_mismatch')} 步出现夹爪相位分叉，最大末端位置分离达到 {fmt(task4['max_pre_pos_l2']['pre_pos_l2_mm'], 1)} mm；但 same-observation replay 的夹爪二值 mismatch 是 0。",
        f"- `task6:init8` 的 same-observation replay 在第 253 步有连续动作 outlier，最大差异 {fmt(focused['runs'][0]['continuous_max_abs'], 6)}，但夹爪仍不翻转；闭环中真正的夹爪相位分叉到第 {event_step(task6, 'first_gripper_binary_mismatch')} 步才出现。",
        "- Step253 outlier 的 fixed-seed repeat span 是 0，说明它是确定性的数值路径差异，不是 runtime 随机噪声。",
        "- compile boundary bisect 把该 outlier 定位到 DiT block0 的 compiled path：只让 block0 eager 后，连续动作最大差异从约 0.055 降到约 0.0018。",
        "",
        "## 分析定义",
        "",
        "为了避免把所有误差混在一起，这里把分叉过程拆成三类事件：",
        "",
        "- first action divergence：第一次连续动作差异超过阈值，例如 `max(|delta x,y,z,roll,pitch,yaw|) >= 0.005`。",
        "- first pose divergence：第一次末端执行器位置差异超过阈值，例如 `0.5mm / 1mm / 5mm / 10mm`。",
        "- first phase split：第一次离散控制相位不同，目前主要看夹爪二值动作是否不同。",
        "",
        "这三个事件的顺序很关键。若同一 observation 已经夹爪翻转，问题更像阈值 jitter；若同一 observation 不翻转、闭环后才翻转，问题更像轨迹 branching。",
        "",
        "## 闭环分叉时间线",
        "",
        closed_loop_table,
        "",
        "读法：",
        "",
        "- `task4:init9` 是较快分叉：第 8 步已有连续动作差异超过 0.005，第 46 步夹爪相位分裂，随后轨迹分离放大到厘米级。",
        "- `task6:init8` 是长潜伏分叉：第 63 步末端位置刚超过 1mm，但第 244 步才出现夹爪相位分裂，第 391 步才超过 5mm。这说明早期误差未必马上失败，但会改变之后进入接触阶段时的状态分布。",
        "",
        "## same-observation 对照",
        "",
        same_obs_table,
        "",
        "same-observation 对照说明：在固定 observation 上，compiled 和 eager 的夹爪二值输出没有发生翻转。尤其 `task6:init8` 的第 253 步连续动作差异很大，但夹爪 margin 仍离阈值很远。因此闭环失败不能简单归因于 `gripper > 0.5` 这种阈值附近的瞬时抖动。",
        "",
        "更合理的解释是：连续动作误差先改变下一帧 observation，新的 observation 再改变后续去噪轨迹和接触几何，最终在夹爪开合、物体接触或恢复路径附近出现相位分叉。",
        "",
        "## Step253 局部 outlier",
        "",
        focused_table,
        "",
        "`action_head_model` 在 248-258 窗口内的连续动作差异如下：",
        "",
        profile_table,
        "",
        "这张表说明 step253 不是慢性累积误差，而是局部 observation 触发的确定性尖峰。附近大部分 step 在 0.001 到 0.004 量级，只有 step253 跳到 0.0549。",
        "",
        "## compile boundary 定位",
        "",
        bisect_table,
        "",
        "定位结果很直接：",
        "",
        "- 让 block0-15、block0-7、block0-3 eager 都能消掉尖峰。",
        "- 只让 block0 eager 也能消掉尖峰。",
        "- 只让 block1、block2、block3、block4-7 或 block8-15 eager，尖峰仍在。",
        "- 所以这个 step253 outlier 的最小已知责任范围是 `DiT block0 compiled path`。",
        "",
        "## 对闭环误差修正的含义",
        "",
        "这组数据把问题从“平均误差大不大”推进到“哪些扰动会触发轨迹分叉”。后续闭环修正应该优先围绕分叉事件做，而不是继续只优化全局 MSE：",
        "",
        "- 在线监控：记录 `first action divergence`、`first pose divergence` 和 `phase split`，把它们作为闭环风险指标。",
        "- 局部 fallback：对已知高风险编译边界，例如 block0 compiled path，做 eager island 或风险触发 fallback。",
        "- 轨迹级校准：校准目标不只看同一 observation 的动作误差，还要看误差是否会把 rollout 推入不同接触区域。",
        "- 反事实 replay：固定某一步 observation，对不同 compile/quant path 做 repeated same-seed replay，区分随机噪声和确定性 outlier。",
        "",
        "## 限制",
        "",
        "当前结论来自已有 trace，不是新 benchmark。闭环 paired case 只有 `task4:init9` 和 `task6:init8` 两个失败病例，所以它适合解释机制和指导下一步工程验证，还不能当成全任务统计结论。真正要形成强 claim，需要继续在更多 task/init 上自动挖 `first divergence -> phase split -> outcome flip` 的链路。",
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
            str(GRIPPER_MARGIN_PATH.relative_to(ROOT)),
            str(FOCUSED_REPLAY_PATH.relative_to(ROOT)),
            str(COMPILE_BISECT_PATH.relative_to(ROOT)),
            str(BLOCK03_BISECT_PATH.relative_to(ROOT)),
        ],
        "closed_loop_branching": summarize_closed_loop(),
        "focused_step253_replay": summarize_focused_replay(),
        "compile_boundary_bisect": summarize_compile_bisect(),
    }
    write_json(OUT_JSON, artifact)
    write_text(OUT_MD, build_report(artifact))
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
