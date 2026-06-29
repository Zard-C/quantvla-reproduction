"""First-divergence analysis for Phase 26 layer/boundary interventions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

OUT_JSON = RESULTS / "phase27_phase26_first_divergence.json"
OUT_MD = DOCS / "phase27_phase26_first_divergence_report_zh.md"

DIMS = ["x", "y", "z", "roll", "pitch", "yaw"]
ACTION_KEYS = [f"action.{key}" for key in DIMS] + ["action.gripper"]

RUNS = {
    "baseline": {
        "label": "FP16 baseline",
        "trace_dir": "phase26_layer_claimC_2case_action_head_model_baseline_trace",
    },
    "full": {
        "label": "compile action_head_model",
        "trace_dir": "phase26_layer_claimC_2case_action_head_model_compiled_trace",
    },
    "block0": {
        "label": "compile action_head_model + block0 eager",
        "trace_dir": "phase26_layer_claimC_2case_blocks_0_0_eager_compiled_trace",
    },
    "block1": {
        "label": "compile action_head_model + block1 eager",
        "trace_dir": "phase26_layer_claimC_2case_blocks_1_1_eager_compiled_trace",
    },
}

CASES = ["4:9", "6:8"]
PAIRS = [
    ("baseline", "full"),
    ("baseline", "block0"),
    ("baseline", "block1"),
    ("full", "block0"),
    ("full", "block1"),
    ("block0", "block1"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def l2(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def quaternion_angle_rad(a: list[float], b: list[float]) -> float:
    norm_a = l2(a)
    norm_b = l2(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)) / (norm_a * norm_b))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def round_float(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def mm(value_m: float) -> float:
    return round(float(value_m) * 1000.0, 3)


def load_trace(run: str, case: str) -> tuple[dict[str, Any], Path]:
    task_id, init_index = (int(part) for part in case.split(":"))
    trace_dir = RESULTS / RUNS[run]["trace_dir"]
    path = next(trace_dir.glob(f"task={task_id:02d}--init={init_index:02d}*.json"))
    return read_json(path), path


def step_map(trace: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(step["policy_step"]): step for step in trace["steps"]}


def compare_steps(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_steps = step_map(left)
    right_steps = step_map(right)
    common_steps = sorted(set(left_steps) & set(right_steps))
    rows: list[dict[str, Any]] = []
    for policy_step in common_steps:
        left_step = left_steps[policy_step]
        right_step = right_steps[policy_step]
        action_delta = vector_sub(right_step["libero_action"], left_step["libero_action"])
        raw_delta = [
            float(right_step["raw_action"][key]) - float(left_step["raw_action"][key])
            for key in ACTION_KEYS
        ]
        continuous_abs = [abs(value) for value in action_delta[:6]]
        raw_continuous_abs = [abs(value) for value in raw_delta[:6]]
        argmax_idx = max(range(6), key=lambda idx: continuous_abs[idx])
        raw_argmax_idx = max(range(6), key=lambda idx: raw_continuous_abs[idx])
        rows.append(
            {
                "policy_step": int(policy_step),
                "pre_pos_l2_m": l2(vector_sub(right_step["pre_robot0_eef_pos"], left_step["pre_robot0_eef_pos"])),
                "post_pos_l2_m": l2(vector_sub(right_step["post_robot0_eef_pos"], left_step["post_robot0_eef_pos"])),
                "pre_quat_angle_rad": quaternion_angle_rad(
                    left_step["pre_robot0_eef_quat"],
                    right_step["pre_robot0_eef_quat"],
                ),
                "pre_gripper_qpos_l2": l2(
                    vector_sub(right_step["pre_robot0_gripper_qpos"], left_step["pre_robot0_gripper_qpos"])
                ),
                "continuous_action_l2": l2(action_delta[:6]),
                "continuous_action_max_abs": max(continuous_abs),
                "continuous_action_argmax_dim": DIMS[argmax_idx],
                "raw_continuous_action_l2": l2(raw_delta[:6]),
                "raw_continuous_action_max_abs": max(raw_continuous_abs),
                "raw_continuous_action_argmax_dim": DIMS[raw_argmax_idx],
                "raw_gripper_diff": raw_delta[6],
                "gripper_bin_mismatch": bool(float(left_step["libero_action"][6]) != float(right_step["libero_action"][6])),
                "left_gripper": float(left_step["libero_action"][6]),
                "right_gripper": float(right_step["libero_action"][6]),
            }
        )
    return rows


def first_event(
    rows: list[dict[str, Any]],
    name: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    for row in rows:
        if predicate(row):
            return event_payload(name, row)
    return None


def max_event(
    rows: list[dict[str, Any]],
    name: str,
    key: Callable[[dict[str, Any]], float],
) -> dict[str, Any] | None:
    if not rows:
        return None
    return event_payload(name, max(rows, key=key))


def event_payload(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": name,
        "policy_step": int(row["policy_step"]),
        "pre_pos_l2_mm": mm(row["pre_pos_l2_m"]),
        "post_pos_l2_mm": mm(row["post_pos_l2_m"]),
        "pre_quat_angle_deg": round_float(math.degrees(row["pre_quat_angle_rad"]), 4),
        "continuous_action_max_abs": round_float(row["continuous_action_max_abs"], 6),
        "continuous_action_argmax_dim": row["continuous_action_argmax_dim"],
        "continuous_action_l2": round_float(row["continuous_action_l2"], 6),
        "raw_gripper_diff": round_float(row["raw_gripper_diff"], 6),
        "gripper_bin_mismatch": bool(row["gripper_bin_mismatch"]),
        "left_gripper": row["left_gripper"],
        "right_gripper": row["right_gripper"],
    }


def summarize_pair(case: str, left_name: str, right_name: str) -> dict[str, Any]:
    left_trace, left_path = load_trace(left_name, case)
    right_trace, right_path = load_trace(right_name, case)
    rows = compare_steps(left_trace, right_trace)
    thresholds = {
        "first_action_max_ge_0p005": first_event(
            rows, "first_action_max_ge_0p005", lambda row: row["continuous_action_max_abs"] >= 0.005
        ),
        "first_action_max_ge_0p01": first_event(
            rows, "first_action_max_ge_0p01", lambda row: row["continuous_action_max_abs"] >= 0.010
        ),
        "first_action_max_ge_0p05": first_event(
            rows, "first_action_max_ge_0p05", lambda row: row["continuous_action_max_abs"] >= 0.050
        ),
        "first_pre_pos_ge_0p5mm": first_event(
            rows, "first_pre_pos_ge_0p5mm", lambda row: row["pre_pos_l2_m"] >= 0.0005
        ),
        "first_pre_pos_ge_1mm": first_event(
            rows, "first_pre_pos_ge_1mm", lambda row: row["pre_pos_l2_m"] >= 0.001
        ),
        "first_pre_pos_ge_5mm": first_event(
            rows, "first_pre_pos_ge_5mm", lambda row: row["pre_pos_l2_m"] >= 0.005
        ),
        "first_pre_pos_ge_10mm": first_event(
            rows, "first_pre_pos_ge_10mm", lambda row: row["pre_pos_l2_m"] >= 0.010
        ),
        "first_gripper_bin_mismatch": first_event(
            rows, "first_gripper_bin_mismatch", lambda row: row["gripper_bin_mismatch"]
        ),
    }
    return {
        "case": case,
        "left": left_name,
        "right": right_name,
        "left_label": RUNS[left_name]["label"],
        "right_label": RUNS[right_name]["label"],
        "left_source": str(left_path.relative_to(ROOT)),
        "right_source": str(right_path.relative_to(ROOT)),
        "left_success": bool(left_trace["success"]),
        "right_success": bool(right_trace["success"]),
        "left_steps": int(left_trace["num_steps"]),
        "right_steps": int(right_trace["num_steps"]),
        "common_steps": len(rows),
        "events": thresholds,
        "max_pre_pos_l2": max_event(rows, "max_pre_pos_l2", lambda row: row["pre_pos_l2_m"]),
        "max_action": max_event(rows, "max_action", lambda row: row["continuous_action_max_abs"]),
    }


def build_artifact() -> dict[str, Any]:
    comparisons = []
    for case in CASES:
        for left, right in PAIRS:
            comparisons.append(summarize_pair(case, left, right))
    return {
        "phase": "phase27_phase26_first_divergence",
        "source_phase": "phase26_layer_claimC",
        "cases": CASES,
        "runs": RUNS,
        "comparisons": comparisons,
    }


def format_step(event: dict[str, Any] | None, field: str = "policy_step") -> str:
    if event is None:
        return "-"
    return str(event[field])


def format_event_short(event: dict[str, Any] | None) -> str:
    if event is None:
        return "-"
    return (
        f"s{event['policy_step']} / {event['pre_pos_l2_mm']}mm / "
        f"{event['continuous_action_max_abs']} {event['continuous_action_argmax_dim']}"
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def selected_comparison(data: dict[str, Any], case: str, left: str, right: str) -> dict[str, Any]:
    for item in data["comparisons"]:
        if item["case"] == case and item["left"] == left and item["right"] == right:
            return item
    raise KeyError((case, left, right))


def build_report(data: dict[str, Any]) -> str:
    rows = []
    for item in data["comparisons"]:
        if item["left"] not in {"baseline", "full"}:
            continue
        events = item["events"]
        rows.append(
            [
                item["case"],
                f"{item['left']} -> {item['right']}",
                f"{item['left_success']}/{item['right_success']}",
                f"{item['left_steps']}/{item['right_steps']}",
                format_step(events["first_action_max_ge_0p01"]),
                format_event_short(events["first_pre_pos_ge_1mm"]),
                format_event_short(events["first_pre_pos_ge_10mm"]),
                format_event_short(events["first_gripper_bin_mismatch"]),
                format_event_short(item["max_pre_pos_l2"]),
            ]
        )

    task4_full_block0 = selected_comparison(data, "4:9", "full", "block0")
    task4_full_block1 = selected_comparison(data, "4:9", "full", "block1")
    task6_full_block0 = selected_comparison(data, "6:8", "full", "block0")
    task6_full_block1 = selected_comparison(data, "6:8", "full", "block1")

    direct_rows = []
    for item in [task4_full_block0, task4_full_block1, task6_full_block0, task6_full_block1]:
        events = item["events"]
        direct_rows.append(
            [
                item["case"],
                f"{item['left']} -> {item['right']}",
                f"{item['left_success']}/{item['right_success']}",
                f"{item['left_steps']}/{item['right_steps']}",
                format_step(events["first_action_max_ge_0p01"]),
                format_event_short(events["first_pre_pos_ge_1mm"]),
                format_event_short(events["first_gripper_bin_mismatch"]),
                format_event_short(item["max_action"]),
            ]
        )

    lines = [
        "# Phase 27: Phase26 First-Divergence Analysis",
        "",
        "本阶段离线读取 Phase26 的 closed-loop traces，按 `policy_step` 对齐不同 layer/boundary variants，寻找最早的 action/state/gripper divergence。",
        "",
        "读表说明：`s17 / 0.505mm / 0.00137 z` 表示 policy step 17，EEF pre-position 差为 0.505mm，该 step 的连续 action 最大差为 0.00137，最大差维度为 z。",
        "",
        "注意：表里的 `steps L/R` 使用 trace 中已写入的 step 数。失败 rollout 在最后一次 server action 请求之后可能因 `executing action in terminated episode` 结束，因此会比 Phase26 latency JSON 的 request count 少 1。",
        "",
        "## Baseline/Full 对各 Eager-Island Variant",
        "",
        md_table(
            [
                "case",
                "pair",
                "success L/R",
                "trace steps L/R",
                "first action>0.01",
                "first pos>1mm",
                "first pos>10mm",
                "first gripper mismatch",
                "max pos",
            ],
            rows,
        ),
        "",
        "## Direct Comparison: Full Compile vs Eager-Island Variants",
        "",
        md_table(
            [
                "case",
                "pair",
                "success L/R",
                "trace steps L/R",
                "first action>0.01",
                "first pos>1mm",
                "first gripper mismatch",
                "max action",
            ],
            direct_rows,
        ),
        "",
        "## 关键解读",
        "",
        "1. 早期 action 分歧很早出现，但状态分叉滞后。`4:9` 中 baseline vs block0 在 step 10-12 就有 `y` 方向 action 差超过 0.005/0.01，但 EEF 位置到 step 44 才超过 1mm，到 step 60 才超过 10mm。",
        "2. full compile 与 block0/block1 的差异也不是立刻造成失败。`4:9` 中 full vs block0 的位置差到 step 59 才超过 1mm，gripper mismatch 到 step 99 才出现；full vs block1 的 gripper mismatch 在 step 97。",
        "3. gripper mismatch 不是充分条件。`4:9` baseline vs full 在 step 46 就有 gripper mismatch，但两者都成功；`6:8` baseline vs full 在 step 262 有 mismatch，也都成功。这说明 gripper phase 是重要分叉信号，但是否 flip 还取决于当时状态 margin。",
        "4. `block0 eager` 的行为是 case-dependent：它让 `4:9` 失败，却让 `6:8` 从 full compile 的 916 步缩短到 404 步成功。这符合 Phase25 的结论：敏感性同时依赖 task、phase、action direction 和 margin。",
        "5. action channel 上，早期 first-action divergence 多出现在 `y`，中后期和最大差常转向 `x/z`。这与 Phase25 的 action-channel anisotropy 能接上：不同阶段的敏感方向并不固定。",
        "",
        "## 对工程路线的含义",
        "",
        "这轮分析说明，不能把 layer 保护策略写成简单规则，比如“发现 block0 spike 就把 block0 留 eager”。局部 spike removal 可能修复某个 same-observation outlier，但闭环里它也可能改变早期 action 方向，经过几十步放大后在接触阶段跨过成功/失败边界。",
        "",
        "更合适的路线是：先用 first-divergence 找出 outcome flip 前的早期 action/state signature，再把它和 Phase25 的 action-channel/phase sensitivity map 结合，形成 sensitivity-guided layer policy。",
        "",
        "## 产物",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Report: `{OUT_MD.relative_to(ROOT)}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build_artifact()
    write_json(OUT_JSON, data)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
