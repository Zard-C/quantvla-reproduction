"""Analyze gripper margin and binarization divergence in LIBERO traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec_sub(left: list[Any], right: list[Any]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def l2(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def max_abs(values: list[float]) -> float:
    return max(abs(float(value)) for value in values) if values else 0.0


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def raw_gripper(action: dict[str, Any]) -> float:
    return float(action["action.gripper"])


def libero_gripper(action: list[Any]) -> float:
    return float(action[6])


def raw_margin(value: float, threshold: float) -> float:
    return abs(float(value) - float(threshold))


def parse_label_path(value: str) -> tuple[str, Path]:
    label, path = value.split(":", 1)
    return label, Path(path)


def parse_pair(value: str) -> tuple[str, Path, Path]:
    label, baseline, compiled = value.split(":", 2)
    return label, Path(baseline), Path(compiled)


def window(rows: list[dict[str, Any]], center_policy_step: int | None, radius: int) -> list[dict[str, Any]]:
    if center_policy_step is None:
        return []
    return [
        row
        for row in rows
        if abs(int(row["policy_step"]) - int(center_policy_step)) <= int(radius)
    ]


def switch_steps(rows: list[dict[str, Any]], key: str) -> list[int]:
    out: list[int] = []
    previous: float | None = None
    for row in rows:
        current = float(row[key])
        if previous is not None and current != previous:
            out.append(int(row["policy_step"]))
        previous = current
    return out


def first_threshold(rows: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any] | None:
    for row in rows:
        if float(row[key]) >= float(threshold):
            return row
    return None


def analyze_online_trace(label: str, path: Path, *, threshold: float, radius: int) -> dict[str, Any]:
    trace = read_json(path)
    rows: list[dict[str, Any]] = []
    for step in trace["steps"]:
        eager_raw = raw_gripper(step["eager_raw_action"])
        compiled_raw = raw_gripper(step["compiled_raw_action"])
        eager_bin = libero_gripper(step["eager_libero_action"])
        compiled_bin = libero_gripper(step["compiled_libero_action"])
        libero_diff = vec_sub(step["compiled_libero_action"], step["eager_libero_action"])
        rows.append(
            {
                "policy_step": int(step["policy_step"]),
                "env_step": int(step["step"]),
                "eager_raw_gripper": eager_raw,
                "compiled_raw_gripper": compiled_raw,
                "raw_gripper_diff": compiled_raw - eager_raw,
                "eager_margin_to_threshold": raw_margin(eager_raw, threshold),
                "compiled_margin_to_threshold": raw_margin(compiled_raw, threshold),
                "eager_libero_gripper": eager_bin,
                "compiled_libero_gripper": compiled_bin,
                "gripper_bin_mismatch": bool(eager_bin != compiled_bin),
                "libero_action_max_abs_diff": max_abs(libero_diff),
                "libero_action_diff": libero_diff,
                "libero_action_argmax_dim": ACTION_KEYS[
                    max(range(len(libero_diff)), key=lambda idx: abs(libero_diff[idx]))
                ],
                "raw_relative_rmse": float(step["raw_action_metrics"]["relative_rmse"]),
                "raw_max_abs_diff": float(step["raw_action_metrics"]["max_abs_diff"]),
            }
        )

    mismatches = [row for row in rows if row["gripper_bin_mismatch"]]
    first_mismatch_step = int(mismatches[0]["policy_step"]) if mismatches else None
    eager_switches = switch_steps(rows, "eager_libero_gripper")
    compiled_switches = switch_steps(rows, "compiled_libero_gripper")
    return {
        "kind": "same_observation",
        "label": label,
        "path": str(path),
        "success": bool(trace.get("success", False)),
        "steps": len(rows),
        "raw_threshold": float(threshold),
        "gripper_bin_mismatch_count": len(mismatches),
        "first_gripper_bin_mismatch_step": first_mismatch_step,
        "eager_gripper_switch_steps": eager_switches,
        "compiled_gripper_switch_steps": compiled_switches,
        "raw_gripper_abs_diff": summarize([abs(float(row["raw_gripper_diff"])) for row in rows]),
        "eager_margin_to_threshold": summarize([float(row["eager_margin_to_threshold"]) for row in rows]),
        "compiled_margin_to_threshold": summarize([float(row["compiled_margin_to_threshold"]) for row in rows]),
        "libero_action_max_abs_diff": summarize([float(row["libero_action_max_abs_diff"]) for row in rows]),
        "top_raw_gripper_diff": sorted(rows, key=lambda row: abs(float(row["raw_gripper_diff"])), reverse=True)[:10],
        "top_libero_action_diff": sorted(rows, key=lambda row: float(row["libero_action_max_abs_diff"]), reverse=True)[:10],
        "first_mismatch_window": window(rows, first_mismatch_step, radius),
    }


def analyze_closed_loop_pair(
    label: str,
    baseline_path: Path,
    compiled_path: Path,
    *,
    threshold: float,
    radius: int,
) -> dict[str, Any]:
    baseline = read_json(baseline_path)
    compiled = read_json(compiled_path)
    baseline_steps = {int(step["policy_step"]): step for step in baseline["steps"]}
    compiled_steps = {int(step["policy_step"]): step for step in compiled["steps"]}
    common_steps = sorted(set(baseline_steps) & set(compiled_steps))
    rows: list[dict[str, Any]] = []
    for policy_step in common_steps:
        eager_step = baseline_steps[policy_step]
        compiled_step = compiled_steps[policy_step]
        baseline_raw = raw_gripper(eager_step["raw_action"])
        compiled_raw = raw_gripper(compiled_step["raw_action"])
        baseline_bin = libero_gripper(eager_step["libero_action"])
        compiled_bin = libero_gripper(compiled_step["libero_action"])
        action_diff = vec_sub(compiled_step["libero_action"], eager_step["libero_action"])
        pre_pos_diff = vec_sub(
            compiled_step["pre_robot0_eef_pos"],
            eager_step["pre_robot0_eef_pos"],
        )
        post_pos_diff = vec_sub(
            compiled_step["post_robot0_eef_pos"],
            eager_step["post_robot0_eef_pos"],
        )
        rows.append(
            {
                "policy_step": int(policy_step),
                "baseline_raw_gripper": baseline_raw,
                "compiled_raw_gripper": compiled_raw,
                "raw_gripper_diff": compiled_raw - baseline_raw,
                "baseline_margin_to_threshold": raw_margin(baseline_raw, threshold),
                "compiled_margin_to_threshold": raw_margin(compiled_raw, threshold),
                "baseline_libero_gripper": baseline_bin,
                "compiled_libero_gripper": compiled_bin,
                "gripper_bin_mismatch": bool(baseline_bin != compiled_bin),
                "action_diff": action_diff,
                "action_max_abs_diff": max_abs(action_diff),
                "action_l2_diff": l2(action_diff),
                "action_argmax_dim": ACTION_KEYS[
                    max(range(len(action_diff)), key=lambda idx: abs(action_diff[idx]))
                ],
                "pre_pos_diff": pre_pos_diff,
                "post_pos_diff": post_pos_diff,
                "pre_pos_l2": l2(pre_pos_diff),
                "post_pos_l2": l2(post_pos_diff),
            }
        )

    mismatches = [row for row in rows if row["gripper_bin_mismatch"]]
    first_mismatch_step = int(mismatches[0]["policy_step"]) if mismatches else None
    baseline_switches = switch_steps(rows, "baseline_libero_gripper")
    compiled_switches = switch_steps(rows, "compiled_libero_gripper")
    threshold_hits = {
        "first_pre_pos_l2_ge_1mm": first_threshold(rows, "pre_pos_l2", 0.001),
        "first_pre_pos_l2_ge_5mm": first_threshold(rows, "pre_pos_l2", 0.005),
        "first_pre_pos_l2_ge_10mm": first_threshold(rows, "pre_pos_l2", 0.01),
        "first_action_max_abs_ge_0p01": first_threshold(rows, "action_max_abs_diff", 0.01),
        "first_action_max_abs_ge_0p05": first_threshold(rows, "action_max_abs_diff", 0.05),
    }
    return {
        "kind": "closed_loop_pair",
        "label": label,
        "baseline_path": str(baseline_path),
        "compiled_path": str(compiled_path),
        "baseline_success": bool(baseline.get("success", False)),
        "compiled_success": bool(compiled.get("success", False)),
        "baseline_steps": int(baseline.get("num_steps", len(baseline["steps"]))),
        "compiled_steps": int(compiled.get("num_steps", len(compiled["steps"]))),
        "common_steps": len(rows),
        "raw_threshold": float(threshold),
        "gripper_bin_mismatch_count": len(mismatches),
        "first_gripper_bin_mismatch_step": first_mismatch_step,
        "baseline_gripper_switch_steps": baseline_switches,
        "compiled_gripper_switch_steps": compiled_switches,
        "raw_gripper_abs_diff": summarize([abs(float(row["raw_gripper_diff"])) for row in rows]),
        "baseline_margin_to_threshold": summarize([float(row["baseline_margin_to_threshold"]) for row in rows]),
        "compiled_margin_to_threshold": summarize([float(row["compiled_margin_to_threshold"]) for row in rows]),
        "pre_pos_l2": summarize([float(row["pre_pos_l2"]) for row in rows]),
        "action_max_abs_diff": summarize([float(row["action_max_abs_diff"]) for row in rows]),
        "threshold_hits": threshold_hits,
        "top_raw_gripper_diff": sorted(rows, key=lambda row: abs(float(row["raw_gripper_diff"])), reverse=True)[:10],
        "top_action_diff": sorted(rows, key=lambda row: float(row["action_max_abs_diff"]), reverse=True)[:10],
        "first_mismatch_window": window(rows, first_mismatch_step, radius),
        "per_step": rows,
    }


def fmt_step(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return "none"
    return f"{row['policy_step']} ({key}={float(row[key]):.6g})"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 14 Gripper Margin Analysis",
        "",
        f"Raw gripper threshold assumption: `{result['raw_threshold']}`.",
        "",
        "## Online Same-Observation Traces",
        "",
        "| case | success | steps | bin mismatch count | first mismatch | eager switches | compiled switches | max raw gripper diff | min compiled margin |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for item in result["online_traces"]:
        lines.append(
            "| {label} | {success} | {steps} | {mismatch} | {first} | `{eager}` | `{compiled}` | {rawmax:.6g} | {margin:.6g} |".format(
                label=item["label"],
                success=str(item["success"]),
                steps=item["steps"],
                mismatch=item["gripper_bin_mismatch_count"],
                first=item["first_gripper_bin_mismatch_step"],
                eager=item["eager_gripper_switch_steps"],
                compiled=item["compiled_gripper_switch_steps"],
                rawmax=item["raw_gripper_abs_diff"]["max"],
                margin=item["compiled_margin_to_threshold"]["min"],
            )
        )
    lines.extend(
        [
            "",
            "## Closed-Loop Pairs",
            "",
            "| case | baseline | compiled | common steps | bin mismatch count | first mismatch | baseline switches | compiled switches | max pre-pos L2 | max action diff |",
            "|---|---:|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )
    for item in result["closed_loop_pairs"]:
        lines.append(
            "| {label} | {base} | {compiled} | {common} | {mismatch} | {first} | `{bs}` | `{cs}` | {premax:.6g} | {actmax:.6g} |".format(
                label=item["label"],
                base=str(item["baseline_success"]),
                compiled=str(item["compiled_success"]),
                common=item["common_steps"],
                mismatch=item["gripper_bin_mismatch_count"],
                first=item["first_gripper_bin_mismatch_step"],
                bs=item["baseline_gripper_switch_steps"],
                cs=item["compiled_gripper_switch_steps"],
                premax=item["pre_pos_l2"]["max"],
                actmax=item["action_max_abs_diff"]["max"],
            )
        )
    lines.extend(["", "## First Threshold Hits", ""])
    for item in result["closed_loop_pairs"]:
        hits = item["threshold_hits"]
        lines.extend(
            [
                f"### {item['label']}",
                "",
                f"- first gripper mismatch: `{item['first_gripper_bin_mismatch_step']}`",
                f"- first pre-position L2 >= 1mm: `{fmt_step(hits['first_pre_pos_l2_ge_1mm'], 'pre_pos_l2')}`",
                f"- first pre-position L2 >= 5mm: `{fmt_step(hits['first_pre_pos_l2_ge_5mm'], 'pre_pos_l2')}`",
                f"- first action max diff >= 0.01: `{fmt_step(hits['first_action_max_abs_ge_0p01'], 'action_max_abs_diff')}`",
                f"- first action max diff >= 0.05: `{fmt_step(hits['first_action_max_abs_ge_0p05'], 'action_max_abs_diff')}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-trace", action="append", default=[], help="label:path")
    parser.add_argument("--closed-loop-pair", action="append", default=[], help="label:baseline_path:compiled_path")
    parser.add_argument("--raw-threshold", type=float, default=0.5)
    parser.add_argument("--window-radius", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "raw_threshold": float(args.raw_threshold),
        "online_traces": [
            analyze_online_trace(label, path, threshold=args.raw_threshold, radius=args.window_radius)
            for label, path in (parse_label_path(value) for value in args.online_trace)
        ],
        "closed_loop_pairs": [
            analyze_closed_loop_pair(label, baseline, compiled, threshold=args.raw_threshold, radius=args.window_radius)
            for label, baseline, compiled in (parse_pair(value) for value in args.closed_loop_pair)
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)


if __name__ == "__main__":
    main()
