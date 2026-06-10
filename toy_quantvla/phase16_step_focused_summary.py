"""Aggregate Phase 16.5 focused replay outputs across compile targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    result["_result_json"] = str(path)
    return result


def flatten_step_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = result["torch_compile"]["target"]
    for case in result["case_summaries"]:
        case_name = f"task{case['task_id']}:init{case['init_index']}"
        for step in case["focus_step_summaries"]:
            summary = step["summary"]
            top = summary.get("top_record") or {}
            rows.append(
                {
                    "target": target,
                    "case": case_name,
                    "policy_step": int(step["policy_step"]),
                    "samples": int(summary["samples"]),
                    "continuous_max_abs_max": float(summary["continuous_max_abs"]["max"]),
                    "continuous_max_abs_p99": float(summary["continuous_max_abs"]["p99"]),
                    "continuous_l2_p99": float(summary["continuous_l2"]["p99"]),
                    "same_seed_repeat_span_max": float(summary["same_seed_repeat_span"]["max"]),
                    "top_dim": top.get("max_dim", ""),
                    "top_seed_offset": top.get("seed_offset"),
                    "top_repeat": top.get("repeat"),
                    "raw_diff": top.get("raw_diff", {}),
                    "compiled_p50_seconds": float(summary["compiled_get_action_seconds"]["p50"]),
                    "eager_p50_seconds": float(summary["eager_get_action_seconds"]["p50"]),
                    "time_ratio_p50": float(summary["compiled_over_eager_time"]["p50"]),
                }
            )
    return rows


def summarize_run(result: dict[str, Any]) -> dict[str, Any]:
    rows = flatten_step_rows(result)
    top = max(rows, key=lambda row: row["continuous_max_abs_max"]) if rows else None
    compiled_p50_values = [row["compiled_p50_seconds"] for row in rows]
    eager_p50_values = [row["eager_p50_seconds"] for row in rows]
    return {
        "result_json": result["_result_json"],
        "target": result["torch_compile"]["target"],
        "case_list": result["case_list"],
        "focus_policy_steps": result["focus_policy_steps"],
        "seed_offsets": result["seed_offsets"],
        "repeats": result["repeats"],
        "step_rows": len(rows),
        "top_step": top,
        "compiled_p50_mean_seconds": sum(compiled_p50_values) / len(compiled_p50_values) if compiled_p50_values else 0.0,
        "eager_p50_mean_seconds": sum(eager_p50_values) / len(eager_p50_values) if eager_p50_values else 0.0,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    runs = summary["runs"]
    lines = [
        "# Phase 16.5 Step-focused Replay Summary",
        "",
        "Boundary: eager-controlled same-observation replay around the `task6:init8` action spike region.",
        "",
        "## Target Ranking",
        "",
        "| target | top case | top step | top dim | cont max | p99 at step | repeat span max | compiled p50 mean | eager p50 mean |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for run in sorted(runs, key=lambda item: (item["top_step"] or {}).get("continuous_max_abs_max", 0.0), reverse=True):
        top = run["top_step"] or {}
        lines.append(
            "| {target} | `{case}` | {step} | `{dim}` | {cmax:.6g} | {p99:.6g} | {span:.6g} | {compiled:.1f} ms | {eager:.1f} ms |".format(
                target=run["target"],
                case=top.get("case", ""),
                step=top.get("policy_step", ""),
                dim=top.get("top_dim", ""),
                cmax=top.get("continuous_max_abs_max", 0.0),
                p99=top.get("continuous_max_abs_p99", 0.0),
                span=top.get("same_seed_repeat_span_max", 0.0),
                compiled=1000.0 * run["compiled_p50_mean_seconds"],
                eager=1000.0 * run["eager_p50_mean_seconds"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-Step Rows",
            "",
            "| target | case | step | samples | cont max | cont p99 | L2 p99 | repeat span max | top dim | seed offset |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in sorted(summary["step_rows"], key=lambda item: (item["target"], item["case"], item["policy_step"])):
        lines.append(
            "| {target} | `{case}` | {step} | {samples} | {cmax:.6g} | {p99:.6g} | {l2:.6g} | {span:.6g} | `{dim}` | {offset} |".format(
                target=row["target"],
                case=row["case"],
                step=row["policy_step"],
                samples=row["samples"],
                cmax=row["continuous_max_abs_max"],
                p99=row["continuous_max_abs_p99"],
                l2=row["continuous_l2_p99"],
                span=row["same_seed_repeat_span_max"],
                dim=row["top_dim"],
                offset=row["top_seed_offset"],
            )
        )

    lines.extend(
        [
            "",
            "## Reading This",
            "",
            "- `cont max` is the largest absolute drift among x/y/z/roll/pitch/yaw for that focused observation.",
            "- `repeat span max` checks fixed-seed determinism across repeated calls; values near zero mean the spike is repeatable once seed and observation are fixed.",
            "- A spike that survives across big-graph targets but disappears in small scopes points to graph-level numerical path rather than gripper thresholding.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase16_step_focused_replay_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase16_step_focused_replay_summary.md"))
    args = parser.parse_args()

    results = [load_result(path) for path in args.result_json]
    runs = [summarize_run(result) for result in results]
    step_rows = [row for result in results for row in flatten_step_rows(result)]
    summary = {"runs": runs, "step_rows": step_rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
