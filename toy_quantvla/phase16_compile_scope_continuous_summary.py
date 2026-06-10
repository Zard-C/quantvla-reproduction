"""Summarize Phase 16 torch.compile continuous-action drift sweeps.

The input files are produced by ``phase13_torch_compile_online_drift.py``.
That probe keeps LIBERO on the eager trajectory and records eager/compiled
actions on the same live observations.  This summarizer focuses on the six
continuous action channels because Phase 15 showed that their joint drift, not
gripper alone, explains the key closed-loop regressions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CONTINUOUS_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
TRANSLATION_KEYS = ["x", "y", "z"]
ROTATION_KEYS = ["roll", "pitch", "yaw"]
ACTION_KEYS = CONTINUOUS_KEYS + ["gripper"]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    ordered = sorted(float(v) for v in values)
    count = len(ordered)

    def pct(q: float) -> float:
        if count == 1:
            return ordered[0]
        idx = int(round((count - 1) * q))
        return ordered[max(0, min(count - 1, idx))]

    return {
        "count": count,
        "mean": float(sum(ordered) / count),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p50": float(pct(0.50)),
        "p90": float(pct(0.90)),
        "p99": float(pct(0.99)),
    }


def action_value(step: dict[str, Any], side: str, key: str) -> float:
    return float(step[f"{side}_raw_action"][f"action.{key}"])


def l2(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def locate_trace(result_path: Path, raw_trace_path: str) -> Path:
    trace_path = Path(raw_trace_path)
    candidates = [
        trace_path,
        result_path.parent / trace_path.name,
        result_path.parent / f"{result_path.stem}_trace" / trace_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate trace {raw_trace_path!r} for {result_path}")


def summarize_episode(trace_path: Path) -> dict[str, Any]:
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    dim_abs: dict[str, list[float]] = {key: [] for key in ACTION_KEYS}
    continuous_l2: list[float] = []
    continuous_max: list[float] = []
    translation_l2: list[float] = []
    rotation_l2: list[float] = []
    eager_seconds: list[float] = []
    compiled_seconds: list[float] = []
    ratio: list[float] = []

    for step in trace["steps"]:
        diffs = {}
        for key in ACTION_KEYS:
            diff = action_value(step, "compiled", key) - action_value(step, "eager", key)
            diffs[key] = diff
            dim_abs[key].append(abs(diff))
        continuous = [diffs[key] for key in CONTINUOUS_KEYS]
        translation = [diffs[key] for key in TRANSLATION_KEYS]
        rotation = [diffs[key] for key in ROTATION_KEYS]
        continuous_l2.append(l2(continuous))
        continuous_max.append(max(abs(v) for v in continuous))
        translation_l2.append(l2(translation))
        rotation_l2.append(l2(rotation))
        timing = step.get("timing", {})
        eager = float(timing.get("eager_get_action_seconds", 0.0))
        compiled = float(timing.get("compiled_get_action_seconds", 0.0))
        eager_seconds.append(eager)
        compiled_seconds.append(compiled)
        ratio.append(compiled / eager if eager > 0 else 0.0)

    return {
        "trace_file": str(trace_path),
        "task_id": trace["task_id"],
        "init_index": trace["init_index"],
        "success": bool(trace["success"]),
        "exception": trace.get("exception"),
        "steps": int(trace["num_steps"]),
        "valid_for_drift": int(trace["num_steps"]) > 0,
        "continuous_abs_by_dim": {key: summarize(dim_abs[key]) for key in CONTINUOUS_KEYS},
        "gripper_abs": summarize(dim_abs["gripper"]),
        "continuous_l2": summarize(continuous_l2),
        "continuous_max_abs": summarize(continuous_max),
        "translation_l2": summarize(translation_l2),
        "rotation_l2": summarize(rotation_l2),
        "eager_get_action_seconds": summarize(eager_seconds),
        "compiled_get_action_seconds": summarize(compiled_seconds),
        "compiled_over_eager_time": summarize(ratio),
    }


def merge_episode_summaries(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    continuous_dim_values: dict[str, list[float]] = {key: [] for key in CONTINUOUS_KEYS}
    merged: dict[str, list[float]] = {
        "continuous_l2": [],
        "continuous_max_abs": [],
        "translation_l2": [],
        "rotation_l2": [],
        "gripper_abs": [],
        "eager_get_action_seconds": [],
        "compiled_get_action_seconds": [],
        "compiled_over_eager_time": [],
    }
    for episode in episodes:
        trace = json.loads(Path(episode["trace_file"]).read_text(encoding="utf-8"))
        for step in trace["steps"]:
            diffs = {
                key: action_value(step, "compiled", key) - action_value(step, "eager", key)
                for key in ACTION_KEYS
            }
            for key in CONTINUOUS_KEYS:
                continuous_dim_values[key].append(abs(diffs[key]))
            continuous = [diffs[key] for key in CONTINUOUS_KEYS]
            translation = [diffs[key] for key in TRANSLATION_KEYS]
            rotation = [diffs[key] for key in ROTATION_KEYS]
            merged["continuous_l2"].append(l2(continuous))
            merged["continuous_max_abs"].append(max(abs(v) for v in continuous))
            merged["translation_l2"].append(l2(translation))
            merged["rotation_l2"].append(l2(rotation))
            merged["gripper_abs"].append(abs(diffs["gripper"]))
            timing = step.get("timing", {})
            eager = float(timing.get("eager_get_action_seconds", 0.0))
            compiled = float(timing.get("compiled_get_action_seconds", 0.0))
            merged["eager_get_action_seconds"].append(eager)
            merged["compiled_get_action_seconds"].append(compiled)
            merged["compiled_over_eager_time"].append(compiled / eager if eager > 0 else 0.0)

    return {
        "continuous_abs_by_dim": {key: summarize(values) for key, values in continuous_dim_values.items()},
        **{key: summarize(values) for key, values in merged.items()},
    }


def summarize_run(result_path: Path) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episodes = [
        summarize_episode(locate_trace(result_path, trace_file))
        for trace_file in result.get("episode_trace_files", [])
    ]
    aggregate = merge_episode_summaries(episodes)
    return {
        "result_json": str(result_path),
        "target": result["torch_compile"]["target"],
        "mode": result["torch_compile"].get("mode"),
        "compile_enabled": bool(result["torch_compile"].get("enabled")),
        "case_list": result.get("case_list"),
        "episodes": len(episodes),
        "valid_drift_episodes": sum(1 for episode in episodes if episode["valid_for_drift"]),
        "valid_drift_steps": sum(int(episode["steps"]) for episode in episodes),
        "eager_successes": result.get("eager_successes"),
        "model_load_seconds": result.get("model_load_seconds"),
        "compile_wrap_seconds": result["torch_compile"].get("wrap_seconds"),
        "aggregate": aggregate,
        "episode_summaries": episodes,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    runs = summary["runs"]
    lines = [
        "# Phase 16 Compile Scope Continuous Drift Sweep",
        "",
        "Boundary: eager-controlled same-observation replay. The simulator is stepped by eager FP16 actions; compiled actions are measured on the same observations with the same denoising seeds.",
        "",
        "## Ranking",
        "",
        "| target | cases | valid steps | window done | cont max p99 | cont max max | cont L2 p99 | compiled p50 | eager p50 | time ratio p50 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        agg = run["aggregate"]
        lines.append(
            "| {target} | `{cases}` | {valid_steps} | {succ}/{eps} | {cmax_p99:.6g} | {cmax_max:.6g} | {l2_p99:.6g} | {compiled:.1f} ms | {eager:.1f} ms | {ratio:.3g} |".format(
                target=run["target"],
                cases=run["case_list"],
                valid_steps=run["valid_drift_steps"],
                succ=run.get("eager_successes"),
                eps=run["episodes"],
                cmax_p99=agg["continuous_max_abs"]["p99"],
                cmax_max=agg["continuous_max_abs"]["max"],
                l2_p99=agg["continuous_l2"]["p99"],
                compiled=1000.0 * agg["compiled_get_action_seconds"]["p50"],
                eager=1000.0 * agg["eager_get_action_seconds"]["p50"],
                ratio=agg["compiled_over_eager_time"]["p50"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Target Episodes",
            "",
        ]
    )
    for run in runs:
        lines.extend(
            [
                f"### `{run['target']}`",
                "",
                "| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |",
                "|---|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for episode in run["episode_summaries"]:
            case = f"task{episode['task_id']}:init{episode['init_index']}"
            exception = ""
            if episode.get("exception"):
                exception = str(episode["exception"]).replace("\n", " ")
                if len(exception) > 180:
                    exception = exception[:177] + "..."
                exception = f"`{exception}`"
            lines.append(
                "| {case} | {success} | {steps} | {exception} | {cmax_p99:.6g} | {cmax_max:.6g} | {tl2:.6g} | {rl2:.6g} |".format(
                    case=case,
                    success=str(episode["success"]),
                    steps=episode["steps"],
                    exception=exception,
                    cmax_p99=episode["continuous_max_abs"]["p99"],
                    cmax_max=episode["continuous_max_abs"]["max"],
                    tl2=episode["translation_l2"]["p99"],
                    rl2=episode["rotation_l2"]["p99"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Reading This",
            "",
            "- Lower continuous p99/max is better for behavior transparency.",
            "- Lower compiled p50 is better for speed, but small scopes can lose the large-graph benefit.",
            "- Candidates should pass this same-observation screen before any closed-loop rollout expansion.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase16_compile_scope_continuous_summary.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase16_compile_scope_continuous_summary.md"))
    args = parser.parse_args()

    runs = [summarize_run(path) for path in args.result_json]
    runs.sort(
        key=lambda run: (
            run["valid_drift_steps"] == 0,
            run["aggregate"]["continuous_max_abs"]["p99"],
            run["aggregate"]["continuous_max_abs"]["max"],
        )
    )
    summary = {
        "phase": "phase16_compile_scope_continuous_drift",
        "continuous_keys": CONTINUOUS_KEYS,
        "runs": runs,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, args.output_md)
    print(json.dumps({"output_json": str(args.output_json), "output_md": str(args.output_md), "runs": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
