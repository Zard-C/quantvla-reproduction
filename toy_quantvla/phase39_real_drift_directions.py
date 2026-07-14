"""Estimate real-backend-drift-aligned perturbation directions from paired traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

DEFAULT_BASELINE_TRACE_DIR = (
    RESULTS / "phase28A_proxy_guided_15case_v1_selected_trace" / "baseline"
)
DEFAULT_VARIANT_TRACE_DIR = (
    RESULTS / "phase28A_proxy_guided_15case_v1_selected_trace" / "speed_only"
)
DEFAULT_OUT_JSON = RESULTS / "phase39_real_drift_directions.json"
DEFAULT_OUT_MD = DOCS / "phase39_real_drift_directions_zh.md"


def parse_case_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def window_bounds(task_id: int, window: str) -> tuple[int | None, int | None]:
    if window == "full":
        return None, None
    table = {
        (4, "early"): (0, 75),
        (4, "mid"): (75, 150),
        (4, "late"): (150, 225),
        (6, "early"): (0, 200),
        (6, "mid"): (200, 450),
        (6, "late"): (450, 700),
    }
    if (task_id, window) not in table:
        raise ValueError(f"Unsupported window {window!r} for task {task_id}")
    return table[(task_id, window)]


def parse_windows(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def find_trace(trace_dir: Path, task_id: int, init_index: int) -> Path | None:
    matches = sorted(trace_dir.glob(f"task={task_id:02d}--init={init_index:02d}--*.json"))
    return matches[0] if matches else None


def l2(vec: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def trace_steps_by_policy_step(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out = {}
    for step in payload.get("steps", []):
        if "policy_step" in step:
            out[int(step["policy_step"])] = step
    return out


def action6(step: dict[str, Any]) -> list[float]:
    return [float(x) for x in step["libero_action"][:6]]


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [mean(vec[i] for vec in vectors) for i in range(width)]


def estimate_case(
    *,
    case: str,
    windows: list[str],
    baseline_trace_dir: Path,
    variant_trace_dir: Path,
    variant_name: str,
) -> list[dict[str, Any]]:
    task_id, init_index = case_key(case)
    baseline_path = find_trace(baseline_trace_dir, task_id, init_index)
    variant_path = find_trace(variant_trace_dir, task_id, init_index)
    if baseline_path is None or variant_path is None:
        return [
            {
                "case": case,
                "task_id": task_id,
                "init_index": init_index,
                "variant": variant_name,
                "direction": f"real_{variant_name}_mean",
                "window": window,
                "status": "missing_trace",
                "baseline_trace": None if baseline_path is None else str(baseline_path.relative_to(ROOT)),
                "variant_trace": None if variant_path is None else str(variant_path.relative_to(ROOT)),
            }
            for window in windows
        ]

    baseline = read_json(baseline_path)
    variant = read_json(variant_path)
    bsteps = trace_steps_by_policy_step(baseline)
    vsteps = trace_steps_by_policy_step(variant)
    common_steps = sorted(set(bsteps) & set(vsteps))
    rows = []
    for window in windows:
        start, end = window_bounds(task_id, window)
        selected_steps = [
            step
            for step in common_steps
            if (start is None or step >= start) and (end is None or step < end)
        ]
        deltas = []
        for step in selected_steps:
            b = action6(bsteps[step])
            v = action6(vsteps[step])
            deltas.append([v[i] - b[i] for i in range(6)])
        vector = mean_vector(deltas)
        norm = l2(vector) if vector else 0.0
        l2_values = [l2(delta) for delta in deltas]
        rows.append(
            {
                "case": case,
                "task_id": task_id,
                "init_index": init_index,
                "variant": variant_name,
                "direction": f"real_{variant_name}_mean",
                "window": window,
                "step_start": start,
                "step_end": end,
                "status": "complete" if norm > 0.0 else "zero_vector",
                "common_steps": len(common_steps),
                "window_steps": len(selected_steps),
                "baseline_success": bool(baseline.get("success")),
                "variant_success": bool(variant.get("success")),
                "baseline_num_steps": int(baseline.get("num_steps", 0)),
                "variant_num_steps": int(variant.get("num_steps", 0)),
                "mean_delta_vector6": vector,
                "mean_delta_l2": norm,
                "delta_l2_p50": percentile(l2_values, 0.50),
                "delta_l2_p95": percentile(l2_values, 0.95),
                "delta_l2_max": max(l2_values) if l2_values else None,
                "baseline_trace": str(baseline_path.relative_to(ROOT)),
                "variant_trace": str(variant_path.relative_to(ROOT)),
                "caveat": (
                    "Estimated from paired closed-loop traces, so steps after trajectory "
                    "divergence mix backend drift with state-distribution shift."
                ),
            }
        )
    return rows


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    baseline_trace_dir = Path(args.baseline_trace_dir)
    variant_trace_dir = Path(args.variant_trace_dir)
    if not baseline_trace_dir.is_absolute():
        baseline_trace_dir = ROOT / baseline_trace_dir
    if not variant_trace_dir.is_absolute():
        variant_trace_dir = ROOT / variant_trace_dir
    windows = parse_windows(args.windows)
    rows = []
    for case in parse_case_list(args.case_list):
        rows.extend(
            estimate_case(
                case=case,
                windows=windows,
                baseline_trace_dir=baseline_trace_dir,
                variant_trace_dir=variant_trace_dir,
                variant_name=args.variant_name,
            )
        )
    return {
        "phase": "phase39_real_backend_drift_direction_estimate",
        "case_list": parse_case_list(args.case_list),
        "windows": windows,
        "variant_name": args.variant_name,
        "baseline_trace_dir": str(baseline_trace_dir.relative_to(ROOT)),
        "variant_trace_dir": str(variant_trace_dir.relative_to(ROOT)),
        "rows": rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(data: dict[str, Any], out_json: Path) -> str:
    table = []
    for row in data["rows"]:
        table.append(
            [
                row["case"],
                row["direction"],
                row["window"],
                row["status"],
                row.get("window_steps", "-"),
                fmt(row.get("mean_delta_l2")),
                fmt(row.get("delta_l2_p50")),
                fmt(row.get("delta_l2_p95")),
                ",".join(fmt(x) for x in row.get("mean_delta_vector6", [])) or "-",
            ]
        )
    return "\n".join(
        [
            "# Phase 39: real backend drift 方向估计",
            "",
            "这个文件从已有 paired closed-loop traces 中估计真实 tactic 的动作差方向，用于后续 `real_speed_only_mean` 方向的 threshold sweep。",
            "",
            "注意：这些方向来自闭环轨迹对齐，不是严格 same-observation replay。first divergence 之后的差异会混入状态分布偏移，因此它适合作为 real tactic footprint，而不是纯函数误差的无偏估计。",
            "",
            f"- Variant: `{data['variant_name']}`",
            f"- Baseline trace dir: `{data['baseline_trace_dir']}`",
            f"- Variant trace dir: `{data['variant_trace_dir']}`",
            f"- JSON artifact: `{out_json.relative_to(ROOT)}`",
            "",
            md_table(
                [
                    "case",
                    "direction",
                    "window",
                    "status",
                    "steps",
                    "mean L2",
                    "p50 L2",
                    "p95 L2",
                    "mean vector6",
                ],
                table,
            ),
            "",
        ]
    )


def emit_runner_tsv(data: dict[str, Any], *, case_list: str, windows: str) -> None:
    wanted_cases = set(parse_case_list(case_list))
    wanted_windows = set(parse_windows(windows))
    for row in data.get("rows", []):
        if row.get("status") != "complete":
            continue
        if row["case"] not in wanted_cases or row["window"] not in wanted_windows:
            continue
        vector = row.get("mean_delta_vector6") or []
        if len(vector) != 6:
            continue
        start = "-" if row.get("step_start") is None else str(row["step_start"])
        end = "-" if row.get("step_end") is None else str(row["step_end"])
        print(
            "\t".join(
                [
                    row["case"],
                    str(row["task_id"]),
                    str(row["init_index"]),
                    row["direction"],
                    row["window"],
                    start,
                    end,
                    ",".join(f"{float(x):.10g}" for x in vector),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace-dir", default=str(DEFAULT_BASELINE_TRACE_DIR.relative_to(ROOT)))
    parser.add_argument("--variant-trace-dir", default=str(DEFAULT_VARIANT_TRACE_DIR.relative_to(ROOT)))
    parser.add_argument("--variant-name", default="speed_only")
    parser.add_argument("--case-list", default="4:9,6:8")
    parser.add_argument("--windows", default="full,early,late")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--input-json", type=Path, help="Read an existing artifact instead of estimating from traces.")
    parser.add_argument("--emit-runner-tsv", action="store_true")
    args = parser.parse_args()

    if args.input_json is not None:
        path = args.input_json if args.input_json.is_absolute() else ROOT / args.input_json
        data = read_json(path)
    else:
        data = build_artifact(args)
        out_json = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
        out_md = args.output_md if args.output_md.is_absolute() else ROOT / args.output_md
        write_json(out_json, data)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_report(data, out_json), encoding="utf-8")
        print(f"Wrote {out_json.relative_to(ROOT)}")
        print(f"Wrote {out_md.relative_to(ROOT)}")

    if args.emit_runner_tsv:
        emit_runner_tsv(data, case_list=args.case_list, windows=args.windows)


if __name__ == "__main__":
    main()
