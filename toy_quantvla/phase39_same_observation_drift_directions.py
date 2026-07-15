"""Estimate real backend drift directions from same-observation online drift traces.

The input is produced by ``phase13_torch_compile_online_drift.py``.  That probe
keeps LIBERO on the eager FP16 trajectory, evaluates both eager/reference and
compiled/tactic actions on each live observation with the same denoising seed,
and records both LIBERO actions.  This script converts those traces into the
Phase 39 vector schema so the closed-loop perturbation runner can inject
normalized real-drift directions.
"""

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


def parse_case_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_windows(spec: str) -> list[str]:
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_path(raw: str, *, result_path: Path) -> Path:
    path = Path(raw)
    candidates = [
        path,
        ROOT / path,
        result_path.parent / path.name,
        result_path.parent / f"{result_path.stem}_trace" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve trace path {raw!r} from {result_path}")


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


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    return [mean(vec[i] for vec in vectors) for i in range(width)]


def action6(step: dict[str, Any], key: str) -> list[float]:
    return [float(x) for x in step[key][:6]]


def trace_by_case(result: dict[str, Any], *, result_path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for raw in result.get("episode_trace_files", []):
        trace_path = resolve_path(raw, result_path=result_path)
        trace = read_json(trace_path)
        case = f"{int(trace['task_id'])}:{int(trace['init_index'])}"
        out[case] = trace_path
    return out


def estimate_case(
    *,
    case: str,
    trace_path: Path | None,
    windows: list[str],
    variant_name: str,
) -> list[dict[str, Any]]:
    task_id, init_index = case_key(case)
    if trace_path is None:
        return [
            {
                "case": case,
                "task_id": task_id,
                "init_index": init_index,
                "variant": variant_name,
                "direction": f"real_{variant_name}_sameobs_mean",
                "window": window,
                "status": "missing_trace",
            }
            for window in windows
        ]

    trace = read_json(trace_path)
    rows = []
    for window in windows:
        start, end = window_bounds(task_id, window)
        selected = [
            step
            for step in trace.get("steps", [])
            if (start is None or int(step["policy_step"]) >= start)
            and (end is None or int(step["policy_step"]) < end)
        ]
        deltas = []
        for step in selected:
            eager = action6(step, "eager_libero_action")
            compiled = action6(step, "compiled_libero_action")
            deltas.append([compiled[i] - eager[i] for i in range(6)])
        vector = mean_vector(deltas)
        norm = l2(vector) if vector else 0.0
        l2_values = [l2(delta) for delta in deltas]
        rows.append(
            {
                "case": case,
                "task_id": task_id,
                "init_index": init_index,
                "variant": variant_name,
                "direction": f"real_{variant_name}_sameobs_mean",
                "window": window,
                "step_start": start,
                "step_end": end,
                "status": "complete" if norm > 0.0 else "zero_vector",
                "window_steps": len(selected),
                "reference_success": bool(trace.get("success")),
                "reference_num_steps": int(trace.get("num_steps", 0)),
                "mean_delta_vector6": vector,
                "mean_delta_l2": norm,
                "delta_l2_p50": percentile(l2_values, 0.50),
                "delta_l2_p95": percentile(l2_values, 0.95),
                "delta_l2_max": max(l2_values) if l2_values else None,
                "trace": str(trace_path.relative_to(ROOT)) if trace_path.is_relative_to(ROOT) else str(trace_path),
                "caveat": (
                    "Estimated from same-observation eager-controlled online drift: "
                    "the reference/eager action controls LIBERO while the tactic action "
                    "is measured on the same observation and denoising seed."
                ),
            }
        )
    return rows


def build_artifact(args: argparse.Namespace) -> dict[str, Any]:
    result_path = args.online_drift_json if args.online_drift_json.is_absolute() else ROOT / args.online_drift_json
    result = read_json(result_path)
    traces = trace_by_case(result, result_path=result_path)
    windows = parse_windows(args.windows)
    rows = []
    for case in parse_case_list(args.case_list):
        rows.extend(
            estimate_case(
                case=case,
                trace_path=traces.get(case),
                windows=windows,
                variant_name=args.variant_name,
            )
        )
    return {
        "phase": "phase39_same_observation_real_backend_drift_direction_estimate",
        "source": "phase13_torch_compile_online_drift",
        "case_list": parse_case_list(args.case_list),
        "windows": windows,
        "variant_name": args.variant_name,
        "online_drift_json": str(result_path.relative_to(ROOT)) if result_path.is_relative_to(ROOT) else str(result_path),
        "torch_compile": result.get("torch_compile", {}),
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
            "# Phase 39 A2: same-observation real backend drift 方向估计",
            "",
            "这个文件从 eager-controlled same-observation drift trace 中估计真实 tactic 的动作差方向。",
            "",
            "区别于 paired closed-loop trace：这里环境始终由 FP16/eager reference action 推进，tactic 只在同一个 observation 和同一个 denoising seed 上旁路查询。因此该方向更接近真实后端的函数误差，而不是 first-divergence 之后混入状态分布偏移的轨迹差异。",
            "",
            f"- Variant: `{data['variant_name']}`",
            f"- Online drift JSON: `{data['online_drift_json']}`",
            f"- JSON artifact: `{out_json.relative_to(ROOT) if out_json.is_relative_to(ROOT) else out_json}`",
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
    parser.add_argument("--online-drift-json", type=Path, required=True)
    parser.add_argument("--variant-name", default="speed_only")
    parser.add_argument("--case-list", default="4:9,6:8")
    parser.add_argument("--windows", default="full,early")
    parser.add_argument("--output-json", type=Path, default=RESULTS / "phase39_sameobs_real_drift_directions.json")
    parser.add_argument("--output-md", type=Path, default=DOCS / "phase39_sameobs_real_drift_directions_zh.md")
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
        print(f"Wrote {out_json.relative_to(ROOT) if out_json.is_relative_to(ROOT) else out_json}")
        print(f"Wrote {out_md.relative_to(ROOT) if out_md.is_relative_to(ROOT) else out_md}")

    if args.emit_runner_tsv:
        emit_runner_tsv(data, case_list=args.case_list, windows=args.windows)


if __name__ == "__main__":
    main()
