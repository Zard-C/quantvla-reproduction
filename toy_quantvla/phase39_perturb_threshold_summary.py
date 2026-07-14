"""Summarize Phase 39 closed-loop perturbation threshold sweeps."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase39_threshold_pilot_reduced_v1")
CASE_LIST = os.environ.get("CASE_LIST", "4:9,6:8")
DIRECTIONS = os.environ.get("DIRECTIONS", "y,z,yaw")
WINDOWS = os.environ.get("WINDOWS", "full,early,late")
EPSILONS = os.environ.get("EPSILONS", "0.006,0.01,0.03,0.05")
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260715"))

MANIFEST_JSONL = RESULTS / f"{TAG_PREFIX}_manifest.jsonl"
OUT_JSON = RESULTS / "phase39_threshold_pilot_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase39_closed_loop_perturbation_budget_pilot_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_case_list(spec: str) -> list[str]:
    return [item.strip() for item in spec.split(",") if item.strip()]


def parse_float_list(spec: str) -> list[float]:
    return [float(item.strip()) for item in spec.split(",") if item.strip()]


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 2)


def client_path(tag: str) -> Path:
    return RESULTS / f"{tag}_client_latency.json"


def trace_for(row: dict[str, Any], task_id: int, init_index: int) -> str | None:
    trace_dir = Path(row.get("trace_dir") or "")
    if not trace_dir:
        return None
    path = trace_dir if trace_dir.is_absolute() else ROOT / trace_dir
    if not path.exists():
        return None
    matches = sorted(path.glob(f"task={task_id:02d}--init={init_index:02d}--*.json"))
    if not matches:
        return None
    return str(matches[0].relative_to(ROOT))


def episode_result(row: dict[str, str], episode: dict[str, Any]) -> dict[str, Any]:
    task_id = int(episode["task_id"])
    init_index = int(episode["init_index"])
    latency = episode.get("policy_latency_seconds", {})
    return {
        "tag": row["tag"],
        "kind": row["kind"],
        "case": f"{task_id}:{init_index}",
        "task_id": task_id,
        "init_index": init_index,
        "direction": row.get("direction") or None,
        "action_keys": row.get("action_keys") or None,
        "window": row.get("window") or None,
        "step_start": int(row["step_start"]) if row.get("step_start") else None,
        "step_end": int(row["step_end"]) if row.get("step_end") else None,
        "epsilon": float(row["epsilon"]) if row.get("epsilon") else None,
        "success": bool(episode.get("success")),
        "steps": int(latency.get("count", 0)),
        "client_p50_ms": ms(latency.get("p50")),
        "client_source": str(client_path(row["tag"]).relative_to(ROOT)),
        "trace_path": trace_for(row, task_id, init_index),
        "status": "complete",
    }


def collect_rows(manifest: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in manifest:
        path = client_path(row["tag"])
        payload = read_json(path)
        if payload is None:
            missing.append(
                {
                    "tag": row["tag"],
                    "kind": row["kind"],
                    "case": row.get("case") or row.get("case_list"),
                    "direction": row.get("direction") or None,
                    "window": row.get("window") or None,
                    "epsilon": float(row["epsilon"]) if row.get("epsilon") else None,
                    "status": "missing",
                    "client_source": str(path.relative_to(ROOT)),
                }
            )
            continue
        for episode in payload.get("episodes", []):
            results.append(episode_result(row, episode))
    return results, missing


def flip_type(baseline_success: bool | None, success: bool) -> str:
    if baseline_success is None:
        return "unknown"
    if baseline_success and success:
        return "same_success"
    if baseline_success and not success:
        return "S->F"
    if not baseline_success and success:
        return "F->S"
    return "same_failure"


def threshold_interval(
    baseline_success: bool | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    complete = sorted(
        [row for row in rows if row.get("epsilon") is not None and row.get("status") == "complete"],
        key=lambda item: float(item["epsilon"]),
    )
    outcomes = []
    flips = []
    for row in complete:
        ft = flip_type(baseline_success, bool(row["success"]))
        is_flip = ft in {"S->F", "F->S"}
        outcomes.append(
            {
                "epsilon": float(row["epsilon"]),
                "success": bool(row["success"]),
                "flip_type": ft,
                "steps": int(row["steps"]),
            }
        )
        if is_flip:
            flips.append(row)

    if baseline_success is None:
        return {
            "status": "missing_baseline",
            "epsilon_crit_lower": None,
            "epsilon_crit_upper": None,
            "threshold_relation": "unknown",
            "outcomes": outcomes,
            "non_monotonic": None,
        }

    if not complete:
        return {
            "status": "missing",
            "epsilon_crit_lower": None,
            "epsilon_crit_upper": None,
            "threshold_relation": "missing",
            "outcomes": outcomes,
            "non_monotonic": None,
        }

    if not flips:
        return {
            "status": "no_flip",
            "epsilon_crit_lower": float(complete[-1]["epsilon"]),
            "epsilon_crit_upper": None,
            "threshold_relation": f"> {complete[-1]['epsilon']}",
            "outcomes": outcomes,
            "non_monotonic": False,
        }

    first_flip = flips[0]
    first_index = complete.index(first_flip)
    lower = 0.0 if first_index == 0 else float(complete[first_index - 1]["epsilon"])
    upper = float(first_flip["epsilon"])
    after = complete[first_index:]
    non_monotonic = any(flip_type(baseline_success, bool(row["success"])) not in {"S->F", "F->S"} for row in after)
    relation = f"<= {upper}" if first_index == 0 else f"({lower}, {upper}]"
    return {
        "status": "flip_found",
        "epsilon_crit_lower": lower,
        "epsilon_crit_upper": upper,
        "threshold_relation": relation,
        "outcomes": outcomes,
        "non_monotonic": bool(non_monotonic),
    }


def build_summary() -> dict[str, Any]:
    manifest = read_manifest(MANIFEST_JSONL)
    rows, missing = collect_rows(manifest)
    server_latency = read_json(RESULTS / f"{TAG_PREFIX}_server_latency.json")
    server_p50_ms = ms((server_latency or {}).get("get_action_seconds", {}).get("p50"))

    baseline_rows = [row for row in rows if row["kind"] == "baseline"]
    baseline_by_case = {row["case"]: row for row in baseline_rows}

    perturb_rows = []
    for row in rows:
        if row["kind"] != "perturb":
            continue
        base = baseline_by_case.get(row["case"])
        row = dict(row)
        row["baseline_success"] = None if base is None else bool(base["success"])
        row["flip_type"] = flip_type(row["baseline_success"], bool(row["success"]))
        row["server_p50_ms"] = server_p50_ms
        perturb_rows.append(row)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in perturb_rows:
        groups[(row["case"], row["direction"], row["window"])].append(row)

    threshold_rows = []
    for key in sorted(groups, key=lambda item: (case_key(item[0]), item[1], item[2])):
        case, direction, window = key
        base = baseline_by_case.get(case)
        interval = threshold_interval(None if base is None else bool(base["success"]), groups[key])
        threshold_rows.append(
            {
                "case": case,
                "direction": direction,
                "window": window,
                "step_start": groups[key][0].get("step_start"),
                "step_end": groups[key][0].get("step_end"),
                "baseline_success": None if base is None else bool(base["success"]),
                **interval,
            }
        )

    return {
        "phase": "phase39_closed_loop_perturbation_threshold_pilot",
        "tag_prefix": TAG_PREFIX,
        "case_list": parse_case_list(CASE_LIST),
        "directions": [item.strip() for item in DIRECTIONS.split(",") if item.strip()],
        "windows": [item.strip() for item in WINDOWS.split(",") if item.strip()],
        "epsilons": parse_float_list(EPSILONS),
        "policy_seed_base": POLICY_SEED_BASE,
        "manifest_source": str(MANIFEST_JSONL.relative_to(ROOT)),
        "server_p50_ms": server_p50_ms,
        "total_manifest_rows": len(manifest),
        "complete_result_rows": len(rows),
        "missing_runs": missing,
        "baseline_rows": baseline_rows,
        "perturb_rows": sorted(
            perturb_rows,
            key=lambda item: (
                case_key(item["case"]),
                str(item["direction"]),
                str(item["window"]),
                float(item["epsilon"]),
            ),
        ),
        "threshold_rows": threshold_rows,
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def yn(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return "success" if value else "fail"


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def outcome_string(outcomes: list[dict[str, Any]]) -> str:
    parts = []
    for item in outcomes:
        mark = "S" if item["success"] else "F"
        parts.append(f"{item['epsilon']:.4g}:{mark}/{item['steps']}")
    return ", ".join(parts)


def build_report(data: dict[str, Any]) -> str:
    baseline_table = [
        [row["case"], yn(row["success"]), row["steps"], fmt(row["client_p50_ms"]), row.get("trace_path") or "-"]
        for row in sorted(data["baseline_rows"], key=lambda item: case_key(item["case"]))
    ]

    threshold_table = []
    for row in data["threshold_rows"]:
        threshold_table.append(
            [
                row["case"],
                row["direction"],
                row["window"],
                f"[{fmt(row.get('step_start'))}, {fmt(row.get('step_end'))})"
                if row.get("step_start") is not None
                else "full",
                yn(row["baseline_success"]),
                row["threshold_relation"],
                "yes" if row.get("non_monotonic") else "no",
                outcome_string(row["outcomes"]),
            ]
        )

    flip_table = []
    for row in data["perturb_rows"]:
        if row["flip_type"] in {"S->F", "F->S"}:
            flip_table.append(
                [
                    row["case"],
                    row["direction"],
                    row["window"],
                    fmt(row["epsilon"]),
                    row["flip_type"],
                    row["steps"],
                    row.get("trace_path") or "-",
                ]
            )

    missing = data["missing_runs"]
    missing_rows = [
        [
            item.get("case") or "-",
            item.get("direction") or "-",
            item.get("window") or "-",
            fmt(item.get("epsilon")),
            item["client_source"],
        ]
        for item in missing[:40]
    ]

    lines = [
        "# Phase 39: 闭环扰动阈值 pilot",
        "",
        "本阶段把后端实现误差转写成显式 action perturbation，并估计不同 case、方向和 rollout 阶段上的临界扰动区间。",
        "",
        "## 配置",
        "",
        f"- Tag: `{data['tag_prefix']}`",
        f"- Cases: `{', '.join(data['case_list'])}`",
        f"- Directions: `{', '.join(data['directions'])}`",
        f"- Windows: `{', '.join(data['windows'])}`",
        f"- Epsilons: `{', '.join(str(x) for x in data['epsilons'])}`",
        f"- Deterministic policy seed base: `{data['policy_seed_base']}`",
        f"- Server p50: `{fmt(data['server_p50_ms'])}` ms",
        f"- Manifest: `{data['manifest_source']}`",
        "",
        "## Baseline",
        "",
        md_table(["case", "outcome", "steps", "client p50 ms", "trace"], baseline_table)
        if baseline_table
        else "_No baseline results yet._",
        "",
        "## 临界扰动区间",
        "",
        "记号：`0.006:S/224` 表示 epsilon=0.006 时 success，episode 有 224 个 policy steps。临界区间是 first flip 的区间，不假设结果严格单调；若 `non-monotonic=yes`，说明更大 epsilon 后结果又翻回来了，需要单独看 trace。",
        "",
        md_table(
            [
                "case",
                "direction",
                "window",
                "steps",
                "baseline",
                "epsilon crit",
                "non-monotonic",
                "outcomes",
            ],
            threshold_table,
        )
        if threshold_table
        else "_No perturbation results yet._",
        "",
        "## 已观察到的 rollout flip",
        "",
        md_table(["case", "direction", "window", "epsilon", "flip", "steps", "trace"], flip_table)
        if flip_table
        else "_No flips observed yet._",
        "",
    ]

    if missing:
        lines.extend(
            [
                "## 未完成项",
                "",
                f"当前 manifest 中还有 `{len(missing)}` 个 run 没有 client latency JSON。前 40 个如下：",
                "",
                md_table(["case", "direction", "window", "epsilon", "expected"], missing_rows),
                "",
            ]
        )

    lines.extend(
        [
            "## 解读口径",
            "",
            "- 如果同一个 case 上 `y/yaw` 的 first-flip epsilon 明显小于 `z`，支持 not all dims are equal。",
            "- 如果 `early` 的 first-flip epsilon 小于 `late`，支持 not all durations are equal。",
            "- 如果某个方向全成功，报告为 `> max epsilon`，表示当前网格还没有碰到闭环失败边界。",
            "- 如果出现非单调，不能把阈值当成确定常数；它更像是这个闭环系统在该扰动族上的局部 basin 边界采样。",
            "",
            f"- JSON artifact: `{OUT_JSON.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    data = build_summary()
    write_json(OUT_JSON, data)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
