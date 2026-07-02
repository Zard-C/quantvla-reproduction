"""Summarize Phase36 N1.7 tactic probe runs."""

from __future__ import annotations

import ast
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase36_n17_tactic_probe_15case_v1")
CASE_LIST = os.environ.get(
    "CASE_LIST",
    "0:21,0:22,0:23,1:21,1:22,1:23,4:21,4:22,4:23,6:21,6:22,6:23,8:21,8:22,8:23",
)
TACTICS = os.environ.get(
    "TACTICS",
    "fp16 speed_only window_0_10 window_0_20 window_0_30 window_10_30 window_20_50",
).split()
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase36b_n17_tactic_probe_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 36B: N1.7 Tactic Probe")
BASELINE_P50_MS = os.environ.get("BASELINE_P50_MS")


TACTIC_LABELS = {
    "fp16": "FP16 baseline",
    "speed_only": "Speed-only compile",
    "window_0_10": "Request window 0-10",
    "window_0_20": "Request window 0-20",
    "window_0_30": "Request window 0-30",
    "window_10_30": "Request window 10-30",
    "window_20_50": "Request window 20-50",
}


def tactic_label(tactic: str) -> str:
    if tactic in TACTIC_LABELS:
        return TACTIC_LABELS[tactic]
    match = re.fullmatch(r"window_(\d+)_(\d+)", tactic)
    if match:
        return f"Request window {match.group(1)}-{match.group(2)}"
    return tactic


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in value.split(","):
        if not item:
            continue
        task, init = item.split(":")
        cases.append((int(task), int(init)))
    return cases


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 2)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def parse_client_result(path: Path) -> dict[str, Any]:
    text = path.read_text(errors="replace", encoding="utf-8") if path.exists() else ""
    success_rate_match = re.search(r"success rate:\s+([0-9.]+)", text)
    result_match = re.search(r"results:\s+\((.*)\)\s*$", text, re.MULTILINE)
    parsed: dict[str, Any] = {}
    if result_match:
        try:
            value = ast.literal_eval("(" + result_match.group(1) + ")")
            parsed["results_tuple"] = value
            parsed["success"] = bool(value[1][0]) if len(value) > 1 and value[1] else None
            details = value[2] if len(value) > 2 and isinstance(value[2], dict) else {}
            parsed["episode_length"] = int(details.get("episode_lengths", [0])[0])
            parsed["episode_reward"] = float(details.get("episode_rewards", [0.0])[0])
        except (SyntaxError, ValueError, TypeError, IndexError):
            parsed["parse_error"] = True
    if success_rate_match:
        parsed["success_rate"] = float(success_rate_match.group(1))
        parsed.setdefault("success", parsed["success_rate"] > 0.0)
    return parsed


def case_tag(tactic: str, task_id: int, init_index: int) -> str:
    return f"{TAG_PREFIX}_{tactic}_task{task_id}_init{init_index}"


def summarize_case(tactic: str, task_id: int, init_index: int) -> dict[str, Any]:
    tag = case_tag(tactic, task_id, init_index)
    summary = read_json(RESULTS / f"{tag}_summary.json")
    latency = read_json(RESULTS / f"{tag}_server_latency.json")
    prepare = read_json(RESULTS / f"{tag}_server_prepare.json")
    status = read_json(RESULTS / f"{tag}_runner_status.json")
    client = parse_client_result(RESULTS / f"{tag}_client.log")

    latency_stats = (latency or {}).get("get_action_seconds", {})
    memory = (latency or {}).get("extra", {}).get("server_memory", {})
    fallback = (latency or {}).get("extra", {}).get("compile_step_window_fallback", {})
    success = client.get("success")
    if success is None and summary is not None:
        rate = summary.get("success_rate")
        success = bool(rate and rate > 0.0)

    return {
        "tag": tag,
        "tactic": tactic,
        "task_id": task_id,
        "init_index": init_index,
        "case": f"{task_id}:{init_index}",
        "status": "complete" if summary is not None else "missing",
        "exit_status": (status or {}).get("exit_status"),
        "success": success,
        "success_rate": (summary or {}).get("success_rate", client.get("success_rate")),
        "episode_length": client.get("episode_length"),
        "episode_reward": client.get("episode_reward"),
        "request_count": latency_stats.get("count"),
        "server_p50_ms": ms(latency_stats.get("p50")),
        "server_mean_ms": ms(latency_stats.get("mean")),
        "server_p90_ms": ms(latency_stats.get("p90")),
        "server_max_ms": ms(latency_stats.get("max")),
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2)
        if prepare
        else None,
        "reserved_mib": memory.get("reserved_mib"),
        "max_reserved_mib": memory.get("max_reserved_mib"),
        "fallback": fallback,
        "summary_path": str((RESULTS / f"{tag}_summary.json").relative_to(ROOT)),
        "client_log": str((RESULTS / f"{tag}_client.log").relative_to(ROOT)),
    }


def summarize_tactic(rows: list[dict[str, Any]], baseline_p50: float | None) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    successes = sum(1 for row in complete if row.get("success") is True)
    p50_values = [row["server_p50_ms"] for row in complete if row.get("server_p50_ms") is not None]
    mean_values = [row["server_mean_ms"] for row in complete if row.get("server_mean_ms") is not None]
    request_values = [row["request_count"] for row in complete if row.get("request_count") is not None]

    by_task: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "total": 0})
    for row in complete:
        bucket = by_task[str(row["task_id"])]
        bucket["total"] += 1
        bucket["successes"] += int(row.get("success") is True)

    avg_p50 = round(sum(p50_values) / len(p50_values), 2) if p50_values else None
    avg_mean = round(sum(mean_values) / len(mean_values), 2) if mean_values else None
    speedup = round(baseline_p50 / avg_p50, 2) if baseline_p50 and avg_p50 else None
    return {
        "tactic": rows[0]["tactic"] if rows else "",
        "label": tactic_label(rows[0]["tactic"]) if rows else "",
        "complete": len(complete),
        "expected": len(rows),
        "successes": successes,
        "success_rate": round(successes / len(complete), 3) if complete else None,
        "avg_server_p50_ms": avg_p50,
        "avg_server_mean_ms": avg_mean,
        "avg_request_count": round(sum(request_values) / len(request_values), 2) if request_values else None,
        "speedup_vs_fp16_avg_p50": speedup,
        "per_task": dict(by_task),
    }


def paired_counts(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["case"]: row for row in left_rows if row["status"] == "complete"}
    right = {row["case"]: row for row in right_rows if row["status"] == "complete"}
    common = sorted(set(left) & set(right), key=lambda item: tuple(map(int, item.split(":"))))
    repaired = [case for case in common if left[case]["success"] is False and right[case]["success"] is True]
    regressed = [case for case in common if left[case]["success"] is True and right[case]["success"] is False]
    same_success = [
        case for case in common if left[case]["success"] is True and right[case]["success"] is True
    ]
    same_failure = [
        case for case in common if left[case]["success"] is False and right[case]["success"] is False
    ]
    return {
        "left": left_rows[0]["tactic"] if left_rows else "",
        "right": right_rows[0]["tactic"] if right_rows else "",
        "common": len(common),
        "repaired": repaired,
        "regressed": regressed,
        "same_success": same_success,
        "same_failure": same_failure,
        "net": len(repaired) - len(regressed),
    }


def build_summary() -> dict[str, Any]:
    cases = parse_cases(CASE_LIST)
    rows_by_tactic = {
        tactic: [summarize_case(tactic, task_id, init_index) for task_id, init_index in cases]
        for tactic in TACTICS
    }
    baseline_tactic = "fp16" if "fp16" in rows_by_tactic else TACTICS[0]
    baseline_rows = rows_by_tactic.get(baseline_tactic, [])
    baseline_summary = summarize_tactic(baseline_rows, None)
    baseline_p50 = float(BASELINE_P50_MS) if BASELINE_P50_MS else baseline_summary.get("avg_server_p50_ms")
    tactic_summaries = [
        summarize_tactic(
            rows_by_tactic[tactic],
            baseline_p50 if BASELINE_P50_MS or tactic != baseline_tactic else None,
        )
        for tactic in TACTICS
    ]

    pairs = []
    for base_name in [baseline_tactic, "speed_only"]:
        if base_name not in rows_by_tactic:
            continue
        for tactic in TACTICS:
            if tactic == base_name:
                continue
            pairs.append(paired_counts(rows_by_tactic[base_name], rows_by_tactic[tactic]))

    return {
        "tag_prefix": TAG_PREFIX,
        "case_list": [f"{task}:{init}" for task, init in cases],
        "tactics": TACTICS,
        "runs": tactic_summaries,
        "pairs": pairs,
        "rows_by_tactic": rows_by_tactic,
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# {REPORT_TITLE}")
    lines.append("")
    lines.append("## 设置")
    lines.append("")
    lines.append(f"- tag prefix: `{summary['tag_prefix']}`")
    lines.append(f"- cases: `{','.join(summary['case_list'])}`")
    lines.append(f"- tactics: `{' '.join(summary['tactics'])}`")
    lines.append("- init 语义: `LIBERO_USE_BENCHMARK_INIT_STATES=1`，`SEED` 表示 LIBERO benchmark init index")
    lines.append("")
    lines.append("## 聚合结果")
    lines.append("")
    lines.append(
        "| Tactic | Complete | Success | Avg p50 ms | Avg mean ms | Speedup vs FP16 p50 | Avg requests |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run in summary["runs"]:
        success = f"{run['successes']}/{run['complete']}" if run["complete"] else "0/0"
        lines.append(
            "| "
            + f"{run['label']} | {run['complete']}/{run['expected']} | {success} | "
            + f"{fmt(run['avg_server_p50_ms'])} | {fmt(run['avg_server_mean_ms'])} | "
            + f"{fmt(run['speedup_vs_fp16_avg_p50'])} | {fmt(run['avg_request_count'])} |"
        )
    lines.append("")
    lines.append("## Paired Repair/Regression")
    lines.append("")
    lines.append("| Base | Candidate | Common | Repairs | Regressions | Net |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for pair in summary["pairs"]:
        lines.append(
            "| "
            + f"{pair['left']} | {pair['right']} | {pair['common']} | "
            + f"{len(pair['repaired'])} | {len(pair['regressed'])} | {pair['net']} |"
        )
    lines.append("")
    lines.append("## Case 明细")
    lines.append("")
    for tactic in summary["tactics"]:
        lines.append(f"### {tactic_label(tactic)}")
        lines.append("")
        lines.append("| Case | Success | Episode steps | Requests | p50 ms | mean ms | max ms |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in summary["rows_by_tactic"][tactic]:
            success = "-" if row["success"] is None else ("1" if row["success"] else "0")
            lines.append(
                "| "
                + f"{row['case']} | {success} | {fmt(row['episode_length'])} | "
                + f"{fmt(row['request_count'])} | {fmt(row['server_p50_ms'])} | "
                + f"{fmt(row['server_mean_ms'])} | {fmt(row['server_max_ms'])} |"
            )
        lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = build_summary()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(json.dumps({"summary": str(OUT_JSON), "report": str(OUT_MD)}, indent=2))


if __name__ == "__main__":
    main()
