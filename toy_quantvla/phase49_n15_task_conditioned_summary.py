"""Summarize Phase49 N1.5 task-conditioned tactic expansion."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase49_n15_task_conditioned_expansion_15case_v1")
CASE_LIST = os.environ.get("CASE_LIST", "")
TACTICS = os.environ.get(
    "TACTICS",
    "fp16 speed_only window_0_60 window_0_120 window_60_180 window_120_260 window_0_240 blocks0_3 combo_blocks0_3_window_0_120 combo_blocks0_3_window_120_260",
).split()
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260716"))
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase49_n15_task_conditioned_expansion_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase49: N1.5 Task-Conditioned Tactic Expansion")


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 2)


def bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024.0**2, 1)


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def tactic_label(tactic: str) -> str:
    if tactic == "fp16":
        return "FP16 baseline"
    if tactic == "speed_only":
        return "Speed-only compile"
    if tactic == "blocks0_3":
        return "Blocks0-3 eager island"
    match = re.fullmatch(r"window_(\d+)_(\d+)", tactic)
    if match:
        return f"Duration {match.group(1)}-{match.group(2)}"
    match = re.fullmatch(r"combo_blocks0_3_window_(\d+)_(\d+)", tactic)
    if match:
        return f"Blocks0-3 + duration {match.group(1)}-{match.group(2)}"
    return tactic


def tactic_policy(tactic: str) -> str:
    if tactic == "fp16":
        return "no compile"
    if tactic == "speed_only":
        return "compile action_head.model"
    if tactic == "blocks0_3":
        return "compile action_head.model with blocks0-3 eager"
    match = re.fullmatch(r"window_(\d+)_(\d+)", tactic)
    if match:
        return f"compile action_head.model; eager fallback for policy steps [{match.group(1)}, {match.group(2)})"
    match = re.fullmatch(r"combo_blocks0_3_window_(\d+)_(\d+)", tactic)
    if match:
        return (
            "compile action_head.model with blocks0-3 eager; "
            f"eager fallback for policy steps [{match.group(1)}, {match.group(2)})"
        )
    return tactic


def run_label(tactic: str) -> str:
    return "baseline" if tactic == "fp16" else "compiled"


def tag_for(tactic: str) -> str:
    return f"{TAG_PREFIX}_{tactic}"


def result_path(tactic: str, suffix: str) -> Path:
    return RESULTS / f"{tag_for(tactic)}_{run_label(tactic)}_{suffix}"


def metric_ms(container: dict[str, Any] | None, key: str) -> float | None:
    if not container:
        return None
    data = container.get("get_action_seconds") or container.get("latency_seconds") or {}
    return ms(data.get(key))


def episode_rows(client: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in client.get("episodes", []):
        latency = item.get("policy_latency_seconds", {})
        out.append(
            {
                "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
                "task_id": int(item["task_id"]),
                "init_index": int(item["init_index"]),
                "success": bool(item["success"]),
                "steps": int(latency.get("count", 0)),
                "p50_ms": ms(latency.get("p50")),
            }
        )
    return out


def eager_fraction(server: dict[str, Any] | None) -> float | None:
    fallback = (server or {}).get("extra", {}).get("compile_step_window_fallback") or {}
    compiled = fallback.get("compiled_requests")
    eager = fallback.get("eager_requests")
    if compiled is None or eager is None or compiled + eager <= 0:
        return None
    return round(eager / (compiled + eager), 3)


def summarize_tactic(tactic: str, baseline_p50_ms: float | None) -> dict[str, Any]:
    client = read_json(result_path(tactic, "client_latency.json"))
    server = read_json(result_path(tactic, "server_latency.json"))
    prepare = read_json(result_path(tactic, "server_prepare.json"))
    label = run_label(tactic)
    if client is None:
        return {
            "name": tactic,
            "label": tactic_label(tactic),
            "policy": tactic_policy(tactic),
            "status": "missing",
            "client_source": str(result_path(tactic, "client_latency.json").relative_to(ROOT)),
            "server_latency_source": str(result_path(tactic, "server_latency.json").relative_to(ROOT)),
            "prepare_source": str(result_path(tactic, "server_prepare.json").relative_to(ROOT)),
            "run_label": label,
        }

    server_memory = (server or {}).get("extra", {}).get("server_memory", {})
    p50_ms = metric_ms(server, "p50")
    p90_ms = metric_ms(server, "p90")
    speedup = round(baseline_p50_ms / p50_ms, 2) if baseline_p50_ms and p50_ms else None
    episodes = episode_rows(client)
    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "total": 0})
    for episode in episodes:
        bucket = per_task[str(episode["task_id"])]
        bucket["total"] += 1
        bucket["successes"] += int(episode["success"])

    return {
        "name": tactic,
        "label": tactic_label(tactic),
        "policy": tactic_policy(tactic),
        "status": "complete",
        "run_label": label,
        "client_source": str(result_path(tactic, "client_latency.json").relative_to(ROOT)),
        "server_latency_source": str(result_path(tactic, "server_latency.json").relative_to(ROOT)),
        "prepare_source": str(result_path(tactic, "server_prepare.json").relative_to(ROOT)),
        "total_successes": int(client.get("total_successes", 0)),
        "total_episodes": int(client.get("total_episodes", 0)),
        "success_rate": float(client.get("success_rate", 0.0)),
        "server_p50_ms": p50_ms,
        "server_p90_ms": p90_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "server_reserved_mib": bytes_to_mib(server_memory.get("reserved_bytes")),
        "server_max_reserved_mib": bytes_to_mib(server_memory.get("max_reserved_bytes")),
        "eager_fraction": eager_fraction(server),
        "server_extra": (server or {}).get("extra", {}),
        "episodes": episodes,
        "per_task": dict(per_task),
    }


def pair_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left.get("status") != "complete" or right.get("status") != "complete":
        return None
    lmap = {ep["case"]: ep for ep in left["episodes"]}
    rmap = {ep["case"]: ep for ep in right["episodes"]}
    common = sorted(set(lmap) & set(rmap), key=case_key)
    repaired = [case for case in common if not lmap[case]["success"] and rmap[case]["success"]]
    regressed = [case for case in common if lmap[case]["success"] and not rmap[case]["success"]]
    return {
        "left": left["name"],
        "right": right["name"],
        "common": len(common),
        "repaired": repaired,
        "regressed": regressed,
        "net": len(repaired) - len(regressed),
    }


def select_task_winners(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted(
        {
            int(task)
            for run in runs
            for task in (run.get("per_task") or {}).keys()
            if run.get("status") == "complete"
        }
    )
    out: list[dict[str, Any]] = []
    for task in tasks:
        candidates = []
        for run in runs:
            if run.get("status") != "complete" or run["name"] == "fp16":
                continue
            bucket = (run.get("per_task") or {}).get(str(task))
            if not bucket:
                continue
            total = int(bucket["total"])
            successes = int(bucket["successes"])
            candidates.append(
                {
                    "task_id": task,
                    "name": run["name"],
                    "successes": successes,
                    "total": total,
                    "success_rate": successes / total if total else 0.0,
                    "server_p50_ms": run.get("server_p50_ms"),
                    "speedup_vs_baseline_server_p50": run.get("speedup_vs_baseline_server_p50"),
                }
            )
        candidates.sort(
            key=lambda row: (
                -row["success_rate"],
                -(row.get("speedup_vs_baseline_server_p50") or 0.0),
                row.get("server_p50_ms") or 1e9,
                row["name"],
            )
        )
        if candidates:
            out.append({"task_id": task, "selected": candidates[0]["name"], "ranking": candidates})
    return out


def build_summary() -> dict[str, Any]:
    baseline_server = read_json(result_path("fp16", "server_latency.json"))
    baseline_p50_ms = metric_ms(baseline_server, "p50")
    runs = [summarize_tactic(tactic, baseline_p50_ms) for tactic in TACTICS]
    by_name = {run["name"]: run for run in runs}
    baseline = by_name.get("fp16")

    pairs = []
    if baseline:
        for tactic in TACTICS:
            if tactic == "fp16":
                continue
            pair = pair_counts(baseline, by_name.get(tactic, {}))
            if pair:
                pairs.append(pair)
    speed = by_name.get("speed_only")
    if speed:
        for tactic in TACTICS:
            if tactic in {"fp16", "speed_only"}:
                continue
            pair = pair_counts(speed, by_name.get(tactic, {}))
            if pair:
                pairs.append(pair)

    return {
        "phase": "phase49_n15_task_conditioned_expansion",
        "tag_prefix": TAG_PREFIX,
        "case_list": CASE_LIST,
        "tactics": TACTICS,
        "policy_seed_base": POLICY_SEED_BASE,
        "baseline_server_p50_ms": baseline_p50_ms,
        "runs": runs,
        "paired_comparisons": pairs,
        "task_winners": select_task_winners(runs),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def fmt_rate(successes: Any, total: Any) -> str:
    if successes is None or total is None:
        return "-"
    return f"{successes}/{total}"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_markdown(summary: dict[str, Any]) -> None:
    runs = summary["runs"]
    complete_runs = [run for run in runs if run.get("status") == "complete"]
    lines: list[str] = []
    lines.append(f"# {REPORT_TITLE}")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append(
        "Phase49 在 N1.5 checkpoint 上围绕 task 4/6/8 扩展 tactic pool，"
        "用于检查 Phase48 中出现的 task-conditioned repair profile 是否稳定。"
    )
    lines.append("")
    lines.append("## 设置")
    lines.append("")
    lines.append(f"- tag prefix: `{summary['tag_prefix']}`")
    lines.append(f"- cases: `{summary['case_list']}`")
    lines.append(f"- tactics: `{' '.join(summary['tactics'])}`")
    lines.append(f"- deterministic policy seed base: `{summary['policy_seed_base']}`")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    lines.append(
        md_table(
            ["run", "policy", "status", "success", "p50 ms", "speedup", "eager frac"],
            [
                [
                    run["label"],
                    run["policy"],
                    run["status"],
                    fmt_rate(run.get("total_successes"), run.get("total_episodes"))
                    if run.get("status") == "complete"
                    else "missing",
                    fmt(run.get("server_p50_ms")),
                    f"{run.get('speedup_vs_baseline_server_p50'):.2f}x"
                    if run.get("speedup_vs_baseline_server_p50") is not None
                    else "-",
                    fmt(run.get("eager_fraction")),
                ]
                for run in runs
            ],
        )
    )
    lines.append("")
    lines.append("## Paired repair/regression vs FP16")
    lines.append("")
    fp16_pairs = [pair for pair in summary["paired_comparisons"] if pair["left"] == "fp16"]
    lines.append(
        md_table(
            ["candidate", "common", "repair", "regress", "net", "repair cases", "regress cases"],
            [
                [
                    pair["right"],
                    pair["common"],
                    len(pair["repaired"]),
                    len(pair["regressed"]),
                    pair["net"],
                    ", ".join(pair["repaired"]) or "-",
                    ", ".join(pair["regressed"]) or "-",
                ]
                for pair in fp16_pairs
            ],
        )
    )
    lines.append("")
    lines.append("## Per-task success")
    lines.append("")
    tasks = sorted(
        {
            int(task)
            for run in complete_runs
            for task in (run.get("per_task") or {}).keys()
        }
    )
    lines.append(
        md_table(
            ["run"] + [f"task {task}" for task in tasks],
            [
                [run["name"]]
                + [
                    fmt_rate(
                        (run.get("per_task") or {}).get(str(task), {}).get("successes"),
                        (run.get("per_task") or {}).get(str(task), {}).get("total"),
                    )
                    for task in tasks
                ]
                for run in complete_runs
            ],
        )
    )
    lines.append("")
    lines.append("## Task-conditioned winners")
    lines.append("")
    lines.append(
        md_table(
            ["task", "selected", "success", "speedup", "top candidates"],
            [
                [
                    item["task_id"],
                    item["selected"],
                    fmt_rate(item["ranking"][0]["successes"], item["ranking"][0]["total"]),
                    f"{item['ranking'][0].get('speedup_vs_baseline_server_p50'):.2f}x"
                    if item["ranking"][0].get("speedup_vs_baseline_server_p50") is not None
                    else "-",
                    "; ".join(
                        f"{row['name']}={fmt_rate(row['successes'], row['total'])}"
                        for row in item["ranking"][:4]
                    ),
                ]
                for item in summary["task_winners"]
            ],
        )
    )
    lines.append("")
    lines.append("## 判读重点")
    lines.append("")
    lines.append("1. 如果 task 4、6、8 选择出不同 winner，说明 N1.5 上 task-conditioned tactic policy 的必要性增强。")
    lines.append("2. 如果 task 6 仍然全失败，说明当前 duration/layer candidate pool 不足，需要转向更晚窗口、动作通道扰动或不加速 fallback。")
    lines.append("3. 如果某个 tactic aggregate 最好但在某个 task 上 regression 明显，论文应继续强调 paired/task-wise validation，而不是 aggregate success。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append(f"- JSON: `{display_path(OUT_JSON)}`")
    lines.append(f"- Report: `{display_path(OUT_MD)}`")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = build_summary()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(f"Wrote {display_path(OUT_JSON)}")
    print(f"Wrote {display_path(OUT_MD)}")


if __name__ == "__main__":
    main()
