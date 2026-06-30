"""Retrospective scoring for sensitivity-guided tactic search.

This script does not run rollouts. It consumes existing phase summary JSON files
and turns them into a compact tactic-search report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DEFAULT_SUMMARIES = [
    RESULTS / "phase28A_proxy_guided_15case_v1_summary.json",
    RESULTS / "phase28B_proxy_guided_33case_v1_summary.json",
    RESULTS / "phase28D_duration_window_33case_v1_summary.json",
    RESULTS / "phase29_finer_duration_proxy_33case_v1_summary.json",
    RESULTS / "phase30_heldout_sanity_30case_v1_summary.json",
]
DEFAULT_PROBE = "phase29_finer_duration_proxy"
DEFAULT_HELDOUT = "phase30_heldout_sanity"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def case_sort_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def phase_id(summary: dict[str, Any], path: Path) -> str:
    return str(summary.get("phase") or path.stem)


def is_complete(run: dict[str, Any]) -> bool:
    return run.get("status") == "complete" and int(run.get("total_episodes") or 0) > 0


def episode_map(run: dict[str, Any]) -> dict[str, bool]:
    return {str(item["case"]): bool(item["success"]) for item in run.get("episodes", [])}


def pair_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if not is_complete(left) or not is_complete(right):
        return None
    left_map = episode_map(left)
    right_map = episode_map(right)
    common = sorted(set(left_map) & set(right_map), key=case_sort_key)
    if not common:
        return None
    repaired = [case for case in common if not left_map[case] and right_map[case]]
    regressed = [case for case in common if left_map[case] and not right_map[case]]
    same_success = [case for case in common if left_map[case] and right_map[case]]
    same_failure = [case for case in common if not left_map[case] and not right_map[case]]
    return {
        "common": len(common),
        "repaired": repaired,
        "regressed": regressed,
        "same_success": same_success,
        "same_failure": same_failure,
        "net": len(repaired) - len(regressed),
    }


def worst_task_success_rate(run: dict[str, Any]) -> float | None:
    buckets: dict[int, list[bool]] = defaultdict(list)
    for item in run.get("episodes", []):
        buckets[int(item["task_id"])].append(bool(item["success"]))
    if not buckets:
        return None
    return min(sum(items) / len(items) for items in buckets.values())


def family_for(run: dict[str, Any]) -> str:
    name = str(run.get("name", ""))
    policy = str(run.get("policy", "")).lower()
    if name == "speed_only":
        return "compile"
    if "window" in name or "fallback" in policy:
        return "duration"
    if "block" in name or "block" in policy:
        return "layer"
    if "random" in name:
        return "sanity"
    return "other"


def speedup_for(run: dict[str, Any], baseline: dict[str, Any]) -> float:
    explicit = run.get("speedup_vs_baseline_server_p50")
    if explicit is not None:
        return float(explicit)
    baseline_p50 = baseline.get("server_p50_ms")
    run_p50 = run.get("server_p50_ms")
    if baseline_p50 and run_p50:
        return float(baseline_p50) / float(run_p50)
    return 1.0


def score_run(
    run: dict[str, Any],
    baseline: dict[str, Any],
    speed_only: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_pair = pair_counts(baseline, run)
    speed_pair = pair_counts(speed_only, run) if speed_only and speed_only is not run else None
    success_rate = float(run.get("success_rate") or 0.0)
    speedup = speedup_for(run, baseline)
    worst_task = worst_task_success_rate(run)
    regress_baseline = len((baseline_pair or {}).get("regressed", []))
    repair_baseline = len((baseline_pair or {}).get("repaired", []))
    regress_speed = len((speed_pair or {}).get("regressed", []))
    repair_speed = len((speed_pair or {}).get("repaired", []))
    baseline_gap = max(0.0, float(baseline.get("success_rate") or 0.0) - success_rate)
    worst_gap = max(0.0, 1.0 - float(worst_task if worst_task is not None else success_rate))

    # Success dominates, speed breaks ties, and regressions/worst-slice risk
    # keep the search from hiding fragile-task failures behind average speed.
    score = (
        100.0 * success_rate
        + 4.0 * speedup
        + 1.0 * repair_baseline
        - 2.5 * regress_baseline
        - 10.0 * baseline_gap
        - 4.0 * worst_gap
    )
    if speed_pair:
        score += 0.5 * repair_speed - 1.0 * regress_speed

    return {
        "name": run["name"],
        "label": run.get("label", run["name"]),
        "family": family_for(run),
        "policy": run.get("policy"),
        "total_successes": int(run.get("total_successes") or 0),
        "total_episodes": int(run.get("total_episodes") or 0),
        "success_rate": success_rate,
        "server_p50_ms": run.get("server_p50_ms"),
        "server_p90_ms": run.get("server_p90_ms"),
        "speedup": round(speedup, 3),
        "worst_task_success_rate": worst_task,
        "repair_vs_baseline": repair_baseline,
        "regress_vs_baseline": regress_baseline,
        "net_vs_baseline": (baseline_pair or {}).get("net"),
        "repair_vs_speed_only": repair_speed if speed_pair else None,
        "regress_vs_speed_only": regress_speed if speed_pair else None,
        "net_vs_speed_only": (speed_pair or {}).get("net") if speed_pair else None,
        "score": round(score, 3),
    }


def rank_source(summary: dict[str, Any], path: Path) -> dict[str, Any]:
    runs = [run for run in summary.get("runs", []) if is_complete(run)]
    by_name = {run["name"]: run for run in runs}
    baseline = by_name.get("baseline")
    if baseline is None:
        raise ValueError(f"{path} has no complete baseline run")
    speed_only = by_name.get("speed_only")
    candidate_runs = [run for run in runs if run.get("name") != "baseline"]
    candidates = [score_run(run, baseline, speed_only) for run in candidate_runs]
    candidates.sort(key=lambda row: (-row["score"], row["server_p50_ms"] or 1e9, row["name"]))
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
    phase = phase_id(summary, path)
    tag_prefix = summary.get("tag_prefix")
    return {
        "phase": phase,
        "display_name": f"{phase} ({tag_prefix})" if tag_prefix else phase,
        "source": str(path.relative_to(ROOT)),
        "tag_prefix": tag_prefix,
        "case_list": summary.get("case_list"),
        "total_cases": int(baseline.get("total_episodes") or 0),
        "baseline": {
            "success": f"{baseline.get('total_successes')}/{baseline.get('total_episodes')}",
            "success_rate": baseline.get("success_rate"),
            "server_p50_ms": baseline.get("server_p50_ms"),
        },
        "candidates": candidates,
    }


def contains_phase(source: dict[str, Any], key: str) -> bool:
    haystack = f"{source.get('phase', '')} {source.get('source', '')} {source.get('tag_prefix', '')}"
    return key in haystack


def build_data(paths: list[Path], probe_key: str, heldout_key: str) -> dict[str, Any]:
    sources = []
    for path in paths:
        if not path.exists():
            continue
        sources.append(rank_source(read_json(path), path))

    probe = next((src for src in sources if contains_phase(src, probe_key)), None)
    heldout = next((src for src in sources if contains_phase(src, heldout_key)), None)

    transfer: list[dict[str, Any]] = []
    if probe and heldout:
        heldout_by_name = {row["name"]: row for row in heldout["candidates"]}
        heldout_rank = {row["name"]: idx for idx, row in enumerate(heldout["candidates"], start=1)}
        for probe_row in probe["candidates"]:
            heldout_row = heldout_by_name.get(probe_row["name"])
            if heldout_row is None:
                continue
            transfer.append(
                {
                    "name": probe_row["name"],
                    "probe_rank": probe_row["rank"],
                    "probe_score": probe_row["score"],
                    "probe_success": f"{probe_row['total_successes']}/{probe_row['total_episodes']}",
                    "probe_speedup": probe_row["speedup"],
                    "heldout_rank": heldout_rank[probe_row["name"]],
                    "heldout_score": heldout_row["score"],
                    "heldout_success": f"{heldout_row['total_successes']}/{heldout_row['total_episodes']}",
                    "heldout_speedup": heldout_row["speedup"],
                    "heldout_net_vs_baseline": heldout_row["net_vs_baseline"],
                }
            )

    selected_probe = probe["candidates"][0] if probe and probe.get("candidates") else None
    selected_heldout = heldout["candidates"][0] if heldout and heldout.get("candidates") else None
    decision = {
        "probe_selected": selected_probe["name"] if selected_probe else None,
        "heldout_selected": selected_heldout["name"] if selected_heldout else None,
        "current_incumbent": selected_heldout["name"] if selected_heldout else selected_probe["name"] if selected_probe else None,
        "interpretation": (
            "Probe search selects window_0_120, but held-out validation selects speed_only. "
            "Treat fixed duration windows as diagnostics unless they survive held-out validation."
        )
        if selected_probe and selected_heldout and selected_probe["name"] != selected_heldout["name"]
        else "Probe and held-out agree on the selected tactic.",
    }

    return {
        "phase": "phase31_tactic_search_retrospective",
        "scoring_rule": {
            "score": "100*success_rate + 4*speedup + repair_vs_baseline - 2.5*regress_vs_baseline - 10*baseline_success_gap - 4*worst_task_gap, with a light speed-only regression penalty when available",
            "priority": "closed-loop success first, speed second, regressions and worst-slice risk as penalties",
        },
        "probe_key": probe_key,
        "heldout_key": heldout_key,
        "sources": sources,
        "transfer": transfer,
        "decision": decision,
    }


def top_rows(source: dict[str, Any], limit: int = 8) -> list[list[Any]]:
    rows = []
    for row in source.get("candidates", [])[:limit]:
        rows.append(
            [
                row["rank"],
                row["name"],
                row["family"],
                f"{row['total_successes']}/{row['total_episodes']}",
                fmt(row["server_p50_ms"]),
                f"{row['speedup']:.2f}x",
                fmt(row["worst_task_success_rate"]),
                row["net_vs_baseline"],
                row["regress_vs_baseline"],
                fmt(row["score"]),
            ]
        )
    return rows


def build_report(data: dict[str, Any]) -> str:
    lines = [
        "# Phase 31: Retrospective Tactic Search",
        "",
        "Phase31 把前面阶段的经验整理成一个轻量的 tactic-search prototype。这里不重新跑 GPU，而是复用已有 closed-loop summary，检查一个打分流程会如何选择候选，以及 probe 上的选择能否通过 held-out validation。",
        "",
        "## Scoring",
        "",
        f"- rule: `{data['scoring_rule']['score']}`",
        f"- priority: {data['scoring_rule']['priority']}",
        "",
        "解释上要注意：这个分数不是理论最优目标，只是一个启发式排序器。它的作用是把 rollout budget 集中到少数候选上，然后必须用 held-out closed-loop validation 验收。",
        "",
    ]

    for source in data["sources"]:
        lines += [
            f"## {source['display_name']}",
            "",
            f"- source: `{source['source']}`",
            f"- baseline: {source['baseline']['success']} at p50 {fmt(source['baseline']['server_p50_ms'])} ms",
            "",
            md_table(
                [
                    "rank",
                    "candidate",
                    "family",
                    "success",
                    "p50 ms",
                    "speedup",
                    "worst task",
                    "net vs base",
                    "regress",
                    "score",
                ],
                top_rows(source),
            ),
            "",
        ]

    if data.get("transfer"):
        lines += [
            "## Probe To Held-Out Transfer",
            "",
            md_table(
                [
                    "candidate",
                    "probe rank",
                    "probe success",
                    "probe speedup",
                    "held-out rank",
                    "held-out success",
                    "held-out speedup",
                    "held-out net",
                ],
                [
                    [
                        row["name"],
                        row["probe_rank"],
                        row["probe_success"],
                        f"{row['probe_speedup']:.2f}x",
                        row["heldout_rank"],
                        row["heldout_success"],
                        f"{row['heldout_speedup']:.2f}x",
                        row["heldout_net_vs_baseline"],
                    ]
                    for row in data["transfer"]
                ],
            ),
            "",
        ]

    decision = data["decision"]
    lines += [
        "## Decision",
        "",
        f"- probe selected: `{decision['probe_selected']}`",
        f"- held-out selected: `{decision['heldout_selected']}`",
        f"- current incumbent: `{decision['current_incumbent']}`",
        f"- interpretation: {decision['interpretation']}",
        "",
        "## 结论",
        "",
        "- `window_0_120` 在 Phase29 probe set 上是合理候选，但 Phase30 held-out 没有通过验证，因此不能写成 universal tactic。",
        "- `speed_only` 在 Phase30 上成为当前 incumbent：它同时给出最高成功率和最高/接近最高的速度收益。",
        "- 这支持论文主线：VLA/world-action 模型的加速应该被表述为 closed-loop tactic search，而不是一次性选择固定窗口或固定层保护。",
        "- 下一轮 GPU 工作应该优先扩大候选集合和 probe/held-out 划分，而不是继续为 `0-120` 做事后解释。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        type=Path,
        default=None,
        help="Summary JSON to include. Can be passed multiple times.",
    )
    parser.add_argument("--probe-key", default=DEFAULT_PROBE)
    parser.add_argument("--heldout-key", default=DEFAULT_HELDOUT)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS / "phase31_tactic_search_retrospective.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "phase31_tactic_search_retrospective_zh.md",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    paths = [resolve(path) for path in (args.summary or DEFAULT_SUMMARIES)]
    data = build_data(paths, args.probe_key, args.heldout_key)

    out_json = resolve(args.out_json)
    out_md = resolve(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(f"Wrote {out_md.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
