"""Validate CLSG-TS v2 selectors on held-out Phase46 results.

This script is intentionally offline. It treats Phase43-45 as search/selection
data and Phase46 as an independent confirmation fold. The goal is to check
whether different tactic-selection policies would have generalized, without
running new simulator episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"

DEFAULT_TRAIN = [
    RESULTS / "phase43_n17_hybrid_bo_probe_10case_v1_summary.json",
    RESULTS / "phase44_n17_hybrid_heldout_15case_v1_summary.json",
    RESULTS / "phase45_n17_hybrid_alltask_stress_20case_v1_summary.json",
]
DEFAULT_TEST = RESULTS / "phase46_n17_alltask_confirmation_30case_v1_summary.json"
DEFAULT_OUT_JSON = RESULTS / "phase47_methodology_v2_validation_summary.json"
DEFAULT_OUT_MD = ROOT / "docs" / "phase47_methodology_v2_validation_report_zh.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(summary: dict[str, Any], tactic: str) -> list[dict[str, Any]]:
    return list((summary.get("rows_by_tactic") or {}).get(tactic) or [])


def fp16_rows(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("case")): row
        for row in rows(summary, "fp16")
        if row.get("status") == "complete" and row.get("case") is not None
    }


def tactic_stats(
    summary: dict[str, Any],
    tactic: str,
    *,
    task_id: int | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for row in rows(summary, tactic)
        if row.get("status") == "complete"
        and (task_id is None or int(row.get("task_id")) == int(task_id))
    ]
    baseline = fp16_rows(summary)
    successes = sum(1 for row in selected if row.get("success") is True)
    p50_values = [
        float(row["server_p50_ms"])
        for row in selected
        if row.get("server_p50_ms") is not None
    ]
    repairs = 0
    regressions = 0
    common = 0
    repaired_cases: list[str] = []
    regressed_cases: list[str] = []
    for row in selected:
        case = str(row.get("case"))
        base = baseline.get(case)
        if not base:
            continue
        common += 1
        if base.get("success") is False and row.get("success") is True:
            repairs += 1
            repaired_cases.append(case)
        elif base.get("success") is True and row.get("success") is False:
            regressions += 1
            regressed_cases.append(case)
    count = len(selected)
    return {
        "tactic": tactic,
        "task_id": task_id,
        "count": count,
        "successes": successes,
        "success_rate": round(successes / count, 6) if count else None,
        "avg_p50_ms": round(sum(p50_values) / len(p50_values), 6)
        if p50_values
        else None,
        "repairs": repairs,
        "regressions": regressions,
        "common": common,
        "repaired_cases": repaired_cases,
        "regressed_cases": regressed_cases,
    }


def aggregate_stats(
    summaries: list[dict[str, Any]],
    tactic: str,
    *,
    task_id: int | None = None,
) -> dict[str, Any]:
    total = 0
    successes = 0
    repairs = 0
    regressions = 0
    common = 0
    p50_values: list[float] = []
    fold_stats: list[dict[str, Any]] = []
    for summary in summaries:
        if tactic not in set(summary.get("tactics") or []):
            continue
        stat = tactic_stats(summary, tactic, task_id=task_id)
        if stat["count"] == 0:
            continue
        fold_stats.append(stat)
        total += int(stat["count"])
        successes += int(stat["successes"])
        repairs += int(stat["repairs"])
        regressions += int(stat["regressions"])
        common += int(stat["common"])
        if stat["avg_p50_ms"] is not None:
            p50_values.append(float(stat["avg_p50_ms"]))
    return {
        "tactic": tactic,
        "task_id": task_id,
        "count": total,
        "successes": successes,
        "success_rate": round(successes / total, 6) if total else None,
        "avg_p50_ms": round(sum(p50_values) / len(p50_values), 6)
        if p50_values
        else None,
        "repairs": repairs,
        "regressions": regressions,
        "common": common,
        "fold_count": len(fold_stats),
        "fold_stats": fold_stats,
    }


def sort_behavior_first(row: dict[str, Any]) -> tuple[Any, ...]:
    success_rate = float(row.get("success_rate") or 0.0)
    avg_p50 = float(row.get("avg_p50_ms") or 1e9)
    return (
        -success_rate,
        int(row.get("regressions") or 0),
        -int(row.get("repairs") or 0),
        avg_p50,
        str(row.get("tactic")),
    )


def evaluate_fixed(test: dict[str, Any], tactic: str, baseline_p50: float) -> dict[str, Any]:
    stat = tactic_stats(test, tactic)
    speedup = baseline_p50 / float(stat["avg_p50_ms"]) if stat["avg_p50_ms"] else None
    return {
        "policy": f"fixed:{tactic}",
        "choices": {"global": tactic},
        **stat,
        "speedup_vs_fp16_p50": round(speedup, 6) if speedup else None,
    }


def evaluate_task_policy(
    test: dict[str, Any],
    choices: dict[int, str],
    baseline_p50: float,
    *,
    policy_name: str,
) -> dict[str, Any]:
    baseline = fp16_rows(test)
    total = 0
    successes = 0
    repairs = 0
    regressions = 0
    p50_values: list[float] = []
    repaired_cases: list[str] = []
    regressed_cases: list[str] = []
    selected_cases: list[dict[str, Any]] = []
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted(choices):
        tactic = choices[task_id]
        task_rows = [
            row
            for row in rows(test, tactic)
            if row.get("status") == "complete" and int(row.get("task_id")) == int(task_id)
        ]
        task_successes = 0
        task_total = 0
        for row in task_rows:
            case = str(row.get("case"))
            task_total += 1
            total += 1
            success = row.get("success") is True
            task_successes += int(success)
            successes += int(success)
            if row.get("server_p50_ms") is not None:
                p50_values.append(float(row["server_p50_ms"]))
            base = baseline.get(case)
            if base and base.get("success") is False and success:
                repairs += 1
                repaired_cases.append(case)
            elif base and base.get("success") is True and not success:
                regressions += 1
                regressed_cases.append(case)
            selected_cases.append(
                {
                    "case": case,
                    "task_id": task_id,
                    "tactic": tactic,
                    "success": success,
                    "fp16_success": base.get("success") if base else None,
                    "server_p50_ms": row.get("server_p50_ms"),
                }
            )
        per_task[str(task_id)] = {
            "tactic": tactic,
            "successes": task_successes,
            "count": task_total,
        }
    avg_p50 = round(sum(p50_values) / len(p50_values), 6) if p50_values else None
    speedup = baseline_p50 / avg_p50 if avg_p50 else None
    return {
        "policy": policy_name,
        "choices": {str(task): tactic for task, tactic in sorted(choices.items())},
        "count": total,
        "successes": successes,
        "success_rate": round(successes / total, 6) if total else None,
        "avg_p50_ms": avg_p50,
        "speedup_vs_fp16_p50": round(speedup, 6) if speedup else None,
        "repairs": repairs,
        "regressions": regressions,
        "repaired_cases": repaired_cases,
        "regressed_cases": regressed_cases,
        "per_task": per_task,
        "selected_cases": selected_cases,
    }


def select_task_conditioned(
    train_summaries: list[dict[str, Any]],
    candidates: list[str],
    test: dict[str, Any],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    test_tasks = sorted({int(row["task_id"]) for row in rows(test, "fp16")})
    choices: dict[int, str] = {}
    ranking: list[dict[str, Any]] = []
    for task_id in test_tasks:
        task_rows = [
            aggregate_stats(train_summaries, tactic, task_id=task_id)
            for tactic in candidates
            if tactic != "fp16"
        ]
        task_rows = [row for row in task_rows if row["count"] > 0]
        task_rows.sort(key=sort_behavior_first)
        ranking.append(
            {
                "task_id": task_id,
                "selected": task_rows[0]["tactic"] if task_rows else None,
                "ranking": task_rows,
            }
        )
        if task_rows:
            choices[task_id] = str(task_rows[0]["tactic"])
    return choices, ranking


def select_oracle_task_policy(
    test: dict[str, Any],
    candidates: list[str],
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    test_tasks = sorted({int(row["task_id"]) for row in rows(test, "fp16")})
    choices: dict[int, str] = {}
    ranking: list[dict[str, Any]] = []
    for task_id in test_tasks:
        task_rows = [
            tactic_stats(test, tactic, task_id=task_id)
            for tactic in candidates
            if tactic != "fp16"
        ]
        task_rows = [row for row in task_rows if row["count"] > 0]
        task_rows.sort(key=sort_behavior_first)
        ranking.append(
            {
                "task_id": task_id,
                "selected": task_rows[0]["tactic"] if task_rows else None,
                "ranking": task_rows,
            }
        )
        if task_rows:
            choices[task_id] = str(task_rows[0]["tactic"])
    return choices, ranking


def format_rate(successes: int, count: int) -> str:
    return f"{successes}/{count}" if count else "-"


def make_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Phase47: CLSG-TS v2 方法论离线验证")
    lines.append("")
    lines.append("## 目的")
    lines.append("")
    lines.append(
        "Phase47 不重新跑仿真，而是把 Phase43-45 当作 tactic search / selection 数据，"
        "把 Phase46 当作独立 confirmation fold，检验几种选择策略是否能泛化。"
    )
    lines.append("")
    lines.append("核心问题不是哪个单点 tactic 最强，而是：")
    lines.append("")
    lines.append(
        "> 在 task-conditioned closed-loop sensitivity 下，global winner、"
        "task-conditioned selector 和 oracle upper bound 的差距有多大？"
    )
    lines.append("")
    lines.append("## 数据划分")
    lines.append("")
    lines.append("| role | summaries |")
    lines.append("| --- | --- |")
    lines.append(
        "| train/search | "
        + "<br>".join(f"`{path}`" for path in payload["train_paths"])
        + " |"
    )
    lines.append(f"| confirmation | `{payload['test_path']}` |")
    lines.append("")
    lines.append("候选 tactic 限制为 Phase46 已经实际跑过的集合，避免离线评估不存在的候选。")
    lines.append("")
    lines.append("```text")
    lines.append(" ".join(payload["candidate_tactics"]))
    lines.append("```")
    lines.append("")
    lines.append("## Phase43-45 global ranking")
    lines.append("")
    lines.append("| tactic | train success | repairs | regressions | avg p50 ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in payload["train_global_ranking"]:
        lines.append(
            f"| `{row['tactic']}` | {format_rate(row['successes'], row['count'])} | "
            f"{row['repairs']} | {row['regressions']} | {row['avg_p50_ms']:.2f} |"
        )
    lines.append("")
    lines.append(
        "按 behavior-first 规则，Phase43-45 会选择 "
        f"`{payload['selected_global_behavior_first']}`。"
    )
    lines.append("")
    lines.append("## Phase46 confirmation 结果")
    lines.append("")
    lines.append("| policy / tactic | success | p50 ms | speedup | repairs | regressions |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["test_fixed_results"]:
        speedup = row.get("speedup_vs_fp16_p50")
        speedup_text = "-" if speedup is None else f"{speedup:.2f}x"
        lines.append(
            f"| `{row['tactic']}` | {format_rate(row['successes'], row['count'])} | "
            f"{row['avg_p50_ms']:.2f} | {speedup_text} | "
            f"{row['repairs']} | {row['regressions']} |"
        )
    lines.append("")
    lines.append("## Selector 对比")
    lines.append("")
    lines.append("| selector | success | p50 ms | speedup | repairs | regressions | comment |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    comments = {
        "global_behavior_first_from_phase43_45": "train folds 的 global winner，Phase46 反转",
        "task_conditioned_behavior_first_from_phase43_45": "按 task 选择 tactic，有速度收益但仍需更稳 probe",
        "phase46_task_oracle_upper_bound": "不可部署 upper bound，只显示 task routing 的潜在空间",
    }
    for row in payload["policy_results"]:
        lines.append(
            f"| `{row['policy']}` | {format_rate(row['successes'], row['count'])} | "
            f"{row['avg_p50_ms']:.2f} | {row['speedup_vs_fp16_p50']:.2f}x | "
            f"{row['repairs']} | {row['regressions']} | "
            f"{comments.get(row['policy'], '')} |"
        )
    lines.append("")
    lines.append("## Task-conditioned selector 选择")
    lines.append("")
    lines.append("| task | selected tactic | train success | train regressions | Phase46 outcome |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    task_policy = next(
        row
        for row in payload["policy_results"]
        if row["policy"] == "task_conditioned_behavior_first_from_phase43_45"
    )
    task_outcomes = task_policy["per_task"]
    for item in payload["task_conditioned_ranking"]:
        task_id = str(item["task_id"])
        selected = item["selected"]
        train_row = item["ranking"][0] if item["ranking"] else {}
        outcome = task_outcomes.get(task_id, {})
        lines.append(
            f"| {task_id} | `{selected}` | "
            f"{format_rate(train_row.get('successes', 0), train_row.get('count', 0))} | "
            f"{train_row.get('regressions', 0)} | "
            f"{format_rate(outcome.get('successes', 0), outcome.get('count', 0))} |"
        )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append("1. Phase43-45 的 global behavior-first selector 会选择 `blocks0_3_window_2_12`，但它在 Phase46 上只有 `24/30`，说明 global local winner 不能直接写成可迁移 tactic。")
    lines.append("2. task-conditioned selector 在 Phase46 上达到 `27/30`、`1.08x`，和 `window_2_12` 的成功率相同但更快，并且明显优于 `speed_only` 的 regression profile。")
    lines.append("3. Phase46 的 task oracle upper bound 是 `29/30`、`1.14x`，说明 task routing 的解空间确实存在，但当前 train probe 还不足以稳定找准每个 task 的 tactic。")
    lines.append("4. 方法论应改写为：CLSG-TS/BO 是 candidate generator 和 task-aware validation protocol，不是 universal tactic oracle。")
    lines.append("5. 下一步迁移到新模型时，不应复用 N1.7 的固定 window；应复用流程：warm-start probe -> task/slice-conditioned selector -> held-out confirmation。")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="*", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    train_paths = [path if path.is_absolute() else ROOT / path for path in args.train]
    test_path = args.test if args.test.is_absolute() else ROOT / args.test
    train_summaries = [load_json(path) for path in train_paths]
    test = load_json(test_path)
    candidate_tactics = [
        tactic
        for tactic in test.get("tactics", [])
        if tactic == "fp16"
        or any(tactic in set(summary.get("tactics") or []) for summary in train_summaries)
    ]
    baseline_p50 = float(tactic_stats(test, "fp16")["avg_p50_ms"])

    train_global_ranking = [
        aggregate_stats(train_summaries, tactic)
        for tactic in candidate_tactics
        if tactic != "fp16"
    ]
    train_global_ranking = [row for row in train_global_ranking if row["count"] > 0]
    train_global_ranking.sort(key=sort_behavior_first)
    selected_global = str(train_global_ranking[0]["tactic"])

    fixed_results = [
        evaluate_fixed(test, tactic, baseline_p50)
        for tactic in candidate_tactics
    ]
    fixed_by_tactic = {row["tactic"]: row for row in fixed_results}

    task_choices, task_ranking = select_task_conditioned(
        train_summaries,
        candidate_tactics,
        test,
    )
    oracle_choices, oracle_ranking = select_oracle_task_policy(test, candidate_tactics)

    policy_results = [
        {
            **fixed_by_tactic[selected_global],
            "policy": "global_behavior_first_from_phase43_45",
        },
        evaluate_task_policy(
            test,
            task_choices,
            baseline_p50,
            policy_name="task_conditioned_behavior_first_from_phase43_45",
        ),
        evaluate_task_policy(
            test,
            oracle_choices,
            baseline_p50,
            policy_name="phase46_task_oracle_upper_bound",
        ),
    ]

    payload = {
        "train_paths": [str(path.relative_to(ROOT)) for path in train_paths],
        "test_path": str(test_path.relative_to(ROOT)),
        "candidate_tactics": candidate_tactics,
        "baseline_p50_ms": baseline_p50,
        "train_global_ranking": train_global_ranking,
        "selected_global_behavior_first": selected_global,
        "test_fixed_results": fixed_results,
        "task_conditioned_ranking": task_ranking,
        "task_conditioned_choices": {str(k): v for k, v in sorted(task_choices.items())},
        "oracle_task_ranking": oracle_ranking,
        "oracle_task_choices": {str(k): v for k, v in sorted(oracle_choices.items())},
        "policy_results": policy_results,
    }

    out_json = args.out_json if args.out_json.is_absolute() else ROOT / args.out_json
    out_md = args.out_md if args.out_md.is_absolute() else ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(make_markdown(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "out_md": str(out_md)}, indent=2))


if __name__ == "__main__":
    main()
