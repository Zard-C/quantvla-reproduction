"""Multi-fold robust tactic selection over closed-loop acceleration results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DEFAULT_FOLDS = [
    RESULTS / "phase33_combo_backtest_phase30_30case_v1_summary.json",
    RESULTS / "phase32_tactic_validation_30case_v1_summary.json",
]
TACTICS = [
    "speed_only",
    "window_0_120",
    "combo_blocks0_3_window_0_120",
]
SPEED_CONSTRAINT = 1.5


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def is_complete(run: dict[str, Any]) -> bool:
    return run.get("status") == "complete" and int(run.get("total_episodes") or 0) > 0


def episode_map(run: dict[str, Any]) -> dict[str, bool]:
    return {str(item["case"]): bool(item["success"]) for item in run.get("episodes", [])}


def pair_counts(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    bmap = episode_map(baseline)
    cmap = episode_map(candidate)
    common = sorted(set(bmap) & set(cmap), key=case_key)
    repaired = [case for case in common if not bmap[case] and cmap[case]]
    regressed = [case for case in common if bmap[case] and not cmap[case]]
    return {
        "common": len(common),
        "repaired": repaired,
        "regressed": regressed,
        "net": len(repaired) - len(regressed),
    }


def fold_name(summary: dict[str, Any], path: Path) -> str:
    phase = str(summary.get("phase") or path.stem)
    if "phase33" in phase:
        return "phase30_slice_backtest"
    if "phase32" in phase:
        return "phase32_slice"
    return phase


def summarize_fold(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    runs = [run for run in summary.get("runs", []) if is_complete(run)]
    by_name = {run["name"]: run for run in runs}
    baseline = by_name.get("baseline")
    if baseline is None:
        raise ValueError(f"{path} has no complete baseline")

    candidates: dict[str, dict[str, Any]] = {}
    for tactic in TACTICS:
        run = by_name.get(tactic)
        if run is None:
            continue
        pair = pair_counts(baseline, run)
        candidates[tactic] = {
            "tactic": tactic,
            "label": run.get("label", tactic),
            "successes": int(run.get("total_successes") or 0),
            "episodes": int(run.get("total_episodes") or 0),
            "success_rate": float(run.get("success_rate") or 0.0),
            "server_p50_ms": run.get("server_p50_ms"),
            "server_p90_ms": run.get("server_p90_ms"),
            "speedup": float(run.get("speedup_vs_baseline_server_p50") or 1.0),
            "repairs": pair["repaired"],
            "regressions": pair["regressed"],
            "net": pair["net"],
        }

    return {
        "name": fold_name(summary, path),
        "source": str(path.relative_to(ROOT)),
        "baseline": {
            "successes": int(baseline.get("total_successes") or 0),
            "episodes": int(baseline.get("total_episodes") or 0),
            "success_rate": float(baseline.get("success_rate") or 0.0),
            "server_p50_ms": baseline.get("server_p50_ms"),
        },
        "candidates": candidates,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def robust_score(item: dict[str, Any]) -> float:
    # Behavior-preserving score: success first, regression penalties second,
    # speed as a tie-breaker. This intentionally prefers low-regression
    # candidates over fastest candidates.
    return (
        100.0 * item["mean_success_rate"]
        + 20.0 * item["worst_success_rate"]
        + 2.0 * item["mean_speedup"]
        - 4.0 * item["total_regressions"]
        - 5.0 * item["worst_fold_regressions"]
    )


def speed_constrained_score(item: dict[str, Any], min_worst_speedup: float) -> float | None:
    if item["worst_speedup"] < min_worst_speedup:
        return None
    return robust_score(item)


def aggregate(folds: list[dict[str, Any]], min_worst_speedup: float) -> list[dict[str, Any]]:
    rows = []
    for tactic in TACTICS:
        present = []
        for fold in folds:
            candidate = fold["candidates"].get(tactic)
            if candidate:
                present.append((fold, candidate))
        if len(present) != len(folds):
            continue

        success_rates = [candidate["success_rate"] for _, candidate in present]
        speedups = [candidate["speedup"] for _, candidate in present]
        regressions = [len(candidate["regressions"]) for _, candidate in present]
        repairs = [len(candidate["repairs"]) for _, candidate in present]
        total_successes = sum(candidate["successes"] for _, candidate in present)
        total_episodes = sum(candidate["episodes"] for _, candidate in present)
        total_regressions = sum(regressions)
        total_repairs = sum(repairs)
        fold_rows = []
        for fold, candidate in present:
            fold_rows.append(
                {
                    "fold": fold["name"],
                    "success": f"{candidate['successes']}/{candidate['episodes']}",
                    "success_rate": candidate["success_rate"],
                    "speedup": candidate["speedup"],
                    "server_p50_ms": candidate["server_p50_ms"],
                    "repairs": len(candidate["repairs"]),
                    "regressions": len(candidate["regressions"]),
                    "net": candidate["net"],
                    "repair_cases": candidate["repairs"],
                    "regression_cases": candidate["regressions"],
                }
            )

        item = {
            "tactic": tactic,
            "label": present[0][1]["label"],
            "folds": fold_rows,
            "fold_count": len(present),
            "total_successes": total_successes,
            "total_episodes": total_episodes,
            "pooled_success_rate": total_successes / total_episodes,
            "mean_success_rate": mean(success_rates),
            "worst_success_rate": min(success_rates),
            "mean_speedup": mean(speedups),
            "worst_speedup": min(speedups),
            "total_repairs": total_repairs,
            "total_regressions": total_regressions,
            "worst_fold_regressions": max(regressions),
            "worst_fold_repairs": max(repairs),
        }
        item["robust_score"] = round(robust_score(item), 3)
        constrained = speed_constrained_score(item, min_worst_speedup)
        item["speed_constrained_score"] = None if constrained is None else round(constrained, 3)
        item["passes_speed_constraint"] = constrained is not None
        rows.append(item)

    rows.sort(key=lambda row: (-row["robust_score"], -row["mean_speedup"], row["tactic"]))
    for idx, item in enumerate(rows, start=1):
        item["robust_rank"] = idx

    constrained_rows = [row for row in rows if row["passes_speed_constraint"]]
    constrained_rows.sort(
        key=lambda row: (-row["speed_constrained_score"], -row["mean_speedup"], row["tactic"])
    )
    for idx, item in enumerate(constrained_rows, start=1):
        item["speed_constrained_rank"] = idx
    for item in rows:
        item.setdefault("speed_constrained_rank", None)
    return rows


def build_data(paths: list[Path], min_worst_speedup: float) -> dict[str, Any]:
    folds = [summarize_fold(path) for path in paths if path.exists()]
    tactics = aggregate(folds, min_worst_speedup)
    behavior_first = next((row for row in tactics if row.get("robust_rank") == 1), None)
    speed_constrained = next((row for row in tactics if row.get("speed_constrained_rank") == 1), None)
    return {
        "phase": "phase34_multifold_tactic_selection",
        "folds": folds,
        "tactics": tactics,
        "min_worst_speedup": min_worst_speedup,
        "decision": {
            "behavior_first": behavior_first["tactic"] if behavior_first else None,
            "speed_constrained": speed_constrained["tactic"] if speed_constrained else None,
            "interpretation": (
                "Use behavior-first ranking when matching FP16 outcomes is primary; "
                "use speed-constrained ranking when every fold must keep a minimum speedup."
            ),
        },
    }


def aggregate_rows(tactics: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for item in sorted(tactics, key=lambda row: row["robust_rank"]):
        rows.append(
            [
                item["robust_rank"],
                item["tactic"],
                f"{item['total_successes']}/{item['total_episodes']}",
                f"{item['mean_success_rate']:.3f}",
                f"{item['worst_success_rate']:.3f}",
                f"{item['mean_speedup']:.2f}x",
                f"{item['worst_speedup']:.2f}x",
                item["total_regressions"],
                item["worst_fold_regressions"],
                fmt(item["robust_score"]),
            ]
        )
    return rows


def speed_rows(tactics: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    candidates = [item for item in tactics if item["speed_constrained_rank"] is not None]
    candidates.sort(key=lambda row: row["speed_constrained_rank"])
    for item in candidates:
        rows.append(
            [
                item["speed_constrained_rank"],
                item["tactic"],
                f"{item['total_successes']}/{item['total_episodes']}",
                f"{item['worst_speedup']:.2f}x",
                item["total_regressions"],
                item["worst_fold_regressions"],
                fmt(item["speed_constrained_score"]),
            ]
        )
    return rows


def fold_rows(folds: list[dict[str, Any]], tactics: list[dict[str, Any]]) -> list[list[Any]]:
    by_tactic = {item["tactic"]: item for item in tactics}
    rows = []
    for fold in folds:
        for tactic in TACTICS:
            item = by_tactic.get(tactic)
            if not item:
                continue
            row = next((fold_row for fold_row in item["folds"] if fold_row["fold"] == fold["name"]), None)
            if not row:
                continue
            rows.append(
                [
                    fold["name"],
                    tactic,
                    row["success"],
                    f"{row['speedup']:.2f}x",
                    fmt(row["server_p50_ms"]),
                    row["repairs"],
                    row["regressions"],
                    row["net"],
                    ", ".join(row["regression_cases"]) if row["regression_cases"] else "-",
                ]
            )
    return rows


def build_report(data: dict[str, Any]) -> str:
    lines = [
        "# Phase 34: Multi-Fold Robust Tactic Selection",
        "",
        "Phase34 把 Phase30 slice 和 Phase32 slice 当作两个 validation folds，重新排序加速 tactic。目标不是找单个 held-out winner，而是在有限 rollout budget 下选择 worst-fold regression 更小的策略。",
        "",
        "## Folds",
        "",
        md_table(
            ["fold", "source", "baseline success", "baseline p50"],
            [
                [
                    fold["name"],
                    f"`{fold['source']}`",
                    f"{fold['baseline']['successes']}/{fold['baseline']['episodes']}",
                    fmt(fold["baseline"]["server_p50_ms"]),
                ]
                for fold in data["folds"]
            ],
        ),
        "",
        "## Per-Fold Outcomes",
        "",
        md_table(
            [
                "fold",
                "tactic",
                "success",
                "speedup",
                "p50 ms",
                "repair",
                "regress",
                "net",
                "regress cases",
            ],
            fold_rows(data["folds"], data["tactics"]),
        ),
        "",
        "## Behavior-First Ranking",
        "",
        "行为优先分数把 paired regression 作为强惩罚，速度只作为 tie-breaker。",
        "",
        md_table(
            [
                "rank",
                "tactic",
                "pooled success",
                "mean success",
                "worst success",
                "mean speedup",
                "worst speedup",
                "total regress",
                "worst regress",
                "score",
            ],
            aggregate_rows(data["tactics"]),
        ),
        "",
        "## Speed-Constrained Ranking",
        "",
        f"只保留 worst-fold speedup >= `{data['min_worst_speedup']:.2f}x` 的 tactic。",
        "",
        md_table(
            [
                "rank",
                "tactic",
                "pooled success",
                "worst speedup",
                "total regress",
                "worst regress",
                "score",
            ],
            speed_rows(data["tactics"]),
        ),
        "",
        "## Decision",
        "",
        f"- behavior-first selected: `{data['decision']['behavior_first']}`",
        f"- speed-constrained selected: `{data['decision']['speed_constrained']}`",
        f"- interpretation: {data['decision']['interpretation']}",
        "",
        "## 结论",
        "",
        "- `speed_only` 不能作为稳定策略：它在 Phase30 胜出，但 Phase32 出现 5 个 FP16 regression。",
        "- `combo_blocks0_3_window_0_120` 的总 regression 最低，但 Phase30 上 worst-fold speedup 只有约 `1.07x`，工程收益偏弱。",
        "- `window_0_120` 是当前更均衡的 speed-constrained candidate：两 fold 上都保持约 `>=1.7x` speedup，总 regression 比 speed-only 少。",
        "- 因此下一步不应继续单押一个固定 tactic，而应把 selection objective 写成多 fold robust search：先限制 worst-fold regression，再按 worst-fold speedup 选 trade-off。",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", action="append", type=Path, default=None)
    parser.add_argument("--min-worst-speedup", type=float, default=SPEED_CONSTRAINT)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS / "phase34_multifold_tactic_selection.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "phase34_multifold_tactic_selection_zh.md",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    args = parse_args()
    paths = [resolve(path) for path in (args.fold or DEFAULT_FOLDS)]
    data = build_data(paths, args.min_worst_speedup)
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
