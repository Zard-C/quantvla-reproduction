"""Closed-loop sensitivity-guided Bayesian tactic selector.

This is a lightweight dependency-free batch Bayesian optimization helper for
duration-window tactics. It treats a tactic ``window_s_e`` as a 2D window
parameter and uses prior rollout summaries to propose the next windows.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WINDOW_RE = re.compile(r"window_(\d+)_(\d+)$")


@dataclass(frozen=True)
class Observation:
    tactic: str
    start: int
    end: int
    successes: int
    complete: int
    success_rate: float
    speedup: float
    avg_p50_ms: float
    repairs: int
    regressions: int
    common: int
    behavior_score: float
    source: str


def parse_window(tactic: str) -> tuple[int, int] | None:
    if tactic == "speed_only":
        return (0, 0)
    match = WINDOW_RE.fullmatch(tactic)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_lookup(summary: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for pair in summary.get("pairs", []):
        if pair.get("left") != "fp16":
            continue
        right = pair.get("right")
        if not right:
            continue
        repaired = len(pair.get("repaired") or [])
        regressed = len(pair.get("regressed") or [])
        common = int(pair.get("common") or 0)
        out[right] = (repaired, regressed, common)
    return out


def baseline_reference(summaries: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], float | None]:
    for summary in summaries:
        rows = (summary.get("rows_by_tactic") or {}).get("fp16") or []
        baseline_rows = {
            str(row.get("case")): row
            for row in rows
            if row.get("case") is not None and row.get("success") is not None
        }
        p50_values = [
            float(row["server_p50_ms"])
            for row in rows
            if row.get("server_p50_ms") is not None
        ]
        if baseline_rows:
            baseline_p50 = sum(p50_values) / len(p50_values) if p50_values else None
            return baseline_rows, baseline_p50
    return {}, None


def paired_against_baseline(
    summary: dict[str, Any],
    tactic: str,
    baseline_rows: dict[str, dict[str, Any]],
) -> tuple[int, int, int] | None:
    rows = ((summary.get("rows_by_tactic") or {}).get(tactic)) or []
    if not rows or not baseline_rows:
        return None
    common = 0
    repairs = 0
    regressions = 0
    for row in rows:
        case = str(row.get("case"))
        if case not in baseline_rows:
            continue
        baseline_success = baseline_rows[case].get("success")
        candidate_success = row.get("success")
        if baseline_success is None or candidate_success is None:
            continue
        common += 1
        if baseline_success is False and candidate_success is True:
            repairs += 1
        elif baseline_success is True and candidate_success is False:
            regressions += 1
    if common == 0:
        return None
    return repairs, regressions, common


def behavior_score(
    success_rate: float,
    repairs: int,
    regressions: int,
    common: int,
    speedup: float,
    *,
    regression_penalty: float,
    repair_bonus: float,
    speed_bonus: float,
    speed_cap: float,
) -> float:
    denom = max(common, 1)
    capped_speed = min(max(speedup, 0.0), speed_cap) / max(speed_cap, 1e-9)
    return (
        success_rate
        - regression_penalty * (regressions / denom)
        + repair_bonus * (repairs / denom)
        + speed_bonus * capped_speed
    )


def collect_observations(
    summaries: list[Path],
    *,
    regression_penalty: float,
    repair_bonus: float,
    speed_bonus: float,
    speed_cap: float,
) -> list[Observation]:
    observations: list[Observation] = []
    seen: set[tuple[str, str]] = set()
    loaded = [(path, load_json(path)) for path in summaries]
    baseline_rows, baseline_p50 = baseline_reference([summary for _, summary in loaded])
    for path, summary in loaded:
        pairs = pair_lookup(summary)
        for run in summary.get("runs", []):
            tactic = run.get("tactic", "")
            bounds = parse_window(tactic)
            if bounds is None:
                continue
            complete = int(run.get("complete") or 0)
            if complete <= 0:
                continue
            successes = int(run.get("successes") or 0)
            success_rate = float(run.get("success_rate") or successes / complete)
            avg_p50 = float(run.get("avg_server_p50_ms") or 0.0)
            speedup = (
                float(baseline_p50) / avg_p50
                if baseline_p50 is not None and avg_p50 > 0
                else float(run.get("speedup_vs_fp16_avg_p50") or 1.0)
            )
            paired = paired_against_baseline(summary, tactic, baseline_rows)
            repairs, regressions, common = paired or pairs.get(tactic, (0, 0, complete))
            score = behavior_score(
                success_rate,
                repairs,
                regressions,
                common,
                speedup,
                regression_penalty=regression_penalty,
                repair_bonus=repair_bonus,
                speed_bonus=speed_bonus,
                speed_cap=speed_cap,
            )
            key = (path.name, tactic)
            if key in seen:
                continue
            seen.add(key)
            observations.append(
                Observation(
                    tactic=tactic,
                    start=bounds[0],
                    end=bounds[1],
                    successes=successes,
                    complete=complete,
                    success_rate=success_rate,
                    speedup=speedup,
                    avg_p50_ms=avg_p50,
                    repairs=repairs,
                    regressions=regressions,
                    common=common,
                    behavior_score=score,
                    source=str(path),
                )
            )
    return observations


def feature(start: int, end: int) -> list[float]:
    duration = max(0, end - start)
    center = start + duration / 2.0 if duration else 0.0
    return [start / 60.0, end / 60.0, duration / 60.0, center / 60.0]


def sqdist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def rbf(a: list[float], b: list[float], length_scale: float) -> float:
    return math.exp(-0.5 * sqdist(a, b) / max(length_scale * length_scale, 1e-9))


def cholesky(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    lower = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            value = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                lower[i][j] = math.sqrt(max(value, 1e-12))
            else:
                lower[i][j] = value / max(lower[j][j], 1e-12)
    return lower


def forward_subst(lower: list[list[float]], b: list[float]) -> list[float]:
    out: list[float] = []
    for i, row in enumerate(lower):
        value = b[i] - sum(row[j] * out[j] for j in range(i))
        out.append(value / max(row[i], 1e-12))
    return out


def backward_subst(lower: list[list[float]], b: list[float]) -> list[float]:
    n = len(lower)
    out = [0.0 for _ in range(n)]
    for i in reversed(range(n)):
        value = b[i] - sum(lower[j][i] * out[j] for j in range(i + 1, n))
        out[i] = value / max(lower[i][i], 1e-12)
    return out


class TinyGP:
    def __init__(
        self,
        xs: list[list[float]],
        ys: list[float],
        *,
        length_scale: float,
        noise: float,
    ) -> None:
        if not xs:
            raise ValueError("TinyGP requires at least one observation")
        self.xs = xs
        self.mean = sum(ys) / len(ys)
        variance = sum((y - self.mean) ** 2 for y in ys) / max(len(ys) - 1, 1)
        self.std = math.sqrt(max(variance, 1e-6))
        self.ys = [(y - self.mean) / self.std for y in ys]
        self.length_scale = length_scale
        n = len(xs)
        kernel = [[rbf(xs[i], xs[j], length_scale) for j in range(n)] for i in range(n)]
        for i in range(n):
            kernel[i][i] += noise
        self.lower = cholesky(kernel)
        tmp = forward_subst(self.lower, self.ys)
        self.alpha = backward_subst(self.lower, tmp)

    def predict(self, x: list[float]) -> tuple[float, float]:
        k = [rbf(x, train_x, self.length_scale) for train_x in self.xs]
        mean_z = sum(ki * ai for ki, ai in zip(k, self.alpha))
        v = forward_subst(self.lower, k)
        var_z = max(1.0 - sum(vi * vi for vi in v), 0.0)
        return self.mean + mean_z * self.std, math.sqrt(var_z) * self.std


def generate_candidates(max_end: int, step: int, durations: list[int]) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for start in range(0, max_end + 1, step):
        for duration in durations:
            end = start + duration
            if end <= max_end and start < end:
                candidates.add((start, end))
    # Add hand-useful off-grid local refinements around the known early boundary.
    for item in [(0, 8), (0, 10), (0, 12), (0, 15), (0, 18), (0, 25), (0, 30), (2, 12), (2, 15), (2, 20), (3, 18), (4, 20)]:
        if item[1] <= max_end:
            candidates.add(item)
    return sorted(candidates)


def distance_window(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def select_diverse(
    ranked: list[dict[str, Any]],
    chosen: list[dict[str, Any]],
    *,
    target: int,
    min_distance: int,
) -> None:
    for row in ranked:
        if len(chosen) >= target:
            return
        tactic = row["tactic"]
        if any(item["tactic"] == tactic for item in chosen):
            continue
        bounds = (row["start"], row["end"])
        if any(distance_window(bounds, (item["start"], item["end"])) < min_distance for item in chosen):
            continue
        chosen.append(row)
    for row in ranked:
        if len(chosen) >= target:
            return
        if not any(item["tactic"] == row["tactic"] for item in chosen):
            chosen.append(row)


def make_report(
    observations: list[Observation],
    ranked: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    lines.append("# CLSG-BO batch selector")
    lines.append("")
    lines.append("## 输入")
    lines.append("")
    for path in args.summary_json:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## 搜索约束")
    lines.append("")
    lines.append(
        f"- request-window search: `end <= {args.max_end}`, grid step `{args.grid_step}`, "
        f"durations `{args.durations}`"
    )
    lines.append("- 更长 horizon、layer islands、precision profiles 应作为下一阶段显式扩展。")
    lines.append("")
    lines.append("## 已观测窗口")
    lines.append("")
    lines.append("| tactic | success | speedup | repair | regress | behavior score |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for obs in sorted(observations, key=lambda item: (item.start, item.end)):
        lines.append(
            f"| `{obs.tactic}` | {obs.successes}/{obs.complete} | {obs.speedup:.2f} | "
            f"{obs.repairs} | {obs.regressions} | {obs.behavior_score:.3f} |"
        )
    lines.append("")
    lines.append("## 推荐下一批")
    lines.append("")
    lines.append("| slot | tactic | reason | predicted behavior | predicted speedup | acq |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: |")
    for idx, row in enumerate(selected, 1):
        lines.append(
            f"| {idx} | `{row['tactic']}` | {row['reason']} | "
            f"{row['behavior_mu']:.3f} +/- {row['behavior_sigma']:.3f} | "
            f"{row['speed_mu']:.2f} +/- {row['speed_sigma']:.2f} | {row['acq']:.3f} |"
        )
    lines.append("")
    lines.append("## 可直接运行的 TACTICS")
    lines.append("")
    lines.append("```text")
    lines.append(" ".join(row["tactic"] for row in selected))
    lines.append("```")
    lines.append("")
    lines.append("## Top candidates")
    lines.append("")
    lines.append("| tactic | reason | predicted behavior | predicted speedup | acq |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in ranked[: min(20, len(ranked))]:
        lines.append(
            f"| `{row['tactic']}` | {row['reason']} | "
            f"{row['behavior_mu']:.3f} +/- {row['behavior_sigma']:.3f} | "
            f"{row['speed_mu']:.2f} +/- {row['speed_sigma']:.2f} | {row['acq']:.3f} |"
        )
    lines.append("")
    lines.append("## 解释")
    lines.append("")
    lines.append(
        "这是 batch BO/active search，而不是最终 deployment selector。它把 Phase40 的窗口结果当作 warm-start，"
        "用一个小 RBF-GP 估计 behavior score 和 speedup，并同时保留 behavior-first、speed-constrained 和 exploration 候选。"
    )
    lines.append(
        "下一批跑完后，应把 Phase40 与 Phase41 的 summary 一起喂回本 selector，再决定是否进入 held-out validation。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("toy_quantvla/results/phase41_clsg_bo_selector.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/phase41_clsg_bo_selector_report_zh.md"))
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-end", type=int, default=30)
    parser.add_argument("--grid-step", type=int, default=2)
    parser.add_argument("--durations", default="5,8,10,12,15,18,20,25,30")
    parser.add_argument("--length-scale", type=float, default=0.22)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--kappa", type=float, default=0.75)
    parser.add_argument("--regression-penalty", type=float, default=0.35)
    parser.add_argument("--repair-bonus", type=float, default=0.10)
    parser.add_argument("--speed-bonus", type=float, default=0.05)
    parser.add_argument("--speed-cap", type=float, default=1.50)
    parser.add_argument("--min-speedup", type=float, default=1.20)
    args = parser.parse_args()

    observations = collect_observations(
        args.summary_json,
        regression_penalty=args.regression_penalty,
        repair_bonus=args.repair_bonus,
        speed_bonus=args.speed_bonus,
        speed_cap=args.speed_cap,
    )
    if len(observations) < 3:
        raise SystemExit("Need at least three observed tactics for BO selection")

    observed_bounds = {(obs.start, obs.end) for obs in observations}
    xs = [feature(obs.start, obs.end) for obs in observations]
    behavior_gp = TinyGP(
        xs,
        [obs.behavior_score for obs in observations],
        length_scale=args.length_scale,
        noise=args.noise,
    )
    speed_gp = TinyGP(xs, [obs.speedup for obs in observations], length_scale=args.length_scale, noise=args.noise)

    durations = [int(item) for item in args.durations.split(",") if item]
    candidates = [item for item in generate_candidates(args.max_end, args.grid_step, durations) if item not in observed_bounds]
    best_behavior = max(obs.behavior_score for obs in observations)
    ranked: list[dict[str, Any]] = []
    for start, end in candidates:
        b_mu, b_sigma = behavior_gp.predict(feature(start, end))
        s_mu, s_sigma = speed_gp.predict(feature(start, end))
        behavior_acq = b_mu + args.kappa * b_sigma + 0.04 * min(s_mu, args.speed_cap)
        feasible = b_mu + 0.5 * b_sigma >= best_behavior - 0.06
        speed_acq = (s_mu + 0.5 * s_sigma + 0.12 * b_mu) if feasible and s_mu >= args.min_speedup else -1e9
        explore_acq = b_sigma + 0.25 * s_sigma
        reason = "behavior_ucb"
        acq = behavior_acq
        if speed_acq > acq:
            reason = "speed_constrained"
            acq = speed_acq
        # Keep one arm of explicit exploration visible when uncertainty is high.
        if explore_acq > 0.55 and explore_acq + 0.75 > acq:
            reason = "explore_boundary"
            acq = explore_acq
        ranked.append(
            {
                "tactic": f"window_{start}_{end}",
                "start": start,
                "end": end,
                "behavior_mu": b_mu,
                "behavior_sigma": b_sigma,
                "speed_mu": s_mu,
                "speed_sigma": s_sigma,
                "behavior_acq": behavior_acq,
                "speed_acq": speed_acq,
                "explore_acq": explore_acq,
                "acq": acq,
                "reason": reason,
            }
        )
    ranked.sort(key=lambda row: row["acq"], reverse=True)

    selected: list[dict[str, Any]] = []
    behavior_ranked = sorted(ranked, key=lambda row: row["behavior_acq"], reverse=True)
    speed_ranked = sorted(ranked, key=lambda row: row["speed_acq"], reverse=True)
    explore_ranked = sorted(ranked, key=lambda row: row["explore_acq"], reverse=True)
    select_diverse(behavior_ranked, selected, target=max(2, args.batch_size // 3), min_distance=4)
    select_diverse(speed_ranked, selected, target=max(4, (2 * args.batch_size) // 3), min_distance=4)
    select_diverse(explore_ranked, selected, target=args.batch_size, min_distance=6)

    payload = {
        "summary_json": [str(path) for path in args.summary_json],
        "batch_size": args.batch_size,
        "observations": [obs.__dict__ for obs in observations],
        "selected_tactics": [row["tactic"] for row in selected],
        "selected": selected,
        "ranked": ranked,
        "objective": {
            "regression_penalty": args.regression_penalty,
            "repair_bonus": args.repair_bonus,
            "speed_bonus": args.speed_bonus,
            "speed_cap": args.speed_cap,
            "min_speedup": args.min_speedup,
        },
        "search_space": {
            "max_end": args.max_end,
            "grid_step": args.grid_step,
            "durations": durations,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report = make_report(observations, ranked, selected, args)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(report, encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "selected_tactics": payload["selected_tactics"]}, indent=2))


if __name__ == "__main__":
    main()
