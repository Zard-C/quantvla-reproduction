"""Select follow-up tactics from a matched rollout summary.

The selector is intentionally simple: keep anchor tactics, then add the best
hybrid candidates according to paired regressions, success rate, and speed.
It is used to queue held-out follow-ups without leaving the GPU idle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_by_right(summary: dict[str, Any], *, left: str = "fp16") -> dict[str, dict[str, Any]]:
    return {
        str(pair.get("right")): pair
        for pair in summary.get("pairs", [])
        if pair.get("left") == left and pair.get("right")
    }


def unique_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def tactic_row(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("tactic")): row for row in summary.get("runs", []) if row.get("tactic")}


def score_candidates(
    summary: dict[str, Any],
    *,
    base_tactics: list[str],
    candidate_prefix: str,
    min_speedup: float,
) -> list[dict[str, Any]]:
    rows = tactic_row(summary)
    pairs = pair_by_right(summary)
    base = set(base_tactics)
    scored: list[dict[str, Any]] = []
    for tactic, row in rows.items():
        if tactic in base or not tactic.startswith(candidate_prefix):
            continue
        pair = pairs.get(tactic, {})
        regressions = len(pair.get("regressed") or [])
        repairs = len(pair.get("repaired") or [])
        success_rate = float(row.get("success_rate") or 0.0)
        speedup = float(row.get("speedup_vs_fp16_avg_p50") or 1.0)
        success = int(row.get("successes") or 0)
        complete = int(row.get("complete") or 0)
        scored.append(
            {
                "tactic": tactic,
                "success": success,
                "complete": complete,
                "success_rate": success_rate,
                "speedup": speedup,
                "repairs": repairs,
                "regressions": regressions,
                "speed_feasible": speedup >= min_speedup,
            }
        )
    scored.sort(
        key=lambda row: (
            row["regressions"],
            not row["speed_feasible"],
            -row["success_rate"],
            -row["speedup"],
            -row["repairs"],
            row["tactic"],
        )
    )
    return scored


def make_report(
    summary_path: Path,
    selected: list[str],
    scored: list[dict[str, Any]],
    *,
    base_tactics: list[str],
    max_tactics: int,
    min_speedup: float,
) -> str:
    lines: list[str] = []
    lines.append("# Follow-up tactic selection")
    lines.append("")
    lines.append(f"- summary: `{summary_path}`")
    lines.append(f"- base tactics: `{' '.join(base_tactics)}`")
    lines.append(f"- max tactics: `{max_tactics}`")
    lines.append(f"- min speedup preference: `{min_speedup:.2f}x`")
    lines.append("")
    lines.append("## Selected")
    lines.append("")
    lines.append("```text")
    lines.append(" ".join(selected))
    lines.append("```")
    lines.append("")
    lines.append("## Candidate ranking")
    lines.append("")
    lines.append("| tactic | success | speedup | repairs | regressions | speed feasible |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in scored:
        lines.append(
            f"| `{row['tactic']}` | {row['success']}/{row['complete']} | "
            f"{row['speedup']:.2f}x | {row['repairs']} | {row['regressions']} | "
            f"{'yes' if row['speed_feasible'] else 'no'} |"
        )
    lines.append("")
    lines.append(
        "Selection rule: keep anchors, then add hybrid candidates with fewer paired regressions first; "
        "within the same regression count prefer speed-feasible, higher-success, faster tactics."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--base-tactics", default="fp16,speed_only,window_0_20,window_2_12")
    parser.add_argument("--candidate-prefix", default="blocks")
    parser.add_argument("--max-tactics", type=int, default=7)
    parser.add_argument("--min-speedup", type=float, default=1.20)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    summary = load_json(args.summary_json)
    rows = tactic_row(summary)
    base_tactics = [item for item in args.base_tactics.split(",") if item]
    selected = [item for item in base_tactics if item in rows]
    scored = score_candidates(
        summary,
        base_tactics=base_tactics,
        candidate_prefix=args.candidate_prefix,
        min_speedup=args.min_speedup,
    )
    for row in scored:
        if len(selected) >= args.max_tactics:
            break
        selected.append(row["tactic"])
    selected = unique_keep_order(selected)

    payload = {
        "summary_json": str(args.summary_json),
        "selected_tactics": selected,
        "base_tactics": base_tactics,
        "candidate_prefix": args.candidate_prefix,
        "max_tactics": args.max_tactics,
        "min_speedup": args.min_speedup,
        "ranked_candidates": scored,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(
            make_report(
                args.summary_json,
                selected,
                scored,
                base_tactics=base_tactics,
                max_tactics=args.max_tactics,
                min_speedup=args.min_speedup,
            ),
            encoding="utf-8",
        )
    print(" ".join(selected))


if __name__ == "__main__":
    main()
