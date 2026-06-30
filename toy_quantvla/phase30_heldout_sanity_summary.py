"""Summarize Phase 30 held-out sanity experiments."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase30_heldout_sanity_30case_v1")
CASE_LIST = os.environ.get("CASE_LIST", "")
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260701"))
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase30_heldout_sanity_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 30: Held-Out Sanity Set")


VARIANTS = [
    {
        "name": "baseline",
        "label": "FP16 baseline",
        "tag": f"{TAG_PREFIX}_baseline",
        "run_label": "baseline",
        "policy": "no compile",
    },
    {
        "name": "speed_only",
        "label": "Speed-only compile",
        "tag": f"{TAG_PREFIX}_speed_only_action_head_model",
        "run_label": "compiled",
        "policy": "compile action_head.model",
    },
    {
        "name": "blocks0_3",
        "label": "Blocks0-3 eager",
        "tag": f"{TAG_PREFIX}_proxy_blocks0_3_eager",
        "run_label": "compiled",
        "policy": "compile action_head.model; keep blocks0-3 eager",
    },
    {
        "name": "window_0_120",
        "label": "Duration 0-120",
        "tag": f"{TAG_PREFIX}_window_0_120",
        "run_label": "compiled",
        "policy": "compile action_head.model; eager fallback for policy steps [0, 120)",
    },
]


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


def client_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_client_latency.json"


def server_latency_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_server_latency.json"


def prepare_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_server_prepare.json"


def metric_ms(container: dict[str, Any] | None, key: str) -> float | None:
    if not container:
        return None
    data = container.get("get_action_seconds") or container.get("latency_seconds") or {}
    return ms(data.get(key))


def episode_rows(client: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in client.get("episodes", []):
        latency = item.get("policy_latency_seconds", {})
        rows.append(
            {
                "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
                "task_id": int(item["task_id"]),
                "init_index": int(item["init_index"]),
                "success": bool(item["success"]),
                "steps": int(latency.get("count", 0)),
                "p50_ms": ms(latency.get("p50")),
            }
        )
    return rows


def eager_fraction(server: dict[str, Any] | None) -> float | None:
    fallback = (server or {}).get("extra", {}).get("compile_step_window_fallback") or {}
    compiled = fallback.get("compiled_requests")
    eager = fallback.get("eager_requests")
    if compiled is None or eager is None or compiled + eager <= 0:
        return None
    return round(eager / (compiled + eager), 3)


def summarize_variant(spec: dict[str, str], baseline_p50_ms: float | None) -> dict[str, Any]:
    client = read_json(client_path(spec))
    server = read_json(server_latency_path(spec))
    prepare = read_json(prepare_path(spec))
    if client is None:
        return {
            "name": spec["name"],
            "label": spec["label"],
            "policy": spec["policy"],
            "status": "missing",
            "client_source": str(client_path(spec).relative_to(ROOT)),
            "server_latency_source": str(server_latency_path(spec).relative_to(ROOT)),
            "prepare_source": str(prepare_path(spec).relative_to(ROOT)),
        }

    server_memory = (server or {}).get("extra", {}).get("server_memory", {})
    model_load_memory = (prepare or {}).get("model_load_memory", {})
    prewarm_memory = (prepare or {}).get("prewarm_memory", {})
    p50_ms = metric_ms(server, "p50")
    p90_ms = metric_ms(server, "p90")
    speedup = round(baseline_p50_ms / p50_ms, 2) if baseline_p50_ms and p50_ms else None
    episodes = episode_rows(client)

    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"successes": 0, "total": 0})
    for ep in episodes:
        row = per_task[str(ep["task_id"])]
        row["total"] += 1
        row["successes"] += int(ep["success"])

    return {
        "name": spec["name"],
        "label": spec["label"],
        "policy": spec["policy"],
        "status": "complete",
        "client_source": str(client_path(spec).relative_to(ROOT)),
        "server_latency_source": str(server_latency_path(spec).relative_to(ROOT)),
        "prepare_source": str(prepare_path(spec).relative_to(ROOT)),
        "total_successes": int(client.get("total_successes", 0)),
        "total_episodes": int(client.get("total_episodes", 0)),
        "success_rate": float(client.get("success_rate", 0.0)),
        "server_p50_ms": p50_ms,
        "server_p90_ms": p90_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2) if prepare else None,
        "model_load_reserved_mib": bytes_to_mib(model_load_memory.get("reserved_bytes")),
        "prewarm_reserved_mib": bytes_to_mib(prewarm_memory.get("reserved_bytes")),
        "server_reserved_mib": bytes_to_mib(server_memory.get("reserved_bytes")),
        "server_max_reserved_mib": bytes_to_mib(server_memory.get("max_reserved_bytes")),
        "eager_fraction": eager_fraction(server),
        "torch_compile": (prepare or {}).get("torch_compile"),
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
    same_success = [case for case in common if lmap[case]["success"] and rmap[case]["success"]]
    same_failure = [case for case in common if not lmap[case]["success"] and not rmap[case]["success"]]
    return {
        "left": left["name"],
        "right": right["name"],
        "common": len(common),
        "repaired": repaired,
        "regressed": regressed,
        "same_success": same_success,
        "same_failure": same_failure,
        "net": len(repaired) - len(regressed),
    }


def build_summary() -> dict[str, Any]:
    baseline_server = read_json(server_latency_path(VARIANTS[0]))
    baseline_p50_ms = metric_ms(baseline_server, "p50")
    runs = [summarize_variant(spec, baseline_p50_ms) for spec in VARIANTS]
    by_name = {run["name"]: run for run in runs}

    pairs = []
    for base_name in ["baseline", "speed_only"]:
        base = by_name.get(base_name)
        if not base:
            continue
        for candidate_name in ["speed_only", "blocks0_3", "window_0_120"]:
            if candidate_name == base_name:
                continue
            candidate = by_name.get(candidate_name)
            if not candidate:
                continue
            pair = pair_counts(base, candidate)
            if pair:
                pairs.append(pair)

    return {
        "phase": "phase30_heldout_sanity",
        "tag_prefix": TAG_PREFIX,
        "case_list": CASE_LIST,
        "policy_seed_base": POLICY_SEED_BASE,
        "baseline_server_p50_ms": baseline_p50_ms,
        "runs": runs,
        "paired_comparisons": pairs,
    }


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


def fmt_rate(run: dict[str, Any]) -> str:
    if run.get("status") != "complete":
        return "missing"
    return f"{run.get('total_successes')}/{run.get('total_episodes')}"


def fmt_cases(cases: list[str]) -> str:
    return ", ".join(cases) if cases else "-"


def outcome_cell(ep: dict[str, Any] | None) -> str:
    if ep is None:
        return "-"
    return ("S" if ep["success"] else "F") + str(ep["steps"])


def build_report(data: dict[str, Any]) -> str:
    run_rows = []
    for run in data["runs"]:
        run_rows.append(
            [
                run["label"],
                run["policy"],
                run.get("status"),
                fmt_rate(run),
                fmt(run.get("server_p50_ms")),
                fmt(run.get("server_p90_ms")),
                f"{fmt(run.get('speedup_vs_baseline_server_p50'))}x" if run.get("speedup_vs_baseline_server_p50") else "-",
                fmt(run.get("eager_fraction")),
                fmt(run.get("server_reserved_mib")),
                fmt(run.get("server_max_reserved_mib")),
            ]
        )

    pair_rows = []
    for pair in data["paired_comparisons"]:
        pair_rows.append(
            [
                f"{pair['right']} vs {pair['left']}",
                pair["common"],
                len(pair["repaired"]),
                len(pair["regressed"]),
                pair["net"],
                fmt_cases(pair["repaired"]),
                fmt_cases(pair["regressed"]),
            ]
        )

    complete = [run for run in data["runs"] if run.get("status") == "complete"]
    cases = sorted({ep["case"] for run in complete for ep in run["episodes"]}, key=case_key)
    case_rows = []
    for case in cases:
        row = [case]
        for run in complete:
            ep = next((item for item in run["episodes"] if item["case"] == case), None)
            row.append(outcome_cell(ep))
        case_rows.append(row)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "Phase30 是 ICRA held-out sanity set。它不参与 Phase29 `0-120` proxy 的选择，用来检查该 proxy 是否只是在 33-case 上偶然有效。",
        "",
        "## 实验设计",
        "",
        f"- tag prefix: `{data['tag_prefix']}`",
        f"- case list: `{data['case_list']}`",
        f"- deterministic policy seed base: `{data['policy_seed_base']}`",
        "- 候选: FP16 baseline, speed-only compile, blocks0-3 eager, duration 0-120。",
        "",
        "## 汇总",
        "",
        md_table(
            [
                "run",
                "policy",
                "status",
                "success",
                "p50 ms",
                "p90 ms",
                "speedup",
                "eager frac",
                "reserved MiB",
                "max reserved MiB",
            ],
            run_rows,
        ),
        "",
        "## Paired repair/regression",
        "",
        md_table(
            ["comparison", "common", "repair", "regress", "net", "repair cases", "regress cases"],
            pair_rows,
        )
        if pair_rows
        else "结果尚未完整。",
    ]
    if case_rows:
        lines += [
            "",
            "## Per-case outcomes",
            "",
            md_table(["case"] + [run["name"] for run in complete], case_rows),
        ]

    lines += [
        "",
        "## 判读方式",
        "",
        "- 如果 `window_0_120` 在 held-out set 上仍然不低于 speed-only，并且 latency 接近 speed-only，则它支持 ICRA 版的 sensitivity-guided acceleration claim。",
        "- 如果 `window_0_120` 明显退化，则论文应把 `0-120` 降级为 33-case diagnostic，而不是泛化候选。",
        "- 如果 `blocks0_3` 比 `window_0_120` 更稳，则下一步应进入 layer x duration 的组合保护。",
        "",
        "## 产物",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Report: `{OUT_MD.relative_to(ROOT)}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = build_summary()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
