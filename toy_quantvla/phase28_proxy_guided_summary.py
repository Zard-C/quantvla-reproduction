"""Summarize Phase 28A proxy-guided mixed-precision experiments."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase28A_proxy_guided_15case_v1")
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = os.environ.get("OUT_MD", "docs/phase28_proxy_guided_mixed_precision_report_zh.md")
OUT_MD = Path(_out_md)
if not OUT_MD.is_absolute():
    OUT_MD = ROOT / OUT_MD
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 28: Sensitivity-Proxy Guided Mixed Precision")
CASE_LIST = os.environ.get(
    "CASE_LIST",
    "4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10",
)
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260613"))

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
        "name": "proxy_block0",
        "label": "Proxy-guided block0 eager",
        "tag": f"{TAG_PREFIX}_proxy_block0_eager",
        "run_label": "compiled",
        "policy": "compile action_head.model; keep block0 eager",
    },
    {
        "name": "proxy_blocks8_15",
        "label": "Proxy-guided blocks8-15 eager",
        "tag": f"{TAG_PREFIX}_proxy_blocks8_15_eager",
        "run_label": "compiled",
        "policy": "compile action_head.model; keep blocks8-15 eager",
    },
    {
        "name": "random_block1",
        "label": "Random/sanity block1 eager",
        "tag": f"{TAG_PREFIX}_random_block1_eager",
        "run_label": "compiled",
        "policy": "compile action_head.model; keep block1 eager",
    },
]

VARIANT_ENV_FLAGS = {
    "baseline": "RUN_BASELINE_VARIANT",
    "speed_only": "RUN_SPEED_ONLY",
    "proxy_block0": "RUN_PROXY_BLOCK0",
    "proxy_blocks8_15": "RUN_PROXY_BLOCKS8_15",
    "random_block1": "RUN_RANDOM_BLOCK1",
}


def enabled_variants() -> list[dict[str, str]]:
    specs = []
    for spec in VARIANTS:
        flag = VARIANT_ENV_FLAGS[spec["name"]]
        if os.environ.get(flag, "1") != "0":
            specs.append(spec)
    return specs


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 2)


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_rate(successes: int | None, total: int | None) -> str:
    if successes is None or total is None:
        return "missing"
    return f"{successes}/{total}"


def bytes_to_mib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 1024.0**2, 1)


def fmt_mib(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def client_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_client_latency.json"


def server_latency_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_server_latency.json"


def prepare_path(spec: dict[str, str]) -> Path:
    return RESULTS / f"{spec['tag']}_{spec['run_label']}_server_prepare.json"


def summarize_variant(spec: dict[str, str], baseline_server_p50_ms: float | None) -> dict[str, Any]:
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
    prewarm_memory = (prepare or {}).get("prewarm_memory", {})
    model_load_memory = (prepare or {}).get("model_load_memory", {})
    client_p50_ms = ms(client.get("policy_latency_seconds", {}).get("p50"))
    server_p50_ms = ms((server or {}).get("get_action_seconds", {}).get("p50"))
    if server_p50_ms is None:
        server_p50_ms = ms((server or {}).get("latency_seconds", {}).get("p50"))
    speedup = None
    if baseline_server_p50_ms and server_p50_ms:
        speedup = round(baseline_server_p50_ms / server_p50_ms, 2)

    episodes = []
    for item in client.get("episodes", []):
        latency = item.get("policy_latency_seconds", {})
        episodes.append(
            {
                "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
                "task_id": int(item["task_id"]),
                "init_index": int(item["init_index"]),
                "success": bool(item["success"]),
                "steps": int(latency.get("count", 0)),
                "p50_ms": ms(latency.get("p50")),
            }
        )

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
        "client_p50_ms": client_p50_ms,
        "server_p50_ms": server_p50_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2)
        if prepare
        else None,
        "model_load_reserved_mib": bytes_to_mib(model_load_memory.get("reserved_bytes")),
        "prewarm_reserved_mib": bytes_to_mib(prewarm_memory.get("reserved_bytes")),
        "prewarm_max_reserved_mib": bytes_to_mib(prewarm_memory.get("max_reserved_bytes")),
        "server_reserved_mib": bytes_to_mib(server_memory.get("reserved_bytes")),
        "server_max_reserved_mib": bytes_to_mib(server_memory.get("max_reserved_bytes")),
        "torch_compile": (prepare or {}).get("torch_compile"),
        "episodes": episodes,
        "per_task": dict(per_task),
    }


def pair_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left.get("status") != "complete" or right.get("status") != "complete":
        return None
    lmap = {ep["case"]: ep for ep in left["episodes"]}
    rmap = {ep["case"]: ep for ep in right["episodes"]}
    common = sorted(set(lmap) & set(rmap))
    repaired = []
    regressed = []
    same_success = []
    same_failure = []
    for case in common:
        l_ok = bool(lmap[case]["success"])
        r_ok = bool(rmap[case]["success"])
        if not l_ok and r_ok:
            repaired.append(case)
        elif l_ok and not r_ok:
            regressed.append(case)
        elif l_ok and r_ok:
            same_success.append(case)
        else:
            same_failure.append(case)
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


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_summary() -> dict[str, Any]:
    specs = enabled_variants()
    baseline_spec = next((spec for spec in specs if spec["name"] == "baseline"), VARIANTS[0])
    baseline_client = read_json(client_path(baseline_spec))
    baseline_server = read_json(server_latency_path(baseline_spec))
    baseline_server_p50_ms = None
    if baseline_server:
        baseline_server_p50_ms = ms(baseline_server.get("get_action_seconds", {}).get("p50"))
        if baseline_server_p50_ms is None:
            baseline_server_p50_ms = ms(baseline_server.get("latency_seconds", {}).get("p50"))
    if baseline_server_p50_ms is None and baseline_client:
        baseline_server_p50_ms = ms(baseline_client.get("policy_latency_seconds", {}).get("p50"))

    runs = [summarize_variant(spec, baseline_server_p50_ms) for spec in specs]
    by_name = {run["name"]: run for run in runs}
    pairs = []
    baseline = by_name.get("baseline")
    speed_only = by_name.get("speed_only")
    if baseline:
        for name in ["speed_only", "proxy_block0", "proxy_blocks8_15", "random_block1"]:
            if name in by_name:
                pair = pair_counts(baseline, by_name[name])
                if pair:
                    pairs.append(pair)
    if speed_only:
        for name in ["proxy_block0", "proxy_blocks8_15", "random_block1"]:
            if name in by_name:
                pair = pair_counts(speed_only, by_name[name])
                if pair:
                    pairs.append(pair)

    return {
        "phase": "phase28A_proxy_guided_mixed_precision",
        "tag_prefix": TAG_PREFIX,
        "purpose": "test whether sensitivity-proxy-guided protection improves closed-loop behavior under similar acceleration budget",
        "case_list": CASE_LIST,
        "policy_seed_base": POLICY_SEED_BASE,
        "baseline_server_p50_ms": baseline_server_p50_ms,
        "runs": runs,
        "paired_comparisons": pairs,
    }


def build_report(data: dict[str, Any]) -> str:
    run_rows = []
    for run in data["runs"]:
        run_rows.append(
            [
                run["label"],
                run["policy"],
                run.get("status", "-"),
                fmt_rate(run.get("total_successes"), run.get("total_episodes")),
                fmt_ms(run.get("server_p50_ms")),
                (
                    f"{run.get('speedup_vs_baseline_server_p50')}x"
                    if run.get("speedup_vs_baseline_server_p50")
                    else "-"
                ),
                fmt_mib(run.get("server_reserved_mib")),
                fmt_mib(run.get("server_max_reserved_mib")),
                run.get("prepare_seconds", "-"),
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
                ", ".join(pair["repaired"]) or "-",
                ", ".join(pair["regressed"]) or "-",
            ]
        )

    case_rows = []
    complete = [run for run in data["runs"] if run.get("status") == "complete"]
    cases = sorted({ep["case"] for run in complete for ep in run["episodes"]})
    for case in cases:
        values = []
        for run in complete:
            ep = next((item for item in run["episodes"] if item["case"] == case), None)
            if ep is None:
                values.append("-")
            else:
                values.append(("S" if ep["success"] else "F") + str(ep["steps"]))
        case_rows.append([case] + values)

    design_lines = [
        "- speed-only: compile whole `action_head.model`",
    ]
    run_names = {run["name"] for run in data["runs"]}
    if "proxy_block0" in run_names:
        design_lines.append("- proxy block0: 来自 same-observation spike proxy，保护 `transformer_blocks.0`")
    if "proxy_blocks8_15" in run_names:
        design_lines.append("- proxy blocks8-15: 来自闭环 repair/regression proxy，保护 `transformer_blocks.8..15`")
    if "random_block1" in run_names:
        design_lines.append("- random/sanity block1: 同样保护一个 block，但不来自 proxy")

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "本阶段验证论文里的 guide 是否能落到工程策略上：用 sensitivity proxy 决定哪些 action-head 边界保留高精度/eager，比较它是否比纯速度导向或随机保护更稳。",
        "",
        "## 实验设计",
        "",
        f"- case list: `{data['case_list']}`",
        f"- deterministic policy seed base: `{data['policy_seed_base']}`",
        *design_lines,
        "",
        "## 汇总",
        "",
        md_table(
            [
                "run",
                "policy",
                "status",
                "success",
                "server p50 ms",
                "speedup",
                "reserved MiB",
                "max reserved MiB",
                "prepare s",
            ],
            run_rows,
        ),
        "",
        "## Paired repair/regression",
        "",
        md_table(
            ["comparison", "common", "repaired", "regressed", "net", "repaired cases", "regressed cases"],
            pair_rows,
        )
        if pair_rows
        else "结果尚未完整，paired comparison 暂不可用。",
        "",
    ]

    if case_rows:
        lines.extend(
            [
                "## Per-case outcomes",
                "",
                md_table(["case"] + [run["name"] for run in complete], case_rows),
                "",
            ]
        )

    lines.extend(
        [
            "## 初步判读规则",
            "",
            "1. 如果 proxy variant 相比 speed-only 减少 regressions，同时保留大部分 speedup，说明 guide 有工程价值。",
            "2. 如果 block0 proxy 改善 same-observation spike 但闭环不提升，说明局部数值 proxy 不足，论文需要强调 proxy 必须闭环校准。",
            "3. 如果 blocks8-15 proxy 优于 random block1，说明 repair/regression/trajectory proxy 比随机保护更有用。",
            "4. 如果所有 proxy 都不优于 speed-only，则 guide 仍可作为诊断框架，但暂不能声称能直接优化策略。",
            "",
            "## 产物",
            "",
            f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
            f"- Report: `{OUT_MD.relative_to(ROOT)}`",
        ]
    )
    return "\n".join(lines) + "\n"


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
