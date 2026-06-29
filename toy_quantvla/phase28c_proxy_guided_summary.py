"""Summarize Phase 28C proxy-guided candidate search."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase28C_proxy_guided_33case_v1")
REFERENCE_TAG_PREFIX = os.environ.get("REFERENCE_TAG_PREFIX", "phase28B_proxy_guided_33case_v1")
CASE_LIST = os.environ.get("CASE_LIST", "")
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260613"))
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase28c_proxy_guided_33case_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 28C: Proxy-Guided Candidate Search")

VARIANTS = [
    {"name": "baseline", "label": "FP16 baseline", "tag": f"{REFERENCE_TAG_PREFIX}_baseline", "run_label": "baseline", "policy": "no compile"},
    {"name": "speed_only", "label": "Speed-only compile", "tag": f"{REFERENCE_TAG_PREFIX}_speed_only_action_head_model", "run_label": "compiled", "policy": "compile action_head.model"},
    {"name": "block0", "label": "A: block0 eager", "tag": f"{TAG_PREFIX}_proxy_block0_eager", "run_label": "compiled", "policy": "compile action_head.model; keep block0 eager"},
    {"name": "block0_blocks8_15", "label": "B: block0 + blocks8-15 eager", "tag": f"{TAG_PREFIX}_proxy_block0_blocks8_15_eager", "run_label": "compiled", "policy": "compile action_head.model; keep block0 and blocks8-15 eager"},
    {"name": "blocks0_3", "label": "C: blocks0-3 eager", "tag": f"{TAG_PREFIX}_proxy_blocks0_3_eager", "run_label": "compiled", "policy": "compile action_head.model; keep blocks0-3 eager"},
    {"name": "duration_window_120_320", "label": "D: eager window 120-320", "tag": f"{TAG_PREFIX}_duration_window_eager_120_320", "run_label": "compiled", "policy": "compile action_head.model; eager fallback for policy steps [120, 320)"},
]


def read_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(float(seconds) * 1000.0, 2)


def bytes_to_mib(value: int | float | None) -> float | None:
    return None if value is None else round(float(value) / 1024.0**2, 1)


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
        return {"name": spec["name"], "label": spec["label"], "policy": spec["policy"], "status": "missing"}
    server_memory = (server or {}).get("extra", {}).get("server_memory", {})
    server_p50_ms = ms((server or {}).get("get_action_seconds", {}).get("p50"))
    speedup = round(baseline_server_p50_ms / server_p50_ms, 2) if baseline_server_p50_ms and server_p50_ms else None
    episodes = []
    for item in client.get("episodes", []):
        latency = item.get("policy_latency_seconds", {})
        episodes.append({
            "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
            "task_id": int(item["task_id"]),
            "init_index": int(item["init_index"]),
            "success": bool(item["success"]),
            "steps": int(latency.get("count", 0)),
            "p50_ms": ms(latency.get("p50")),
        })
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
        "total_successes": int(client.get("total_successes", 0)),
        "total_episodes": int(client.get("total_episodes", 0)),
        "success_rate": float(client.get("success_rate", 0.0)),
        "server_p50_ms": server_p50_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2) if prepare else None,
        "server_reserved_mib": bytes_to_mib(server_memory.get("reserved_bytes")),
        "server_max_reserved_mib": bytes_to_mib(server_memory.get("max_reserved_bytes")),
        "torch_compile": (prepare or {}).get("torch_compile"),
        "server_extra": (server or {}).get("extra", {}),
        "episodes": episodes,
        "per_task": dict(per_task),
        "client_source": str(client_path(spec).relative_to(ROOT)),
        "server_latency_source": str(server_latency_path(spec).relative_to(ROOT)),
    }


def pair_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left.get("status") != "complete" or right.get("status") != "complete":
        return None
    lmap = {ep["case"]: ep for ep in left["episodes"]}
    rmap = {ep["case"]: ep for ep in right["episodes"]}
    common = sorted(set(lmap) & set(rmap), key=lambda x: tuple(map(int, x.split(":"))))
    repaired = [c for c in common if not lmap[c]["success"] and rmap[c]["success"]]
    regressed = [c for c in common if lmap[c]["success"] and not rmap[c]["success"]]
    return {"left": left["name"], "right": right["name"], "common": len(common), "repaired": repaired, "regressed": regressed, "net": len(repaired) - len(regressed)}


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def fmt_ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def fmt_mib(x: float | None) -> str:
    return "-" if x is None else f"{x:.1f}"


def fmt_rate(s: int | None, t: int | None) -> str:
    return "missing" if s is None or t is None else f"{s}/{t}"


def build_summary() -> dict[str, Any]:
    baseline_server = read_json(server_latency_path(VARIANTS[0]))
    baseline_server_p50_ms = ms((baseline_server or {}).get("get_action_seconds", {}).get("p50"))
    runs = [summarize_variant(v, baseline_server_p50_ms) for v in VARIANTS]
    by_name = {r["name"]: r for r in runs}
    pairs = []
    for base_name in ["baseline", "speed_only"]:
        base = by_name.get(base_name)
        if not base:
            continue
        for cand_name in ["block0", "block0_blocks8_15", "blocks0_3", "duration_window_120_320"]:
            if cand_name in by_name:
                pair = pair_counts(base, by_name[cand_name])
                if pair:
                    pairs.append(pair)
    return {
        "phase": "phase28C_proxy_guided_candidate_search",
        "tag_prefix": TAG_PREFIX,
        "reference_tag_prefix": REFERENCE_TAG_PREFIX,
        "case_list": CASE_LIST,
        "policy_seed_base": POLICY_SEED_BASE,
        "baseline_server_p50_ms": baseline_server_p50_ms,
        "runs": runs,
        "paired_comparisons": pairs,
    }


def build_report(data: dict[str, Any]) -> str:
    run_rows = []
    for run in data["runs"]:
        run_rows.append([
            run["label"], run["policy"], run.get("status", "-"),
            fmt_rate(run.get("total_successes"), run.get("total_episodes")),
            fmt_ms(run.get("server_p50_ms")),
            f"{run.get('speedup_vs_baseline_server_p50')}x" if run.get("speedup_vs_baseline_server_p50") else "-",
            fmt_mib(run.get("server_reserved_mib")), fmt_mib(run.get("server_max_reserved_mib")),
            run.get("prepare_seconds", "-"),
        ])
    pair_rows = [[f"{p['right']} vs {p['left']}", p["common"], len(p["repaired"]), len(p["regressed"]), p["net"], ", ".join(p["repaired"]) or "-", ", ".join(p["regressed"]) or "-"] for p in data["paired_comparisons"]]
    complete = [r for r in data["runs"] if r.get("status") == "complete"]
    cases = sorted({ep["case"] for r in complete for ep in r["episodes"]}, key=lambda x: tuple(map(int, x.split(":"))))
    case_rows = []
    for case in cases:
        row = [case]
        for run in complete:
            ep = next((e for e in run["episodes"] if e["case"] == case), None)
            row.append("-" if ep is None else ("S" if ep["success"] else "F") + str(ep["steps"]))
        case_rows.append(row)
    lines = [
        f"# {REPORT_TITLE}", "",
        "本阶段把 Phase28B 的 33-case matched set 固定住，继续搜索更细的 proxy-guided 加速边界。Baseline 和 speed-only 复用 Phase28B 同一批 task/init/seed；A-D 是新增候选。", "",
        "## 实验设计", "",
        f"- reference tag: `{data['reference_tag_prefix']}`",
        f"- candidate tag: `{data['tag_prefix']}`",
        f"- case list: `{data['case_list']}`",
        f"- deterministic policy seed base: `{data['policy_seed_base']}`",
        "- A block0: same-observation spike proxy",
        "- B block0 + blocks8-15: combine spike proxy and earlier repair/regression proxy",
        "- C blocks0-3: wider early-block protection",
        "- D eager window 120-320: duration-aware fallback; whole action_head eager only in this policy-step window",
        "", "## 汇总", "",
        md_table(["run", "policy", "status", "success", "server p50 ms", "speedup", "reserved MiB", "max reserved MiB", "prepare s"], run_rows),
        "", "## Paired repair/regression", "",
        md_table(["comparison", "common", "repaired", "regressed", "net", "repaired cases", "regressed cases"], pair_rows) if pair_rows else "结果尚未完整。",
    ]
    if case_rows:
        lines += ["", "## Per-case outcomes", "", md_table(["case"] + [r["name"] for r in complete], case_rows)]
    lines += ["", "## 产物", "", f"- JSON: `{OUT_JSON.relative_to(ROOT)}`", f"- Report: `{OUT_MD.relative_to(ROOT)}`", ""]
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
