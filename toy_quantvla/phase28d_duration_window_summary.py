"""Summarize Phase 28D duration-window fallback sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase28D_duration_window_33case_v1")
REFERENCE_TAG_PREFIX = os.environ.get("REFERENCE_TAG_PREFIX", "phase28B_proxy_guided_33case_v1")
CASE_LIST = os.environ.get("CASE_LIST", "")
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260613"))
WINDOWS = os.environ.get("WINDOWS", "80:220,120:320,180:420,0:250")
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase28d_duration_window_sweep_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 28D: Duration-Window Fallback Sweep")


def parse_windows(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected window START:END, got {item!r}")
        start, end = item.split(":", 1)
        out.append((int(start), int(end)))
    return out


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


def client_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_client_latency.json"


def server_latency_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_server_latency.json"


def prepare_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_server_prepare.json"


def episode_rows(client: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
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


def summarize_variant(
    *,
    name: str,
    label: str,
    policy: str,
    tag: str,
    run_label: str,
    baseline_server_p50_ms: float | None,
) -> dict[str, Any]:
    client = read_json(client_path(tag, run_label))
    server = read_json(server_latency_path(tag, run_label))
    prepare = read_json(prepare_path(tag, run_label))
    if client is None:
        return {
            "name": name,
            "label": label,
            "policy": policy,
            "status": "missing",
            "client_source": str(client_path(tag, run_label).relative_to(ROOT)),
            "server_latency_source": str(server_latency_path(tag, run_label).relative_to(ROOT)),
        }

    server_memory = (server or {}).get("extra", {}).get("server_memory", {})
    fallback = (server or {}).get("extra", {}).get("compile_step_window_fallback")
    server_p50_ms = ms((server or {}).get("get_action_seconds", {}).get("p50"))
    speedup = None
    if baseline_server_p50_ms and server_p50_ms:
        speedup = round(baseline_server_p50_ms / server_p50_ms, 2)

    return {
        "name": name,
        "label": label,
        "policy": policy,
        "status": "complete",
        "client_source": str(client_path(tag, run_label).relative_to(ROOT)),
        "server_latency_source": str(server_latency_path(tag, run_label).relative_to(ROOT)),
        "prepare_source": str(prepare_path(tag, run_label).relative_to(ROOT)),
        "total_successes": int(client.get("total_successes", 0)),
        "total_episodes": int(client.get("total_episodes", 0)),
        "success_rate": float(client.get("success_rate", 0.0)),
        "server_p50_ms": server_p50_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2) if prepare else None,
        "server_reserved_mib": bytes_to_mib(server_memory.get("reserved_bytes")),
        "server_max_reserved_mib": bytes_to_mib(server_memory.get("max_reserved_bytes")),
        "torch_compile": (prepare or {}).get("torch_compile"),
        "fallback": fallback,
        "episodes": episode_rows(client),
    }


def pair_counts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    if left.get("status") != "complete" or right.get("status") != "complete":
        return None
    lmap = {ep["case"]: ep for ep in left["episodes"]}
    rmap = {ep["case"]: ep for ep in right["episodes"]}
    common = sorted(set(lmap) & set(rmap), key=lambda x: tuple(map(int, x.split(":"))))
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


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def fmt_rate(successes: int | None, total: int | None) -> str:
    if successes is None or total is None:
        return "missing"
    return f"{successes}/{total}"


def fmt_speedup(value: float | None) -> str:
    return "-" if value is None else f"{value}x"


def fmt_requests(fallback: dict[str, Any] | None, key: str) -> str:
    if not fallback:
        return "-"
    return str(fallback.get(key, "-"))


def build_summary() -> dict[str, Any]:
    baseline_tag = f"{REFERENCE_TAG_PREFIX}_baseline"
    baseline_server = read_json(server_latency_path(baseline_tag, "baseline"))
    baseline_server_p50_ms = ms((baseline_server or {}).get("get_action_seconds", {}).get("p50"))

    variants: list[dict[str, Any]] = [
        summarize_variant(
            name="baseline",
            label="FP16 baseline",
            policy="no compile",
            tag=baseline_tag,
            run_label="baseline",
            baseline_server_p50_ms=baseline_server_p50_ms,
        ),
        summarize_variant(
            name="speed_only",
            label="Speed-only compile",
            policy="compile action_head.model",
            tag=f"{REFERENCE_TAG_PREFIX}_speed_only_action_head_model",
            run_label="compiled",
            baseline_server_p50_ms=baseline_server_p50_ms,
        ),
    ]

    for start, end in parse_windows(WINDOWS):
        variants.append(
            summarize_variant(
                name=f"window_{start}_{end}",
                label=f"D window {start}-{end}",
                policy=f"compile action_head.model; eager fallback for policy steps [{start}, {end})",
                tag=f"{TAG_PREFIX}_window_{start}_{end}",
                run_label="compiled",
                baseline_server_p50_ms=baseline_server_p50_ms,
            )
        )

    by_name = {run["name"]: run for run in variants}
    pairs = []
    for base_name in ["baseline", "speed_only"]:
        base = by_name.get(base_name)
        if not base:
            continue
        for start, end in parse_windows(WINDOWS):
            candidate = by_name.get(f"window_{start}_{end}")
            if candidate:
                pair = pair_counts(base, candidate)
                if pair:
                    pairs.append(pair)

    return {
        "phase": "phase28D_duration_window_sweep",
        "tag_prefix": TAG_PREFIX,
        "reference_tag_prefix": REFERENCE_TAG_PREFIX,
        "case_list": CASE_LIST,
        "policy_seed_base": POLICY_SEED_BASE,
        "windows": [{"start": start, "end": end} for start, end in parse_windows(WINDOWS)],
        "baseline_server_p50_ms": baseline_server_p50_ms,
        "runs": variants,
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
                fmt_speedup(run.get("speedup_vs_baseline_server_p50")),
                fmt_requests(run.get("fallback"), "compiled_requests"),
                fmt_requests(run.get("fallback"), "eager_requests"),
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

    complete = [run for run in data["runs"] if run.get("status") == "complete"]
    cases = sorted({ep["case"] for run in complete for ep in run["episodes"]}, key=lambda x: tuple(map(int, x.split(":"))))
    case_rows = []
    for case in cases:
        row = [case]
        for run in complete:
            ep = next((item for item in run["episodes"] if item["case"] == case), None)
            row.append("-" if ep is None else ("S" if ep["success"] else "F") + str(ep["steps"]))
        case_rows.append(row)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "本阶段只重跑 duration-aware fallback 候选，验证 `not all durations are equal` 是否能转化为可用的推理后端策略。Baseline 和 speed-only 复用 Phase28B。",
        "",
        "## 实验设计",
        "",
        f"- reference tag: `{data['reference_tag_prefix']}`",
        f"- candidate tag: `{data['tag_prefix']}`",
        f"- case list: `{data['case_list']}`",
        f"- deterministic policy seed base: `{data['policy_seed_base']}`",
        f"- windows: `{WINDOWS}`",
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
                "compiled requests",
                "eager requests",
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
        else "结果尚未完整。",
    ]

    if case_rows:
        lines.extend(
            [
                "",
                "## Per-case outcomes",
                "",
                md_table(["case"] + [run["name"] for run in complete], case_rows),
            ]
        )

    lines.extend(
        [
            "",
            "## 产物",
            "",
            f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
            f"- Report: `{OUT_MD.relative_to(ROOT)}`",
            "",
        ]
    )
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
