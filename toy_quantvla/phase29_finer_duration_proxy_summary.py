"""Summarize Phase 29 finer duration proxy sweeps."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
TAG_PREFIX = os.environ.get("TAG_PREFIX", "phase29_finer_duration_proxy_33case_v1")
REFERENCE_TAG_PREFIX = os.environ.get("REFERENCE_TAG_PREFIX", "phase28B_proxy_guided_33case_v1")
PHASE28D_SUMMARY = RESULTS / os.environ.get(
    "PHASE28D_SUMMARY", "phase28D_duration_window_33case_v1_summary.json"
)
CASE_LIST = os.environ.get("CASE_LIST", "")
POLICY_SEED_BASE = int(os.environ.get("POLICY_SEED_BASE", "20260613"))
WINDOWS = os.environ.get("WINDOWS", "0:120,0:180,0:220,80:240,120:280,160:240,240:320")
OUT_JSON = RESULTS / f"{TAG_PREFIX}_summary.json"
_out_md = Path(os.environ.get("OUT_MD", "docs/phase29_finer_duration_proxy_report_zh.md"))
OUT_MD = _out_md if _out_md.is_absolute() else ROOT / _out_md
REPORT_TITLE = os.environ.get("REPORT_TITLE", "Phase 29: Finer Duration Proxy")


def parse_windows(spec: str) -> list[tuple[int, int]]:
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
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


def client_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_client_latency.json"


def server_latency_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_server_latency.json"


def prepare_path(tag: str, run_label: str) -> Path:
    return RESULTS / f"{tag}_{run_label}_server_prepare.json"


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
    source = str(client_path(tag, run_label).relative_to(ROOT))
    if client is None:
        return {
            "name": name,
            "label": label,
            "policy": policy,
            "status": "missing",
            "client_source": source,
        }

    server_p50_ms = ms((server or {}).get("get_action_seconds", {}).get("p50"))
    speedup = None
    if baseline_server_p50_ms and server_p50_ms:
        speedup = round(baseline_server_p50_ms / server_p50_ms, 2)
    fallback = (server or {}).get("extra", {}).get("compile_step_window_fallback")

    return {
        "name": name,
        "label": label,
        "policy": policy,
        "status": "complete",
        "client_source": source,
        "server_latency_source": str(server_latency_path(tag, run_label).relative_to(ROOT)),
        "prepare_source": str(prepare_path(tag, run_label).relative_to(ROOT)),
        "total_successes": int(client.get("total_successes", 0)),
        "total_episodes": int(client.get("total_episodes", 0)),
        "success_rate": float(client.get("success_rate", 0.0)),
        "server_p50_ms": server_p50_ms,
        "speedup_vs_baseline_server_p50": speedup,
        "prepare_seconds": round(float((prepare or {}).get("prepare_seconds", 0.0)), 2) if prepare else None,
        "torch_compile": (prepare or {}).get("torch_compile"),
        "fallback": fallback,
        "episodes": episode_rows(client),
    }


def phase28d_reference() -> dict[str, Any] | None:
    data = read_json(PHASE28D_SUMMARY)
    if not data:
        return None
    for run in data.get("runs", []):
        if run.get("name") == "window_0_250":
            copy = dict(run)
            copy["name"] = "phase28d_window_0_250"
            copy["label"] = "Phase28D window 0-250"
            copy["policy"] = "existing Phase28D broad early fallback reference"
            return copy
    return None


def case_key(case: str) -> tuple[int, int]:
    task, init = case.split(":")
    return int(task), int(init)


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


def build_summary() -> dict[str, Any]:
    baseline_tag = f"{REFERENCE_TAG_PREFIX}_baseline"
    baseline_server = read_json(server_latency_path(baseline_tag, "baseline"))
    baseline_server_p50_ms = ms((baseline_server or {}).get("get_action_seconds", {}).get("p50"))

    runs = [
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

    ref = phase28d_reference()
    if ref:
        runs.append(ref)

    for start, end in parse_windows(WINDOWS):
        runs.append(
            summarize_variant(
                name=f"window_{start}_{end}",
                label=f"Window {start}-{end}",
                policy=f"compile action_head.model; eager fallback for policy steps [{start}, {end})",
                tag=f"{TAG_PREFIX}_window_{start}_{end}",
                run_label="compiled",
                baseline_server_p50_ms=baseline_server_p50_ms,
            )
        )

    by_name = {run["name"]: run for run in runs}
    candidate_names = [f"window_{start}_{end}" for start, end in parse_windows(WINDOWS)]
    if "phase28d_window_0_250" in by_name:
        candidate_names.insert(0, "phase28d_window_0_250")

    pairs = []
    for base_name in ["baseline", "speed_only"]:
        for candidate_name in candidate_names:
            pair = pair_counts(by_name[base_name], by_name[candidate_name])
            if pair:
                pairs.append(pair)

    return {
        "phase": "phase29_finer_duration_proxy",
        "tag_prefix": TAG_PREFIX,
        "reference_tag_prefix": REFERENCE_TAG_PREFIX,
        "case_list": CASE_LIST,
        "policy_seed_base": POLICY_SEED_BASE,
        "windows": [{"start": start, "end": end} for start, end in parse_windows(WINDOWS)],
        "baseline_server_p50_ms": baseline_server_p50_ms,
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


def build_report(data: dict[str, Any]) -> str:
    run_rows = []
    for run in data["runs"]:
        fallback = run.get("fallback") or {}
        compiled = fallback.get("compiled_requests")
        eager = fallback.get("eager_requests")
        eager_frac = None
        if compiled is not None and eager is not None and compiled + eager:
            eager_frac = round(eager / (compiled + eager), 3)
        run_rows.append(
            [
                run["name"],
                run["policy"],
                run.get("status"),
                fmt_rate(run),
                fmt(run.get("server_p50_ms")),
                f"{fmt(run.get('speedup_vs_baseline_server_p50'))}x" if run.get("speedup_vs_baseline_server_p50") else "-",
                fmt(eager_frac),
            ]
        )

    pair_rows = []
    for pair in data["paired_comparisons"]:
        pair_rows.append(
            [
                f"{pair['right']} vs {pair['left']}",
                len(pair["repaired"]),
                len(pair["regressed"]),
                pair["net"],
                fmt_cases(pair["repaired"]),
                fmt_cases(pair["regressed"]),
            ]
        )

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "Phase29 是 Phase28D 之后的独立 finer duration proxy 搜索。它不改变已有 Phase28D tag；默认窗口围绕 `0-250` 拆细，用来判断收益是否来自接近、接触/抓取、初始搬起，还是后续搬运阶段。",
        "",
        "## Runs",
        "",
        md_table(["run", "policy", "status", "success", "server p50 ms", "speedup", "eager frac"], run_rows),
        "",
        "## Paired Repair / Regress",
        "",
        md_table(["pair", "repair", "regress", "net", "repair cases", "regress cases"], pair_rows),
        "",
        "## 判读方式",
        "",
        "- 如果某个窄窗口接近 `phase28d_window_0_250` 的成功率但 p50 更低，它就是更好的 duration proxy。",
        "- 如果 prefix 窗口有效而中后段 atomic bucket 无效，说明敏感性主要来自 grasp-critical prefix。",
        "- 如果不同 case 被不同窗口修复，就需要进入 layer x duration 或 task-conditioned duration guard。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = build_summary()
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
