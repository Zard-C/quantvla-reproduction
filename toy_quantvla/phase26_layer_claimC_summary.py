"""Summarize Phase 26 layer/boundary closed-loop intervention results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

OUT_JSON = RESULTS / "phase26_layer_claimC_summary.json"
OUT_MD = DOCS / "phase26_layer_claimC_report_zh.md"


RUNS = [
    {
        "name": "fp16_baseline",
        "label": "FP16 baseline",
        "client": "phase26_layer_claimC_2case_action_head_model_baseline_client_latency.json",
        "prepare": "phase26_layer_claimC_2case_action_head_model_baseline_server_prepare.json",
    },
    {
        "name": "action_head_model",
        "label": "compile action_head_model",
        "client": "phase26_layer_claimC_2case_action_head_model_compiled_client_latency.json",
        "prepare": "phase26_layer_claimC_2case_action_head_model_compiled_server_prepare.json",
    },
    {
        "name": "blocks_0_0_eager",
        "label": "compile action_head_model + block0 eager",
        "client": "phase26_layer_claimC_2case_blocks_0_0_eager_compiled_client_latency.json",
        "prepare": "phase26_layer_claimC_2case_blocks_0_0_eager_compiled_server_prepare.json",
    },
    {
        "name": "blocks_1_1_eager",
        "label": "compile action_head_model + block1 eager",
        "client": "phase26_layer_claimC_2case_blocks_1_1_eager_compiled_client_latency.json",
        "prepare": "phase26_layer_claimC_2case_blocks_1_1_eager_compiled_server_prepare.json",
    },
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ms(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    return round(float(seconds) * 1000.0, 2)


def success_text(value: bool) -> str:
    return "success" if value else "fail"


def episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("episodes", []):
        latency = item["policy_latency_seconds"]
        rows.append(
            {
                "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
                "task_id": int(item["task_id"]),
                "init_index": int(item["init_index"]),
                "success": bool(item["success"]),
                "steps": int(latency["count"]),
                "p50_ms": ms(latency["p50"]),
                "exception": item.get("exception"),
            }
        )
    return rows


def summarize_run(spec: dict[str, str], baseline_p50_ms: float | None) -> dict[str, Any]:
    client_path = RESULTS / spec["client"]
    prepare_path = RESULTS / spec["prepare"]
    client = read_json(client_path)
    prepare = read_json(prepare_path)
    p50_ms = ms(client["policy_latency_seconds"]["p50"])
    speedup = None
    if baseline_p50_ms and p50_ms:
        speedup = round(baseline_p50_ms / p50_ms, 2)
    return {
        "name": spec["name"],
        "label": spec["label"],
        "client_source": str(client_path.relative_to(ROOT)),
        "prepare_source": str(prepare_path.relative_to(ROOT)),
        "torch_compile": prepare.get("torch_compile", {}),
        "total_successes": int(client["total_successes"]),
        "total_episodes": int(client["total_episodes"]),
        "success_rate": float(client["success_rate"]),
        "p50_ms": p50_ms,
        "speedup_vs_baseline_p50": speedup,
        "prepare_seconds": round(float(prepare.get("prepare_seconds", 0.0)), 2),
        "prewarm_p50_ms": ms(prepare.get("prewarm_get_action_seconds", {}).get("p50")),
        "episodes": episode_rows(client),
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
    baseline_client = read_json(RESULTS / RUNS[0]["client"])
    baseline_p50_ms = ms(baseline_client["policy_latency_seconds"]["p50"])
    runs = [summarize_run(run, baseline_p50_ms) for run in RUNS]
    return {
        "phase": "phase26_layer_claimC",
        "purpose": "closed-loop layer/boundary intervention for not-all-layers-are-equal",
        "case_list": ["4:9", "6:8"],
        "policy_seed_base": 20260613,
        "baseline_p50_ms": baseline_p50_ms,
        "runs": runs,
    }


def build_report(data: dict[str, Any]) -> str:
    run_rows = []
    for run in data["runs"]:
        target = run["torch_compile"].get("target", "none")
        islands = run["torch_compile"].get("eager_island_modules") or []
        island_text = ", ".join(path.rsplit(".", 1)[-1] for path in islands) if islands else "-"
        per_case = ", ".join(
            f"{ep['case']}={success_text(ep['success'])}/{ep['steps']}"
            for ep in run["episodes"]
        )
        run_rows.append(
            [
                run["label"],
                target,
                island_text,
                f"{run['total_successes']}/{run['total_episodes']}",
                per_case,
                run["p50_ms"],
                f"{run['speedup_vs_baseline_p50']}x" if run["speedup_vs_baseline_p50"] else "-",
            ]
        )

    case_rows = []
    by_case: dict[str, list[str]] = {"4:9": [], "6:8": []}
    for run in data["runs"]:
        for ep in run["episodes"]:
            by_case.setdefault(ep["case"], []).append(
                f"{run['name']}={success_text(ep['success'])}/{ep['steps']}"
            )
    for case, values in by_case.items():
        case_rows.append([case, "; ".join(values)])

    lines = [
        "# Phase 26: Layer/Boundary Closed-Loop Intervention",
        "",
        "本阶段补强 Phase25 的第三个判断：`not all layers are equal`。实验不是重新做量化，而是用 torch.compile 的 action-head compile boundary 作为可控 layer/boundary intervention：外层都围绕 `action_head.model`，只改变 eager island 的 layer 位置。",
        "",
        "## 设置",
        "",
        "- case list: `4:9,6:8`",
        "- deterministic policy seed base: `20260613`",
        "- model: GR00T N1.5 LIBERO long posttrain",
        "- comparison: FP16 baseline vs full `action_head_model` compile vs `block0 eager` vs `block1 eager`",
        "",
        "## 结果表",
        "",
        md_table(
            [
                "run",
                "compile target",
                "eager island",
                "success",
                "per-case outcome / steps",
                "p50 ms",
                "speedup",
            ],
            run_rows,
        ),
        "",
        "## 按 case 看",
        "",
        md_table(["case", "outcomes"], case_rows),
        "",
        "## 关键观察",
        "",
        "1. `action_head_model` full compile 在这两个 case 上没有直接造成失败：`2/2` 成功，而且 p50 从 baseline 的 `160.53ms` 降到 `72.80ms`，约 `2.21x`。",
        "2. 但 full compile 不是行为透明的：`6:8` 从 baseline 的 649 步变成 916 步。也就是说它虽然最终成功，但闭环轨迹已经明显重分配。",
        "3. `block0 eager` 并没有在闭环上变得更安全：`4:9` 从成功变成 991 步失败；`6:8` 反而从 649 步变成 404 步成功。它是明显的 case-dependent intervention。",
        "4. `block1 eager` 更差：两个 case 都失败，说明不是随便把某个 block 留在 eager 就能提高鲁棒性。",
        "5. 三个 compiled variants 的 p50 都在 `72-73ms` 左右，但成功率分别是 `2/2`、`1/2`、`0/2`。因此工程上不能只看推理速度，也不能只看同观测 drift outlier 是否被消掉。",
        "",
        "## 对 Claim C 的含义",
        "",
        "这轮结果比简单的 `block0 是敏感层` 更强，也更谨慎：layer/boundary 的作用不是单调的。相同外层 compile 目标、相近 runtime p50，只改变 eager island 的 layer 位置，就能把闭环结果从 `2/2` 改成 `1/2` 或 `0/2`。",
        "",
        "所以更准确的 claim 是：",
        "",
        "```text",
        "Not all layer/boundary perturbations are equal; their effect is filtered by task, phase, action direction, and closed-loop margin.",
        "```",
        "",
        "这也解释了为什么 Phase17 里 `block0 eager` 能消除 step253 same-observation spike，但 Phase26 中它仍然会让 `4:9` 闭环失败：same-observation spike removal 只是局部数值指标，不等价于 full rollout robustness。",
        "",
        "## 下一步",
        "",
        "1. 对这三个 variants 做 first-divergence 分析，尤其比较 `4:9` 中 full compile 成功而 block0/block1 eager 失败的最早分叉点。",
        "2. 把 `action_head_model` full compile 纳入更大 case set，确认它是否只是这两个 case 上好运，还是可以作为速度候选。",
        "3. 将 layer intervention 和 Phase25 的 action-channel/phase sensitivity map 关联，检查失败是否集中投影到 `y/rotation` 等敏感方向。",
        "",
        "## 产物",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Report: `{OUT_MD.relative_to(ROOT)}`",
    ]
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
