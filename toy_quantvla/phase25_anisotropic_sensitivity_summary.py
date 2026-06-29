"""Summarize Phase 25 closed-loop anisotropic sensitivity experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs"

OUT_JSON = RESULTS / "phase25_anisotropic_sensitivity_summary.json"
OUT_MD = DOCS / "phase25_anisotropic_sensitivity_report_zh.md"


DIM_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
DURATION_TAGS = [
    ("task4_early", "task4:init9", "early", "0:75"),
    ("task4_mid", "task4:init9", "mid", "75:150"),
    ("task4_late", "task4:init9", "late", "150:225"),
    ("task6_early", "task6:init8", "early", "0:200"),
    ("task6_mid", "task6:init8", "mid", "200:450"),
    ("task6_late", "task6:init8", "late", "450:700"),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("episodes", []):
        latency = item["policy_latency_seconds"]
        rows.append(
            {
                "task_id": int(item["task_id"]),
                "init_index": int(item["init_index"]),
                "case": f"{int(item['task_id'])}:{int(item['init_index'])}",
                "success": bool(item["success"]),
                "steps": int(latency["count"]),
                "p50_ms": round(float(latency["p50"]) * 1000.0, 2),
            }
        )
    return rows


def summarize_baseline() -> dict[str, Any]:
    path = RESULTS / "phase24_cuda_graph_service_2case_client_latency.json"
    payload = read_json(path)
    return {
        "source": str(path.relative_to(ROOT)),
        "total_successes": int(payload["total_successes"]),
        "total_episodes": int(payload["total_episodes"]),
        "episodes": episode_rows(payload),
    }


def summarize_continuous_calibration() -> dict[str, Any]:
    path = RESULTS / "phase25_action_perturb_calib_continuous_amp003_2case_client_latency.json"
    payload = read_json(path)
    return {
        "source": str(path.relative_to(ROOT)),
        "perturbation": payload["action_perturbation"],
        "total_successes": int(payload["total_successes"]),
        "total_episodes": int(payload["total_episodes"]),
        "episodes": episode_rows(payload),
    }


def summarize_dim_grid() -> list[dict[str, Any]]:
    out = []
    for key in DIM_KEYS:
        path = RESULTS / f"phase25_action_dim_amp003_2case_{key}_client_latency.json"
        payload = read_json(path)
        rows = episode_rows(payload)
        out.append(
            {
                "dimension": key,
                "source": str(path.relative_to(ROOT)),
                "perturbation": payload["action_perturbation"],
                "total_successes": int(payload["total_successes"]),
                "total_episodes": int(payload["total_episodes"]),
                "success_rate": float(payload["success_rate"]),
                "episodes": rows,
            }
        )
    return out


def summarize_duration_grid() -> list[dict[str, Any]]:
    out = []
    for tag, case, phase, window in DURATION_TAGS:
        path = RESULTS / f"phase25_duration_y_amp003_{tag}_client_latency.json"
        payload = read_json(path)
        episode = episode_rows(payload)[0]
        out.append(
            {
                "tag": tag,
                "case": case,
                "phase": phase,
                "window": window,
                "source": str(path.relative_to(ROOT)),
                "perturbation": payload["action_perturbation"],
                "success": bool(episode["success"]),
                "steps": int(episode["steps"]),
                "p50_ms": episode["p50_ms"],
            }
        )
    return out


def summarize_layer_evidence() -> dict[str, Any]:
    path = RESULTS / "phase17_module_layer_sensitivity_map.json"
    payload = read_json(path)
    heatmap = payload["layer_heatmap"]
    compile_scope = payload["compile_scope_drift"]
    step253 = payload["step253_bisect"]
    return {
        "source": str(path.relative_to(ROOT)),
        "high_scope_rows": [
            item
            for item in compile_scope
            if item.get("sensitivity") == "high"
        ],
        "low_scope_rows": [
            item
            for item in compile_scope
            if item.get("sensitivity") == "low-drift-slow"
        ],
        "layer_heatmap": heatmap,
        "step253_top_rows": step253[:10],
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def success_mark(value: bool) -> str:
    return "success" if value else "fail"


def build_report(data: dict[str, Any]) -> str:
    baseline_rows = []
    for row in data["baseline"]["episodes"]:
        baseline_rows.append([row["case"], success_mark(row["success"]), row["steps"]])

    continuous_rows = []
    for row in data["continuous_calibration"]["episodes"]:
        continuous_rows.append([row["case"], success_mark(row["success"]), row["steps"]])

    dim_rows = []
    for item in data["dim_grid"]:
        per_case = ", ".join(
            f"{row['case']}={success_mark(row['success'])}/{row['steps']}"
            for row in item["episodes"]
        )
        dim_rows.append(
            [
                item["dimension"],
                f"{item['total_successes']}/{item['total_episodes']}",
                per_case,
            ]
        )

    duration_rows = [
        [item["case"], item["phase"], item["window"], success_mark(item["success"]), item["steps"]]
        for item in data["duration_grid"]
    ]

    layer_rows = []
    for item in data["layer_evidence"]["layer_heatmap"]:
        layer_rows.append(
            [
                item["block_or_scope"],
                item["risk"],
                item["continuous_max_abs"],
                item["spike_removed"],
                item["p50_speedup"],
            ]
        )

    scope_rows = []
    for item in data["layer_evidence"]["high_scope_rows"] + data["layer_evidence"]["low_scope_rows"]:
        scope_rows.append(
            [
                item["target"],
                item["sensitivity"],
                item["continuous_max_max"],
                item["p50_speedup"],
                item["reason"],
            ]
        )

    lines = [
        "# Phase 25: 闭环敏感性的三重各向异性",
        "",
        "本阶段验证三个判断：",
        "",
        "- not all dims are equal",
        "- not all durations / phases are equal",
        "- not all layers are equal",
        "",
        "实验使用 Phase24 的无损 CUDA Graph FP16 service 作为 baseline。这个 baseline 在 Phase24 已验证与 FP16 hygiene success map 一致，因此这里可以把 rollout 变化归因到显式注入的 action perturbation 或已有 layer/scope 变体，而不是推理加速路径本身。",
        "",
        "## 1. Baseline",
        "",
        md_table(["case", "outcome", "steps"], baseline_rows),
        "",
        "两个重点病例在无扰动 baseline 下都是成功：`4:9` 为 224 步，`6:8` 为 649 步。",
        "",
        "## 2. Calibration: continuous 6D 全程扰动",
        "",
        "设置：`keys=continuous`，`amplitude=0.03`，对 6 个连续 action 维度按 `sqrt(6)` 归一化，因此每步扰动 L2 约为 0.03。",
        "",
        md_table(["case", "outcome", "steps"], continuous_rows),
        "",
        "这个扰动预算会把两个 baseline 成功病例都推成失败，说明幅度足以触发闭环分叉。后续单维实验使用同样 L2 预算，比较不同维度的相对敏感性。",
        "",
        "## 3. Claim A: not all dims are equal",
        "",
        "设置：固定两个 case、固定 deterministic policy seed、固定每步扰动 L2 为 0.03，只改变扰动注入的 action dimension。",
        "",
        md_table(["dimension", "success", "per-case outcome / steps"], dim_rows),
        "",
        "结论非常清楚：同样的 L2 扰动预算下，`z` 维度保持 `2/2` 成功，`x` 为 `1/2`，而 `y/roll/pitch/yaw` 全部 `0/2`。这说明闭环敏感性不是 action-space isotropic 的，不能只看整体 action MSE 或 L2 drift。",
        "",
        "更有意思的是，`z` 并不是完全无影响：`6:8` 从 baseline 649 步变成 666 步；但它没有跨过失败边界。`y/roll/pitch/yaw` 则更容易把轨迹推过成功/失败 margin。",
        "",
        "## 4. Claim B: not all durations / phases are equal",
        "",
        "设置：固定扰动维度为 `y`，固定 amplitude 为 0.03，只改变注入时间窗口。窗口按 policy step 计数。",
        "",
        md_table(["case", "phase", "window", "outcome", "steps"], duration_rows),
        "",
        "task4:init9 中，同样的 y 扰动打在 early 或 mid 会失败，但打在 late 仍成功。这说明任务后段对这个方向的扰动更可吸收，或者该扰动已经不能再把轨迹推过关键失败边界。",
        "",
        "task6:init8 中，early 失败，但 mid 和 late 都成功，只是步数从 baseline 649 变为 697/752。这里同样说明闭环系统对误差的敏感性随阶段变化，不是简单按扰动累计时间决定。",
        "",
        "## 5. Claim C: not all layers are equal",
        "",
        "本轮没有重新跑 layer-level intervention；这里先引用 Phase17 的 module/layer sensitivity map 作为已有证据。它来自 compile scope、step253 focused replay、block-level bisect 和 matched rollout artifacts。",
        "",
        md_table(["scope", "sensitivity", "max drift", "p50 speedup", "reason"], scope_rows),
        "",
        md_table(["block/scope", "risk", "max abs", "spike removed", "p50 speedup"], layer_rows),
        "",
        "Phase17 的关键发现是：`action_head_model` 和 `blocks_8_15_eager` 等 scope 包含 deterministic continuous-action spike，而 `ff_all` / `attn_all` 这类 scope 同观测 drift 很低但没有实际速度收益。更细的 step253 bisect 显示，保留 `block0` eager 就能消除该 spike，而单独保留 block1/2/3 或 blocks4-7 不能消除。这已经支持 layer/scope sensitivity 不均匀。",
        "",
        "不过，Claim C 目前仍比 A/B 弱一些：A/B 是本阶段新跑的 controlled intervention；C 主要依赖已有 compile/layer artifacts。下一步如果要把 Claim C 写成强实验，需要重新做 matched-norm layer perturbation 或 layer-wise fake quant sweep，并用 Phase24 baseline service 加速闭环评估。",
        "",
        "## 6. 统一解释",
        "",
        "三个结果可以统一写成：",
        "",
        "```text",
        "Δh ≈ Σ_t c_t^T η_t",
        "```",
        "",
        "其中 `η_t` 的来源可以是 action-level 注入、量化误差，或者某层扰动经过 policy Jacobian 后映射到 action space 的结果。Phase25 的实验证明：",
        "",
        "- `c_t` 在 action dimension 上不均匀：同样 L2 的 z/y/roll 扰动闭环结果不同。",
        "- `c_t` 随 rollout phase 改变：同样 y 扰动在 early/mid/late 的结果不同。",
        "- layer perturbation 通过 `J_{t,l}` 投影到 action-sensitive directions，因此不同 layer/scope 风险不同。",
        "",
        "所以真正需要优化的不是全局 action MSE，而是 sensitivity-weighted closed-loop perturbation。",
        "",
        "## 7. 当前结论",
        "",
        "Phase25 已经给出足够强的初步证据来支撑：闭环量化风险在 action dimension、task phase、model layer 上都是各向异性的。",
        "",
        "最强新证据：",
        "",
        "- 同样 L2=0.03 的单维扰动，`z=2/2` 成功，`y/roll/pitch/yaw=0/2`。",
        "- 同样 `y+0.03`，task4 early/mid 失败但 late 成功；task6 early 失败但 mid/late 成功。",
        "",
        "这为后续 optimize guide 提供了方向：保护预算应优先分配给闭环敏感 action directions、敏感 task phases，以及能投影到这些方向的敏感 layer/scope。",
        "",
        "## 8. 下一步",
        "",
        "1. 对 action dim 做双符号和多幅度 sweep，例如 `±0.01/0.02/0.03`，区分方向性和维度性。",
        "2. 对 phase windows 做更细分的 sliding-window sweep，估计每个 step 区间的 empirical sensitivity curve。",
        "3. 做 layer-wise matched-norm perturbation 或 fake-quant sweep，把 Claim C 从已有证据升级成 controlled intervention。",
        "4. 将 empirical sensitivity map 转成量化策略：敏感维度/阶段/层保留更高精度或使用闭环校正。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    data = {
        "phase": "phase25_anisotropic_sensitivity",
        "baseline": summarize_baseline(),
        "continuous_calibration": summarize_continuous_calibration(),
        "dim_grid": summarize_dim_grid(),
        "duration_grid": summarize_duration_grid(),
        "layer_evidence": summarize_layer_evidence(),
    }
    write_json(OUT_JSON, data)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build_report(data), encoding="utf-8")
    print(f"Wrote {OUT_JSON.relative_to(ROOT)}")
    print(f"Wrote {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
