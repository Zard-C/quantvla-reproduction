"""Offline audit for lossless Torch-level acceleration opportunities.

This phase does not run GR00T or LIBERO. It consolidates existing latency
artifacts and the current wrapper code paths into a small JSON artifact plus a
Chinese Markdown report. The goal is to separate truly lossless acceleration
candidates from paths that already showed closed-loop behavior changes.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "toy_quantvla" / "results"
DOCS = ROOT / "docs" / "phase18_lossless_torch"

OUT_JSON = RESULTS / "phase18_lossless_torch_acceleration_audit.json"
OUT_MD = DOCS / "lossless_torch_acceleration_audit_zh.md"

BASELINE_CLIENT_30 = RESULTS / "phase13_block_island_30case_mem_v1_baseline_client_latency.json"
BASELINE_SERVER_30 = RESULTS / "phase13_block_island_30case_mem_v1_baseline_server_latency.json"
BASELINE_PREPARE_30 = RESULTS / "phase13_block_island_30case_mem_v1_baseline_server_prepare.json"
COMPILED_CLIENT_30 = RESULTS / "phase13_block_island_30case_mem_v1_compiled_client_latency.json"
COMPILED_SERVER_30 = RESULTS / "phase13_block_island_30case_mem_v1_compiled_server_latency.json"
COMPILED_PREPARE_30 = RESULTS / "phase13_block_island_30case_mem_v1_compiled_server_prepare.json"
CUDA_GRAPH_REPLAY = RESULTS / "phase14_cuda_graph_dit_probe_obs4_r10_v1.json"
FP16_DIT_PROFILE_SERVER = RESULTS / "phase8_fp16_profiled_server_dit_mlp_task6_init1_latency.json"
FP16_HOTSPOT = RESULTS / "phase10_fp16_hotspot_llm_dit_mlp_obs4_r3.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_get(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def seconds_to_ms(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value) * 1000.0, 3)


def ratio(num: Any, den: Any) -> float | None:
    if num is None or den is None or float(den) == 0.0:
        return None
    return round(float(num) / float(den), 4)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.1f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{fmt(value)} ms"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def summarize_fp16_baseline() -> dict[str, Any]:
    client = read_json(BASELINE_CLIENT_30)
    server = read_json(BASELINE_SERVER_30)
    prepare = read_json(BASELINE_PREPARE_30)
    client_latency = client["policy_latency_seconds"]
    server_latency = server["get_action_seconds"]
    breakdown = client["policy_breakdown_seconds"]
    memory = safe_get(server, "extra", "server_memory") or {}
    return {
        "sources": [rel(BASELINE_CLIENT_30), rel(BASELINE_SERVER_30), rel(BASELINE_PREPARE_30)],
        "episodes": int(client["total_episodes"]),
        "successes": int(client["total_successes"]),
        "success_rate": round(float(client["success_rate"]), 4),
        "client_policy_p50_ms": seconds_to_ms(client_latency["p50"]),
        "client_policy_p90_ms": seconds_to_ms(client_latency["p90"]),
        "server_get_action_p50_ms": seconds_to_ms(server_latency["p50"]),
        "server_get_action_p90_ms": seconds_to_ms(server_latency["p90"]),
        "client_preprocess_p50_ms": seconds_to_ms(breakdown["preprocess_seconds"]["p50"]),
        "client_postprocess_p50_ms": seconds_to_ms(breakdown["postprocess_seconds"]["p50"]),
        "client_remote_p50_ms": seconds_to_ms(breakdown["remote_get_action_seconds"]["p50"]),
        "client_minus_server_p50_ms": round(
            seconds_to_ms(breakdown["remote_get_action_seconds"]["p50"])
            - seconds_to_ms(server_latency["p50"]),
            3,
        ),
        "prepare_seconds": round(float(prepare["prepare_seconds"]), 3),
        "prewarm_total_seconds": round(float(prepare["prewarm_total_seconds"]), 3),
        "prewarm_p50_ms": seconds_to_ms(prepare["prewarm_get_action_seconds"]["p50"]),
        "reserved_mib": round(float(memory.get("reserved_bytes", 0)) / 1024**2, 1),
        "allocated_mib": round(float(memory.get("allocated_bytes", 0)) / 1024**2, 1),
    }


def summarize_fp16_profiles(baseline_server_p50_ms: float) -> dict[str, Any]:
    dit_server = read_json(FP16_DIT_PROFILE_SERVER)
    module_results = safe_get(dit_server, "extra", "profile_module_results") or {}
    dit_total_seconds = 0.0
    dit_calls = 0
    for row in module_results.values():
        stats = row["stats"]
        calls = int(stats["calls"])
        dit_calls += calls
        dit_total_seconds += float(stats["forward_seconds"]["mean"]) * calls
    dit_per_get_action = dit_total_seconds / max(int(dit_server["get_action_seconds"]["count"]), 1)

    hotspot = read_json(FP16_HOTSPOT)
    profile_requests = int(hotspot["profile_requests"])
    selected_linear_seconds = float(hotspot["module_summary"]["total_profiled_linear_seconds"])
    selected_per_request = selected_linear_seconds / max(profile_requests, 1)
    families = hotspot["module_summary"]["by_family"]

    return {
        "sources": [rel(FP16_DIT_PROFILE_SERVER), rel(FP16_HOTSPOT)],
        "dit_mlp_linear_modules": len(module_results),
        "dit_mlp_calls": int(dit_calls),
        "dit_mlp_per_get_action_ms": seconds_to_ms(dit_per_get_action),
        "dit_mlp_share_of_profiled_server_mean": round(
            dit_per_get_action / float(dit_server["get_action_seconds"]["mean"]), 4
        ),
        "dit_mlp_share_of_baseline_p50": round(
            seconds_to_ms(dit_per_get_action) / baseline_server_p50_ms,
            4,
        ),
        "llm_dit_selected_modules": int(hotspot["profiled_modules"]),
        "llm_dit_selected_per_request_ms": seconds_to_ms(selected_per_request),
        "llm_dit_selected_share_of_offline_profile_p50": round(
            selected_per_request / float(hotspot["profile_request_seconds"]["p50"]), 4
        ),
        "family_breakdown_ms_per_request": {
            family: seconds_to_ms(float(row["total_seconds"]) / max(profile_requests, 1))
            for family, row in families.items()
        },
    }


def summarize_existing_acceleration_routes() -> dict[str, Any]:
    baseline_client = read_json(BASELINE_CLIENT_30)
    baseline_server = read_json(BASELINE_SERVER_30)
    compiled_client = read_json(COMPILED_CLIENT_30)
    compiled_server = read_json(COMPILED_SERVER_30)
    compiled_prepare = read_json(COMPILED_PREPARE_30)
    graph = read_json(CUDA_GRAPH_REPLAY)

    baseline_server_p50 = float(baseline_server["get_action_seconds"]["p50"])
    compiled_server_p50 = float(compiled_server["get_action_seconds"]["p50"])
    eager_p50 = float(graph["eager_replay"]["get_action_seconds"]["p50"])
    graph_p50 = float(graph["cuda_graph_replay"]["get_action_seconds"]["p50"])
    eager_p90 = float(graph["eager_replay"]["get_action_seconds"]["p90"])
    graph_p90 = float(graph["cuda_graph_replay"]["get_action_seconds"]["p90"])

    return {
        "torch_compile_block_island": {
            "sources": [rel(COMPILED_CLIENT_30), rel(COMPILED_SERVER_30), rel(COMPILED_PREPARE_30)],
            "successes": int(compiled_client["total_successes"]),
            "episodes": int(compiled_client["total_episodes"]),
            "baseline_successes": int(baseline_client["total_successes"]),
            "baseline_episodes": int(baseline_client["total_episodes"]),
            "server_p50_ms": seconds_to_ms(compiled_server_p50),
            "baseline_server_p50_ms": seconds_to_ms(baseline_server_p50),
            "p50_speedup": ratio(baseline_server_p50, compiled_server_p50),
            "server_p90_ms": seconds_to_ms(compiled_server["get_action_seconds"]["p90"]),
            "prepare_seconds": round(float(compiled_prepare["prepare_seconds"]), 3),
            "prewarm_total_seconds": round(float(compiled_prepare["prewarm_total_seconds"]), 3),
            "prewarm_p50_ms": seconds_to_ms(compiled_prepare["prewarm_get_action_seconds"]["p50"]),
            "lossless_judgement": "not_lossless",
        },
        "cuda_graph_dit_forward": {
            "sources": [rel(CUDA_GRAPH_REPLAY)],
            "cases": len(graph["cases"]),
            "eager_p50_ms": seconds_to_ms(eager_p50),
            "graph_p50_ms": seconds_to_ms(graph_p50),
            "p50_speedup": ratio(eager_p50, graph_p50),
            "eager_p90_ms": seconds_to_ms(eager_p90),
            "graph_p90_ms": seconds_to_ms(graph_p90),
            "p90_speedup": ratio(eager_p90, graph_p90),
            "max_abs_diff": float(graph["cuda_graph_replay"]["metrics"]["max_abs_diff"]["max"]),
            "rel_rmse_mean": float(graph["cuda_graph_replay"]["metrics"]["relative_rmse"]["mean"]),
            "graph_cache_size": int(graph["cuda_graph"]["graph_cache_size"]),
            "capture_seconds": round(float(graph["cuda_graph"]["capture_seconds"]), 3),
            "lossless_judgement": "fixed_observation_bit_exact",
        },
    }


def build_static_findings() -> list[dict[str, Any]]:
    return [
        {
            "topic": "client preprocessing/postprocessing",
            "finding": "LIBERO client 图像/状态预处理和 action 后处理的 p50 都约为 0.1 ms。",
            "judgement": "not_a_primary_bottleneck",
            "evidence": [rel(BASELINE_CLIENT_30), "toy_quantvla/libero_eval_init_range.py"],
        },
        {
            "topic": "env.step(action.tolist())",
            "finding": "这里的 action 已经是 simulator client 侧的 NumPy 数组，不是 server GPU tensor 的 D2H。",
            "judgement": "do_not_optimize_first",
            "evidence": ["toy_quantvla/libero_eval_init_range.py"],
        },
        {
            "topic": "request trace and summary flushing",
            "finding": "per-request JSONL trace 每行都会 flush；latency summary flush 也可能在 rollout 中写文件。",
            "judgement": "disable_for_deployment_or_buffer",
            "evidence": ["toy_quantvla/timing_utils.py", "toy_quantvla/timed_fp16_inference_service.py"],
        },
        {
            "topic": "explicit CUDA synchronize",
            "finding": "server request CUDA sync 是显式 opt-in，应继续只用于 profiling；module profiler 的 synchronize 也是诊断用途。",
            "judgement": "safe_when_disabled",
            "evidence": ["toy_quantvla/timing_utils.py", "toy_quantvla/fp16_linear_profiler.py"],
        },
        {
            "topic": "fixed shapes",
            "finding": "task description 长度会改变 encoder_hidden_states shape；已有 warmdesc prewarm，应在 rollout 前覆盖所有 eval task description。",
            "judgement": "required_for_compile_or_kernel_cache_paths",
            "evidence": [
                "toy_quantvla/timed_fp16_inference_service.py",
                "docs/phase8_cutlass_fp4_server_rollout_smoke.md",
            ],
        },
        {
            "topic": "hidden D2H / scalar sync inside GR00T policy",
            "finding": "当前 wrapper 层审计不能证明 Isaac-GR00T 内部没有隐藏 D2H/scalar sync，必须在 5090 上用 torch.profiler 验证。",
            "judgement": "needs_gpu_profile",
            "evidence": ["external GR00T policy internals not vendored in this repo"],
        },
    ]


def build_candidates() -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "name": "低扰动 torch.profiler 审计",
            "lossless": True,
            "expected_speedup": "先测量",
            "why": "改代码前先找隐藏 cudaMemcpyDtoH、aten::item、aten::_local_scalar_dense、CPU wait 和 shape-cache miss。",
            "risk": "profiler 本身会扰动 timing，所以只跑短 fixed-observation 或短 rollout。",
            "gate": "找到每 request 反复出现且超过 1 ms 的 sync/D2H，或确认这类问题不存在。",
        },
        {
            "rank": 2,
            "name": "部署态 timing hygiene",
            "lossless": True,
            "expected_speedup": "小",
            "why": "部署跑法里禁用 per-request JSONL trace、summary flush、CUDA sync 和 module profiler。",
            "risk": "不减少模型计算量，主要移除测量开销。",
            "gate": "同一 30-case set 下 server p50/p90 不变差，成功率不变。",
        },
        {
            "rank": 3,
            "name": "shape-aware prewarm contract",
            "lossless": True,
            "expected_speedup": "只改善冷启动/长尾",
            "why": "把 task-description shape miss 和 kernel/graph cache 构造前移到 server 启动阶段。",
            "risk": "需要 serving 前知道 task set；未见过的 description 仍可能 miss。",
            "gate": "request-time 没有 cold spike，也没有新的 graph/compile/cache miss 记录。",
        },
        {
            "rank": 4,
            "name": "eager DiT forward 的 CUDA graph replay",
            "lossless": True,
            "expected_speedup": "p50 ~1.02x, p90 better",
            "why": "已有 fixed-observation replay 是 bit-exact，可以减少 launch/Python 调度开销。",
            "risk": "p50 收益小；server 集成需要 shape cache 和 fallback 策略。",
            "gate": "matched rollout 的 success/action trace 与 FP16 对齐，同时 server p90 改善。",
        },
        {
            "rank": 5,
            "name": "torch.compile block-island",
            "lossless": False,
            "expected_speedup": "p50 ~2.04x",
            "why": "目前最强的速度信号。",
            "risk": "已经把 30-case success 从 16/30 改成 13/30。",
            "gate": "不是无损路径；只能配合 behavior guard 或闭环修正使用。",
        },
    ]


def build_remote_commands() -> list[str]:
    prepare = read_json(BASELINE_PREPARE_30)
    descriptions = safe_get(prepare, "prewarm_observation_meta", "prewarm_task_descriptions") or []
    desc_flags = " ".join(f"--prewarm-task-description {shlex.quote(str(desc))}" for desc in descriptions)
    return [
        "# 1. 纯 FP16 hygiene run：不打开 request trace、不打开 server cuda sync、不打开 module profiler。",
        f"python toy_quantvla/timed_fp16_inference_service.py --port 5555 --prewarm-observations 1 --prewarm-indices 115 {desc_flags} --server-latency-json toy_quantvla/results/phase18_fp16_hygiene_server_latency.json",
        "python toy_quantvla/libero_eval_init_range.py --task-suite-name libero_10 --case-list 4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10 --headless --port 5555 --deterministic-policy-seeds --policy-seed-base 20260613 --latency-json toy_quantvla/results/phase18_fp16_hygiene_client_latency.json",
        "# 2. profiler probe：包住短 fixed-observation 或短 rollout，重点 grep cudaMemcpyDtoH、aten::item、aten::_local_scalar_dense、cudaDeviceSynchronize。",
        "# 3. 如果 profiler 没有发现隐藏 sync/D2H，就不要继续在 D2H 上耗时间；改测 CUDA graph DiT 的 p90-only clean optimization。",
    ]


def build_report(data: dict[str, Any]) -> str:
    baseline = data["fp16_baseline"]
    profiles = data["fp16_profiles"]
    routes = data["existing_routes"]
    graph = routes["cuda_graph_dit_forward"]
    compile_route = routes["torch_compile_block_island"]

    lines = [
        "# Phase 18 无损 Torch 加速审计",
        "",
        "## 边界",
        "",
        "这一阶段只讨论量化之前的无损加速：不改权重、不改低比特 kernel、不接受同观测 action drift。当前产物基于已有 5090 结果和本地代码静态审计，没有重新跑 GPU。",
        "",
        "## 当前 FP16 基线",
        "",
        md_table(
            ["item", "value"],
            [
                ["30-case success", f"{baseline['successes']}/{baseline['episodes']}"],
                ["server get_action p50", fmt_ms(baseline["server_get_action_p50_ms"])],
                ["server get_action p90", fmt_ms(baseline["server_get_action_p90_ms"])],
                ["client policy p50", fmt_ms(baseline["client_policy_p50_ms"])],
                ["client preprocess p50", fmt_ms(baseline["client_preprocess_p50_ms"])],
                ["client postprocess p50", fmt_ms(baseline["client_postprocess_p50_ms"])],
                ["client-server overhead p50", fmt_ms(baseline["client_minus_server_p50_ms"])],
                ["reserved memory", f"{fmt(baseline['reserved_mib'], 1)} MiB"],
            ],
        ),
        "",
        "直接结论：client 侧 preprocess/postprocess 是 0.1 ms 量级，通信/序列化差额约 4.7 ms，主瓶颈仍是 server `policy.get_action`。",
        "",
        "## 覆盖面事实",
        "",
        md_table(
            ["profile scope", "per get_action", "share"],
            [
                [
                    "DiT MLP Linear only",
                    fmt_ms(profiles["dit_mlp_per_get_action_ms"]),
                    f"{fmt(profiles['dit_mlp_share_of_profiled_server_mean'] * 100, 2)}% of profiled server mean",
                ],
                [
                    "LLM attn + LLM MLP + DiT MLP selected linears",
                    fmt_ms(profiles["llm_dit_selected_per_request_ms"]),
                    f"{fmt(profiles['llm_dit_selected_share_of_offline_profile_p50'] * 100, 2)}% of offline profile p50",
                ],
            ],
        ),
        "",
        "这解释了之前 `dit_mlp_only` 很难端到端提速：它在 FP16 server 里的覆盖面只有约 8.4%。如果不改变更大范围的算子执行，单独优化这块的理论上限很低。",
        "",
        "## 已验证路线对比",
        "",
        md_table(
            ["route", "speed", "behavior", "judgement"],
            [
                [
                    "CUDA graph DiT forward",
                    f"p50 {fmt(graph['p50_speedup'], 3)}x, p90 {fmt(graph['p90_speedup'], 3)}x",
                    f"fixed replay max diff {fmt(graph['max_abs_diff'])}",
                    "clean but small",
                ],
                [
                    "torch.compile block-island",
                    f"server p50 {fmt(compile_route['p50_speedup'], 2)}x",
                    f"success {compile_route['baseline_successes']}/{compile_route['baseline_episodes']} -> {compile_route['successes']}/{compile_route['episodes']}",
                    "fast but not lossless",
                ],
            ],
        ),
        "",
        "CUDA Graph 是目前唯一已经证明同观测 bit-exact 的加速尝试，但 p50 基本不动，主要改善 p90。`torch.compile` 速度强，但已经不是无损替换。",
        "",
        "## 静态审计结论",
        "",
    ]
    for row in data["static_findings"]:
        lines.append(f"- `{row['topic']}`: {row['finding']} 判断：`{row['judgement']}`。")

    lines.extend(
        [
            "",
            "## 优先级",
            "",
            md_table(
                ["rank", "candidate", "lossless", "expected", "gate"],
                [
                    [row["rank"], row["name"], "yes" if row["lossless"] else "no", row["expected_speedup"], row["gate"]]
                    for row in data["candidates"]
                ],
            ),
            "",
            "当前建议不是先写 kernel，而是先在 5090 上做低扰动 profiler：如果没有隐藏 D2H/同步，Torch 层无损空间大概率只剩 hygiene、prewarm、CUDA Graph p90 优化；主 p50 加速仍要回到 kernel/compile/闭环修正路线。",
            "",
            "## 5090 下一步命令草案",
            "",
            "```bash",
            *data["remote_commands"],
            "```",
            "",
            "## 产物",
            "",
            f"- JSON: `{rel(OUT_JSON)}`",
            f"- Report: `{rel(OUT_MD)}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    baseline = summarize_fp16_baseline()
    data = {
        "boundary": "Offline lossless Torch-level acceleration audit; no GPU execution in this phase.",
        "fp16_baseline": baseline,
        "fp16_profiles": summarize_fp16_profiles(float(baseline["server_get_action_p50_ms"])),
        "existing_routes": summarize_existing_acceleration_routes(),
        "static_findings": build_static_findings(),
        "candidates": build_candidates(),
        "remote_commands": build_remote_commands(),
    }
    write_json(OUT_JSON, data)
    write_text(OUT_MD, build_report(data))
    print(f"Wrote {rel(OUT_JSON)}")
    print(f"Wrote {rel(OUT_MD)}")


if __name__ == "__main__":
    main()
