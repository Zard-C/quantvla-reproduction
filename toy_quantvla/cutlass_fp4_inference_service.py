"""GR00T inference service patched with CUTLASS blockscaled FP4 Linear modules."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from phase3_atm_ohb_forward import (
    collect_attention_stats,
    compute_atm_ohb_scales,
    install_attention_processors,
    scale_summary,
    select_scales,
)
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase4_real_data_validation import load_observations, parse_indices, sample_indices
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase8_cutlass_blockscaled_fp4_forward_smoke import (
    cuda_memory,
    module_results,
    parse_name_contains,
    patch_cutlass_fp4_modules,
    reset_module_stats,
    reset_cuda_peak,
    run_actions,
    synchronize,
)
from phase8_cutlass_blockscaled_fp4_real_activation_bench import build_observations
from phase8_cutlass_blockscaled_fp4_smoke import parse_tuple
from timing_utils import TimedPolicyWrapper


DEFAULT_CALIBRATION_INDICES = (
    "115,462,632,1063,1273,1482,1823,2034,"
    "2406,2536,3053,3198,3492,3824,3980,4299"
)


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def write_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def summarize_patch(records: dict[str, Any]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    for row in records.values():
        family = str(row["family"])
        by_family[family] = by_family.get(family, 0) + 1
    return {"modules": len(records), "by_family": dict(sorted(by_family.items()))}


def build_module_delta_tracer(patched_modules: dict[str, Any]):
    """Return a compact per-request module delta tracer for JSONL latency tails."""

    last: dict[str, dict[str, Any]] = {}
    for name, module in patched_modules.items():
        last[name] = {
            "calls": int(module.stats.calls),
            "local_hits": int(module.stats.local_compile_cache_hits),
            "shared_hits": int(module.stats.shared_compile_cache_hits),
            "compile_count": len(module.stats.compile_seconds),
            "compile_seconds_sum": float(sum(module.stats.compile_seconds)),
        }

    def trace(_request_index: int, _seconds: float) -> dict[str, Any]:
        total_call_delta = 0
        total_local_hit_delta = 0
        total_shared_hit_delta = 0
        total_compile_delta = 0
        total_compile_seconds_delta = 0.0
        compile_events = []
        for name, module in patched_modules.items():
            stats = module.stats
            prev = last[name]
            calls = int(stats.calls)
            local_hits = int(stats.local_compile_cache_hits)
            shared_hits = int(stats.shared_compile_cache_hits)
            compile_count = len(stats.compile_seconds)
            compile_seconds_sum = float(sum(stats.compile_seconds))

            call_delta = calls - int(prev["calls"])
            local_hit_delta = local_hits - int(prev["local_hits"])
            shared_hit_delta = shared_hits - int(prev["shared_hits"])
            compile_delta = compile_count - int(prev["compile_count"])
            compile_seconds_delta = compile_seconds_sum - float(prev["compile_seconds_sum"])

            total_call_delta += call_delta
            total_local_hit_delta += local_hit_delta
            total_shared_hit_delta += shared_hit_delta
            total_compile_delta += compile_delta
            total_compile_seconds_delta += compile_seconds_delta
            if compile_delta:
                compile_events.append(
                    {
                        "module": name,
                        "compile_count_delta": int(compile_delta),
                        "compile_seconds_delta": float(compile_seconds_delta),
                        "compiled_m_values": sorted(int(m) for m in module._compiled_by_m),
                    }
                )

            prev["calls"] = calls
            prev["local_hits"] = local_hits
            prev["shared_hits"] = shared_hits
            prev["compile_count"] = compile_count
            prev["compile_seconds_sum"] = compile_seconds_sum

        return {
            "fp4_module_calls_delta": int(total_call_delta),
            "fp4_local_compile_cache_hits_delta": int(total_local_hit_delta),
            "fp4_shared_compile_cache_hits_delta": int(total_shared_hit_delta),
            "fp4_compile_count_delta": int(total_compile_delta),
            "fp4_compile_seconds_delta": float(total_compile_seconds_delta),
            "fp4_compile_events": compile_events,
        }

    return trace


def build_prewarm_observations(args: argparse.Namespace, data_config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.prewarm_observations <= 0:
        return [], {
            "observation_source": args.prewarm_observation_source,
            "prewarm_observations": 0,
        }
    prewarm_args = SimpleNamespace(
        observation_source=args.prewarm_observation_source,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        num_observations=args.prewarm_observations,
        indices=args.prewarm_indices,
        start_index=args.prewarm_start_index,
        sample_stride=args.prewarm_sample_stride,
        synthetic_variants=args.synthetic_variants,
        base_seed=args.prewarm_base_seed,
    )
    return build_observations(prewarm_args, data_config)


def build_calibration_observations(args: argparse.Namespace, data_config: Any) -> list[dict[str, Any]]:
    from gr00t.data.dataset import LeRobotSingleDataset

    dataset = LeRobotSingleDataset(
        dataset_path=str(args.dataset_path),
        modality_configs=data_config.modality_config(),
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
    )
    calibration_indices = sample_indices(
        len(dataset),
        num_observations=args.num_calibration_observations,
        start_index=args.calibration_start_index,
        stride=args.calibration_sample_stride,
        explicit_indices=parse_indices(args.calibration_indices),
    )
    observations = load_observations(dataset, calibration_indices, base_seed=args.calibration_base_seed)
    for item, index in zip(observations, calibration_indices, strict=True):
        item["dataset_index"] = int(index)
    return observations


def add_prewarm_task_descriptions(
    observations: list[dict[str, Any]],
    descriptions: list[str],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Add warmup observations with controlled task text to cover LLM sequence lengths."""

    if not descriptions:
        return observations
    if not observations:
        raise ValueError("--prewarm-task-description requires at least one base prewarm observation")

    out = list(observations)
    base = observations[0]
    base_obs = dict(base["obs"])
    for offset, description in enumerate(descriptions, start=1):
        obs = dict(base_obs)
        obs["annotation.human.action.task_description"] = [description]
        item = dict(base)
        item["obs"] = obs
        item["seed"] = int(base_seed) + offset
        item["prewarm_task_description"] = description
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--max-modules", type=int, default=0)
    parser.add_argument("--name-contains")
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN", "Float8E8M0FNU"], default="Float8E4M3FN")
    parser.add_argument("--tile-shape-mnk", type=lambda s: parse_tuple(s, 3), default=(128, 128, 128))
    parser.add_argument("--epi-tile", type=lambda s: parse_tuple(s, 2), default=(64, 32))
    parser.add_argument("--pack-backend", choices=["helper", "torch", "triton"], default="triton")
    parser.add_argument("--no-share-compile-cache", action="store_true")
    parser.add_argument("--cache-output-tensor", action="store_true")
    parser.add_argument("--profile-modules", action="store_true")
    parser.add_argument("--atm-ohb-mode", choices=["none", "identity", "atm", "ohb", "atm_ohb"], default="none")
    parser.add_argument("--log-clamp", type=float, default=0.3)
    parser.add_argument("--ohb-skip-epsilon", type=float, default=0.0)
    parser.add_argument("--calibration-indices", default=DEFAULT_CALIBRATION_INDICES)
    parser.add_argument("--num-calibration-observations", type=int, default=16)
    parser.add_argument("--calibration-start-index", type=int, default=0)
    parser.add_argument("--calibration-sample-stride", type=int, default=100)
    parser.add_argument("--calibration-base-seed", type=int, default=360204)
    parser.add_argument("--prewarm-observations", type=int, default=1)
    parser.add_argument("--prewarm-observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--prewarm-indices", default="115")
    parser.add_argument("--prewarm-start-index", type=int, default=0)
    parser.add_argument("--prewarm-sample-stride", type=int, default=100)
    parser.add_argument("--prewarm-base-seed", type=int, default=20260608)
    parser.add_argument("--prewarm-task-description", action="append", default=[])
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_fp4_server_prepare.json"))
    parser.add_argument("--server-latency-json", type=Path, help="Optional server-side get_action latency JSON.")
    parser.add_argument("--server-latency-flush-every", type=int, default=0)
    parser.add_argument("--server-request-trace-jsonl", type=Path, help="Optional per-request server latency JSONL.")
    parser.add_argument("--server-request-trace-min-seconds", type=float, default=0.0)
    parser.add_argument("--server-request-trace-module-deltas", action="store_true")
    parser.add_argument("--server-request-trace-cuda-sync", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    result: dict[str, Any] = {
        "boundary": "CUTLASS blockscaled FP4 inference server with optional policy prewarm",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "max_modules": args.max_modules,
        "name_contains": parse_name_contains(args.name_contains),
        "cutlass_root": str(args.cutlass_root),
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "epi_tile": list(args.epi_tile),
        "pack_backend": args.pack_backend,
        "share_compile_cache": not args.no_share_compile_cache,
        "cache_output_tensor": args.cache_output_tensor,
        "atm_ohb_mode": args.atm_ohb_mode,
        "log_clamp": args.log_clamp,
        "ohb_skip_epsilon": args.ohb_skip_epsilon,
        "video_backend": args.video_backend,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
    }

    started = time.perf_counter()
    reset_cuda_peak(args.device)
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    synchronize(args.device)
    result["model_load_seconds"] = time.perf_counter() - started
    result["model_load_memory"] = cuda_memory(args.device)

    calibration_observations: list[dict[str, Any]] = []
    teacher_stats = {}
    if args.atm_ohb_mode != "none":
        calibration_started = time.perf_counter()
        calibration_observations = build_calibration_observations(args, data_config)
        result["calibration_dataset_indices"] = [
            int(item["dataset_index"]) for item in calibration_observations
        ]
        result["calibration_base_seed"] = args.calibration_base_seed
        result["calibration_load_seconds"] = time.perf_counter() - calibration_started

        attention_started = time.perf_counter()
        teacher_stats = collect_attention_stats(policy, calibration_observations)
        synchronize(args.device)
        result["teacher_attention_calibration_seconds"] = time.perf_counter() - attention_started
    else:
        result["calibration_dataset_indices"] = []
        result["calibration_base_seed"] = args.calibration_base_seed
        result["calibration_load_seconds"] = 0.0
        result["teacher_attention_calibration_seconds"] = 0.0

    patch_started = time.perf_counter()
    records, patched_modules = patch_cutlass_fp4_modules(
        policy.model,
        args.scope,
        cutlass_root=args.cutlass_root,
        sf_vec_size=args.sf_vec_size,
        sf_dtype=args.sf_dtype,
        tile_shape_mnk=args.tile_shape_mnk,
        epi_tile=args.epi_tile,
        pack_backend=args.pack_backend,
        share_compile_cache=not args.no_share_compile_cache,
        cache_output_tensor=args.cache_output_tensor,
        max_modules=args.max_modules,
        name_contains=parse_name_contains(args.name_contains),
        profile=args.profile_modules,
    )
    synchronize(args.device)
    result["patch_seconds"] = time.perf_counter() - patch_started
    result["patched_modules"] = len(records)
    result["patch_summary"] = summarize_patch(records)
    result["post_patch_memory"] = cuda_memory(args.device)

    scales = {}
    if args.atm_ohb_mode != "none":
        attention_started = time.perf_counter()
        student_stats = collect_attention_stats(policy, calibration_observations)
        synchronize(args.device)
        scales = compute_atm_ohb_scales(teacher_stats, student_stats, log_clamp=args.log_clamp)
        selected_scales = select_scales(scales, args.atm_ohb_mode)
        if args.ohb_skip_epsilon > 0.0:
            selected_scales = {
                name: scale
                for name, scale in selected_scales.items()
                if ("beta" not in scale) or abs(float(scale["beta"]) - 1.0) > args.ohb_skip_epsilon
            }
        install_attention_processors(
            policy.model,
            mode="apply",
            scales=selected_scales,
        )
        result["student_attention_calibration_seconds"] = time.perf_counter() - attention_started
        result["scale_summary"] = scale_summary(scales)
        result["attention_processors_applied"] = len(selected_scales)
    else:
        result["student_attention_calibration_seconds"] = 0.0
        result["scale_summary"] = None
        result["attention_processors_applied"] = 0
    result["module_results_after_calibration"] = module_results(patched_modules)

    prewarm_started = time.perf_counter()
    observations, observation_meta = build_prewarm_observations(args, data_config)
    observations = add_prewarm_task_descriptions(
        observations,
        args.prewarm_task_description,
        base_seed=args.prewarm_base_seed,
    )
    if args.prewarm_task_description:
        observation_meta["prewarm_task_descriptions"] = list(args.prewarm_task_description)
        observation_meta["prewarm_observations_with_task_descriptions"] = len(observations)
    result["prewarm_observation_meta"] = observation_meta
    if observations:
        _actions, prewarm_seconds, prewarm_memory = run_actions(policy, observations, device=args.device)
    else:
        prewarm_seconds = 0.0
        prewarm_memory = cuda_memory(args.device)
    result["prewarm_seconds"] = prewarm_seconds
    result["prewarm_total_seconds"] = time.perf_counter() - prewarm_started
    result["prewarm_memory"] = prewarm_memory
    result["module_results_after_prewarm"] = module_results(patched_modules)
    result["prepare_seconds"] = time.perf_counter() - started
    write_json(args.output_json, result)
    reset_module_stats(patched_modules)

    if args.prepare_only:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting CUTLASS FP4 server on port {args.port}", flush=True)
    set_seed(args.prewarm_base_seed)
    timed_policy = TimedPolicyWrapper(
        policy,
        output_json=args.server_latency_json,
        label=f"cutlass_fp4:{args.scope}:{args.atm_ohb_mode}",
        flush_every=args.server_latency_flush_every,
        extra_summary=lambda: {"module_results": module_results(patched_modules)},
        request_trace_jsonl=args.server_request_trace_jsonl,
        request_trace_min_seconds=args.server_request_trace_min_seconds,
        request_extra=build_module_delta_tracer(patched_modules)
        if args.server_request_trace_module_deltas
        else None,
        cuda_sync_device=args.device if args.server_request_trace_cuda_sync else None,
    )
    server = RobotInferenceServer(timed_policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
