"""GR00T FP16 inference service with static DiT residual-cache blocks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase8_cutlass_blockscaled_fp4_forward_smoke import cuda_memory, reset_cuda_peak, run_actions, synchronize
from phase8_cutlass_blockscaled_fp4_real_activation_bench import build_observations
from phase11_dit_residual_cache_smoke import (
    ResidualCacheBlock,
    parse_int_set,
    patch_residual_cache_blocks,
)
from timing_utils import TimedPolicyWrapper


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


def wrapper_results(wrappers: dict[str, ResidualCacheBlock]) -> dict[str, Any]:
    return {name: wrapper.to_result() for name, wrapper in wrappers.items()}


def reset_wrappers(wrappers: dict[str, ResidualCacheBlock]) -> None:
    for wrapper in wrappers.values():
        wrapper.reset_request()


def cache_summary(wrappers: dict[str, ResidualCacheBlock], *, denoising_steps: int) -> dict[str, Any]:
    results = wrapper_results(wrappers)
    cache_hits = sum(int(row["cache_hits"]) for row in results.values())
    full_compute_calls = sum(int(row["full_compute_calls"]) for row in results.values())
    patched_calls = sum(int(row["calls"]) for row in results.values())
    all_dit_block_calls = int(16 * int(denoising_steps) * max(1, patched_calls // max(1, len(wrappers) * int(denoising_steps))))
    return {
        "patched_blocks": len(wrappers),
        "patched_block_calls": int(patched_calls),
        "full_compute_calls": int(full_compute_calls),
        "cache_hits": int(cache_hits),
        "cache_misses": int(sum(int(row["cache_misses"]) for row in results.values())),
        "patched_block_hit_rate": float(cache_hits / patched_calls) if patched_calls else 0.0,
        "estimated_all_dit_block_calls": int(all_dit_block_calls),
        "estimated_all_dit_block_skip_rate": float(cache_hits / all_dit_block_calls) if all_dit_block_calls else 0.0,
        "wrapper_results": results,
    }


class ResidualCachePolicy:
    """Proxy that resets DiT residual-cache wrappers once per policy request."""

    def __init__(self, policy: Any, wrappers: dict[str, ResidualCacheBlock]):
        self._policy = policy
        self._wrappers = wrappers

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        reset_wrappers(self._wrappers)
        return self._policy.get_action(observation)


def build_cache_delta_tracer(wrappers: dict[str, ResidualCacheBlock]):
    last: dict[str, dict[str, int]] = {
        name: {
            "calls": int(wrapper.calls),
            "full_compute_calls": int(wrapper.full_compute_calls),
            "cache_hits": int(wrapper.cache_hits),
            "cache_misses": int(wrapper.cache_misses),
        }
        for name, wrapper in wrappers.items()
    }

    def trace(_request_index: int, _seconds: float) -> dict[str, Any]:
        total_calls = 0
        total_full = 0
        total_hits = 0
        total_misses = 0
        per_block: dict[str, Any] = {}
        for name, wrapper in wrappers.items():
            prev = last[name]
            calls = int(wrapper.calls)
            full = int(wrapper.full_compute_calls)
            hits = int(wrapper.cache_hits)
            misses = int(wrapper.cache_misses)
            delta = {
                "calls_delta": calls - int(prev["calls"]),
                "full_compute_calls_delta": full - int(prev["full_compute_calls"]),
                "cache_hits_delta": hits - int(prev["cache_hits"]),
                "cache_misses_delta": misses - int(prev["cache_misses"]),
            }
            per_block[name] = {
                "block_index": int(wrapper.block_index),
                **delta,
            }
            total_calls += int(delta["calls_delta"])
            total_full += int(delta["full_compute_calls_delta"])
            total_hits += int(delta["cache_hits_delta"])
            total_misses += int(delta["cache_misses_delta"])
            prev["calls"] = calls
            prev["full_compute_calls"] = full
            prev["cache_hits"] = hits
            prev["cache_misses"] = misses
        return {
            "residual_cache_block_calls_delta": int(total_calls),
            "residual_cache_full_compute_delta": int(total_full),
            "residual_cache_hits_delta": int(total_hits),
            "residual_cache_misses_delta": int(total_misses),
            "residual_cache_per_block": per_block,
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


def add_prewarm_task_descriptions(
    observations: list[dict[str, Any]],
    descriptions: list[str],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
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
    parser.add_argument("--cache-blocks", default="12,15")
    parser.add_argument("--cache-steps", default="1-6")
    parser.add_argument("--prewarm-observations", type=int, default=1)
    parser.add_argument("--prewarm-observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--prewarm-indices", default="115")
    parser.add_argument("--prewarm-start-index", type=int, default=0)
    parser.add_argument("--prewarm-sample-stride", type=int, default=100)
    parser.add_argument("--prewarm-base-seed", type=int, default=20260610)
    parser.add_argument("--prewarm-task-description", action="append", default=[])
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase11_residual_cache_server_prepare.json"))
    parser.add_argument("--server-latency-json", type=Path, help="Optional server-side get_action latency JSON.")
    parser.add_argument("--server-latency-flush-every", type=int, default=0)
    parser.add_argument("--server-request-trace-jsonl", type=Path, help="Optional per-request server latency JSONL.")
    parser.add_argument("--server-request-trace-min-seconds", type=float, default=0.0)
    parser.add_argument("--server-request-trace-cache-deltas", action="store_true")
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
    cache_blocks = parse_int_set(args.cache_blocks)
    cache_steps = parse_int_set(args.cache_steps)
    result: dict[str, Any] = {
        "boundary": "FP16 GR00T inference server with static DiT residual-cache blocks",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": int(args.denoising_steps),
        "cache_blocks": sorted(int(v) for v in cache_blocks),
        "cache_steps": sorted(int(v) for v in cache_steps),
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

    patch_started = time.perf_counter()
    wrappers = patch_residual_cache_blocks(policy.model, cache_blocks=cache_blocks, cache_steps=cache_steps)
    synchronize(args.device)
    if not wrappers:
        raise RuntimeError(f"No DiT blocks patched for cache_blocks={sorted(cache_blocks)}")
    result["patch_seconds"] = time.perf_counter() - patch_started
    result["patched_blocks"] = len(wrappers)
    result["wrapper_results_after_patch"] = wrapper_results(wrappers)
    result["post_patch_memory"] = cuda_memory(args.device)

    residual_policy = ResidualCachePolicy(policy, wrappers)
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
        _actions, prewarm_seconds, prewarm_memory = run_actions(residual_policy, observations, device=args.device)
    else:
        prewarm_seconds = 0.0
        prewarm_memory = cuda_memory(args.device)
    result["prewarm_seconds"] = float(prewarm_seconds)
    result["prewarm_total_seconds"] = time.perf_counter() - prewarm_started
    result["prewarm_memory"] = prewarm_memory
    result["cache_summary_after_prewarm"] = cache_summary(wrappers, denoising_steps=args.denoising_steps)
    result["prepare_seconds"] = time.perf_counter() - started
    write_json(args.output_json, result)

    if args.prepare_only:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting DiT residual-cache server on port {args.port}", flush=True)
    set_seed(args.prewarm_base_seed)
    timed_policy = TimedPolicyWrapper(
        residual_policy,
        output_json=args.server_latency_json,
        label="fp16_dit_residual_cache",
        flush_every=args.server_latency_flush_every,
        extra_summary=lambda: {"cache_summary": cache_summary(wrappers, denoising_steps=args.denoising_steps)},
        request_trace_jsonl=args.server_request_trace_jsonl,
        request_trace_min_seconds=args.server_request_trace_min_seconds,
        request_extra=build_cache_delta_tracer(wrappers) if args.server_request_trace_cache_deltas else None,
        cuda_sync_device=args.device if args.server_request_trace_cuda_sync else None,
    )
    server = RobotInferenceServer(timed_policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
