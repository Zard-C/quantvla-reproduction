"""Offline FP16 Linear hotspot profile for GR00T get_action.

This avoids simulator rollout noise. It runs a small number of real LIBERO
observations through the FP16 policy while wrapping selected Linear modules
with synchronized timers, then ranks modules and projection suffixes by total
forward time.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from fp16_linear_profiler import (
    module_results,
    parse_name_contains,
    patch_timed_fp16_modules,
    reset_module_stats,
)
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, module_family, scope_description
from phase8_cutlass_blockscaled_fp4_real_activation_bench import (
    build_observations,
    cuda_memory,
    reset_cuda_peak,
    synchronize,
)
from timing_utils import summarize_float


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def expanded_observations(observations: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for item in observations:
            copied = dict(item)
            copied["seed"] = int(item["seed"]) + repeat * 10_000
            copied["repeat_index"] = int(repeat)
            out.append(copied)
    return out


def run_requests(policy: Any, observations: list[dict[str, Any]], *, device: str) -> tuple[dict[str, Any], dict[str, int]]:
    request_seconds: list[float] = []
    reset_cuda_peak(device)
    for item in observations:
        set_seed(int(item["seed"]))
        started = time.perf_counter()
        policy.get_action(item["obs"])
        synchronize(device)
        request_seconds.append(time.perf_counter() - started)
    return summarize_float(request_seconds), cuda_memory(device)


def summarize_modules(records: dict[str, Any], raw_results: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, float | int]] = {}
    by_suffix: dict[str, dict[str, float | int]] = {}

    for name, result in raw_results.items():
        stats = result["stats"]["forward_seconds"]
        calls = int(stats["count"])
        total_seconds = float(stats["mean"]) * calls
        family = module_family(name) or str(records.get(name, {}).get("family", "unknown"))
        suffix = name.rsplit(".", 1)[-1]
        row = {
            "module": name,
            "family": family,
            "suffix": suffix,
            "calls": calls,
            "total_seconds": total_seconds,
            "mean_seconds": float(stats["mean"]),
            "p50_seconds": float(stats["p50"]),
            "p90_seconds": float(stats["p90"]),
            "p99_seconds": float(stats["p99"]),
            "in_features": int(result["in_features"]),
            "out_features": int(result["out_features"]),
            "bias": bool(result["bias"]),
        }
        rows.append(row)

        for key, bucket in ((family, by_family), (suffix, by_suffix)):
            if key not in bucket:
                bucket[key] = {"modules": 0, "calls": 0, "total_seconds": 0.0}
            bucket[key]["modules"] = int(bucket[key]["modules"]) + 1
            bucket[key]["calls"] = int(bucket[key]["calls"]) + calls
            bucket[key]["total_seconds"] = float(bucket[key]["total_seconds"]) + total_seconds

    rows.sort(key=lambda row: float(row["total_seconds"]), reverse=True)
    total_profiled_seconds = sum(float(row["total_seconds"]) for row in rows)
    for bucket in (by_family, by_suffix):
        for value in bucket.values():
            value["share_of_profiled_linear_seconds"] = (
                float(value["total_seconds"]) / total_profiled_seconds if total_profiled_seconds else 0.0
            )

    return {
        "total_profiled_linear_seconds": total_profiled_seconds,
        "top_modules": rows[: int(top_k)],
        "by_family": dict(sorted(by_family.items())),
        "by_suffix": dict(sorted(by_suffix.items())),
    }


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
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="llm_dit_mlp")
    parser.add_argument("--name-contains")
    parser.add_argument("--max-modules", type=int, default=0)
    parser.add_argument("--observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--indices", default="115")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260609)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase10_fp16_linear_hotspot_profile.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    observations, observation_meta = build_observations(args, data_config)
    warmup_observations = expanded_observations(observations, args.warmup_repeats)
    profile_observations = expanded_observations(observations, args.profile_repeats)

    load_started = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    synchronize(args.device)
    load_seconds = time.perf_counter() - load_started

    patch_started = time.perf_counter()
    records, profiled_modules = patch_timed_fp16_modules(
        policy.model,
        args.scope,
        max_modules=args.max_modules,
        name_contains=parse_name_contains(args.name_contains),
        profile=True,
    )
    synchronize(args.device)
    patch_seconds = time.perf_counter() - patch_started

    warmup_summary, warmup_memory = run_requests(policy, warmup_observations, device=args.device)
    reset_module_stats(profiled_modules)
    profile_summary, profile_memory = run_requests(policy, profile_observations, device=args.device)
    raw_results = module_results(profiled_modules)
    module_summary = summarize_modules(records, raw_results, top_k=args.top_k)

    payload = {
        "boundary": "Offline FP16 synchronized Linear hotspot profile",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "name_contains": parse_name_contains(args.name_contains),
        "max_modules": int(args.max_modules),
        "observation_meta": observation_meta,
        "warmup_requests": len(warmup_observations),
        "profile_requests": len(profile_observations),
        "load_seconds": float(load_seconds),
        "patch_seconds": float(patch_seconds),
        "profiled_modules": len(records),
        "profile_patch_records": records,
        "warmup_request_seconds": warmup_summary,
        "profile_request_seconds": profile_summary,
        "warmup_memory": warmup_memory,
        "profile_memory": profile_memory,
        "module_summary": module_summary,
        "raw_module_results": raw_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "scope": args.scope,
        "profiled_modules": len(records),
        "profile_requests": len(profile_observations),
        "profile_request_seconds": profile_summary,
        "by_family": module_summary["by_family"],
        "by_suffix": module_summary["by_suffix"],
        "top_modules": module_summary["top_modules"][:10],
        "output_json": str(args.output_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
