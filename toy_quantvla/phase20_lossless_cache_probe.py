"""A/B probe for semantic-preserving GR00T cache patches.

This script compares stock FP16 eager inference against the low-risk cache
patches in ``lossless_cache_patches.py`` on fixed real observations.  It does
not use torch.compile and does not modify model weights.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from lossless_cache_patches import install_lossless_cache_patches, lossless_cache_stats
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
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


def flatten_action(action: dict[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for key in sorted(action):
        value = action[key]
        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        chunks.append(arr.astype(np.float64, copy=False).reshape(-1))
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks)


def compare_action_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    diff = candidate - reference
    ref_norm = float(np.linalg.norm(reference))
    cand_norm = float(np.linalg.norm(candidate))
    denom = max(ref_norm * cand_norm, 1e-12)
    cosine = float(np.dot(reference, candidate) / denom)
    return {
        "shape_equal": True,
        "numel": int(reference.size),
        "max_abs_diff": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "reference_norm": ref_norm,
        "candidate_norm": cand_norm,
        "cosine": cosine,
        "allclose_rtol0_atol0": bool(np.array_equal(reference, candidate)),
        "allclose_rtol1e_6_atol1e_6": bool(np.allclose(reference, candidate, rtol=1e-6, atol=1e-6)),
        "allclose_rtol1e_5_atol1e_5": bool(np.allclose(reference, candidate, rtol=1e-5, atol=1e-5)),
    }


def run_requests(
    policy: Any,
    observations: list[dict[str, Any]],
    *,
    device: str,
    collect_actions: bool,
) -> tuple[dict[str, Any], dict[str, int], list[np.ndarray]]:
    latencies: list[float] = []
    actions: list[np.ndarray] = []
    reset_cuda_peak(device)
    for item in observations:
        set_seed(int(item["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            action = policy.get_action(item["obs"])
        synchronize(device)
        latencies.append(time.perf_counter() - started)
        if collect_actions:
            actions.append(flatten_action(action))
    return summarize_float(latencies), cuda_memory(device), actions


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
    parser.add_argument("--observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--indices", default="115")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260620)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--compare-repeats", type=int, default=3)
    parser.add_argument("--prepare-input-pruning", action="store_true")
    parser.add_argument("--action-head-static-cache", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase20_lossless_cache_probe.json"))
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
    compare_observations = expanded_observations(observations, args.compare_repeats)

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

    baseline_warmup, baseline_warmup_memory, _ = run_requests(
        policy,
        warmup_observations,
        device=args.device,
        collect_actions=False,
    )
    baseline_compare, _baseline_compare_memory, baseline_actions = run_requests(
        policy,
        compare_observations,
        device=args.device,
        collect_actions=True,
    )
    baseline_profile, baseline_profile_memory, _ = run_requests(
        policy,
        profile_observations,
        device=args.device,
        collect_actions=False,
    )

    patch_result = install_lossless_cache_patches(
        policy,
        prepare_input_pruning=bool(args.prepare_input_pruning),
        action_head_static_cache=bool(args.action_head_static_cache),
    )
    cached_warmup, cached_warmup_memory, _ = run_requests(
        policy,
        warmup_observations,
        device=args.device,
        collect_actions=False,
    )
    cached_compare, _cached_compare_memory, cached_actions = run_requests(
        policy,
        compare_observations,
        device=args.device,
        collect_actions=True,
    )
    cached_profile, cached_profile_memory, _ = run_requests(
        policy,
        profile_observations,
        device=args.device,
        collect_actions=False,
    )

    comparisons = [
        compare_action_arrays(reference, candidate)
        for reference, candidate in zip(baseline_actions, cached_actions)
    ]
    max_abs = max((float(item.get("max_abs_diff", float("inf"))) for item in comparisons), default=0.0)
    all_equal = all(bool(item.get("allclose_rtol0_atol0", False)) for item in comparisons)
    all_close_1e6 = all(bool(item.get("allclose_rtol1e_6_atol1e_6", False)) for item in comparisons)

    baseline_p50 = float(baseline_profile.get("p50") or 0.0)
    cached_p50 = float(cached_profile.get("p50") or 0.0)
    speedup_p50 = baseline_p50 / cached_p50 if baseline_p50 > 0 and cached_p50 > 0 else None

    payload = {
        "boundary": "Lossless cache A/B probe; no torch.compile",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "observation_meta": observation_meta,
        "warmup_requests": len(warmup_observations),
        "compare_requests": len(compare_observations),
        "profile_requests": len(profile_observations),
        "load_seconds": float(load_seconds),
        "patch_request": {
            "prepare_input_pruning": bool(args.prepare_input_pruning),
            "action_head_static_cache": bool(args.action_head_static_cache),
        },
        "patch_result": patch_result,
        "baseline_warmup_seconds": baseline_warmup,
        "baseline_compare_seconds": baseline_compare,
        "baseline_profile_seconds": baseline_profile,
        "baseline_profile_memory": baseline_profile_memory,
        "cached_warmup_seconds": cached_warmup,
        "cached_compare_seconds": cached_compare,
        "cached_profile_seconds": cached_profile,
        "cached_profile_memory": cached_profile_memory,
        "cached_warmup_memory": cached_warmup_memory,
        "action_comparisons": comparisons,
        "action_summary": {
            "all_bitwise_equal_after_unapply": bool(all_equal),
            "allclose_1e_6": bool(all_close_1e6),
            "max_abs_diff": float(max_abs),
        },
        "speedup": {
            "p50": speedup_p50,
            "baseline_p50_seconds": baseline_p50,
            "cached_p50_seconds": cached_p50,
        },
        "cache_stats": lossless_cache_stats(policy),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "profile_requests": payload["profile_requests"],
                "patch_request": payload["patch_request"],
                "action_summary": payload["action_summary"],
                "speedup": payload["speedup"],
                "baseline_profile_seconds": baseline_profile,
                "cached_profile_seconds": cached_profile,
                "cache_stats": payload["cache_stats"],
                "output_json": str(args.output_json),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

