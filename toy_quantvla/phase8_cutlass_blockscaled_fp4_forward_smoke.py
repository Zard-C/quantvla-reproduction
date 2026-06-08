"""Offline GR00T get_action smoke with CUTLASS blockscaled FP4 Linear modules."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from cutlass_blockscaled_fp4_linear import CutlassBlockscaledFP4Linear
from phase3_fake_quant_forward import action_to_vector, compare_actions, set_seed, set_submodule
from phase3_gr00t_smoke import _insert_paths
from phase4_real_data_validation import load_observations, parse_indices, sample_indices
from phase6_w4a16_scopes import SCOPE_CHOICES, include_module_for_scope, module_family, scope_description
from phase8_cutlass_blockscaled_fp4_smoke import parse_tuple
from phase8_cutlass_blockscaled_fp4_real_activation_bench import build_observations


def synchronize(device: str = "cuda") -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_cuda_peak(device: str = "cuda") -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory(device: str = "cuda") -> dict[str, int]:
    if not (torch.device(device).type == "cuda" and torch.cuda.is_available()):
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def parse_name_contains(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def patch_cutlass_fp4_modules(
    model: Any,
    scope: str,
    *,
    cutlass_root: Path,
    sf_vec_size: int,
    sf_dtype: str,
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    pack_backend: str,
    share_compile_cache: bool,
    max_modules: int,
    name_contains: list[str],
    profile: bool,
) -> tuple[dict[str, Any], dict[str, CutlassBlockscaledFP4Linear]]:
    import torch.nn as nn

    records: dict[str, Any] = {}
    patched_modules: dict[str, CutlassBlockscaledFP4Linear] = {}
    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in list(model.named_modules()):
        family = module_family(name)
        if family is None or not include_module_for_scope(name, scope) or not isinstance(module, nn.Linear):
            continue
        if name_contains and not any(token in name for token in name_contains):
            continue
        targets.append((name, module, family))
        if max_modules > 0 and len(targets) >= max_modules:
            break

    for name, module, family in targets:
        started = time.perf_counter()
        patched = CutlassBlockscaledFP4Linear.from_linear(
            module,
            cutlass_root=cutlass_root,
            sf_vec_size=sf_vec_size,
            sf_dtype=sf_dtype,
            tile_shape_mnk=tile_shape_mnk,
            epi_tile=epi_tile,
            pack_backend=pack_backend,
            share_compile_cache=share_compile_cache,
            profile=profile,
        )
        synchronize("cuda")
        patch_seconds = time.perf_counter() - started
        records[name] = {
            "module": name,
            "family": family,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": module.bias is not None,
            "original_weight_dtype": str(module.weight.dtype),
            "patch_seconds": patch_seconds,
            "weight_pack_seconds": patched.weight_pack_seconds,
        }
        set_submodule(model, name, patched)
        patched_modules[name] = patched
    return records, patched_modules


def run_actions(policy: Any, observations: list[dict[str, Any]], *, device: str) -> tuple[list[dict[str, Any]], float, dict[str, int]]:
    actions = []
    reset_cuda_peak(device)
    started = time.perf_counter()
    for item in observations:
        set_seed(int(item["seed"]))
        actions.append(policy.get_action(item["obs"]))
        synchronize(device)
    seconds = time.perf_counter() - started
    return actions, seconds, cuda_memory(device)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["mse", "nmse", "rmse", "relative_rmse", "cosine", "max_abs_diff", "teacher_rms", "student_rms"]
    out: dict[str, Any] = {"observations": len(rows)}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {"mean": float(vals.mean()), "max": float(vals.max()), "min": float(vals.min())}
    return out


def module_results(modules: dict[str, CutlassBlockscaledFP4Linear]) -> dict[str, Any]:
    return {name: module.to_result() for name, module in modules.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--indices")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260608)
    parser.add_argument("--max-modules", type=int, default=1)
    parser.add_argument("--name-contains")
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN", "Float8E8M0FNU"], default="Float8E4M3FN")
    parser.add_argument("--tile-shape-mnk", type=lambda s: parse_tuple(s, 3), default=(128, 128, 128))
    parser.add_argument("--epi-tile", type=lambda s: parse_tuple(s, 2), default=(64, 32))
    parser.add_argument("--pack-backend", choices=["helper", "torch", "triton"], default="helper")
    parser.add_argument("--no-share-compile-cache", action="store_true")
    parser.add_argument("--profile-modules", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    data_config = LiberoDataConfig()
    observations, observation_meta = build_observations(args, data_config)
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

    teacher_actions, teacher_seconds, teacher_memory = run_actions(policy, observations, device=args.device)
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
        max_modules=args.max_modules,
        name_contains=parse_name_contains(args.name_contains),
        profile=args.profile_modules,
    )
    synchronize(args.device)
    patch_seconds = time.perf_counter() - patch_started
    cold_actions, cold_student_seconds, cold_student_memory = run_actions(policy, observations, device=args.device)
    warm_actions, warm_student_seconds, warm_student_memory = run_actions(policy, observations, device=args.device)

    comparisons = []
    for item, teacher, student in zip(observations, teacher_actions, warm_actions, strict=True):
        row = compare_actions(teacher, student)
        row["seed"] = int(item["seed"])
        if "dataset_index" in item:
            row["dataset_index"] = int(item["dataset_index"])
        row["action_size"] = int(action_to_vector(teacher).size)
        comparisons.append(row)

    result = {
        "boundary": f"CUTLASS blockscaled FP4 module wrapper get_action smoke; pack_backend={args.pack_backend}",
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "observation_meta": observation_meta,
        "denoising_steps": args.denoising_steps,
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "pack_backend": args.pack_backend,
        "share_compile_cache": not args.no_share_compile_cache,
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "epi_tile": list(args.epi_tile),
        "max_modules": args.max_modules,
        "name_contains": parse_name_contains(args.name_contains),
        "load_seconds": load_seconds,
        "teacher_seconds": teacher_seconds,
        "patch_seconds": patch_seconds,
        "cold_student_seconds": cold_student_seconds,
        "warm_student_seconds": warm_student_seconds,
        "cold_student_over_teacher_time": float(cold_student_seconds / teacher_seconds) if teacher_seconds else None,
        "warm_student_over_teacher_time": float(warm_student_seconds / teacher_seconds) if teacher_seconds else None,
        "teacher_memory": teacher_memory,
        "cold_student_memory": cold_student_memory,
        "warm_student_memory": warm_student_memory,
        "patched_modules": len(records),
        "patch_records": records,
        "module_results": module_results(patched_modules),
        "comparisons": comparisons,
        "summary": aggregate_metrics(comparisons),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": result["summary"],
        "timing": {
            "teacher_seconds": teacher_seconds,
            "patch_seconds": patch_seconds,
            "cold_student_seconds": cold_student_seconds,
            "warm_student_seconds": warm_student_seconds,
            "warm_student_over_teacher_time": result["warm_student_over_teacher_time"],
        },
        "memory": {
            "teacher": teacher_memory,
            "cold_student": cold_student_memory,
            "warm_student": warm_student_memory,
        },
        "patched_modules": len(records),
        "output_json": str(args.output_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
