"""Offline GR00T get_action smoke with CUTLASS dense Linear modules."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from cutlass_dense_linear import CutlassDenseLinear, CutlassDenseLinearCache
from phase3_activation_capture import build_variant_observation
from phase3_fake_quant_forward import (
    action_to_vector,
    compare_actions,
    set_seed,
    set_submodule,
)
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, include_module_for_scope, module_family, scope_description


def parse_tile(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3:
        raise ValueError(f"tile must be M,N,K, got {value!r}")
    return parts


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def patch_cutlass_dense_modules(
    model: Any,
    scope: str,
    *,
    cutlass_root: Path,
    tile_shape_mnk: tuple[int, int, int],
) -> dict[str, Any]:
    import torch.nn as nn

    records: dict[str, Any] = {}
    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in list(model.named_modules()):
        family = module_family(name)
        if family is not None and include_module_for_scope(name, scope) and isinstance(module, nn.Linear):
            targets.append((name, module, family))

    for name, module, family in targets:
        patched = CutlassDenseLinear.from_linear(
            module,
            cutlass_root=cutlass_root,
            tile_shape_mnk=tile_shape_mnk,
            fallback=True,
        )
        records[name] = {
            "module": name,
            "family": family,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": module.bias is not None,
            "weight_dtype": str(module.weight.dtype),
        }
        set_submodule(model, name, patched)
    return records


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["mse", "nmse", "rmse", "relative_rmse", "cosine", "max_abs_diff", "teacher_rms", "student_rms"]
    out: dict[str, Any] = {"observations": len(rows)}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "min": float(vals.min()),
        }
    return out


def run_actions(policy: Any, observations: list[dict[str, Any]], *, device: str) -> tuple[list[dict[str, Any]], float]:
    actions = []
    started = time.perf_counter()
    for item in observations:
        set_seed(int(item["seed"]))
        actions.append(policy.get_action(item["obs"]))
    synchronize(device)
    return actions, time.perf_counter() - started


def summarize_patch(records: dict[str, Any]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    for row in records.values():
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
    return {
        "modules": len(records),
        "by_family": dict(sorted(by_family.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260606)
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--tile-shape-mnk", type=parse_tile, default=(64, 128, 64))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_dense_forward_smoke.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    CutlassDenseLinearCache.clear()
    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    data_config = LiberoDataConfig()
    load_started = time.time()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    load_seconds = time.time() - load_started

    observations = [
        {
            "variant": variants[idx % len(variants)],
            "seed": args.base_seed + idx,
            "obs": build_variant_observation(args.model_path, args.embodiment_tag, variants[idx % len(variants)], idx),
        }
        for idx in range(args.num_observations)
    ]

    teacher_actions, teacher_seconds = run_actions(policy, observations, device=args.device)

    patch_started = time.perf_counter()
    records = patch_cutlass_dense_modules(
        policy.model,
        args.scope,
        cutlass_root=args.cutlass_root,
        tile_shape_mnk=args.tile_shape_mnk,
    )
    synchronize(args.device)
    patch_seconds = time.perf_counter() - patch_started

    cold_actions, cold_student_seconds = run_actions(policy, observations, device=args.device)
    warm_actions, warm_student_seconds = run_actions(policy, observations, device=args.device)

    comparisons = []
    for item, teacher, student in zip(observations, teacher_actions, warm_actions, strict=True):
        row = compare_actions(teacher, student)
        row["variant"] = item["variant"]
        row["seed"] = int(item["seed"])
        row["action_size"] = int(action_to_vector(teacher).size)
        comparisons.append(row)

    result = {
        "boundary": "CUTLASS dense fp16/bf16 module replacement get_action smoke; not FP4 quantized yet",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "load_seconds": load_seconds,
        "teacher_seconds": teacher_seconds,
        "patch_seconds": patch_seconds,
        "cold_student_seconds": cold_student_seconds,
        "warm_student_seconds": warm_student_seconds,
        "cold_student_over_teacher_time": float(cold_student_seconds / teacher_seconds) if teacher_seconds else None,
        "warm_student_over_teacher_time": float(warm_student_seconds / teacher_seconds) if teacher_seconds else None,
        "patched_modules": len(records),
        "patch_summary": summarize_patch(records),
        "cutlass_cache": CutlassDenseLinearCache.stats(),
        "comparisons": comparisons,
        "summary": aggregate_metrics(comparisons),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "timing": {
        "teacher_seconds": teacher_seconds,
        "cold_student_seconds": cold_student_seconds,
        "warm_student_seconds": warm_student_seconds,
        "warm_student_over_teacher_time": result["warm_student_over_teacher_time"],
    }, "cache": result["cutlass_cache"], "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
