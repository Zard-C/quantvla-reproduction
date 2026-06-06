"""Offline GR00T get_action smoke with packed W4A16 Linear modules."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase3_activation_capture import build_variant_observation, group_for_module
from phase3_fake_quant_forward import (
    action_to_vector,
    compare_actions,
    config_groups,
    set_seed,
    set_submodule,
)
from phase3_gr00t_smoke import _insert_paths
from triton_w4a16 import PackedW4A16Linear


def patch_w4a16_modules(
    model: Any,
    groups_to_quantize: set[str],
    *,
    prefer_triton: bool,
) -> dict[str, Any]:
    import torch.nn as nn

    records: dict[str, Any] = {}
    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in list(model.named_modules()):
        group = group_for_module(name)
        if group in groups_to_quantize and isinstance(module, nn.Linear):
            targets.append((name, module, group))

    for name, module, group in targets:
        packed = PackedW4A16Linear.from_linear(module, prefer_triton=prefer_triton)
        records[name] = {
            "module": name,
            "group": group,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": module.bias is not None,
            "fp16_weight_bytes": 2 * int(module.in_features) * int(module.out_features),
            "packed_weight_bytes": int(packed.qweight.numel() * packed.qweight.element_size()),
            "scale_bytes": int(packed.scale.numel() * packed.scale.element_size()),
            "bias_bytes": int(packed.bias.numel() * packed.bias.element_size()) if packed.bias is not None else 0,
        }
        set_submodule(model, name, packed)
    return records


def summarize_patch(records: dict[str, Any]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in records.values():
        by_group.setdefault(row["group"], []).append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        fp16 = sum(int(row["fp16_weight_bytes"]) for row in rows)
        packed = sum(int(row["packed_weight_bytes"]) for row in rows)
        scale = sum(int(row["scale_bytes"]) for row in rows)
        bias = sum(int(row["bias_bytes"]) for row in rows)
        return {
            "modules": len(rows),
            "fp16_weight_bytes": fp16,
            "packed_weight_bytes": packed,
            "scale_bytes": scale,
            "bias_bytes": bias,
            "packed_plus_scale_ratio_vs_fp16": float((packed + scale) / max(1, fp16)),
        }

    return {
        "all": summarize(list(records.values())),
        "by_group": {group: summarize(rows) for group, rows in sorted(by_group.items())},
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--config", default="llm_dit_mlp")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260606)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase6_w4a16_forward_smoke.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

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

    teacher_actions: list[dict[str, Any]] = []
    teacher_started = time.time()
    for item in observations:
        set_seed(int(item["seed"]))
        teacher_actions.append(policy.get_action(item["obs"]))
    if args.device == "cuda":
        torch.cuda.synchronize()
    teacher_seconds = time.time() - teacher_started

    patch_started = time.time()
    records = patch_w4a16_modules(
        policy.model,
        config_groups(args.config),
        prefer_triton=not args.force_reference,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    patch_seconds = time.time() - patch_started

    student_actions: list[dict[str, Any]] = []
    student_started = time.time()
    for item in observations:
        set_seed(int(item["seed"]))
        student_actions.append(policy.get_action(item["obs"]))
    if args.device == "cuda":
        torch.cuda.synchronize()
    student_seconds = time.time() - student_started

    comparisons = []
    for item, teacher, student in zip(observations, teacher_actions, student_actions, strict=True):
        row = compare_actions(teacher, student)
        row["variant"] = item["variant"]
        row["seed"] = int(item["seed"])
        row["action_size"] = int(action_to_vector(teacher).size)
        comparisons.append(row)

    result = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "config": args.config,
        "prefer_triton": not args.force_reference,
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "load_seconds": load_seconds,
        "teacher_seconds": teacher_seconds,
        "patch_seconds": patch_seconds,
        "student_seconds": student_seconds,
        "patched_modules": len(records),
        "patch_summary": summarize_patch(records),
        "comparisons": comparisons,
        "summary": aggregate_metrics(comparisons),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "patch_summary": result["patch_summary"], "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
