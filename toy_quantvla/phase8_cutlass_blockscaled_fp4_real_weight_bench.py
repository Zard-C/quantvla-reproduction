"""Benchmark CUTLASS blockscaled FP4 GEMM with real GR00T Linear weights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from phase3_activation_capture import build_variant_observation
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_real_shape_bench import capture_runtime_m, collect_selected_modules, representative_cases
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase8_cutlass_blockscaled_fp4_smoke import (
    load_blockscaled_example,
    make_blockscaled_fp4_operand,
    parse_tuple,
    time_ms,
)


def run_fp4_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    blockscaled = load_blockscaled_example(args.cutlass_root)
    module = case["module"]
    m, k, n = int(case["m"]), int(case["k"]), int(case["n"])
    device = torch.device(args.device)
    dtype = torch.float32
    torch.manual_seed(args.seed + m + 3 * k + 17 * n)
    x = (torch.randn((m, k, 1), device=device, dtype=dtype) / math.sqrt(k)).contiguous()
    w = module.weight.detach().to(device=device, dtype=dtype).contiguous().view(n, k, 1)
    c_ref = torch.zeros((m, n, 1), device=device, dtype=torch.float32).contiguous()
    sf_dtype = getattr(cutlass, args.sf_dtype)

    c_tensor, c_storage = cutlass_torch.cute_tensor_like(
        c_ref,
        cutlass.Float16,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
    a = make_blockscaled_fp4_operand(
        x,
        fp4_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,
        sf_vec_size=args.sf_vec_size,
        blockscaled_module=blockscaled,
        cutlass_torch=cutlass_torch,
        from_dlpack=from_dlpack,
    )
    b = make_blockscaled_fp4_operand(
        w,
        fp4_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,
        sf_vec_size=args.sf_vec_size,
        blockscaled_module=blockscaled,
        cutlass_torch=cutlass_torch,
        from_dlpack=from_dlpack,
    )

    gemm = blockscaled.Sm120BlockScaledGemmKernel(
        cutlass.Float32,
        args.sf_vec_size,
        args.tile_shape_mnk,
        args.epi_tile,
    )
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()
    compile_started = time.perf_counter()
    compiled = cute.compile(
        gemm,
        a["fp4_tensor"],
        b["fp4_tensor"],
        a["scale_tensor"],
        b["scale_tensor"],
        c_tensor,
        max_active_clusters,
        stream,
    )
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - compile_started

    def cutlass_call() -> None:
        compiled(a["fp4_tensor"], b["fp4_tensor"], a["scale_tensor"], b["scale_tensor"], c_tensor, stream)

    cutlass_call()
    torch.cuda.synchronize()
    out = c_storage.float()
    decoded_ref = torch.einsum(
        "mkl,nkl->mnl",
        a["decoded"] * a["decoded_scale_expanded"],
        b["decoded"] * b["decoded_scale_expanded"],
    )
    fp32_ref = torch.einsum("mkl,nkl->mnl", x, w)
    fp16_ref = torch.einsum("mkl,nkl->mnl", x.half(), w.half())

    cutlass_ms = time_ms(cutlass_call, args.repeats, args.warmup)
    torch_fp16_ms = time_ms(lambda: torch.einsum("mkl,nkl->mnl", x.half(), w.half()), args.repeats, args.warmup)

    kernel_diff = (out - decoded_ref).float()
    quant_diff = (out - fp32_ref).float()
    fp16_diff = (fp16_ref.float() - fp32_ref).float()
    return {
        "name": case["name"],
        "group": case["group"],
        "m": m,
        "k": k,
        "n": n,
        "input_shapes": case["input_shapes"],
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "compile_seconds": compile_seconds,
        "cutlass_fp4_ms": cutlass_ms,
        "torch_fp16_ms": torch_fp16_ms,
        "speedup_vs_torch_fp16": float(torch_fp16_ms / cutlass_ms) if cutlass_ms else 0.0,
        "kernel_relative_rmse_vs_decoded_ref": float(
            torch.sqrt(kernel_diff.square().mean() / decoded_ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "quant_relative_rmse_vs_fp32": float(
            torch.sqrt(quant_diff.square().mean() / fp32_ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "torch_fp16_relative_rmse_vs_fp32": float(
            torch.sqrt(fp16_diff.square().mean() / fp32_ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "max_abs_diff_vs_decoded_ref": float(kernel_diff.abs().max().item()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    out = {}
    for group, group_rows in by_group.items():
        out[group] = {
            "cases": len(group_rows),
            "cutlass_fp4_ms_mean": float(np.mean([row["cutlass_fp4_ms"] for row in group_rows])),
            "torch_fp16_ms_mean": float(np.mean([row["torch_fp16_ms"] for row in group_rows])),
            "speedup_vs_torch_fp16_mean": float(np.mean([row["speedup_vs_torch_fp16"] for row in group_rows])),
            "quant_relative_rmse_vs_fp32_mean": float(np.mean([row["quant_relative_rmse_vs_fp32"] for row in group_rows])),
            "kernel_relative_rmse_vs_decoded_ref_mean": float(np.mean([row["kernel_relative_rmse_vs_decoded_ref"] for row in group_rows])),
        }
    return out


def strip_modules(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        item = dict(case)
        item.pop("module", None)
        rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="llm_dit_mlp")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260606)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN", "Float8E8M0FNU"], default="Float8E4M3FN")
    parser.add_argument("--tile-shape-mnk", type=lambda s: parse_tuple(s, 3), default=(128, 128, 128))
    parser.add_argument("--epi-tile", type=lambda s: parse_tuple(s, 2), default=(64, 32))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_blockscaled_fp4_real_weight_bench.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

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
    selected = collect_selected_modules(policy.model, args.scope)
    observations = [
        {
            "variant": variants[idx % len(variants)],
            "seed": args.base_seed + idx,
            "obs": build_variant_observation(args.model_path, args.embodiment_tag, variants[idx % len(variants)], idx),
        }
        for idx in range(args.num_observations)
    ]
    for item in observations:
        set_seed(int(item["seed"]))
    captured = capture_runtime_m(policy, selected, observations)
    cases = representative_cases(selected, captured)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] {case['group']} M={case['m']} K={case['k']} N={case['n']} {case['name']}", flush=True)
        rows.append(run_fp4_case(case, args))

    result = {
        "boundary": "CUTLASS blockscaled FP4 GEMM with real GR00T weights and synthetic activations",
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "epi_tile": list(args.epi_tile),
        "load_seconds": load_seconds,
        "selected_modules": len(selected),
        "captured_modules": len(captured),
        "representative_cases": len(cases),
        "case_descriptors": strip_modules(cases),
        "summary": summarize(rows),
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
