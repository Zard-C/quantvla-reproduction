"""Byte-check Triton blockscaled FP4 packing on real GR00T operands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_real_shape_bench import collect_selected_modules
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase8_blockscaled_fp4_packer_smoke import compare_pack, make_fast_pack, make_helper_pack
from phase8_cutlass_blockscaled_fp4_real_activation_bench import (
    build_observations,
    capture_activation_cases,
    strip_cases,
)
from phase8_cutlass_blockscaled_fp4_smoke import load_blockscaled_example


def synchronize(device: str = "cuda") -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def check_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import cutlass
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    blockscaled = load_blockscaled_example(args.cutlass_root)
    sf_dtype = getattr(cutlass, args.sf_dtype)
    module = case["module"]
    m, k, n = int(case["m"]), int(case["k"]), int(case["n"])
    device = torch.device(args.device)
    x = case["activation_cpu"].to(device=device, dtype=torch.float32).contiguous()
    w = module.weight.detach().to(device=device, dtype=torch.float32).contiguous().view(n, k, 1)

    helper_kwargs = {
        "cutlass": cutlass,
        "cutlass_torch": cutlass_torch,
        "from_dlpack": from_dlpack,
        "blockscaled": blockscaled,
        "sf_dtype": sf_dtype,
        "sf_vec_size": args.sf_vec_size,
    }
    triton_kwargs = {
        "backend": "triton",
        "cutlass": cutlass,
        "cutlass_torch": cutlass_torch,
        "sf_dtype": sf_dtype,
        "sf_vec_size": args.sf_vec_size,
    }

    started = time.perf_counter()
    a_helper = make_helper_pack(x, **helper_kwargs)
    b_helper = make_helper_pack(w, **helper_kwargs)
    a_triton = make_fast_pack(x, **triton_kwargs)
    b_triton = make_fast_pack(w, **triton_kwargs)
    synchronize(args.device)
    pack_seconds = time.perf_counter() - started

    row = {
        "name": case["name"],
        "group": case["group"],
        "m": m,
        "k": k,
        "n": n,
        "input_shape": case["input_shape"],
        "input_dtype": case["input_dtype"],
        "pack_seconds": pack_seconds,
    }
    row.update(compare_pack("activation", a_helper, a_triton, blockscaled=blockscaled, from_dlpack=from_dlpack))
    row.update(compare_pack("weight", b_helper, b_triton, blockscaled=blockscaled, from_dlpack=from_dlpack))
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bool_keys = [
        "activation_fp4_bytes_equal",
        "activation_scale_storage_equal",
        "weight_fp4_bytes_equal",
        "weight_scale_storage_equal",
    ]
    max_keys = [
        "activation_decoded_max_abs_diff",
        "weight_decoded_max_abs_diff",
    ]
    return {
        "cases": len(rows),
        **{key: all(bool(row[key]) for row in rows) for key in bool_keys},
        **{key: max(float(row[key]) for row in rows) if rows else 0.0 for key in max_keys},
    }


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
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN"], default="Float8E4M3FN")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_triton_real_operand_bytecheck.json"))
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
    model_load_seconds = time.perf_counter() - load_started

    selected = collect_selected_modules(policy.model, args.scope)
    cases, capture_meta = capture_activation_cases(
        policy,
        selected,
        observations,
        max_cases=args.max_cases,
        device=args.device,
    )
    rows = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] byte-check {case['group']} M={case['m']} K={case['k']} N={case['n']} {case['name']}", flush=True)
        rows.append(check_case(case, args))

    result = {
        "boundary": "real GR00T activation and weight byte-check: helper pack vs Triton pack",
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "denoising_steps": args.denoising_steps,
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "model_load_seconds": model_load_seconds,
        "observation_meta": observation_meta,
        "selected_modules": len(selected),
        "captured_cases": len(cases),
        "capture_meta": capture_meta,
        "case_descriptors": strip_cases(cases),
        "summary": summarize(rows),
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
