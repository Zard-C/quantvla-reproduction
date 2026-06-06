"""Benchmark bitsandbytes Linear4bit on real selected GR00T module shapes."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase6_w4a16_real_shape_bench import (
    capture_runtime_m,
    collect_selected_modules,
    representative_cases,
)
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase3_activation_capture import build_variant_observation
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from triton_w4a16 import cosine_similarity, relative_rmse


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Any, device: str, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def make_bnb_linear(
    weight: Any,
    bias: Any,
    *,
    quant_type: str,
    compute_dtype: Any,
    device: str,
) -> Any:
    import bitsandbytes as bnb
    import torch

    out_features, in_features = weight.shape
    module = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=bias is not None,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
        compress_statistics=True,
    )
    module.weight = bnb.nn.Params4bit(
        weight.detach().cpu(),
        requires_grad=False,
        quant_type=quant_type,
        compress_statistics=True,
    )
    if bias is not None:
        module.bias = torch.nn.Parameter(bias.detach().cpu(), requires_grad=False)
    return module.to(device)


def benchmark_case(
    case: dict[str, Any],
    *,
    device: str,
    repeats: int,
    warmup: int,
    quant_types: list[str],
    compute_dtype_name: str,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    module = case["module"]
    compute_dtype = getattr(torch, compute_dtype_name)
    weight = module.weight.detach().to(device=device, dtype=compute_dtype).contiguous()
    bias = module.bias.detach().to(device=device, dtype=compute_dtype).contiguous() if module.bias is not None else None
    torch.manual_seed(20260606 + case["m"] + 5 * case["k"] + 19 * case["n"])
    x = (torch.randn((case["m"], case["k"]), device=device, dtype=compute_dtype) / (case["k"] ** 0.5)).contiguous()
    ref = F.linear(x, weight, bias)
    torch_ms = time_ms(lambda: F.linear(x, weight, bias), device, repeats, warmup)

    rows = []
    for quant_type in quant_types:
        try:
            bnb_linear = make_bnb_linear(
                weight,
                bias,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
                device=device,
            )
            out = bnb_linear(x)
            row = {
                "quant_type": quant_type,
                "compute_dtype": compute_dtype_name,
                "max_abs_diff_vs_fp": float((out - ref).detach().float().abs().max().item()),
                "relative_rmse_vs_fp": relative_rmse(ref, out),
                "cosine_vs_fp": cosine_similarity(ref, out),
                "bnb_ms": time_ms(lambda: bnb_linear(x), device, repeats, warmup),
                "error": None,
            }
            row["speedup_vs_torch_fp"] = float(torch_ms / row["bnb_ms"])
        except Exception as exc:
            row = {
                "quant_type": quant_type,
                "compute_dtype": compute_dtype_name,
                "error": repr(exc),
            }
        rows.append(row)

    best = None
    for row in rows:
        if row.get("error") is None and (best is None or row["bnb_ms"] < best["bnb_ms"]):
            best = row
    return {
        "name": case["name"],
        "group": case["group"],
        "m": case["m"],
        "k": case["k"],
        "n": case["n"],
        "input_shapes": case["input_shapes"],
        "torch_fp_ms": torch_ms,
        "bnb": rows,
        "best": best,
    }


def strip_modules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item.pop("module", None)
        out.append(item)
    return out


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
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--quant-types", default="fp4,nf4")
    parser.add_argument("--compute-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase7_bnb4_real_shape_bench.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import bitsandbytes as bnb
    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    quant_types = [v.strip() for v in args.quant_types.split(",") if v.strip()]
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
    captured = capture_runtime_m(policy, selected, observations)
    cases = representative_cases(selected, captured)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] {case['group']} M={case['m']} K={case['k']} N={case['n']} {case['name']}",
            flush=True,
        )
        rows.append(
            benchmark_case(
                case,
                device=args.device,
                repeats=args.repeats,
                warmup=args.warmup,
                quant_types=quant_types,
                compute_dtype_name=args.compute_dtype,
            )
        )

    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    summary = {}
    for group, group_rows in by_group.items():
        valid = [row for row in group_rows if row["best"] is not None]
        summary[group] = {
            "cases": len(group_rows),
            "valid_cases": len(valid),
            "best_speedup_vs_torch_fp_mean": float(np.mean([row["best"]["speedup_vs_torch_fp"] for row in valid])) if valid else 0.0,
            "best_relative_rmse_vs_fp_mean": float(np.mean([row["best"]["relative_rmse_vs_fp"] for row in valid])) if valid else 0.0,
        }

    result = {
        "torch_version": torch.__version__,
        "bnb_version": bnb.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "compute_dtype": args.compute_dtype,
        "quant_types": quant_types,
        "selected_modules": len(selected),
        "captured_modules": len(captured),
        "representative_cases": len(cases),
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "load_seconds": load_seconds,
        "case_descriptors": strip_modules(cases),
        "summary": summary,
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

