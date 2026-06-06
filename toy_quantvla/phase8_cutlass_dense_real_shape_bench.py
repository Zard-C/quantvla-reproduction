"""Benchmark CUTLASS dense Linear on real selected GR00T module shapes."""

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
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_real_shape_bench import (
    capture_runtime_m,
    collect_selected_modules,
    representative_cases,
)
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from triton_w4a16 import cosine_similarity, relative_rmse


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Any, device: str, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def parse_tile(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3:
        raise ValueError(f"tile must be M,N,K, got {value!r}")
    return parts


def benchmark_case(
    case: dict[str, Any],
    *,
    device: str,
    repeats: int,
    warmup: int,
    cutlass_root: Path,
    tile_shape_mnk: tuple[int, int, int],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    module = case["module"]
    weight = module.weight.detach().to(device=device, dtype=torch.float16).contiguous()
    bias = module.bias.detach().to(device=device, dtype=torch.float16).contiguous() if module.bias is not None else None
    torch.manual_seed(20260606 + case["m"] + 7 * case["k"] + 31 * case["n"])
    x = (torch.randn((case["m"], case["k"]), device=device, dtype=torch.float16) / (case["k"] ** 0.5)).contiguous()
    ref = F.linear(x, weight, bias)
    cutlass_linear = CutlassDenseLinear(
        weight,
        bias,
        cutlass_root=cutlass_root,
        tile_shape_mnk=tile_shape_mnk,
        fallback=False,
    ).to(device)

    compile_started = time.perf_counter()
    out = cutlass_linear(x)
    synchronize(device)
    cold_ms = (time.perf_counter() - compile_started) * 1000.0

    cutlass_ms = time_ms(lambda: cutlass_linear(x), device, repeats, warmup)
    torch_ms = time_ms(lambda: F.linear(x, weight, bias), device, repeats, warmup)

    return {
        "name": case["name"],
        "group": case["group"],
        "m": case["m"],
        "k": case["k"],
        "n": case["n"],
        "input_shapes": case["input_shapes"],
        "torch_fp16_ms": torch_ms,
        "cutlass_dense_ms": cutlass_ms,
        "cutlass_cold_first_call_ms": cold_ms,
        "speedup_vs_torch_fp16": float(torch_ms / cutlass_ms) if cutlass_ms else 0.0,
        "max_abs_diff_vs_torch": float((out - ref).detach().float().abs().max().item()),
        "relative_rmse_vs_torch": relative_rmse(ref, out),
        "cosine_vs_torch": cosine_similarity(ref, out),
    }


def strip_modules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item.pop("module", None)
        out.append(item)
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    summary: dict[str, Any] = {}
    for group, group_rows in by_group.items():
        summary[group] = {
            "cases": len(group_rows),
            "torch_fp16_ms_mean": float(np.mean([row["torch_fp16_ms"] for row in group_rows])),
            "cutlass_dense_ms_mean": float(np.mean([row["cutlass_dense_ms"] for row in group_rows])),
            "speedup_vs_torch_fp16_mean": float(np.mean([row["speedup_vs_torch_fp16"] for row in group_rows])),
            "speedup_vs_torch_fp16_min": float(np.min([row["speedup_vs_torch_fp16"] for row in group_rows])),
            "speedup_vs_torch_fp16_max": float(np.max([row["speedup_vs_torch_fp16"] for row in group_rows])),
            "relative_rmse_vs_torch_mean": float(np.mean([row["relative_rmse_vs_torch"] for row in group_rows])),
        }
    return summary


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
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--tile-shape-mnk", type=parse_tile, default=(64, 128, 64))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_dense_real_shape_bench.json"))
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
                cutlass_root=args.cutlass_root,
                tile_shape_mnk=args.tile_shape_mnk,
            )
        )

    result = {
        "boundary": "CUTLASS dense fp16 backend op benchmark on real GR00T Linear shapes; not FP4 quantized yet",
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "selected_modules": len(selected),
        "captured_modules": len(captured),
        "representative_cases": len(cases),
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "load_seconds": load_seconds,
        "cutlass_root": str(args.cutlass_root),
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "case_descriptors": strip_modules(cases),
        "summary": summarize(rows),
        "cutlass_cache": CutlassDenseLinearCache.stats(),
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "cache": result["cutlass_cache"], "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
