"""Smoke tests and microbenchmarks for the Phase 6 Triton W4A16 Linear path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from quant import fake_quant_weight
from triton_w4a16 import (
    PackedW4A16Linear,
    cosine_similarity,
    dequantize_w4_weight,
    pack_weight_w4_per_row,
    relative_rmse,
    w4a16_linear,
)


DEFAULT_SHAPES = (
    (1, 256, 256),
    (8, 512, 512),
    (32, 1024, 1024),
    (64, 2048, 1024),
    (16, 1024, 4096),
)


def parse_shapes(spec: str | None) -> list[tuple[int, int, int]]:
    if not spec:
        return list(DEFAULT_SHAPES)
    shapes: list[tuple[int, int, int]] = []
    for item in spec.split(","):
        parts = item.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"shape must be MxKxN, got {item!r}")
        shapes.append(tuple(int(p) for p in parts))  # type: ignore[arg-type]
    return shapes


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Any, device: torch.device, repeats: int) -> float:
    for _ in range(3):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def run_case(
    m: int,
    k: int,
    n: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    repeats: int,
    prefer_triton: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed + m + 17 * k + 101 * n)
    x = torch.randn((m, k), device=device, dtype=dtype) / (k**0.5)
    weight = torch.randn((n, k), device=device, dtype=dtype) / (k**0.5)
    bias = torch.randn((n,), device=device, dtype=dtype) / (n**0.5)

    packed = pack_weight_w4_per_row(weight.detach().cpu(), scale_dtype=torch.float32)
    qweight = packed.qweight.to(device)
    scale = packed.scale.to(device)
    w_dequant = dequantize_w4_weight(qweight, scale, k, dtype=dtype)
    ref = F.linear(x, w_dequant, bias)
    fake_w, _ = fake_quant_weight(weight, bits=4)
    fake_ref = F.linear(x, fake_w, bias)
    out = w4a16_linear(x, qweight, scale, bias, k, prefer_triton=prefer_triton)

    result: dict[str, Any] = {
        "shape": {"m": m, "k": k, "n": n},
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "prefer_triton": prefer_triton,
        "triton_used": bool(prefer_triton and device.type == "cuda"),
        "max_abs_diff_vs_dequant_ref": float((out - ref).detach().float().abs().max().item()),
        "mean_abs_diff_vs_dequant_ref": float((out - ref).detach().float().abs().mean().item()),
        "relative_rmse_vs_dequant_ref": relative_rmse(ref, out),
        "cosine_vs_dequant_ref": cosine_similarity(ref, out),
        "max_abs_diff_pack_vs_fake_weight_ref": float((ref - fake_ref).detach().float().abs().max().item()),
        "relative_rmse_pack_vs_fake_weight_ref": relative_rmse(fake_ref, ref),
        "fp16_weight_bytes": packed.fp16_weight_bytes(),
        "packed_weight_bytes": packed.packed_weight_bytes(),
        "scale_bytes": packed.scale_bytes(),
        "packed_plus_scale_ratio_vs_fp16": packed.compression_ratio_vs_fp16(),
    }

    if repeats > 0:
        result["timing_ms"] = {
            "torch_fp16_linear": time_ms(lambda: F.linear(x, weight, bias), device, repeats),
            "dense_dequant_ref": time_ms(lambda: F.linear(x, w_dequant, bias), device, repeats),
            "w4a16_dispatch": time_ms(
                lambda: w4a16_linear(x, qweight, scale, bias, k, prefer_triton=prefer_triton),
                device,
                repeats,
            ),
        }
    return result


def run_module_wrapper_case(device: torch.device, dtype: torch.dtype, prefer_triton: bool) -> dict[str, Any]:
    torch.manual_seed(20260606)
    base = torch.nn.Linear(257, 123, bias=True, device=device, dtype=dtype)
    x = torch.randn((3, 5, 257), device=device, dtype=dtype)
    packed = PackedW4A16Linear.from_linear(base, prefer_triton=prefer_triton).to(device)
    y = packed(x)
    ref = F.linear(x, packed.dequantized_weight(dtype=dtype).to(device), packed.bias)
    return {
        "shape": list(x.shape),
        "out_shape": list(y.shape),
        "max_abs_diff": float((y - ref).detach().float().abs().max().item()),
        "relative_rmse": relative_rmse(ref, y),
        "cosine": cosine_similarity(ref, y),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--shapes", help="Comma-separated MxKxN cases, e.g. 8x512x512,64x2048x1024")
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--force-reference", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase6_w4a16_smoke.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    prefer_triton = not args.force_reference

    results = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "dtype": args.dtype,
        "prefer_triton": prefer_triton,
        "shapes": [],
        "module_wrapper": None,
    }
    for m, k, n in parse_shapes(args.shapes):
        item = run_case(
            m,
            k,
            n,
            device=device,
            dtype=dtype,
            seed=args.seed,
            repeats=args.repeats,
            prefer_triton=prefer_triton,
        )
        results["shapes"].append(item)
        print(
            f"M={m} K={k} N={n} max_abs={item['max_abs_diff_vs_dequant_ref']:.4g} "
            f"rrmse={item['relative_rmse_vs_dequant_ref']:.4g} cos={item['cosine_vs_dequant_ref']:.8f}",
            flush=True,
        )
    results["module_wrapper"] = run_module_wrapper_case(device, dtype, prefer_triton)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
