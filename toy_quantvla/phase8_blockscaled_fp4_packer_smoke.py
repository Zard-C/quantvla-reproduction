"""Validate torch-side blockscaled FP4 packing against CUTLASS helpers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import torch

from blockscaled_fp4_packer import make_torch_blockscaled_fp4_operand
from phase8_cutlass_blockscaled_fp4_smoke import (
    load_blockscaled_example,
    make_blockscaled_fp4_operand,
    parse_tuple,
)


def synchronize(device: str = "cuda") -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def time_pack(fn: Callable[[], dict[str, Any]], *, repeats: int, warmup: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / repeats


def decode_fp4_with_cutlass(pack: dict[str, Any], *, blockscaled: Any, from_dlpack: Any) -> torch.Tensor:
    decoded = torch.empty_like(pack["normalized"])
    decoded_tensor = from_dlpack(decoded, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    blockscaled.cute.testing.convert(pack["fp4_tensor"], decoded_tensor)
    return decoded


def meaningful_fp4_bytes(pack: dict[str, Any]) -> torch.Tensor:
    count = int(pack.get("packed_fp4_bytes", math.ceil(pack["normalized"].numel() / 2)))
    return pack["fp4_storage"].reshape(-1)[:count]


def make_helper_pack(
    x: torch.Tensor,
    *,
    cutlass: Any,
    cutlass_torch: Any,
    from_dlpack: Any,
    blockscaled: Any,
    sf_dtype: Any,
    sf_vec_size: int,
) -> dict[str, Any]:
    return make_blockscaled_fp4_operand(
        x,
        fp4_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        blockscaled_module=blockscaled,
        cutlass_torch=cutlass_torch,
        from_dlpack=from_dlpack,
    )


def make_fast_pack(
    x: torch.Tensor,
    *,
    cutlass: Any,
    cutlass_torch: Any,
    sf_dtype: Any,
    sf_vec_size: int,
) -> dict[str, Any]:
    return make_torch_blockscaled_fp4_operand(
        x,
        fp4_dtype=cutlass.Float4E2M1FN,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        cutlass_torch=cutlass_torch,
    )


def run_gemm_pair(
    *,
    a_helper: dict[str, Any],
    b_helper: dict[str, Any],
    a_fast: dict[str, Any],
    b_fast: dict[str, Any],
    m: int,
    n: int,
    args: argparse.Namespace,
    cutlass: Any,
    cute: Any,
    cutlass_torch: Any,
    blockscaled: Any,
) -> dict[str, Any]:
    c_ref = torch.zeros((m, n, 1), device=args.device, dtype=torch.float32).contiguous()
    c_tensor, c_storage = cutlass_torch.cute_tensor_like(
        c_ref,
        cutlass.Float16,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
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
        a_helper["fp4_tensor"],
        b_helper["fp4_tensor"],
        a_helper["scale_tensor"],
        b_helper["scale_tensor"],
        c_tensor,
        max_active_clusters,
        stream,
    )
    synchronize(args.device)
    compile_seconds = time.perf_counter() - compile_started

    compiled(
        a_helper["fp4_tensor"],
        b_helper["fp4_tensor"],
        a_helper["scale_tensor"],
        b_helper["scale_tensor"],
        c_tensor,
        stream,
    )
    synchronize(args.device)
    helper_out = c_storage.float().clone()

    c_storage.zero_()
    compiled(
        a_fast["fp4_tensor"],
        b_fast["fp4_tensor"],
        a_fast["scale_tensor"],
        b_fast["scale_tensor"],
        c_tensor,
        stream,
    )
    synchronize(args.device)
    fast_out = c_storage.float().clone()
    diff = (fast_out - helper_out).float()
    return {
        "compile_seconds": float(compile_seconds),
        "max_abs_diff_fast_vs_helper": float(diff.abs().max().item()),
        "mean_abs_diff_fast_vs_helper": float(diff.abs().mean().item()),
        "relative_rmse_fast_vs_helper": float(
            torch.sqrt(diff.square().mean() / helper_out.float().square().mean().clamp_min(1e-12)).item()
        ),
    }


def compare_pack(prefix: str, helper: dict[str, Any], fast: dict[str, Any], *, blockscaled: Any, from_dlpack: Any) -> dict[str, Any]:
    helper_fp4 = meaningful_fp4_bytes(helper)
    fast_fp4 = meaningful_fp4_bytes(fast)
    decoded_fast_by_cutlass = decode_fp4_with_cutlass(fast, blockscaled=blockscaled, from_dlpack=from_dlpack)
    decoded_diff = (decoded_fast_by_cutlass - helper["decoded"]).float()
    scale_diff = (fast["decoded_scale_expanded"] - helper["decoded_scale_expanded"]).float()
    fp4_len = min(int(helper_fp4.numel()), int(fast_fp4.numel()))
    fp4_equal = torch.equal(helper_fp4[:fp4_len].detach().cpu(), fast_fp4[:fp4_len].detach().cpu())
    scale_equal = torch.equal(helper["scale_storage"].detach().cpu(), fast["scale_storage"].detach().cpu())
    return {
        f"{prefix}_meaningful_fp4_bytes": int(fp4_len),
        f"{prefix}_fp4_bytes_equal": bool(fp4_equal),
        f"{prefix}_scale_storage_equal": bool(scale_equal),
        f"{prefix}_decoded_max_abs_diff": float(decoded_diff.abs().max().item()),
        f"{prefix}_decoded_mean_abs_diff": float(decoded_diff.abs().mean().item()),
        f"{prefix}_decoded_scale_max_abs_diff": float(scale_diff.abs().max().item()),
        f"{prefix}_decoded_scale_mean_abs_diff": float(scale_diff.abs().mean().item()),
        f"{prefix}_fp4_storage_shape": list(fast["fp4_storage"].shape),
        f"{prefix}_scale_storage_shape": list(fast["scale_storage"].shape),
        f"{prefix}_scale_storage_stride": list(fast["scale_storage"].stride()),
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    blockscaled = load_blockscaled_example(args.cutlass_root)
    device = torch.device(args.device)
    sf_dtype = getattr(cutlass, args.sf_dtype)
    torch.manual_seed(args.seed)
    x = (torch.randn((args.m, args.k, 1), device=device, dtype=torch.float32) / math.sqrt(args.k)).contiguous()
    w = (torch.randn((args.n, args.k, 1), device=device, dtype=torch.float32) / math.sqrt(args.k)).contiguous()

    helper_kwargs = {
        "cutlass": cutlass,
        "cutlass_torch": cutlass_torch,
        "from_dlpack": from_dlpack,
        "blockscaled": blockscaled,
        "sf_dtype": sf_dtype,
        "sf_vec_size": args.sf_vec_size,
    }
    fast_kwargs = {
        "cutlass": cutlass,
        "cutlass_torch": cutlass_torch,
        "sf_dtype": sf_dtype,
        "sf_vec_size": args.sf_vec_size,
    }

    a_helper = make_helper_pack(x, **helper_kwargs)
    b_helper = make_helper_pack(w, **helper_kwargs)
    a_fast = make_fast_pack(x, **fast_kwargs)
    b_fast = make_fast_pack(w, **fast_kwargs)
    synchronize(args.device)

    result: dict[str, Any] = {
        "boundary": "torch-side blockscaled FP4 packer smoke against CUTLASS helper conversion",
        "m": args.m,
        "k": args.k,
        "n": args.n,
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "epi_tile": list(args.epi_tile),
    }
    result.update(compare_pack("a", a_helper, a_fast, blockscaled=blockscaled, from_dlpack=from_dlpack))
    result.update(compare_pack("b", b_helper, b_fast, blockscaled=blockscaled, from_dlpack=from_dlpack))

    helper_a_ms = time_pack(lambda: make_helper_pack(x, **helper_kwargs), repeats=args.helper_repeats, warmup=args.warmup, device=args.device)
    fast_a_ms = time_pack(lambda: make_fast_pack(x, **fast_kwargs), repeats=args.fast_repeats, warmup=args.warmup, device=args.device)
    helper_b_ms = time_pack(lambda: make_helper_pack(w, **helper_kwargs), repeats=args.helper_repeats, warmup=args.warmup, device=args.device)
    fast_b_ms = time_pack(lambda: make_fast_pack(w, **fast_kwargs), repeats=args.fast_repeats, warmup=args.warmup, device=args.device)
    result.update(
        {
            "helper_activation_pack_ms": helper_a_ms,
            "torch_activation_pack_ms": fast_a_ms,
            "activation_pack_speedup_vs_helper": float(helper_a_ms / fast_a_ms) if fast_a_ms else None,
            "helper_weight_pack_ms": helper_b_ms,
            "torch_weight_pack_ms": fast_b_ms,
            "weight_pack_speedup_vs_helper": float(helper_b_ms / fast_b_ms) if fast_b_ms else None,
        }
    )

    if args.run_gemm:
        result["gemm"] = run_gemm_pair(
            a_helper=a_helper,
            b_helper=b_helper,
            a_fast=a_fast,
            b_fast=b_fast,
            m=args.m,
            n=args.n,
            args=args,
            cutlass=cutlass,
            cute=cute,
            cutlass_torch=cutlass_torch,
            blockscaled=blockscaled,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=49)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN"], default="Float8E4M3FN")
    parser.add_argument("--tile-shape-mnk", type=lambda s: parse_tuple(s, 3), default=(128, 128, 128))
    parser.add_argument("--epi-tile", type=lambda s: parse_tuple(s, 2), default=(64, 32))
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--helper-repeats", type=int, default=3)
    parser.add_argument("--fast-repeats", type=int, default=20)
    parser.add_argument("--run-gemm", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke.json"))
    args = parser.parse_args()
    result = run_case(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
