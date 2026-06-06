"""Smoke-test real-tensor CUTLASS SM120 blockscaled FP4 GEMM.

This is the missing bridge between the Phase 7 standalone CUTLASS example and
an eventual real QuantVLA backend.  It starts from ordinary torch fp32/fp16
activation and weight matrices, creates blockscaled FP4 operands plus scale
factor tensors in the layout expected by the NVIDIA GeForce example, then runs
the CUTLASS blockscaled GEMM.

The conversion path still uses CUTLASS DSL helper kernels (`cute.testing.convert`)
and is not yet a production runtime packer.  The goal is to prove numerical
and callable integration before investing in a fused/compiled packer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F


def load_blockscaled_example(cutlass_root: Path) -> Any:
    example_dir = (
        cutlass_root
        / "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm"
    )
    example = example_dir / "dense_blockscaled_gemm_persistent_pingpong.py"
    if not example.exists():
        raise FileNotFoundError(f"CUTLASS blockscaled example not found: {example}")
    import sys

    sys.path.insert(0, str(example_dir))
    spec = importlib.util.spec_from_file_location("quantvla_blockscaled_example", example)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CUTLASS blockscaled example: {example}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_tuple(value: str, n: int) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != n:
        raise ValueError(f"expected {n} comma-separated ints, got {value!r}")
    return parts


def time_ms(fn: Any, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0 / repeats


def make_scale_factor_tensor(
    *,
    scale_blocks_mkl: torch.Tensor,
    sf_vec_size: int,
    sf_dtype: Any,
    blockscaled_module: Any,
    cutlass_torch: Any,
    from_dlpack: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
    """Create CUTLASS scale-factor tensor and expanded f32 reference.

    `scale_blocks_mkl` is shaped `(mn, ceil_div(k, sf_vec_size), l)`.
    CUTLASS wants the backing storage in an MMA-specific
    `(32, 4, rest_m, 4, rest_k, l)`-like permutation.
    """

    mn, sf_k, l = (int(v) for v in scale_blocks_mkl.shape)
    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        math.ceil(mn / (atom_m[0] * atom_m[1])),
        math.ceil(sf_k / atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    scale_ref_cpu = scale_blocks_mkl.detach().float().cpu()
    scale_mma_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(mma_permute_order)
    blockscaled_module.cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(scale_ref_cpu),
        from_dlpack(scale_mma_cpu),
    )
    scale_mma_gpu = scale_mma_cpu.cuda()
    scale_tensor, scale_storage = cutlass_torch.cute_tensor_like(
        scale_mma_cpu,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    scale_tensor = cutlass_torch.convert_cute_tensor(
        scale_mma_gpu,
        scale_tensor,
        sf_dtype,
        is_dynamic_layout=True,
    )
    decoded_mma = torch.empty_strided(
        tuple(scale_mma_gpu.shape),
        tuple(scale_mma_gpu.stride()),
        device=scale_mma_gpu.device,
        dtype=torch.float32,
    )
    decoded_mma_tensor = from_dlpack(decoded_mma, assumed_align=16).mark_layout_dynamic(
        leading_dim=cutlass_torch.get_leading_dim(decoded_mma)
    )
    blockscaled_module.cute.testing.convert(scale_tensor, decoded_mma_tensor)

    decoded_blocks = torch.empty_like(scale_blocks_mkl, device=scale_mma_gpu.device, dtype=torch.float32)
    for row in range(mn):
        row0 = row % atom_m[0]
        row1 = (row // atom_m[0]) % atom_m[1]
        rest_m = row // (atom_m[0] * atom_m[1])
        for kk in range(sf_k):
            kk0 = kk % atom_k
            rest_k = kk // atom_k
            decoded_blocks[row, kk, :] = decoded_mma[row0, row1, rest_m, kk0, rest_k, :]

    scale_expanded = (
        scale_ref_cpu.unsqueeze(-1)
        .expand(mn, sf_k, l, sf_vec_size)
        .reshape(mn, sf_k * sf_vec_size, l)
    )
    decoded_expanded = (
        decoded_blocks.cpu().unsqueeze(-1)
        .expand(mn, sf_k, l, sf_vec_size)
        .reshape(mn, sf_k * sf_vec_size, l)
    )
    return scale_expanded, decoded_expanded, scale_tensor, scale_storage


def make_blockscaled_fp4_operand(
    x_mkl: torch.Tensor,
    *,
    fp4_dtype: Any,
    sf_dtype: Any,
    sf_vec_size: int,
    blockscaled_module: Any,
    cutlass_torch: Any,
    from_dlpack: Any,
) -> dict[str, Any]:
    if x_mkl.ndim != 3 or x_mkl.shape[2] != 1:
        raise ValueError(f"expected tensor shape (mn,k,1), got {tuple(x_mkl.shape)}")
    mn, k, l = (int(v) for v in x_mkl.shape)
    pad = (-k) % sf_vec_size
    x_padded = F.pad(x_mkl.detach().float(), (0, 0, 0, pad)) if pad else x_mkl.detach().float()
    sf_k = x_padded.shape[1] // sf_vec_size
    blocks = x_padded.reshape(mn, sf_k, sf_vec_size, l)
    scale_blocks = blocks.abs().amax(dim=2).clamp_min(1e-8) / 6.0
    scale_expanded, decoded_scale_expanded, scale_tensor, scale_storage = make_scale_factor_tensor(
        scale_blocks_mkl=scale_blocks,
        sf_vec_size=sf_vec_size,
        sf_dtype=sf_dtype,
        blockscaled_module=blockscaled_module,
        cutlass_torch=cutlass_torch,
        from_dlpack=from_dlpack,
    )
    decoded_scale_expanded = decoded_scale_expanded[:, :k, :].to(device=x_mkl.device)
    scale_expanded = scale_expanded[:, :k, :].to(device=x_mkl.device)
    normalized = (x_mkl.detach().float() / decoded_scale_expanded).clamp(-6.0, 6.0).contiguous()
    fp4_tensor, fp4_storage = cutlass_torch.cute_tensor_like(
        normalized,
        fp4_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    fp4_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=2,
    )
    fp4_tensor = cutlass_torch.convert_cute_tensor(
        normalized,
        fp4_tensor,
        fp4_dtype,
        is_dynamic_layout=True,
    )

    decoded = torch.empty_like(normalized)
    decoded_tensor = from_dlpack(decoded, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    blockscaled_module.cute.testing.convert(fp4_tensor, decoded_tensor)

    return {
        "normalized": normalized,
        "decoded": decoded,
        "scale_expanded": scale_expanded,
        "decoded_scale_expanded": decoded_scale_expanded,
        "fp4_tensor": fp4_tensor,
        "fp4_storage": fp4_storage,
        "scale_tensor": scale_tensor,
        "scale_storage": scale_storage,
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cutlass.cute.runtime import from_dlpack

    blockscaled = load_blockscaled_example(args.cutlass_root)
    m, k, n = args.m, args.k, args.n
    l = 1
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    x = (torch.randn((m, k, l), device=device, dtype=torch.float32) / math.sqrt(k)).contiguous()
    w = (torch.randn((n, k, l), device=device, dtype=torch.float32) / math.sqrt(k)).contiguous()
    c_ref = torch.zeros((m, n, l), device=device, dtype=torch.float32).contiguous()
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
        compiled(
            a["fp4_tensor"],
            b["fp4_tensor"],
            a["scale_tensor"],
            b["scale_tensor"],
            c_tensor,
            stream,
        )

    cutlass_call()
    torch.cuda.synchronize()
    out = c_storage.float()
    ref = torch.einsum(
        "mkl,nkl->mnl",
        a["decoded"] * a["scale_expanded"],
        b["decoded"] * b["scale_expanded"],
    )
    decoded_scale_ref = torch.einsum(
        "mkl,nkl->mnl",
        a["decoded"] * a["decoded_scale_expanded"],
        b["decoded"] * b["decoded_scale_expanded"],
    )
    original_ref = torch.einsum("mkl,nkl->mnl", x, w)

    fp4_ref_ms = time_ms(lambda: torch.einsum("mkl,nkl->mnl", a["decoded"] * a["scale_expanded"], b["decoded"] * b["scale_expanded"]), args.repeats, args.warmup)
    torch_fp16_ms = time_ms(lambda: torch.einsum("mkl,nkl->mnl", x.half(), w.half()), args.repeats, args.warmup)
    cutlass_ms = time_ms(cutlass_call, args.repeats, args.warmup)

    diff = (out - decoded_scale_ref).float()
    raw_scale_diff = (out - ref).float()
    original_diff = (out - original_ref).float()
    return {
        "boundary": "real torch tensor -> blockscaled FP4 CUTLASS GEMM smoke; conversion uses CUTLASS helper pack path",
        "m": m,
        "k": k,
        "n": n,
        "sf_vec_size": args.sf_vec_size,
        "sf_dtype": args.sf_dtype,
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "epi_tile": list(args.epi_tile),
        "compile_seconds": compile_seconds,
        "cutlass_fp4_ms": cutlass_ms,
        "torch_fp16_einsum_ms": torch_fp16_ms,
        "torch_decoded_fp4_ref_ms": fp4_ref_ms,
        "speedup_vs_torch_fp16_einsum": float(torch_fp16_ms / cutlass_ms) if cutlass_ms else 0.0,
        "max_abs_diff_vs_decoded_fp4_decoded_scale_ref": float(diff.abs().max().item()),
        "mean_abs_diff_vs_decoded_fp4_decoded_scale_ref": float(diff.abs().mean().item()),
        "relative_rmse_vs_decoded_fp4_decoded_scale_ref": float(
            torch.sqrt(diff.square().mean() / decoded_scale_ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "relative_rmse_vs_decoded_fp4_raw_scale_ref": float(
            torch.sqrt(raw_scale_diff.square().mean() / ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "relative_rmse_vs_original_fp32": float(
            torch.sqrt(original_diff.square().mean() / original_ref.float().square().mean().clamp_min(1e-12)).item()
        ),
        "a_scale_min": float(a["scale_expanded"].min().item()),
        "a_scale_max": float(a["scale_expanded"].max().item()),
        "b_scale_min": float(b["scale_expanded"].min().item()),
        "b_scale_max": float(b["scale_expanded"].max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--m", type=int, default=49)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--n", type=int, default=6144)
    parser.add_argument("--sf-vec-size", type=int, default=16)
    parser.add_argument("--sf-dtype", choices=["Float8E4M3FN", "Float8E8M0FNU"], default="Float8E4M3FN")
    parser.add_argument("--tile-shape-mnk", type=lambda s: parse_tuple(s, 3), default=(128, 128, 128))
    parser.add_argument("--epi-tile", type=lambda s: parse_tuple(s, 2), default=(64, 32))
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_blockscaled_fp4_smoke.json"))
    args = parser.parse_args()
    result = run_case(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
