"""Torch-side packer for CUTLASS SM120 blockscaled FP4 operands.

This module replaces the slow CuTe helper conversion in the runtime activation
path while keeping the exact same CUTLASS operand contract:

- values are quantized to Float4E2M1FN and stored two nibbles per byte;
- scale factors are quantized to Float8E4M3FN;
- scale storage is written in the SM120 MMA layout expected by the NVIDIA
  GeForce blockscaled GEMM example.

The implementation is intentionally torch-only for now.  It is a correctness
bridge toward a fused Triton/CUDA packer, not the final optimal pack kernel.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on machines without Triton.
    triton = None
    tl = None


def _check_e4m3(sf_dtype: Any) -> None:
    if "Float8E4M3FN" not in str(sf_dtype):
        raise ValueError(f"torch packer currently supports only Float8E4M3FN scales, got {sf_dtype}")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("this PyTorch build does not expose torch.float8_e4m3fn")


def quantize_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Return unsigned 4-bit Float4E2M1FN codes in an int16 tensor.

    CUTLASS uses round-to-nearest-even for exact midpoints.  The small odd-code
    penalty below makes ties choose codes with an even low bit, matching the
    helper conversion behavior observed on SM120.
    """

    values = torch.tensor(FP4_E2M1_VALUES, device=x.device, dtype=x.dtype)
    x = torch.where(torch.isnan(x), torch.full_like(x, 6.0), x)
    abs_x = x.abs().clamp(max=6.0)
    distances = (abs_x.unsqueeze(-1) - values).abs()
    odd_penalty = torch.tensor((0.0, 1e-7, 0.0, 1e-7, 0.0, 1e-7, 0.0, 1e-7), device=x.device, dtype=x.dtype)
    magnitude_codes = torch.argmin(distances + odd_penalty, dim=-1).to(torch.int16)
    sign_codes = torch.signbit(x).to(torch.int16) << 3
    return magnitude_codes | sign_codes


def decode_e2m1_codes(codes: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode unsigned 4-bit Float4E2M1FN codes to a torch float tensor."""

    values = torch.tensor(FP4_E2M1_VALUES, device=codes.device, dtype=dtype)
    magnitude = values[(codes.to(torch.long) & 0x7)]
    sign = torch.where((codes.to(torch.int16) & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign.to(dtype=dtype, device=codes.device)


def pack_e2m1_codes_into_storage(codes: torch.Tensor, storage: torch.Tensor) -> int:
    """Pack two 4-bit codes per byte into an int8 CUTLASS backing storage.

    `storage` is the int8 tensor returned by `cutlass.torch.cute_tensor_like`.
    Only the first ceil(num_values / 2) bytes are meaningful for the compact
    FP4 layout; the remaining logical bytes are zeroed for reproducibility.
    """

    flat_codes = codes.reshape(-1).to(torch.int16)
    if flat_codes.numel() % 2:
        flat_codes = F.pad(flat_codes, (0, 1))
    packed = (flat_codes[0::2] & 0xF) | ((flat_codes[1::2] & 0xF) << 4)
    packed_i8 = torch.where(packed >= 128, packed - 256, packed).to(torch.int8)
    flat_storage = storage.reshape(-1)
    flat_storage.zero_()
    flat_storage[: packed_i8.numel()].copy_(packed_i8)
    return int(packed_i8.numel())


def _scale_mma_template(mn: int, sf_k: int, l: int) -> torch.Tensor:
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
    return torch.empty(mma_shape, dtype=torch.float32).permute(3, 4, 1, 5, 2, 0)


def _make_cutlass_fp4_tensor(
    *,
    shape_tensor: torch.Tensor,
    fp4_dtype: Any,
    cutlass_torch: Any,
) -> tuple[Any, torch.Tensor]:
    fp4_tensor, fp4_storage = cutlass_torch.cute_tensor_like(
        shape_tensor,
        fp4_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    fp4_tensor.mark_compact_shape_dynamic(
        mode=1,
        stride_order=(2, 0, 1),
        divisibility=2,
    )
    return fp4_tensor, fp4_storage


def _make_cutlass_scale_tensor(
    *,
    mn: int,
    sf_k: int,
    l: int,
    sf_dtype: Any,
    cutlass_torch: Any,
) -> tuple[Any, torch.Tensor]:
    scale_template = _scale_mma_template(mn, sf_k, l)
    return cutlass_torch.cute_tensor_like(
        scale_template,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )


def scale_blocks_to_mma_storage(scale_bytes_mkl: torch.Tensor) -> torch.Tensor:
    """Reorder `(mn, sf_k, l)` scale bytes to CUTLASS MMA scale layout."""

    if scale_bytes_mkl.ndim != 3:
        raise ValueError(f"expected scale bytes shape (mn, sf_k, l), got {tuple(scale_bytes_mkl.shape)}")
    mn, sf_k, l = (int(v) for v in scale_bytes_mkl.shape)
    rest_m = math.ceil(mn / (32 * 4))
    rest_k = math.ceil(sf_k / 4)
    padded = torch.zeros((rest_m * 32 * 4, rest_k * 4, l), device=scale_bytes_mkl.device, dtype=torch.int8)
    padded[:mn, :sf_k, :].copy_(scale_bytes_mkl)

    # padded row order is rm, row1, row0; k order is rest_k, kk0.
    packed = padded.view(rest_m, 4, 32, rest_k, 4, l)
    base = packed.permute(5, 0, 3, 2, 1, 4).contiguous()
    return base.permute(3, 4, 1, 5, 2, 0)


def make_torch_blockscaled_fp4_operand(
    x_mkl: torch.Tensor,
    *,
    fp4_dtype: Any,
    sf_dtype: Any,
    sf_vec_size: int,
    cutlass_torch: Any,
) -> dict[str, Any]:
    """Create a CUTLASS blockscaled FP4 operand with torch-side packing."""

    _check_e4m3(sf_dtype)
    if x_mkl.ndim != 3 or x_mkl.shape[2] != 1:
        raise ValueError(f"expected tensor shape (mn, k, 1), got {tuple(x_mkl.shape)}")

    mn, k, l = (int(v) for v in x_mkl.shape)
    pad = (-k) % sf_vec_size
    x_float = x_mkl.detach().float()
    x_padded = F.pad(x_float, (0, 0, 0, pad)) if pad else x_float
    sf_k = int(x_padded.shape[1] // sf_vec_size)

    blocks = x_padded.reshape(mn, sf_k, sf_vec_size, l)
    scale_blocks = blocks.abs().amax(dim=2).clamp_min(1e-8) / 6.0
    scale_blocks_f8 = scale_blocks.to(torch.float8_e4m3fn)
    decoded_scale_blocks = scale_blocks_f8.to(torch.float32)

    scale_expanded = (
        scale_blocks.unsqueeze(2)
        .expand(mn, sf_k, sf_vec_size, l)
        .reshape(mn, sf_k * sf_vec_size, l)[:, :k, :]
        .contiguous()
    )
    decoded_scale_expanded = (
        decoded_scale_blocks.unsqueeze(2)
        .expand(mn, sf_k, sf_vec_size, l)
        .reshape(mn, sf_k * sf_vec_size, l)[:, :k, :]
        .contiguous()
    )
    normalized = (x_float / decoded_scale_expanded).clamp(-6.0, 6.0).contiguous()

    fp4_tensor, fp4_storage = _make_cutlass_fp4_tensor(
        shape_tensor=normalized,
        fp4_dtype=fp4_dtype,
        cutlass_torch=cutlass_torch,
    )
    codes = quantize_e2m1_codes(normalized)
    packed_fp4_bytes = pack_e2m1_codes_into_storage(codes, fp4_storage)
    decoded = decode_e2m1_codes(codes, dtype=torch.float32).view_as(normalized)

    scale_tensor, scale_storage = _make_cutlass_scale_tensor(
        mn=mn,
        sf_k=sf_k,
        l=l,
        sf_dtype=sf_dtype,
        cutlass_torch=cutlass_torch,
    )
    scale_bytes = scale_blocks_f8.view(torch.int8)
    manual_scale_storage = scale_blocks_to_mma_storage(scale_bytes)
    if tuple(scale_storage.shape) != tuple(manual_scale_storage.shape):
        raise RuntimeError(
            f"scale storage shape mismatch: cutlass={tuple(scale_storage.shape)} manual={tuple(manual_scale_storage.shape)}"
        )
    scale_storage.zero_()
    scale_storage.copy_(manual_scale_storage.to(device=scale_storage.device))

    return {
        "normalized": normalized,
        "decoded": decoded,
        "scale_expanded": scale_expanded,
        "decoded_scale_expanded": decoded_scale_expanded,
        "fp4_tensor": fp4_tensor,
        "fp4_storage": fp4_storage,
        "scale_tensor": scale_tensor,
        "scale_storage": scale_storage,
        "packed_fp4_bytes": packed_fp4_bytes,
        "pack_backend": "torch_e4m3",
    }


if tl is not None:

    @triton.jit
    def _round_nearest_even(x):
        floor_x = tl.floor(x)
        frac = x - floor_x
        floor_i = floor_x.to(tl.int32)
        round_up = (frac > 0.5) | ((frac == 0.5) & ((floor_i & 1) == 1))
        return (floor_i + round_up.to(tl.int32)).to(tl.int32)

    @triton.jit
    def _encode_e4m3fn_positive(x):
        # Positive Float8E4M3FN, matching PyTorch for the scale values used here.
        min_normal = 0.015625
        max_finite = 448.0
        x_clamped = tl.minimum(tl.maximum(x, 0.0), max_finite)

        sub_q = _round_nearest_even(x_clamped * 512.0)
        # Values just below the normal range can round up to the minimum normal
        # code 0x08.  PyTorch's float8_e4m3fn conversion does this, so allow 8.
        sub_q = tl.minimum(tl.maximum(sub_q, 0), 8)

        exp_unbiased = tl.floor(tl.log2(tl.maximum(x_clamped, min_normal))).to(tl.int32)
        step = tl.exp2(exp_unbiased.to(tl.float32) - 3.0)
        sig_q = _round_nearest_even(x_clamped / step)
        carry = sig_q == 16
        exp_unbiased = exp_unbiased + carry.to(tl.int32)
        sig_q = tl.where(carry, 8, sig_q)
        exp_field = exp_unbiased + 7
        exp_field = tl.minimum(tl.maximum(exp_field, 1), 15)
        normal_code = (exp_field << 3) | (sig_q - 8)
        normal_code = tl.minimum(normal_code, 0x7E)

        return tl.where(x_clamped < min_normal, sub_q, normal_code).to(tl.int32)

    @triton.jit
    def _decode_e4m3fn_positive(code):
        exp_field = (code >> 3) & 0xF
        mant = code & 0x7
        sub = mant.to(tl.float32) * 0.001953125
        normal = (1.0 + mant.to(tl.float32) * 0.125) * tl.exp2(exp_field.to(tl.float32) - 7.0)
        return tl.where(exp_field == 0, sub, normal)

    @triton.jit
    def _quantize_e2m1_code(x):
        abs_x = tl.minimum(tl.abs(x), 6.0)
        eps = 0.000001
        mag = tl.full(abs_x.shape, 7, dtype=tl.int32)
        mag = tl.where(abs_x <= 5.0 + eps, 6, mag)
        mag = tl.where(abs_x < 3.5 - eps, 5, mag)
        mag = tl.where(abs_x <= 2.5 + eps, 4, mag)
        mag = tl.where(abs_x < 1.75 - eps, 3, mag)
        mag = tl.where(abs_x <= 1.25 + eps, 2, mag)
        mag = tl.where(abs_x < 0.75 - eps, 1, mag)
        mag = tl.where(abs_x <= 0.25 + eps, 0, mag)
        negative_zero = (x == 0.0) & ((1.0 / x) < 0.0)
        sign = tl.where((x < 0.0) | negative_zero, 8, 0)
        return (mag | sign).to(tl.int32)

    @triton.jit
    def _triton_blockscaled_fp4_pack_kernel(
        x_ptr,
        fp4_ptr,
        scale_ptr,
        mn: tl.constexpr,
        k_size: tl.constexpr,
        sf_k: tl.constexpr,
        scale_stride0: tl.constexpr,
        scale_stride1: tl.constexpr,
        scale_stride2: tl.constexpr,
        scale_stride3: tl.constexpr,
        scale_stride4: tl.constexpr,
        scale_stride5: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        sf_block = tl.program_id(1)
        offs = tl.arange(0, BLOCK_K)
        k0 = sf_block * BLOCK_K
        k = k0 + offs
        value_offsets = row * k_size + k
        valid = k < k_size
        x = tl.load(x_ptr + value_offsets, mask=valid, other=0.0).to(tl.float32)
        raw_scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 6.0, 0.00000001)
        scale_code = _encode_e4m3fn_positive(raw_scale)
        decoded_scale = _decode_e4m3fn_positive(scale_code)
        normalized = tl.where(decoded_scale > 0.0, x / decoded_scale, tl.where(x < 0.0, -6.0, 6.0))
        normalized = tl.minimum(tl.maximum(normalized, -6.0), 6.0)

        pair = tl.arange(0, BLOCK_K // 2)
        k_lo = k0 + pair * 2
        k_hi = k_lo + 1
        x_lo = tl.load(x_ptr + row * k_size + k_lo, mask=k_lo < k_size, other=0.0).to(tl.float32)
        x_hi = tl.load(x_ptr + row * k_size + k_hi, mask=k_hi < k_size, other=0.0).to(tl.float32)
        n_lo = tl.where(decoded_scale > 0.0, x_lo / decoded_scale, tl.where(x_lo < 0.0, -6.0, 6.0))
        n_hi = tl.where(decoded_scale > 0.0, x_hi / decoded_scale, tl.where(x_hi < 0.0, -6.0, 6.0))
        n_lo = tl.minimum(tl.maximum(n_lo, -6.0), 6.0)
        n_hi = tl.minimum(tl.maximum(n_hi, -6.0), 6.0)
        lo = _quantize_e2m1_code(n_lo)
        hi = _quantize_e2m1_code(n_hi)
        packed = (lo & 0xF) | ((hi & 0xF) << 4)
        fp4_offsets = (row * k_size + k0) // 2 + pair
        pair_valid = (k0 + pair * 2) < k_size
        tl.store(fp4_ptr + fp4_offsets, packed.to(tl.int8), mask=pair_valid)

        row0 = row % 32
        row1 = (row // 32) % 4
        rest_m = row // 128
        kk0 = sf_block % 4
        rest_k = sf_block // 4
        scale_offset = (
            row0 * scale_stride0
            + row1 * scale_stride1
            + rest_m * scale_stride2
            + kk0 * scale_stride3
            + rest_k * scale_stride4
            + 0 * scale_stride5
        )
        tl.store(scale_ptr + scale_offset, scale_code.to(tl.int8))


def make_triton_blockscaled_fp4_operand(
    x_mkl: torch.Tensor,
    *,
    fp4_dtype: Any,
    sf_dtype: Any,
    sf_vec_size: int,
    cutlass_torch: Any,
) -> dict[str, Any]:
    """Create a CUTLASS blockscaled FP4 operand with a Triton pack kernel."""

    _check_e4m3(sf_dtype)
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if x_mkl.ndim != 3 or x_mkl.shape[2] != 1:
        raise ValueError(f"expected tensor shape (mn, k, 1), got {tuple(x_mkl.shape)}")
    if not x_mkl.is_cuda:
        raise ValueError("Triton blockscaled FP4 packer requires a CUDA tensor")
    if sf_vec_size != 16:
        raise ValueError("Triton blockscaled FP4 packer currently supports sf_vec_size=16")

    mn, k, l = (int(v) for v in x_mkl.shape)
    if l != 1:
        raise ValueError("Triton blockscaled FP4 packer currently supports l=1")
    if k % 2 != 0:
        raise ValueError("Triton blockscaled FP4 packer currently requires even K")

    x_float = x_mkl.detach().float().contiguous()
    sf_k = math.ceil(k / sf_vec_size)
    fp4_tensor, fp4_storage = _make_cutlass_fp4_tensor(
        shape_tensor=x_float,
        fp4_dtype=fp4_dtype,
        cutlass_torch=cutlass_torch,
    )
    scale_tensor, scale_storage = _make_cutlass_scale_tensor(
        mn=mn,
        sf_k=sf_k,
        l=l,
        sf_dtype=sf_dtype,
        cutlass_torch=cutlass_torch,
    )
    fill_triton_blockscaled_fp4_operand(
        x_float,
        fp4_storage=fp4_storage,
        scale_storage=scale_storage,
        sf_vec_size=sf_vec_size,
        zero_storage=True,
    )
    return {
        "normalized": x_float,
        "fp4_tensor": fp4_tensor,
        "fp4_storage": fp4_storage,
        "scale_tensor": scale_tensor,
        "scale_storage": scale_storage,
        "packed_fp4_bytes": int(math.ceil(x_float.numel() / 2)),
        "pack_backend": "triton_e4m3",
    }


def make_empty_triton_blockscaled_fp4_operand(
    x_mkl: torch.Tensor,
    *,
    fp4_dtype: Any,
    sf_dtype: Any,
    sf_vec_size: int,
    cutlass_torch: Any,
) -> dict[str, Any]:
    """Allocate CUTLASS operand tensors for repeated Triton storage fills."""

    _check_e4m3(sf_dtype)
    if x_mkl.ndim != 3 or x_mkl.shape[2] != 1:
        raise ValueError(f"expected tensor shape (mn, k, 1), got {tuple(x_mkl.shape)}")
    if not x_mkl.is_cuda:
        raise ValueError("Triton blockscaled FP4 packer requires a CUDA tensor")
    if sf_vec_size != 16:
        raise ValueError("Triton blockscaled FP4 packer currently supports sf_vec_size=16")
    mn, k, l = (int(v) for v in x_mkl.shape)
    if l != 1:
        raise ValueError("Triton blockscaled FP4 packer currently supports l=1")
    if k % 2 != 0:
        raise ValueError("Triton blockscaled FP4 packer currently requires even K")

    shape_tensor = torch.empty_like(x_mkl.detach().float().contiguous())
    sf_k = math.ceil(k / sf_vec_size)
    fp4_tensor, fp4_storage = _make_cutlass_fp4_tensor(
        shape_tensor=shape_tensor,
        fp4_dtype=fp4_dtype,
        cutlass_torch=cutlass_torch,
    )
    scale_tensor, scale_storage = _make_cutlass_scale_tensor(
        mn=mn,
        sf_k=sf_k,
        l=l,
        sf_dtype=sf_dtype,
        cutlass_torch=cutlass_torch,
    )
    fp4_storage.zero_()
    scale_storage.zero_()
    return {
        "normalized": shape_tensor,
        "fp4_tensor": fp4_tensor,
        "fp4_storage": fp4_storage,
        "scale_tensor": scale_tensor,
        "scale_storage": scale_storage,
        "packed_fp4_bytes": int(math.ceil(shape_tensor.numel() / 2)),
        "pack_backend": "triton_e4m3_reusable",
    }


def fill_triton_blockscaled_fp4_operand(
    x_mkl: torch.Tensor,
    *,
    fp4_storage: torch.Tensor,
    scale_storage: torch.Tensor,
    sf_vec_size: int,
    zero_storage: bool = False,
) -> None:
    """Fill preallocated FP4 and scale storage using the Triton pack kernel."""

    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if x_mkl.ndim != 3 or x_mkl.shape[2] != 1:
        raise ValueError(f"expected tensor shape (mn, k, 1), got {tuple(x_mkl.shape)}")
    if not x_mkl.is_cuda:
        raise ValueError("Triton blockscaled FP4 packer requires a CUDA tensor")
    if sf_vec_size != 16:
        raise ValueError("Triton blockscaled FP4 packer currently supports sf_vec_size=16")
    mn, k, l = (int(v) for v in x_mkl.shape)
    if l != 1:
        raise ValueError("Triton blockscaled FP4 packer currently supports l=1")
    if k % 2 != 0:
        raise ValueError("Triton blockscaled FP4 packer currently requires even K")

    x_float = x_mkl.detach().float().contiguous()
    sf_k = math.ceil(k / sf_vec_size)
    if zero_storage:
        fp4_storage.zero_()
        scale_storage.zero_()
    strides = tuple(int(v) for v in scale_storage.stride())
    _triton_blockscaled_fp4_pack_kernel[(mn, sf_k)](
        x_float,
        fp4_storage,
        scale_storage,
        mn,
        k,
        sf_k,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        BLOCK_K=sf_vec_size,
        num_warps=1,
    )
