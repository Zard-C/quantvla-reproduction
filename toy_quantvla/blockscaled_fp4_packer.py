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
    normalized = torch.where(
        decoded_scale_expanded > 0,
        x_float / decoded_scale_expanded,
        torch.zeros_like(x_float),
    ).clamp(-6.0, 6.0).contiguous()

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
    codes = quantize_e2m1_codes(normalized)
    packed_fp4_bytes = pack_e2m1_codes_into_storage(codes, fp4_storage)
    decoded = decode_e2m1_codes(codes, dtype=torch.float32).view_as(normalized)

    scale_template = _scale_mma_template(mn, sf_k, l)
    scale_tensor, scale_storage = cutlass_torch.cute_tensor_like(
        scale_template,
        sf_dtype,
        is_dynamic_layout=True,
        assumed_align=16,
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
