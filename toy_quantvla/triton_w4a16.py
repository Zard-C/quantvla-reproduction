"""Packed W4A16 Linear utilities and Triton forward kernel.

This module implements the first real-kernel target for Phase 6:
per-output-channel signed int4 weights with fp16/bf16 activations.

The quantization rule intentionally matches the Phase 3/5 fake quant path:

    scale[out] = max(abs(W[out, :])) / 7
    q[out, in] = round(W[out, in] / scale[out]).clamp(-7, 7)

Two signed int4 values are packed into one uint8 byte by storing q + 8 in each
nibble. The value 8 therefore represents quantized zero, which is also used as
padding for odd K.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
W4_QMAX = 7


try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on machines without Triton.
    triton = None
    tl = None


@dataclass(frozen=True)
class PackedW4A16Weight:
    qweight: torch.Tensor
    scale: torch.Tensor
    in_features: int
    out_features: int

    @property
    def packed_in_features(self) -> int:
        return (self.in_features + 1) // 2

    def fp16_weight_bytes(self) -> int:
        return 2 * self.out_features * self.in_features

    def packed_weight_bytes(self) -> int:
        return int(self.qweight.numel() * self.qweight.element_size())

    def scale_bytes(self) -> int:
        return int(self.scale.numel() * self.scale.element_size())

    def compression_ratio_vs_fp16(self) -> float:
        packed_total = self.packed_weight_bytes() + self.scale_bytes()
        return float(packed_total / max(1, self.fp16_weight_bytes()))


def _check_weight_2d(weight: torch.Tensor) -> None:
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D [out_features, in_features], got {tuple(weight.shape)}")
    if not torch.is_floating_point(weight):
        raise TypeError(f"weight must be floating point, got {weight.dtype}")


def quantize_weight_w4_per_row(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed int quantized weight and per-row scales.

    The returned q tensor is int8 with values in [-7, 7]. This is a debug and
    reference helper; deployment should use :func:`pack_weight_w4_per_row`.
    """
    _check_weight_2d(weight)
    weight_f = weight.detach().float()
    scale = (weight_f.abs().amax(dim=1) / W4_QMAX).clamp_min(EPS)
    q = torch.round(weight_f / scale[:, None]).clamp(-W4_QMAX, W4_QMAX).to(torch.int8)
    return q, scale


def pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values in [-7, 7] into uint8 nibbles.

    Layout is row-major [N, ceil(K / 2)]. Even K positions are stored in the low
    nibble, odd positions in the high nibble. Odd K is padded with q=0.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be 2D, got {tuple(q.shape)}")
    if q.min().item() < -W4_QMAX or q.max().item() > W4_QMAX:
        raise ValueError("q values must be in [-7, 7]")

    n, k = q.shape
    if k % 2:
        pad = torch.zeros((n, 1), dtype=q.dtype, device=q.device)
        q = torch.cat([q, pad], dim=1)
    uq = (q.to(torch.int16) + 8).to(torch.uint8)
    low = uq[:, 0::2]
    high = uq[:, 1::2]
    return low | (high << 4)


def unpack_int4_nibbles(qweight: torch.Tensor, in_features: int) -> torch.Tensor:
    """Unpack uint8 nibbles into signed int8 values in [-8, 7]."""
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2D, got {tuple(qweight.shape)}")
    if qweight.dtype != torch.uint8:
        raise TypeError(f"qweight must be torch.uint8, got {qweight.dtype}")
    if in_features <= 0:
        raise ValueError("in_features must be positive")

    low = (qweight & 0x0F).to(torch.int16) - 8
    high = ((qweight >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.empty((qweight.shape[0], qweight.shape[1] * 2), dtype=torch.int8, device=qweight.device)
    q[:, 0::2] = low.to(torch.int8)
    q[:, 1::2] = high.to(torch.int8)
    return q[:, :in_features]


def pack_weight_w4_per_row(
    weight: torch.Tensor,
    *,
    scale_dtype: torch.dtype = torch.float32,
) -> PackedW4A16Weight:
    """Quantize and pack a Linear weight shaped [out_features, in_features]."""
    q, scale = quantize_weight_w4_per_row(weight)
    qweight = pack_int4_nibbles(q).contiguous()
    return PackedW4A16Weight(
        qweight=qweight,
        scale=scale.to(dtype=scale_dtype).contiguous(),
        in_features=int(weight.shape[1]),
        out_features=int(weight.shape[0]),
    )


def dequantize_w4_weight(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    in_features: int,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Materialize the packed W4 weight as a dense floating-point tensor."""
    q = unpack_int4_nibbles(qweight, in_features).float()
    w = q * scale.float()[:, None]
    if dtype is not None:
        w = w.to(dtype=dtype)
    return w


def w4a16_linear_reference(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
) -> torch.Tensor:
    """PyTorch reference path using dense dequantization."""
    weight = dequantize_w4_weight(qweight, scale, in_features, dtype=x.dtype)
    return F.linear(x, weight, bias)


if tl is not None:

    @triton.jit
    def _w4a16_linear_kernel(
        x_ptr,
        qweight_ptr,
        scale_ptr,
        bias_ptr,
        y_ptr,
        m_size,
        n_size,
        k_size,
        k_packed,
        has_bias: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ) -> None:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in tl.range(0, k_size, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * k_size + k[None, :],
                mask=(offs_m[:, None] < m_size) & (k[None, :] < k_size),
                other=0.0,
            )

            packed_col = k // 2
            packed = tl.load(
                qweight_ptr + offs_n[None, :] * k_packed + packed_col[:, None],
                mask=(offs_n[None, :] < n_size) & (k[:, None] < k_size),
                other=8,
            ).to(tl.int32)
            low = packed & 0x0F
            high = (packed >> 4) & 0x0F
            nibble = tl.where((k[:, None] & 1) == 0, low, high)
            q = nibble - 8
            scale = tl.load(scale_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            w = (q.to(tl.float32) * scale[None, :]).to(tl.float16)
            acc += tl.dot(x, w, out_dtype=tl.float32)

        if has_bias:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < n_size, other=0.0)
            acc += bias[None, :]

        tl.store(
            y_ptr + offs_m[:, None] * n_size + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size),
        )


def _next_power_of_2_at_least(value: int, minimum: int, maximum: int) -> int:
    block = 1 << (max(value, minimum) - 1).bit_length()
    return min(maximum, block)


def w4a16_linear_triton(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
    *,
    block_m: int = 16,
    block_n: int = 32,
    block_k: int = 64,
) -> torch.Tensor:
    """Run the Triton fused dequantization matmul path."""
    if triton is None or tl is None:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise ValueError("Triton W4A16 kernel requires a CUDA input tensor")
    if x.dtype != torch.float16:
        raise TypeError(f"first Triton W4A16 kernel supports torch.float16 activation, got {x.dtype}")
    if qweight.dtype != torch.uint8:
        raise TypeError(f"qweight must be torch.uint8, got {qweight.dtype}")
    if x.shape[-1] != in_features:
        raise ValueError(f"input last dim {x.shape[-1]} does not match in_features {in_features}")

    original_shape = tuple(x.shape[:-1])
    x_2d = x.contiguous().reshape(-1, in_features)
    m_size = x_2d.shape[0]
    n_size = qweight.shape[0]
    k_packed = qweight.shape[1]
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    qweight = qweight.to(device=x.device).contiguous()
    scale = scale.to(device=x.device).contiguous()
    if bias is not None:
        bias = bias.to(device=x.device, dtype=x.dtype).contiguous()

    bm = _next_power_of_2_at_least(min(m_size, block_m), 16, block_m)
    grid = (triton.cdiv(m_size, bm), triton.cdiv(n_size, block_n))
    _w4a16_linear_kernel[grid](
        x_2d,
        qweight,
        scale,
        bias if bias is not None else out,
        out,
        m_size,
        n_size,
        in_features,
        k_packed,
        bias is not None,
        BLOCK_M=bm,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return out.reshape(*original_shape, n_size)


def w4a16_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None,
    in_features: int,
    *,
    prefer_triton: bool = True,
) -> torch.Tensor:
    """Dispatch W4A16 Linear to Triton on CUDA, otherwise PyTorch reference."""
    if prefer_triton and x.is_cuda and triton is not None:
        return w4a16_linear_triton(x, qweight, scale, bias, in_features)
    return w4a16_linear_reference(x, qweight, scale, bias, in_features)


class PackedW4A16Linear(nn.Module):
    """Drop-in inference-only replacement for ``nn.Linear``."""

    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        *,
        prefer_triton: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.prefer_triton = bool(prefer_triton)
        self.register_buffer("qweight", qweight.contiguous())
        self.register_buffer("scale", scale.contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone().contiguous())

    @classmethod
    def from_linear(
        cls,
        base: nn.Linear,
        *,
        scale_dtype: torch.dtype = torch.float32,
        prefer_triton: bool = True,
    ) -> "PackedW4A16Linear":
        packed = pack_weight_w4_per_row(base.weight.detach(), scale_dtype=scale_dtype)
        bias = base.bias.detach().clone() if base.bias is not None else None
        return cls(
            packed.qweight,
            packed.scale,
            bias,
            packed.in_features,
            packed.out_features,
            prefer_triton=prefer_triton,
        )

    def dequantized_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        return dequantize_w4_weight(self.qweight, self.scale, self.in_features, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w4a16_linear(
            x,
            self.qweight,
            self.scale,
            self.bias,
            self.in_features,
            prefer_triton=self.prefer_triton,
        )

    def extra_repr(self) -> str:
        total_bytes = self.qweight.numel() * self.qweight.element_size()
        total_bytes += self.scale.numel() * self.scale.element_size()
        return (
            f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
            f"packed_bytes={total_bytes}, prefer_triton={self.prefer_triton}"
        )


def relative_rmse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    ref = reference.detach().float()
    err = estimate.detach().float() - ref
    denom = torch.mean(ref.square()).clamp_min(EPS)
    return float(torch.sqrt(torch.mean(err.square()) / denom).item())


def cosine_similarity(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(reference.detach().float().flatten(), estimate.detach().float().flatten(), dim=0).item()
    )
