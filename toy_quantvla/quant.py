from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


@dataclass
class QuantStats:
    bits: int
    qmax: int
    scale_mean: float
    scale_max: float
    clip_abs: float
    saturation_ratio: float


def _signed_qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError("signed quantization needs at least 2 bits")
    return (1 << (bits - 1)) - 1


def percentile_abs(x: torch.Tensor, percentile: Optional[float]) -> torch.Tensor:
    if percentile is None or percentile >= 100.0:
        return x.detach().abs().amax()
    q = torch.tensor(percentile / 100.0, device=x.device, dtype=x.dtype)
    return torch.quantile(x.detach().abs().flatten(), q)


def fake_quant_symmetric(
    x: torch.Tensor,
    bits: int,
    *,
    scale_dim: Optional[int] = None,
    clip_percentile: Optional[float] = None,
) -> tuple[torch.Tensor, QuantStats]:
    """Symmetric fake quantization with optional per-dimension scales."""
    qmax = _signed_qmax(bits)

    if clip_percentile is None or clip_percentile >= 100.0:
        if scale_dim is None:
            clip_abs = x.detach().abs().amax().clamp_min(EPS)
            scale = clip_abs / qmax
        else:
            clip_abs = x.detach().abs().amax(dim=scale_dim, keepdim=True).clamp_min(EPS)
            scale = clip_abs / qmax
    else:
        clip_abs = percentile_abs(x, clip_percentile).clamp_min(EPS)
        scale = clip_abs / qmax

    x_clip = torch.clamp(x, -clip_abs, clip_abs)
    q = torch.clamp(torch.round(x_clip / scale), -qmax, qmax)
    y = q * scale
    sat = (x.detach().abs() > clip_abs).float().mean().item()

    stats = QuantStats(
        bits=bits,
        qmax=qmax,
        scale_mean=float(scale.detach().float().mean().item()),
        scale_max=float(scale.detach().float().amax().item()),
        clip_abs=float(clip_abs.detach().float().amax().item()),
        saturation_ratio=float(sat),
    )
    return y, stats


def fake_quant_weight(weight: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, QuantStats]:
    return fake_quant_symmetric(weight, bits, scale_dim=1)


def fake_quant_activation(
    activation: torch.Tensor,
    bits: int = 8,
    clip_percentile: float = 99.9,
) -> tuple[torch.Tensor, QuantStats]:
    return fake_quant_symmetric(
        activation,
        bits,
        scale_dim=None,
        clip_percentile=clip_percentile,
    )


def fake_quant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    weight_bits: int = 4,
    activation_bits: int = 8,
    activation_percentile: float = 99.9,
    quantize_weight: bool = True,
    quantize_activation: bool = True,
) -> tuple[torch.Tensor, dict[str, QuantStats]]:
    stats: dict[str, QuantStats] = {}
    x_q = x
    w_q = weight
    if quantize_activation:
        x_q, stats["activation"] = fake_quant_activation(
            x, bits=activation_bits, clip_percentile=activation_percentile
        )
    if quantize_weight:
        w_q, stats["weight"] = fake_quant_weight(weight, bits=weight_bits)
    return F.linear(x_q, w_q, bias), stats


def smooth_linear_pair(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    migration_strength: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SmoothQuant-style diagonal migration for Y = X @ W.T.

    The transformation keeps the floating-point linear output unchanged:
    (X / s) @ (W * s).T == X @ W.T.
    """
    if x.shape[-1] != weight.shape[1]:
        raise ValueError("activation hidden dimension must match weight input dimension")

    act_max = x.detach().abs().reshape(-1, x.shape[-1]).amax(dim=0).clamp_min(EPS)
    weight_max = weight.detach().abs().amax(dim=0).clamp_min(EPS)
    strength = float(migration_strength)
    scale = (act_max.pow(strength) / weight_max.pow(1.0 - strength)).clamp_min(EPS)
    return x / scale, weight * scale.unsqueeze(0), scale


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.detach().float().flatten()
    b_f = b.detach().float().flatten()
    return float(F.cosine_similarity(a_f, b_f, dim=0).item())


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.detach().float() - b.detach().float()) ** 2).item())


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x.float() ** 2).clamp_min(EPS))


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p = p.detach().float().clamp_min(EPS)
    q = q.detach().float().clamp_min(EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    return float((0.5 * (kl_pm + kl_qm)).mean().item())


def entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.detach().float().clamp_min(EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    return -torch.sum(p * torch.log(p), dim=-1)
