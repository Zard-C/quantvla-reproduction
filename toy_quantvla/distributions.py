from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DistributionPreset:
    name: str
    layer_scale_sigma: float
    channel_scale_sigma: float
    input_channel_scale_sigma: float
    outlier_prob: float
    outlier_min: float
    outlier_max: float
    activation_tail: str
    tail_prob: float
    tail_scale: float
    activation_channel_sigma: float
    student_drift_sigma: float
    student_noise: float


PRESETS: dict[str, DistributionPreset] = {
    "standard_normal": DistributionPreset(
        name="standard_normal",
        layer_scale_sigma=0.0,
        channel_scale_sigma=0.0,
        input_channel_scale_sigma=0.0,
        outlier_prob=0.0,
        outlier_min=1.0,
        outlier_max=1.0,
        activation_tail="gaussian",
        tail_prob=0.0,
        tail_scale=1.0,
        activation_channel_sigma=0.0,
        student_drift_sigma=0.04,
        student_noise=0.01,
    ),
    "llm_attn": DistributionPreset(
        name="llm_attn",
        layer_scale_sigma=0.18,
        channel_scale_sigma=0.35,
        input_channel_scale_sigma=0.30,
        outlier_prob=0.015,
        outlier_min=4.0,
        outlier_max=8.0,
        activation_tail="student_t",
        tail_prob=0.04,
        tail_scale=2.0,
        activation_channel_sigma=0.32,
        student_drift_sigma=0.16,
        student_noise=0.025,
    ),
    "llm_mlp": DistributionPreset(
        name="llm_mlp",
        layer_scale_sigma=0.22,
        channel_scale_sigma=0.50,
        input_channel_scale_sigma=0.36,
        outlier_prob=0.025,
        outlier_min=5.0,
        outlier_max=10.0,
        activation_tail="laplace",
        tail_prob=0.06,
        tail_scale=2.8,
        activation_channel_sigma=0.40,
        student_drift_sigma=0.20,
        student_noise=0.035,
    ),
    "dit_attn": DistributionPreset(
        name="dit_attn",
        layer_scale_sigma=0.26,
        channel_scale_sigma=0.45,
        input_channel_scale_sigma=0.42,
        outlier_prob=0.030,
        outlier_min=5.0,
        outlier_max=12.0,
        activation_tail="student_t",
        tail_prob=0.07,
        tail_scale=2.7,
        activation_channel_sigma=0.46,
        student_drift_sigma=0.26,
        student_noise=0.045,
    ),
    "dit_mlp": DistributionPreset(
        name="dit_mlp",
        layer_scale_sigma=0.24,
        channel_scale_sigma=0.55,
        input_channel_scale_sigma=0.48,
        outlier_prob=0.035,
        outlier_min=5.0,
        outlier_max=12.0,
        activation_tail="laplace",
        tail_prob=0.08,
        tail_scale=3.0,
        activation_channel_sigma=0.52,
        student_drift_sigma=0.24,
        student_noise=0.050,
    ),
}


def get_preset(name: str | DistributionPreset) -> DistributionPreset:
    if isinstance(name, DistributionPreset):
        return name
    if name not in PRESETS:
        raise KeyError(f"unknown distribution preset {name!r}")
    return PRESETS[name]


def _lognormal(shape: tuple[int, ...], sigma: float, device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if sigma == 0.0:
        return torch.ones(shape, device=device)
    return torch.exp(torch.randn(shape, device=device, generator=generator) * sigma)


def _uniform(shape: tuple[int, ...], low: float, high: float, device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
    return torch.empty(shape, device=device).uniform_(low, high, generator=generator)


def sample_weight(
    out_features: int,
    in_features: int,
    preset: str | DistributionPreset,
    *,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    preset = get_preset(preset)
    device = torch.device(device)
    base = torch.randn((out_features, in_features), device=device, generator=generator)
    base = base / max(in_features, 1) ** 0.5

    layer_scale = _lognormal((1, 1), preset.layer_scale_sigma, device, generator)
    out_scale = _lognormal((out_features, 1), preset.channel_scale_sigma, device, generator)
    in_scale = _lognormal((1, in_features), preset.input_channel_scale_sigma, device, generator)
    weight = base * layer_scale * out_scale * in_scale

    if preset.outlier_prob > 0.0:
        out_mask = torch.rand((out_features, 1), device=device, generator=generator) < preset.outlier_prob
        in_mask = torch.rand((1, in_features), device=device, generator=generator) < preset.outlier_prob
        out_mult = _uniform((out_features, 1), preset.outlier_min, preset.outlier_max, device, generator)
        in_mult = _uniform((1, in_features), preset.outlier_min, preset.outlier_max, device, generator)
        weight = weight * torch.where(out_mask, out_mult, torch.ones_like(out_mult))
        weight = weight * torch.where(in_mask, in_mult, torch.ones_like(in_mult))

    return weight


def sample_activation(
    batch: int,
    seq_len: int,
    hidden: int,
    preset: str | DistributionPreset,
    *,
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    preset = get_preset(preset)
    device = torch.device(device)
    x = torch.randn((batch, seq_len, hidden), device=device, generator=generator)

    if preset.tail_prob > 0.0:
        mask = torch.rand(x.shape, device=device, generator=generator) < preset.tail_prob
        if preset.activation_tail == "laplace":
            tail_dist = torch.distributions.Laplace(
                torch.tensor(0.0, device=device),
                torch.tensor(preset.tail_scale, device=device),
            )
            tail = tail_dist.sample(x.shape)
        elif preset.activation_tail == "student_t":
            tail_dist = torch.distributions.StudentT(
                df=torch.tensor(3.0, device=device),
                loc=torch.tensor(0.0, device=device),
                scale=torch.tensor(preset.tail_scale, device=device),
            )
            tail = tail_dist.sample(x.shape)
        else:
            tail = torch.randn(x.shape, device=device, generator=generator) * preset.tail_scale
        x = torch.where(mask, tail, x)

    channel_scale = _lognormal((1, 1, hidden), preset.activation_channel_sigma, device, generator)
    x = x * channel_scale

    if preset.outlier_prob > 0.0:
        mask = torch.rand((1, 1, hidden), device=device, generator=generator) < preset.outlier_prob
        mult = _uniform((1, 1, hidden), preset.outlier_min, preset.outlier_max, device, generator)
        x = x * torch.where(mask, mult, torch.ones_like(mult))

    return x


def apply_student_drift(
    x: torch.Tensor,
    preset: str | DistributionPreset,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    preset = get_preset(preset)
    hidden = x.shape[-1]
    channel_shift = _lognormal((1, 1, hidden), preset.student_drift_sigma, x.device, generator)
    noise = torch.randn(x.shape, device=x.device, generator=generator) * preset.student_noise
    return x * channel_shift + noise * x.detach().float().std().clamp_min(1e-6)


def summarize_tensor(x: torch.Tensor) -> dict[str, float | list[int]]:
    x_f = x.detach().float().flatten()
    abs_x = x_f.abs()
    p99 = torch.quantile(abs_x, 0.99).item()
    p999 = torch.quantile(abs_x, 0.999).item()
    return {
        "shape": list(x.shape),
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()),
        "max_abs": float(abs_x.max().item()),
        "p99_abs": float(p99),
        "p99_9_abs": float(p999),
        "tail_ratio_max_over_p99_9": float(abs_x.max().item() / max(p999, 1e-8)),
    }


def summarize_weight(weight: torch.Tensor, topk: int = 8) -> dict[str, object]:
    summary = summarize_tensor(weight)
    per_out = weight.detach().float().abs().amax(dim=1)
    values, indices = torch.topk(per_out, k=min(topk, per_out.numel()))
    summary.update(
        {
            "per_output_channel_max_abs_mean": float(per_out.mean().item()),
            "per_output_channel_max_abs_p99": float(torch.quantile(per_out, 0.99).item()),
            "top_outlier_channels": [
                {"index": int(i.item()), "max_abs": float(v.item())}
                for v, i in zip(values, indices)
            ],
        }
    )
    return summary


def preset_summary() -> dict[str, dict[str, object]]:
    return {name: asdict(preset) for name, preset in PRESETS.items()}


WEIGHT_STATS_SCHEMA = {
    "type": "object",
    "required": ["layers"],
    "properties": {
        "layers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "name",
                    "shape",
                    "mean",
                    "std",
                    "max_abs",
                    "p99_abs",
                    "p99_9_abs",
                    "tail_ratio_max_over_p99_9",
                    "per_output_channel_max_abs_mean",
                    "per_output_channel_max_abs_p99",
                    "top_outlier_channels",
                ],
            },
        }
    },
}
