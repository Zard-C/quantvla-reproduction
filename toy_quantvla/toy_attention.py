from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F

from distributions import apply_student_drift, sample_weight
from quant import entropy, fake_quant_linear, js_divergence, mse, nmse, relative_rms_error, rms


@dataclass
class AttentionWeights:
    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    b_q: torch.Tensor
    b_k: torch.Tensor
    b_v: torch.Tensor
    b_o: torch.Tensor


@dataclass
class MLPWeights:
    w_up: torch.Tensor
    w_down: torch.Tensor
    b_up: torch.Tensor
    b_down: torch.Tensor


@dataclass
class TransformerWeights:
    attention: AttentionWeights
    mlp: MLPWeights


@dataclass
class AttentionOutput:
    output: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    post_o: torch.Tensor


def make_attention_weights(
    hidden: int,
    preset: str,
    *,
    device: torch.device | str,
    generator: Optional[torch.Generator],
) -> AttentionWeights:
    zero = lambda: torch.zeros(hidden, device=device)
    return AttentionWeights(
        w_q=sample_weight(hidden, hidden, preset, device=device, generator=generator),
        w_k=sample_weight(hidden, hidden, preset, device=device, generator=generator),
        w_v=sample_weight(hidden, hidden, preset, device=device, generator=generator),
        w_o=sample_weight(hidden, hidden, preset, device=device, generator=generator),
        b_q=zero(),
        b_k=zero(),
        b_v=zero(),
        b_o=zero(),
    )


def make_mlp_weights(
    hidden: int,
    mlp_hidden: int,
    preset: str,
    *,
    device: torch.device | str,
    generator: Optional[torch.Generator],
) -> MLPWeights:
    return MLPWeights(
        w_up=sample_weight(mlp_hidden, hidden, preset, device=device, generator=generator),
        w_down=sample_weight(hidden, mlp_hidden, preset, device=device, generator=generator),
        b_up=torch.zeros(mlp_hidden, device=device),
        b_down=torch.zeros(hidden, device=device),
    )


def make_transformer_weights(
    hidden: int,
    mlp_hidden: int,
    attn_preset: str,
    mlp_preset: str,
    *,
    device: torch.device | str,
    generator: Optional[torch.Generator],
) -> TransformerWeights:
    return TransformerWeights(
        attention=make_attention_weights(hidden, attn_preset, device=device, generator=generator),
        mlp=make_mlp_weights(hidden, mlp_hidden, mlp_preset, device=device, generator=generator),
    )


def _split_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, seq_len, hidden = x.shape
    head_dim = hidden // heads
    return x.reshape(batch, seq_len, heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(batch, seq_len, heads * head_dim)


def attention_forward(
    x: torch.Tensor,
    weights: AttentionWeights,
    *,
    heads: int,
    quantize_attention: bool = False,
    activation_percentile: float = 99.9,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
) -> AttentionOutput:
    if x.shape[-1] % heads != 0:
        raise ValueError("hidden size must be divisible by head count")

    q, _ = fake_quant_linear(
        x,
        weights.w_q,
        weights.b_q,
        quantize_weight=quantize_attention,
        quantize_activation=quantize_attention,
        activation_percentile=activation_percentile,
    )
    k, _ = fake_quant_linear(
        x,
        weights.w_k,
        weights.b_k,
        quantize_weight=quantize_attention,
        quantize_activation=quantize_attention,
        activation_percentile=activation_percentile,
    )
    v, _ = fake_quant_linear(
        x,
        weights.w_v,
        weights.b_v,
        quantize_weight=quantize_attention,
        quantize_activation=quantize_attention,
        activation_percentile=activation_percentile,
    )

    q_h = _split_heads(q, heads)
    k_h = _split_heads(k, heads)
    v_h = _split_heads(v, heads)
    head_dim = q_h.shape[-1]

    if alpha is not None:
        q_h = q_h * alpha.reshape(1, heads, 1, 1).to(q_h.device)

    logits = torch.matmul(q_h, k_h.transpose(-2, -1)) / head_dim**0.5
    probs = torch.softmax(logits, dim=-1)
    context = _merge_heads(torch.matmul(probs, v_h))
    post_o, _ = fake_quant_linear(
        context,
        weights.w_o,
        weights.b_o,
        quantize_weight=quantize_attention,
        quantize_activation=quantize_attention,
        activation_percentile=activation_percentile,
    )
    if beta is not None:
        post_o = post_o * beta.to(post_o.device)
    return AttentionOutput(output=x + post_o, logits=logits, probs=probs, post_o=post_o)


def mlp_forward(
    x: torch.Tensor,
    weights: MLPWeights,
    *,
    quantize_mlp: bool = False,
    activation_percentile: float = 99.9,
) -> torch.Tensor:
    up, _ = fake_quant_linear(
        x,
        weights.w_up,
        weights.b_up,
        quantize_weight=quantize_mlp,
        quantize_activation=quantize_mlp,
        activation_percentile=activation_percentile,
    )
    hidden = F.gelu(up)
    down, _ = fake_quant_linear(
        hidden,
        weights.w_down,
        weights.b_down,
        quantize_weight=quantize_mlp,
        quantize_activation=quantize_mlp,
        activation_percentile=activation_percentile,
    )
    return x + down


def transformer_block_forward(
    x: torch.Tensor,
    weights: TransformerWeights,
    *,
    heads: int,
    quantize_attention: bool = False,
    quantize_mlp: bool = False,
    upstream_drift_preset: Optional[str] = None,
    activation_percentile: float = 99.9,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, AttentionOutput]:
    x_in = x
    if upstream_drift_preset is not None:
        x_in = apply_student_drift(x_in, upstream_drift_preset, generator=generator)
    attn = attention_forward(
        x_in,
        weights.attention,
        heads=heads,
        quantize_attention=quantize_attention,
        activation_percentile=activation_percentile,
    )
    out = mlp_forward(
        attn.output,
        weights.mlp,
        quantize_mlp=quantize_mlp,
        activation_percentile=activation_percentile,
    )
    return out, attn


def compare_attention_outputs(teacher: AttentionOutput, student: AttentionOutput) -> dict[str, float]:
    teacher_entropy = entropy(teacher.probs).mean()
    student_entropy = entropy(student.probs).mean()
    return {
        "logits_std_teacher": float(teacher.logits.detach().float().std(unbiased=False).item()),
        "logits_std_student": float(student.logits.detach().float().std(unbiased=False).item()),
        "logits_std_abs_error": abs(
            float(teacher.logits.detach().float().std(unbiased=False).item())
            - float(student.logits.detach().float().std(unbiased=False).item())
        ),
        "attention_js": js_divergence(teacher.probs, student.probs),
        "entropy_abs_error": float(torch.abs(teacher_entropy - student_entropy).item()),
        "post_o_rms_teacher": float(rms(teacher.post_o).item()),
        "post_o_rms_student": float(rms(student.post_o).item()),
        "post_o_rms_abs_error": abs(float(rms(teacher.post_o).item()) - float(rms(student.post_o).item())),
        "post_o_rms_relative_error": abs(float(rms(teacher.post_o).item()) - float(rms(student.post_o).item()))
        / max(float(rms(teacher.post_o).item()), 1e-8),
        "output_mse": mse(teacher.output, student.output),
        "output_nmse": nmse(teacher.output, student.output),
        "output_relative_rms_error": relative_rms_error(teacher.output, student.output),
    }


def per_head_logits_std(logits: torch.Tensor) -> torch.Tensor:
    return logits.detach().float().flatten(2).std(dim=-1, unbiased=False).mean(dim=0)


def calibrate_alpha(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    *,
    log_clamp: float = 0.30,
    neutral_band: float = 0.03,
) -> torch.Tensor:
    std_t = per_head_logits_std(teacher_logits).clamp_min(1e-6)
    std_s = per_head_logits_std(student_logits).clamp_min(1e-6)
    log_alpha = torch.log(std_t / std_s).clamp(-log_clamp, log_clamp)
    return torch.where(torch.abs(log_alpha) < neutral_band, torch.ones_like(log_alpha), torch.exp(log_alpha))


def calibrate_beta(
    teacher_post_o: torch.Tensor,
    student_post_o: torch.Tensor,
    *,
    log_clamp: float = 0.30,
    neutral_band: float = 0.03,
) -> torch.Tensor:
    log_beta = torch.log(rms(teacher_post_o) / rms(student_post_o).clamp_min(1e-6)).clamp(
        -log_clamp, log_clamp
    )
    return torch.where(torch.abs(log_beta) < neutral_band, torch.ones_like(log_beta), torch.exp(log_beta))


def apply_direction(
    value: torch.Tensor,
    factor: torch.Tensor,
    direction: Literal["multiply", "divide"],
) -> torch.Tensor:
    factor = factor.to(value.device)
    if direction == "multiply":
        return value * factor
    if direction == "divide":
        return value / factor.clamp_min(1e-6)
    raise ValueError(f"unknown direction {direction}")
