"""ATM/OHB calibration probe for GR00T W4A8 fake-quant students.

ATM is implemented as a per-DiT-attention query scale that matches teacher and
student attention-logit standard deviations.

OHB is implemented as a per-DiT-attention output scale that matches teacher and
student attention-output RMS before the residual add in BasicTransformerBlock.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phase3_activation_capture import build_variant_observation
from phase3_fake_quant_forward import (
    aggregate_metrics,
    collect_activation_max,
    compare_actions,
    config_groups,
    make_smooth_scales,
    patch_modules,
    restore_modules,
    set_seed,
)
from phase3_gr00t_smoke import _insert_paths


EPS = 1e-8
DIT_ATTN_MODULE_RE = re.compile(r"^action_head\.model\.transformer_blocks\.\d+\.attn1$")


@dataclass
class AttentionStat:
    logits_std: list[float] = field(default_factory=list)
    output_rms: list[float] = field(default_factory=list)

    def add(self, logits_std: float, output_rms: float) -> None:
        self.logits_std.append(float(logits_std))
        self.output_rms.append(float(output_rms))

    def mean_logits_std(self) -> float:
        return float(np.asarray(self.logits_std, dtype=np.float64).mean()) if self.logits_std else 0.0

    def mean_output_rms(self) -> float:
        return float(np.asarray(self.output_rms, dtype=np.float64).mean()) if self.output_rms else 0.0

    def to_result(self) -> dict[str, Any]:
        return {
            "calls": len(self.logits_std),
            "logits_std_mean": self.mean_logits_std(),
            "output_rms_mean": self.mean_output_rms(),
        }


class ATMOHBProcessor:
    def __init__(
        self,
        name: str,
        stats: AttentionStat | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ):
        self.name = name
        self.stats = stats
        self.alpha = alpha
        self.beta = beta

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        temb: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        import torch
        import torch.nn.functional as F

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        logits_std = 0.0
        if self.stats is not None:
            with torch.no_grad():
                logits = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(head_dim)
                logits_std = float(logits.std(unbiased=False).item())

        if self.alpha is not None:
            query = query * float(self.alpha)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        with torch.no_grad():
            output_rms = float(torch.sqrt(torch.mean(hidden_states.float().square()).clamp_min(EPS)).item())
            if self.stats is not None:
                self.stats.add(logits_std, output_rms)

        if self.beta is not None:
            hidden_states = hidden_states * float(self.beta)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class OHBOutputWrapperProcessor:
    """Apply OHB while preserving the original attention processor fast path."""

    def __init__(self, base_processor: Any, beta: float):
        self.base_processor = base_processor
        self.beta = float(beta)

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any | None = None,
        attention_mask: Any | None = None,
        temb: Any | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        base_output = self.base_processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            *args,
            **kwargs,
        )
        if self.beta == 1.0:
            return base_output
        if attn.residual_connection:
            attention_branch = base_output * attn.rescale_output_factor - hidden_states
            return (attention_branch * self.beta + hidden_states) / attn.rescale_output_factor
        return base_output * self.beta


def install_attention_processors(
    model: Any,
    *,
    mode: str,
    scales: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, Any], dict[str, AttentionStat]]:
    stats: dict[str, AttentionStat] = {}
    originals: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not DIT_ATTN_MODULE_RE.match(name) or not hasattr(module, "set_processor"):
            continue
        if mode == "collect":
            originals[name] = module.processor
            stat = AttentionStat()
            stats[name] = stat
            module.set_processor(ATMOHBProcessor(name=name, stats=stat))
        elif mode == "apply":
            scale = scales.get(name, {}) if scales else {}
            if not scale:
                continue
            originals[name] = module.processor
            if "beta" in scale and "alpha" not in scale:
                module.set_processor(OHBOutputWrapperProcessor(module.processor, beta=scale["beta"]))
            else:
                module.set_processor(
                    ATMOHBProcessor(name=name, alpha=scale.get("alpha"), beta=scale.get("beta"))
                )
        else:
            raise ValueError(f"Unknown processor mode: {mode}")
    return originals, stats


def restore_attention_processors(model: Any, originals: dict[str, Any]) -> None:
    for name, processor in originals.items():
        module = get_module(model, name)
        module.set_processor(processor)


def get_module(root: Any, name: str) -> Any:
    current = root
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def collect_attention_stats(policy: Any, observations: list[dict[str, Any]]) -> dict[str, AttentionStat]:
    originals, stats = install_attention_processors(policy.model, mode="collect")
    try:
        for item in observations:
            set_seed(int(item["seed"]))
            policy.get_action(item["obs"])
    finally:
        restore_attention_processors(policy.model, originals)
    return stats


def clamp_ratio(ratio: float, log_clamp: float | None) -> float:
    if log_clamp is None:
        return float(ratio)
    return float(math.exp(max(-log_clamp, min(log_clamp, math.log(max(ratio, EPS))))))


def compute_atm_ohb_scales(
    teacher: dict[str, AttentionStat],
    student: dict[str, AttentionStat],
    *,
    log_clamp: float | None,
) -> dict[str, dict[str, float]]:
    scales: dict[str, dict[str, float]] = {}
    for name, t in teacher.items():
        s = student.get(name)
        if s is None:
            continue
        alpha_raw = t.mean_logits_std() / max(s.mean_logits_std(), EPS)
        beta_raw = t.mean_output_rms() / max(s.mean_output_rms(), EPS)
        scales[name] = {
            "alpha_raw": float(alpha_raw),
            "beta_raw": float(beta_raw),
            "alpha": clamp_ratio(alpha_raw, log_clamp),
            "beta": clamp_ratio(beta_raw, log_clamp),
            "teacher_logits_std": t.mean_logits_std(),
            "student_logits_std": s.mean_logits_std(),
            "teacher_output_rms": t.mean_output_rms(),
            "student_output_rms": s.mean_output_rms(),
        }
    return scales


def scale_summary(scales: dict[str, dict[str, float]]) -> dict[str, Any]:
    out: dict[str, Any] = {"modules": len(scales)}
    for key in ["alpha", "beta", "alpha_raw", "beta_raw"]:
        vals = np.asarray([v[key] for v in scales.values()], dtype=np.float64)
        out[key] = {
            "mean": float(vals.mean()) if vals.size else 0.0,
            "min": float(vals.min()) if vals.size else 0.0,
            "max": float(vals.max()) if vals.size else 0.0,
        }
    return out


def select_scales(scales: dict[str, dict[str, float]], mode: str) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    identity_modes = {"identity", "identity_processor"}
    for name, value in scales.items():
        selected[name] = {}
        if mode in identity_modes:
            selected[name]["alpha"] = 1.0
            selected[name]["beta"] = 1.0
            continue
        if "atm" in mode:
            selected[name]["alpha"] = value["alpha"]
        if "ohb" in mode:
            selected[name]["beta"] = value["beta"]
    return selected


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# QuantVLA Phase 3 ATM/OHB Forward",
        "",
        "Boundary: synthetic-input W4A8 fake quantization through the official GR00T policy path. ATM/OHB scales are calibrated from teacher/student attention statistics on the same synthetic calibration set.",
        "",
        "## Run",
        "",
        f"- Model: `{result['model_path']}`",
        f"- Configs: `{', '.join(result['configs'])}`",
        f"- Synthetic variants: `{', '.join(result['synthetic_variants'])}`",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Activation scale: `{result['activation_scale']}`",
        f"- Smoothing alpha: `{result['smoothing']}`",
        f"- ATM/OHB log clamp: `{result['log_clamp']}`",
        "",
        "## Output Drift",
        "",
        "| config | mode | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["config_results"]:
        s = row["summary"]
        lines.append(
            "| {config} | {mode} | {modules} | {nmse:.6g} | {nmse_max:.6g} | {rrmse:.6g} | {cos:.6g} | {mad:.6g} |".format(
                config=row["config"],
                mode=row["mode"],
                modules=row["quantized_modules"],
                nmse=s["nmse"]["mean"],
                nmse_max=s["nmse"]["max"],
                rrmse=s["relative_rmse"]["mean"],
                cos=s["cosine"]["mean"],
                mad=s["max_abs_diff"]["max"],
            )
        )
    lines.extend(
        [
            "",
            "## Scale Summary",
            "",
            "| config | alpha mean | alpha min | alpha max | beta mean | beta min | beta max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["config_results"]:
        if row["mode"] != "none":
            continue
        sc = row["scale_summary"]
        lines.append(
            f"| {row['config']} | {sc['alpha']['mean']:.6g} | {sc['alpha']['min']:.6g} | {sc['alpha']['max']:.6g} | {sc['beta']['mean']:.6g} | {sc['beta']['min']:.6g} | {sc['beta']['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ATM is applied by multiplying the attention query, which scales attention logits by `alpha = std_teacher / std_student`.",
            "- OHB is applied by multiplying the DiT attention output before residual addition, with `beta = rms_teacher / rms_student`.",
            "- Scales are log-clamped by default to avoid a calibration artifact dominating the synthetic probe.",
            "- This is still not a LIBERO benchmark; it tests whether ATM/OHB reduce teacher/student action drift under controlled synthetic inputs.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--num-observations", type=int, default=3)
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--configs", default="llm_only,dit_mlp_only,llm_dit_mlp")
    parser.add_argument("--activation-scale", default="absmax")
    parser.add_argument("--activation-percentile", type=float, default=0.999)
    parser.add_argument("--smoothing-alpha", default="none")
    parser.add_argument("--log-clamp", type=float, default=0.3)
    parser.add_argument("--base-seed", type=int, default=260203)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase3_atm_ohb_forward.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase3_atm_ohb_forward.md"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    configs = [v.strip() for v in args.configs.split(",") if v.strip()]
    smoothing_alpha = None if args.smoothing_alpha.lower() in {"none", "off", "naive"} else float(args.smoothing_alpha)

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

    observations = [
        {
            "variant": variants[idx % len(variants)],
            "seed": args.base_seed + idx,
            "obs": build_variant_observation(
                args.model_path, args.embodiment_tag, variants[idx % len(variants)], idx
            ),
        }
        for idx in range(args.num_observations)
    ]

    teacher_actions = []
    for item in observations:
        set_seed(int(item["seed"]))
        teacher_actions.append(policy.get_action(item["obs"]))

    teacher_stats = collect_attention_stats(policy, observations)

    smoothing_groups = set().union(*(config_groups(config) for config in configs))
    smoothing_groups.discard("dit_attention_excluded")
    smooth_scales = None
    if smoothing_alpha is not None:
        act_max = collect_activation_max(policy, observations, smoothing_groups)
        smooth_scales = make_smooth_scales(policy.model, act_max, smoothing_alpha)

    config_results = []
    for config in configs:
        groups = config_groups(config)
        originals, records = patch_modules(
            policy.model,
            groups,
            weight_bits=4,
            activation_bits=8,
            activation_scale=args.activation_scale,
            activation_percentile=args.activation_percentile,
            smooth_scales=smooth_scales,
            smooth_alpha=smoothing_alpha,
        )
        try:
            student_stats = collect_attention_stats(policy, observations)
            scales = compute_atm_ohb_scales(
                teacher_stats, student_stats, log_clamp=args.log_clamp
            )
            for mode in ["none", "atm", "ohb", "atm_ohb"]:
                attn_originals = {}
                if mode != "none":
                    attn_originals, _ = install_attention_processors(
                        policy.model, mode="apply", scales=select_scales(scales, mode)
                    )
                rows = []
                try:
                    for idx, item in enumerate(observations):
                        set_seed(int(item["seed"]))
                        student = policy.get_action(item["obs"])
                        metrics = compare_actions(teacher_actions[idx], student)
                        metrics.update({"variant": item["variant"], "seed": item["seed"]})
                        rows.append(metrics)
                finally:
                    if attn_originals:
                        restore_attention_processors(policy.model, attn_originals)
                config_results.append(
                    {
                        "config": config,
                        "mode": mode,
                        "groups": sorted(groups),
                        "quantized_modules": len(originals),
                        "summary": aggregate_metrics(rows),
                        "per_observation": rows,
                        "scale_summary": scale_summary(scales),
                        "scales": scales,
                    }
                )
        finally:
            restore_modules(policy.model, originals)

    result = {
        "model_path": str(args.model_path),
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "synthetic_variants": [item["variant"] for item in observations],
        "configs": configs,
        "activation_scale": args.activation_scale,
        "activation_percentile": args.activation_percentile,
        "smoothing": "none" if smoothing_alpha is None else f"sq_alpha_{smoothing_alpha:g}",
        "log_clamp": args.log_clamp,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "load_seconds": load_seconds,
        "teacher_attention_stats": {k: v.to_result() for k, v in teacher_stats.items()},
        "config_results": config_results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "config_results": [
            {
                "config": row["config"],
                "mode": row["mode"],
                "nmse_mean": row["summary"]["nmse"]["mean"],
                "relative_rmse_mean": row["summary"]["relative_rmse"]["mean"],
                "cosine_mean": row["summary"]["cosine"]["mean"],
            }
            for row in config_results
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
