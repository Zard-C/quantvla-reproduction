from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from distributions import (
    WEIGHT_STATS_SCHEMA,
    apply_student_drift,
    preset_summary,
    sample_activation,
    sample_weight,
    summarize_tensor,
    summarize_weight,
)
from quant import (
    cosine_similarity,
    entropy,
    fake_quant_linear,
    js_divergence,
    mse,
    nmse,
    relative_rms_error,
    rms,
    rms_error,
    smooth_linear_pair,
)
from toy_attention import (
    attention_forward,
    calibrate_alpha,
    calibrate_beta,
    compare_attention_outputs,
    make_attention_weights,
    make_transformer_weights,
    per_head_logits_std,
    transformer_block_forward,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 20260604


def make_generator(device: torch.device, offset: int = 0) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED + offset)
    return gen


def to_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_float(v) for v in value]
    if isinstance(value, tuple):
        return [to_float(v) for v in value]
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def linear_quantization_experiment(device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [
        ("standard_normal", "standard_normal"),
        ("vla_like_dit_mlp", "dit_mlp"),
    ]
    for idx, (label, preset) in enumerate(configs):
        gen = make_generator(device, 10 + idx)
        x = sample_activation(16, 32, 128, preset, device=device, generator=gen)
        weight = sample_weight(192, 128, preset, device=device, generator=gen)
        if preset != "standard_normal":
            stress_channels = torch.tensor([3, 19, 77], device=device)
            x[..., stress_channels] *= 20.0
            weight[:, stress_channels] /= 20.0**0.5
        bias = torch.zeros(192, device=device)
        teacher = F.linear(x, weight, bias)

        naive, naive_stats = fake_quant_linear(x, weight, bias)
        x_s, w_s, smooth_scale = smooth_linear_pair(x, weight, migration_strength=0.15)
        smoothed, smooth_stats = fake_quant_linear(x_s, w_s, bias)

        for method, output, stats in [
            ("naive_w4a8", naive, naive_stats),
            ("smoothed_w4a8", smoothed, smooth_stats),
        ]:
            rows.append(
                {
                    "distribution": label,
                    "method": method,
                    "output_mse": mse(teacher, output),
                    "output_nmse": nmse(teacher, output),
                    "output_rms": float(rms(output).item()),
                    "teacher_rms": float(rms(teacher).item()),
                    "error_rms": rms_error(teacher, output),
                    "relative_rms_error": relative_rms_error(teacher, output),
                    "cosine": cosine_similarity(teacher, output),
                    "activation_saturation_ratio": stats["activation"].saturation_ratio,
                    "weight_scale_max": stats["weight"].scale_max,
                    "smooth_scale_max": float(smooth_scale.detach().float().amax().item())
                    if method == "smoothed_w4a8"
                    else 1.0,
                }
            )
    return rows


def selective_quantization_experiment(device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs = [
        ("standard_normal", "standard_normal", "standard_normal", None),
        ("vla_like", "dit_attn", "dit_mlp", "dit_attn"),
    ]
    variants = [
        ("mlp_only", False, True, None),
        ("attention_only", True, False, None),
        ("attention_and_mlp", True, True, None),
        ("upstream_drift_fp_attention", False, False, "use_config_drift"),
    ]

    for cfg_idx, (label, attn_preset, mlp_preset, drift_preset) in enumerate(configs):
        gen = make_generator(device, 100 + cfg_idx)
        x = sample_activation(8, 24, 128, attn_preset, device=device, generator=gen)
        weights = make_transformer_weights(
            128,
            256,
            attn_preset,
            mlp_preset,
            device=device,
            generator=gen,
        )
        teacher_out, teacher_attn = transformer_block_forward(x, weights, heads=4)
        for name, q_attn, q_mlp, drift_mode in variants:
            student_drift = drift_preset if drift_mode == "use_config_drift" else None
            student_out, student_attn = transformer_block_forward(
                x,
                weights,
                heads=4,
                quantize_attention=q_attn,
                quantize_mlp=q_mlp,
                upstream_drift_preset=student_drift,
                generator=gen,
            )
            metrics = compare_attention_outputs(teacher_attn, student_attn)
            rows.append(
                {
                    "distribution": label,
                    "variant": name,
                    "quantize_attention": q_attn,
                    "quantize_mlp": q_mlp,
                    "final_output_mse": mse(teacher_out, student_out),
                    "final_output_nmse": nmse(teacher_out, student_out),
                    "final_output_cosine": cosine_similarity(teacher_out, student_out),
                    "teacher_final_rms": float(rms(teacher_out).item()),
                    "student_final_rms": float(rms(student_out).item()),
                    "final_relative_rms_error": relative_rms_error(teacher_out, student_out),
                    **metrics,
                }
            )
    return rows


def atm_direction_experiment(device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (label, preset) in enumerate([("standard_normal", "standard_normal"), ("vla_like", "dit_attn")]):
        gen = make_generator(device, 200 + idx)
        x = sample_activation(10, 24, 128, preset, device=device, generator=gen)
        weights = make_attention_weights(128, preset, device=device, generator=gen)
        teacher = attention_forward(x, weights, heads=4)
        x_student = apply_student_drift(x, preset, generator=gen)
        student = attention_forward(x_student, weights, heads=4, quantize_attention=False)
        alpha = calibrate_alpha(teacher.logits, student.logits)

        for direction in ["none", "multiply", "divide"]:
            if direction == "none":
                calibrated = student.logits
            elif direction == "multiply":
                calibrated = student.logits * alpha.reshape(1, -1, 1, 1).to(device)
            else:
                calibrated = student.logits / alpha.reshape(1, -1, 1, 1).to(device).clamp_min(1e-6)
            teacher_probs = torch.softmax(teacher.logits, dim=-1)
            probs = torch.softmax(calibrated, dim=-1)
            std_t = per_head_logits_std(teacher.logits)
            std_s = per_head_logits_std(calibrated)
            logits_std_mae = float(torch.mean(torch.abs(std_t - std_s)).item())
            rows.append(
                {
                    "distribution": label,
                    "direction": direction,
                    "alpha_mean": float(alpha.mean().item()),
                    "alpha_min": float(alpha.min().item()),
                    "alpha_max": float(alpha.max().item()),
                    "teacher_logits_std_mean": float(std_t.mean().item()),
                    "logits_std_mae": logits_std_mae,
                    "logits_std_relative_mae": logits_std_mae / max(float(std_t.mean().item()), 1e-8),
                    "attention_js": js_divergence(teacher_probs, probs),
                    "entropy_abs_error": float(
                        torch.abs(entropy(teacher_probs).mean() - entropy(probs).mean()).item()
                    ),
                }
            )
    return rows


def ohb_direction_experiment(device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (label, preset) in enumerate([("standard_normal", "standard_normal"), ("vla_like", "dit_attn")]):
        gen = make_generator(device, 300 + idx)
        x = sample_activation(10, 24, 128, preset, device=device, generator=gen)
        weights = make_attention_weights(128, preset, device=device, generator=gen)
        teacher = attention_forward(x, weights, heads=4)
        x_student = apply_student_drift(x, preset, generator=gen)
        student = attention_forward(x_student, weights, heads=4, quantize_attention=False)
        beta = calibrate_beta(teacher.post_o, student.post_o)

        for direction in ["none", "multiply", "divide"]:
            if direction == "none":
                calibrated = student.post_o
            elif direction == "multiply":
                calibrated = student.post_o * beta.to(device)
            else:
                calibrated = student.post_o / beta.to(device).clamp_min(1e-6)
            teacher_rms = float(rms(teacher.post_o).item())
            student_rms = float(rms(calibrated).item())
            rms_abs_error = abs(teacher_rms - student_rms)
            rows.append(
                {
                    "distribution": label,
                    "direction": direction,
                    "beta": float(beta.item()),
                    "teacher_rms": teacher_rms,
                    "student_rms": student_rms,
                    "rms_abs_error": rms_abs_error,
                    "rms_relative_error": rms_abs_error / max(teacher_rms, 1e-8),
                    "post_o_mse": mse(teacher.post_o, calibrated),
                    "post_o_nmse": nmse(teacher.post_o, calibrated),
                    "post_o_relative_rms_error": relative_rms_error(teacher.post_o, calibrated),
                }
            )
    return rows


def calibration_noise_experiment(device: torch.device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_counts = [4, 8, 32, 128]
    for idx, preset in enumerate(["standard_normal", "dit_attn"]):
        label = "standard_normal" if preset == "standard_normal" else "vla_like"
        gen = make_generator(device, 400 + idx)
        x = sample_activation(128, 16, 128, preset, device=device, generator=gen)
        x_student = apply_student_drift(x, preset, generator=gen)
        weights = make_attention_weights(128, preset, device=device, generator=gen)
        teacher_all = attention_forward(x, weights, heads=4)
        student_all = attention_forward(x_student, weights, heads=4)

        eval_teacher = teacher_all
        eval_student = student_all
        base_std_error = float(
            torch.mean(torch.abs(per_head_logits_std(eval_teacher.logits) - per_head_logits_std(eval_student.logits))).item()
        )
        teacher_std_mean = float(per_head_logits_std(eval_teacher.logits).mean().item())
        base_rms_error = abs(float(rms(eval_teacher.post_o).item()) - float(rms(eval_student.post_o).item()))
        teacher_post_o_rms = float(rms(eval_teacher.post_o).item())

        for count in sample_counts:
            teacher_cal = attention_forward(x[:count], weights, heads=4)
            student_cal = attention_forward(x_student[:count], weights, heads=4)
            alpha = calibrate_alpha(teacher_cal.logits, student_cal.logits)
            beta = calibrate_beta(teacher_cal.post_o, student_cal.post_o)

            eval_logits = eval_student.logits * alpha.reshape(1, -1, 1, 1).to(device)
            eval_post_o = eval_student.post_o * beta.to(device)
            alpha_log = torch.log(alpha)
            calibrated_std_error = float(
                torch.mean(
                    torch.abs(per_head_logits_std(eval_teacher.logits) - per_head_logits_std(eval_logits))
                ).item()
            )
            calibrated_rms_error = abs(float(rms(eval_teacher.post_o).item()) - float(rms(eval_post_o).item()))
            rows.append(
                {
                    "distribution": label,
                    "samples": count,
                    "alpha_mean": float(alpha.mean().item()),
                    "alpha_std": float(alpha.std(unbiased=False).item()),
                    "alpha_neutral_heads": int((alpha == 1.0).sum().item()),
                    "alpha_clamp_hits": int((torch.abs(alpha_log) >= 0.299).sum().item()),
                    "beta": float(beta.item()),
                    "beta_is_neutral": bool(beta.item() == 1.0),
                    "base_logits_std_mae": base_std_error,
                    "base_logits_std_relative_mae": base_std_error / max(teacher_std_mean, 1e-8),
                    "calibrated_logits_std_mae": calibrated_std_error,
                    "calibrated_logits_std_relative_mae": calibrated_std_error / max(teacher_std_mean, 1e-8),
                    "base_post_o_rms_abs_error": base_rms_error,
                    "base_post_o_rms_relative_error": base_rms_error / max(teacher_post_o_rms, 1e-8),
                    "calibrated_post_o_rms_abs_error": calibrated_rms_error,
                    "calibrated_post_o_rms_relative_error": calibrated_rms_error / max(teacher_post_o_rms, 1e-8),
                }
            )
    return rows


def distribution_sensitivity_rows(
    selective_rows: list[dict[str, Any]],
    atm_rows: list[dict[str, Any]],
    ohb_rows: list[dict[str, Any]],
    linear_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for distribution in ["standard_normal", "vla_like"]:
        sel = [r for r in selective_rows if r["distribution"] == distribution]
        if sel:
            mlp = next(r for r in sel if r["variant"] == "mlp_only")
            attn = next(r for r in sel if r["variant"] == "attention_only")
            both = next(r for r in sel if r["variant"] == "attention_and_mlp")
            rows.append(
                {
                    "distribution": distribution,
                    "metric_group": "selective_quantization",
                    "mlp_only_final_output_mse": mlp["final_output_mse"],
                    "mlp_only_final_output_nmse": mlp["final_output_nmse"],
                    "attention_only_final_output_mse": attn["final_output_mse"],
                    "attention_only_final_output_nmse": attn["final_output_nmse"],
                    "both_final_output_mse": both["final_output_mse"],
                    "both_final_output_nmse": both["final_output_nmse"],
                    "attention_vs_mlp_mse_ratio": attn["final_output_mse"] / max(mlp["final_output_mse"], 1e-12),
                    "attention_vs_mlp_nmse_ratio": attn["final_output_nmse"] / max(mlp["final_output_nmse"], 1e-12),
                    "attention_js_attention_only": attn["attention_js"],
                }
            )

        atm = [r for r in atm_rows if r["distribution"] == distribution]
        if atm:
            none = next(r for r in atm if r["direction"] == "none")
            mult = next(r for r in atm if r["direction"] == "multiply")
            div = next(r for r in atm if r["direction"] == "divide")
            rows.append(
                {
                    "distribution": distribution,
                    "metric_group": "atm_direction",
                    "none_logits_std_mae": none["logits_std_mae"],
                    "none_logits_std_relative_mae": none["logits_std_relative_mae"],
                    "multiply_logits_std_mae": mult["logits_std_mae"],
                    "multiply_logits_std_relative_mae": mult["logits_std_relative_mae"],
                    "divide_logits_std_mae": div["logits_std_mae"],
                    "divide_logits_std_relative_mae": div["logits_std_relative_mae"],
                    "multiply_improvement": none["logits_std_mae"] - mult["logits_std_mae"],
                    "divide_delta": div["logits_std_mae"] - none["logits_std_mae"],
                }
            )

        ohb = [r for r in ohb_rows if r["distribution"] == distribution]
        if ohb:
            none = next(r for r in ohb if r["direction"] == "none")
            mult = next(r for r in ohb if r["direction"] == "multiply")
            div = next(r for r in ohb if r["direction"] == "divide")
            rows.append(
                {
                    "distribution": distribution,
                    "metric_group": "ohb_direction",
                    "none_rms_abs_error": none["rms_abs_error"],
                    "none_rms_relative_error": none["rms_relative_error"],
                    "multiply_rms_abs_error": mult["rms_abs_error"],
                    "multiply_rms_relative_error": mult["rms_relative_error"],
                    "divide_rms_abs_error": div["rms_abs_error"],
                    "divide_rms_relative_error": div["rms_relative_error"],
                    "multiply_improvement": none["rms_abs_error"] - mult["rms_abs_error"],
                    "divide_delta": div["rms_abs_error"] - none["rms_abs_error"],
                }
            )

    for label, distribution in [("standard_normal", "standard_normal"), ("vla_like_dit_mlp", "vla_like")]:
        dist_rows = [r for r in linear_rows if r["distribution"] == label]
        if dist_rows:
            naive = next(r for r in dist_rows if r["method"] == "naive_w4a8")
            smooth = next(r for r in dist_rows if r["method"] == "smoothed_w4a8")
            rows.append(
                {
                    "distribution": distribution,
                    "metric_group": "linear_smoothing",
                    "naive_output_mse": naive["output_mse"],
                    "naive_output_nmse": naive["output_nmse"],
                    "smoothed_output_mse": smooth["output_mse"],
                    "smoothed_output_nmse": smooth["output_nmse"],
                    "smoothing_improvement": naive["output_mse"] - smooth["output_mse"],
                    "smoothing_nmse_improvement": naive["output_nmse"] - smooth["output_nmse"],
                    "naive_saturation_ratio": naive["activation_saturation_ratio"],
                    "smoothed_saturation_ratio": smooth["activation_saturation_ratio"],
                }
            )
    return rows


def best_direction(rows: list[dict[str, Any]], distribution: str, metric: str) -> str:
    candidates = [r for r in rows if r["distribution"] == distribution and r["direction"] in {"multiply", "divide"}]
    return min(candidates, key=lambda row: row[metric])["direction"]


def make_markdown(summary: dict[str, Any]) -> str:
    def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
        out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in rows:
            vals = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    vals.append(f"{value:.6g}")
                else:
                    vals.append(str(value))
            out.append("|" + "|".join(vals) + "|")
        return "\n".join(out)

    linear = summary["experiments"]["linear_quantization"]
    selective = summary["experiments"]["selective_quantization"]
    atm = summary["experiments"]["atm_direction"]
    ohb = summary["experiments"]["ohb_direction"]
    noise = summary["experiments"]["calibration_noise"]
    sensitivity = summary["experiments"]["distribution_sensitivity"]
    conclusions = summary["conclusions"]
    sensitivity_columns: list[str] = []
    for row in sensitivity:
        for key in row:
            if key not in sensitivity_columns:
                sensitivity_columns.append(key)

    lines = [
        "# QuantVLA Toy Experiment Summary",
        "",
        "## Environment",
        "",
        table([summary["environment"]], ["command", "python", "torch", "cuda", "cuda_available", "device"]),
        "",
        "## Conclusions",
        "",
    ]
    for key, value in conclusions.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## W4A8 Linear Smoothing",
            "",
            table(
                linear,
                [
                    "distribution",
                    "method",
                    "output_mse",
                    "output_nmse",
                    "cosine",
                    "relative_rms_error",
                    "activation_saturation_ratio",
                    "weight_scale_max",
                ],
            ),
            "",
            "## Selective Quantization",
            "",
            table(
                selective,
                [
                    "distribution",
                    "variant",
                    "final_output_mse",
                    "final_output_nmse",
                    "final_output_cosine",
                    "logits_std_abs_error",
                    "attention_js",
                    "post_o_rms_abs_error",
                    "post_o_rms_relative_error",
                ],
            ),
            "",
            "## ATM Direction",
            "",
            table(
                atm,
                [
                    "distribution",
                    "direction",
                    "alpha_mean",
                    "logits_std_mae",
                    "logits_std_relative_mae",
                    "attention_js",
                    "entropy_abs_error",
                ],
            ),
            "",
            "## OHB Direction",
            "",
            table(
                ohb,
                [
                    "distribution",
                    "direction",
                    "beta",
                    "teacher_rms",
                    "student_rms",
                    "rms_abs_error",
                    "rms_relative_error",
                    "post_o_mse",
                    "post_o_nmse",
                ],
            ),
            "",
            "## Calibration Noise",
            "",
            table(
                noise,
                [
                    "distribution",
                    "samples",
                    "alpha_mean",
                    "alpha_std",
                    "alpha_neutral_heads",
                    "alpha_clamp_hits",
                    "beta",
                    "calibrated_logits_std_mae",
                    "calibrated_logits_std_relative_mae",
                    "calibrated_post_o_rms_abs_error",
                    "calibrated_post_o_rms_relative_error",
                ],
            ),
            "",
            "## Distribution Sensitivity",
            "",
            table(sensitivity, sensitivity_columns),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    linear_rows = linear_quantization_experiment(device)
    selective_rows = selective_quantization_experiment(device)
    atm_rows = atm_direction_experiment(device)
    ohb_rows = ohb_direction_experiment(device)
    noise_rows = calibration_noise_experiment(device)
    sensitivity = distribution_sensitivity_rows(selective_rows, atm_rows, ohb_rows, linear_rows)

    atm_multiply_ok = all(
        best_direction(atm_rows, distribution, "logits_std_relative_mae") == "multiply"
        for distribution in ["standard_normal", "vla_like"]
    )
    ohb_multiply_ok = all(
        best_direction(ohb_rows, distribution, "rms_relative_error") == "multiply"
        for distribution in ["standard_normal", "vla_like"]
    )
    vla_selective = [r for r in selective_rows if r["distribution"] == "vla_like"]
    vla_mlp = next(r for r in vla_selective if r["variant"] == "mlp_only")
    vla_attn = next(r for r in vla_selective if r["variant"] == "attention_only")
    empirical_stats_available = (RESULTS / "weight_stats.json").exists()

    standard_linear = [r for r in linear_rows if r["distribution"] == "standard_normal"]
    vla_linear = [r for r in linear_rows if r["distribution"] == "vla_like_dit_mlp"]
    standard_smoothing_gain = next(r for r in standard_linear if r["method"] == "naive_w4a8")["output_mse"] - next(
        r for r in standard_linear if r["method"] == "smoothed_w4a8"
    )["output_mse"]
    vla_smoothing_gain = next(r for r in vla_linear if r["method"] == "naive_w4a8")["output_mse"] - next(
        r for r in vla_linear if r["method"] == "smoothed_w4a8"
    )["output_mse"]
    standard_smoothing_nmse_gain = next(r for r in standard_linear if r["method"] == "naive_w4a8")[
        "output_nmse"
    ] - next(r for r in standard_linear if r["method"] == "smoothed_w4a8")["output_nmse"]
    vla_smoothing_nmse_gain = next(r for r in vla_linear if r["method"] == "naive_w4a8")["output_nmse"] - next(
        r for r in vla_linear if r["method"] == "smoothed_w4a8"
    )["output_nmse"]
    normalized_metric_gate = (
        atm_multiply_ok
        and ohb_multiply_ok
        and vla_attn["final_output_nmse"] > vla_mlp["final_output_nmse"]
        and vla_smoothing_nmse_gain > standard_smoothing_nmse_gain
    )

    env = {
        "command": "python toy_quantvla/run_toy_experiments.py",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "seed": SEED,
    }

    synthetic_weight_stats = {
        "standard_normal": summarize_weight(
            sample_weight(128, 128, "standard_normal", device=device, generator=make_generator(device, 900))
        ),
        "vla_like_dit_mlp": summarize_weight(
            sample_weight(128, 128, "dit_mlp", device=device, generator=make_generator(device, 901))
        ),
    }
    synthetic_activation_stats = {
        "standard_normal": summarize_tensor(
            sample_activation(8, 16, 128, "standard_normal", device=device, generator=make_generator(device, 902))
        ),
        "vla_like_dit_attn": summarize_tensor(
            sample_activation(8, 16, 128, "dit_attn", device=device, generator=make_generator(device, 903))
        ),
    }

    summary = {
        "environment": env,
        "distribution_presets": preset_summary(),
        "synthetic_stats": {
            "weights": synthetic_weight_stats,
            "activations": synthetic_activation_stats,
        },
        "experiments": {
            "linear_quantization": linear_rows,
            "selective_quantization": selective_rows,
            "atm_direction": atm_rows,
            "ohb_direction": ohb_rows,
            "calibration_noise": noise_rows,
            "distribution_sensitivity": sensitivity,
        },
        "conclusions": {
            "atm_multiply_direction_wins": atm_multiply_ok,
            "ohb_multiply_direction_wins": ohb_multiply_ok,
            "vla_attention_quantization_more_fragile_than_mlp": vla_attn["final_output_mse"] > vla_mlp["final_output_mse"],
            "vla_attention_quantization_more_fragile_than_mlp_nmse": vla_attn["final_output_nmse"] > vla_mlp["final_output_nmse"],
            "smoothing_gain_larger_under_vla_like": vla_smoothing_gain > standard_smoothing_gain,
            "smoothing_nmse_gain_larger_under_vla_like": vla_smoothing_nmse_gain > standard_smoothing_nmse_gain,
            "empirical_weight_stats_available": empirical_stats_available,
            "normalized_metric_gate": normalized_metric_gate,
            "phase3_ready": empirical_stats_available and normalized_metric_gate,
        },
    }

    write_csv(RESULTS / "atm_direction.csv", atm_rows)
    write_csv(RESULTS / "ohb_direction.csv", ohb_rows)
    write_csv(RESULTS / "distribution_sensitivity.csv", sensitivity)
    write_csv(RESULTS / "linear_quantization.csv", linear_rows)
    write_csv(RESULTS / "selective_quantization.csv", selective_rows)
    write_csv(RESULTS / "calibration_noise.csv", noise_rows)
    (RESULTS / "weight_stats_schema.json").write_text(
        json.dumps(WEIGHT_STATS_SCHEMA, indent=2),
        encoding="utf-8",
    )
    (RESULTS / "summary.json").write_text(
        json.dumps(to_float(summary), indent=2),
        encoding="utf-8",
    )
    (RESULTS / "summary.md").write_text(make_markdown(to_float(summary)), encoding="utf-8")

    print(f"wrote {RESULTS / 'summary.md'}")
    print(json.dumps(to_float(summary["conclusions"]), indent=2))


if __name__ == "__main__":
    main()
