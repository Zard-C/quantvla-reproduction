"""Teacher/student fake-quant forward comparison for GR00T LIBERO.

This script patches selected nn.Linear modules with a W4A8 fake-quant wrapper,
runs the official Gr00tPolicy.get_action path, and compares action outputs
against the unpatched teacher under matched RNG seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phase3_activation_capture import build_variant_observation, group_for_module
from phase3_gr00t_smoke import _insert_paths


EPS = 1e-8


def signed_qmax(bits: int) -> int:
    if bits < 2:
        raise ValueError("signed quantization requires at least 2 bits")
    return (1 << (bits - 1)) - 1


def fake_quant_symmetric(
    x: Any,
    bits: int,
    *,
    scale_dim: int | None = None,
    activation_scale: str = "absmax",
    percentile: float = 0.999,
) -> tuple[Any, dict[str, float]]:
    import torch

    qmax = float(signed_qmax(bits))
    xf = x.detach().float()
    if scale_dim is not None:
        clip_abs = xf.abs().amax(dim=scale_dim, keepdim=True).clamp_min(EPS)
    elif activation_scale == "absmax":
        clip_abs = xf.abs().amax().clamp_min(EPS)
    elif activation_scale == "p999":
        clip_abs = torch.quantile(xf.abs().reshape(-1), percentile).clamp_min(EPS)
    else:
        raise ValueError(f"Unknown activation scale: {activation_scale}")
    q = torch.clamp(
        torch.round(torch.clamp(xf, -clip_abs, clip_abs) / (clip_abs / qmax)), -qmax, qmax
    )
    y = q * (clip_abs / qmax)
    clip_frac = (xf.abs() > clip_abs).float().mean()
    return y.to(dtype=x.dtype), {
        "clip_abs": float(torch.as_tensor(clip_abs).float().amax().item()),
        "clip_frac": float(clip_frac.item()),
    }


@dataclass
class QuantRecord:
    module: str
    group: str
    calls: int = 0
    activation_clip_abs: list[float] = field(default_factory=list)
    activation_clip_frac: list[float] = field(default_factory=list)

    def add_activation(self, stats: dict[str, float]) -> None:
        self.calls += 1
        self.activation_clip_abs.append(stats["clip_abs"])
        self.activation_clip_frac.append(stats["clip_frac"])

    def to_result(self) -> dict[str, Any]:
        def summarize(vals: list[float]) -> dict[str, float]:
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                return {"mean": 0.0, "max": 0.0}
            return {"mean": float(arr.mean()), "max": float(arr.max())}

        return {
            "module": self.module,
            "group": self.group,
            "calls": self.calls,
            "activation_clip_abs": summarize(self.activation_clip_abs),
            "activation_clip_frac": summarize(self.activation_clip_frac),
        }


class FakeQuantLinear:
    """Small wrapper that preserves the original nn.Linear module as `base`."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch.nn as nn

        class _FakeQuantLinear(nn.Module):
            def __init__(
                self,
                base: nn.Linear,
                name: str,
                group: str,
                record: QuantRecord,
                weight_bits: int,
                activation_bits: int,
                activation_scale: str,
                activation_percentile: float,
                smooth_scale: Any | None,
                smooth_alpha: float | None,
            ):
                super().__init__()
                self.base = base
                self.name = name
                self.group = group
                self.record = record
                self.weight_bits = weight_bits
                self.activation_bits = activation_bits
                self.activation_scale = activation_scale
                self.activation_percentile = activation_percentile
                self.smooth_scale = smooth_scale
                self.smooth_alpha = smooth_alpha
                self.in_features = base.in_features
                self.out_features = base.out_features

            @property
            def weight(self) -> Any:
                return self.base.weight

            @property
            def bias(self) -> Any:
                return self.base.bias

            def forward(self, x: Any) -> Any:
                import torch.nn.functional as F

                weight = self.base.weight
                if self.smooth_scale is not None:
                    scale = self.smooth_scale.to(device=x.device, dtype=x.dtype).clamp_min(EPS)
                    x = x / scale
                    weight = weight * scale.to(dtype=weight.dtype).unsqueeze(0)

                x_q, act_stats = fake_quant_symmetric(
                    x,
                    self.activation_bits,
                    activation_scale=self.activation_scale,
                    percentile=self.activation_percentile,
                )
                w_q, _ = fake_quant_symmetric(weight, self.weight_bits, scale_dim=1)
                self.record.add_activation(act_stats)
                return F.linear(x_q, w_q.to(dtype=self.base.weight.dtype), self.base.bias)

            def extra_repr(self) -> str:
                smooth = "none" if self.smooth_alpha is None else f"{self.smooth_alpha:.3g}"
                return (
                    f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
                    f"group={self.group}, W{self.weight_bits}A{self.activation_bits}, smooth={smooth}"
                )

        return _FakeQuantLinear(*args, **kwargs)


def get_submodule(root: Any, name: str) -> Any:
    current = root
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def set_submodule(root: Any, name: str, module: Any) -> None:
    parts = name.split(".")
    parent = get_submodule(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def patch_modules(
    model: Any,
    groups_to_quantize: set[str],
    *,
    weight_bits: int,
    activation_bits: int,
    activation_scale: str,
    activation_percentile: float,
    smooth_scales: dict[str, Any] | None = None,
    smooth_alpha: float | None = None,
) -> tuple[dict[str, Any], dict[str, QuantRecord]]:
    import torch.nn as nn

    originals: dict[str, Any] = {}
    records: dict[str, QuantRecord] = {}
    for name, module in list(model.named_modules()):
        group = group_for_module(name)
        if group is None or group not in groups_to_quantize or not isinstance(module, nn.Linear):
            continue
        record = QuantRecord(module=name, group=group)
        originals[name] = module
        records[name] = record
        set_submodule(
            model,
            name,
            FakeQuantLinear(
                module,
                name=name,
                group=group,
                record=record,
                weight_bits=weight_bits,
                activation_bits=activation_bits,
                activation_scale=activation_scale,
                activation_percentile=activation_percentile,
                smooth_scale=smooth_scales.get(name) if smooth_scales else None,
                smooth_alpha=smooth_alpha,
            ),
        )
    return originals, records


def restore_modules(model: Any, originals: dict[str, Any]) -> None:
    for name, module in originals.items():
        set_submodule(model, name, module)


def collect_activation_max(
    policy: Any,
    observations: list[dict[str, Any]],
    groups_to_collect: set[str],
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    act_max: dict[str, Any] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_floating_point(inputs[0]):
                return
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).abs().amax(dim=0).cpu()
            if name in act_max:
                act_max[name] = torch.maximum(act_max[name], x)
            else:
                act_max[name] = x

        return hook

    for name, module in policy.model.named_modules():
        group = group_for_module(name)
        if group is None or group not in groups_to_collect or not isinstance(module, nn.Linear):
            continue
        handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for item in observations:
            set_seed(int(item["seed"]))
            policy.get_action(item["obs"])
    finally:
        for handle in handles:
            handle.remove()
    return act_max


def make_smooth_scales(model: Any, act_max: dict[str, Any], alpha: float) -> dict[str, Any]:
    import torch

    scales: dict[str, Any] = {}
    for name, activation_max in act_max.items():
        module = get_submodule(model, name)
        weight_max = module.weight.detach().float().abs().amax(dim=0).cpu().clamp_min(EPS)
        activation_max = activation_max.float().clamp_min(EPS)
        scale = activation_max.pow(alpha) / weight_max.pow(1.0 - alpha)
        scales[name] = scale.clamp_min(EPS)
    return scales


def action_to_vector(action: dict[str, Any]) -> np.ndarray:
    keys = sorted(action)
    arrays = [np.asarray(action[k], dtype=np.float64).reshape(-1) for k in keys]
    return np.concatenate(arrays, axis=0)


def compare_actions(teacher: dict[str, Any], student: dict[str, Any]) -> dict[str, Any]:
    t = action_to_vector(teacher)
    s = action_to_vector(student)
    diff = s - t
    mse = float(np.mean(diff**2))
    ref = float(np.mean(t**2) + EPS)
    dot = float(np.dot(t, s))
    denom = float(np.linalg.norm(t) * np.linalg.norm(s) + EPS)
    out = {
        "mse": mse,
        "nmse": mse / ref,
        "rmse": float(np.sqrt(mse)),
        "relative_rmse": float(np.sqrt(mse) / np.sqrt(ref)),
        "cosine": dot / denom,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "teacher_rms": float(np.sqrt(ref)),
        "student_rms": float(np.sqrt(float(np.mean(s**2) + EPS))),
    }
    per_key: dict[str, Any] = {}
    for key in sorted(teacher):
        tk = np.asarray(teacher[key], dtype=np.float64).reshape(-1)
        sk = np.asarray(student[key], dtype=np.float64).reshape(-1)
        dk = sk - tk
        mse_k = float(np.mean(dk**2))
        ref_k = float(np.mean(tk**2) + EPS)
        per_key[key] = {
            "mse": mse_k,
            "nmse": mse_k / ref_k,
            "max_abs_diff": float(np.max(np.abs(dk))),
            "teacher_rms": float(np.sqrt(ref_k)),
            "student_rms": float(np.sqrt(float(np.mean(sk**2) + EPS))),
        }
    out["per_key"] = per_key
    return out


def set_seed(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["mse", "nmse", "rmse", "relative_rmse", "cosine", "max_abs_diff", "teacher_rms", "student_rms"]
    out: dict[str, Any] = {"observations": len(rows)}
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        out[key] = {"mean": float(vals.mean()), "max": float(vals.max()), "min": float(vals.min())}
    return out


def aggregate_records(records: dict[str, QuantRecord]) -> dict[str, Any]:
    rows = [record.to_result() for record in records.values()]
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
    group_summary: dict[str, Any] = {}
    for group, group_rows in by_group.items():
        clip = np.asarray(
            [float(row["activation_clip_frac"]["mean"]) for row in group_rows], dtype=np.float64
        )
        clip_abs = np.asarray(
            [float(row["activation_clip_abs"]["mean"]) for row in group_rows], dtype=np.float64
        )
        group_summary[group] = {
            "modules": len(group_rows),
            "calls": int(sum(int(row["calls"]) for row in group_rows)),
            "activation_clip_frac_mean": float(clip.mean()) if clip.size else 0.0,
            "activation_clip_frac_max": float(clip.max()) if clip.size else 0.0,
            "activation_clip_abs_mean": float(clip_abs.mean()) if clip_abs.size else 0.0,
            "activation_clip_abs_max": float(clip_abs.max()) if clip_abs.size else 0.0,
        }
    return {"modules": rows, "group_summary": group_summary}


def parse_smoothing_alphas(value: str) -> list[float | None]:
    out: list[float | None] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if raw.lower() in {"none", "off", "naive"}:
            out.append(None)
        else:
            out.append(float(raw))
    if not out:
        raise ValueError("--smoothing-alphas must include at least one value")
    return out


def smooth_label(alpha: float | None) -> str:
    return "none" if alpha is None else f"sq_alpha_{alpha:g}"


def config_groups(config_name: str) -> set[str]:
    if config_name == "llm_only":
        return {"llm_selected"}
    if config_name == "dit_mlp_only":
        return {"dit_mlp_selected"}
    if config_name == "llm_dit_mlp":
        return {"llm_selected", "dit_mlp_selected"}
    raise ValueError(f"Unknown quant config: {config_name}")


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# QuantVLA Phase 3 Fake-Quant Forward",
        "",
        "Boundary: this is W4A8 fake quantization on synthetic inputs through the official GR00T policy path. It is not packed-kernel inference and not a LIBERO benchmark.",
        "",
        "## Run",
        "",
        f"- Model: `{result['model_path']}`",
        f"- Observations: `{result['num_observations']}`",
        f"- Synthetic variants: `{', '.join(result['synthetic_variants'])}`",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Weight bits: `{result['weight_bits']}`",
        f"- Activation bits: `{result['activation_bits']}`",
        f"- Smoothing alphas: `{', '.join(result['smoothing_labels'])}`",
        f"- Seeds: `{result['base_seed']} + observation_index`",
        "",
        "## Output Drift",
        "",
        "| config | smoothing | act scale | quantized modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["config_results"]:
        s = row["summary"]
        lines.append(
            "| {config} | {smooth} | {scale} | {modules} | {nmse:.6g} | {nmse_max:.6g} | {rrmse:.6g} | {cos:.6g} | {mad:.6g} |".format(
                config=row["config"],
                smooth=row["smoothing"],
                scale=row["activation_scale"],
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
            "## Interpretation",
            "",
            "- The comparison uses matched RNG seeds because GR00T action denoising starts from random Gaussian actions.",
            "- `llm_only` quantizes all selected LLM attention and MLP linears; `dit_mlp_only` quantizes only DiT feed-forward linears; `llm_dit_mlp` is the intended QuantVLA selected set.",
            "- `absmax` activation scale is conservative dynamic A8. `p999` clips roughly 0.1% of activation values per module call and is included to expose outlier sensitivity.",
            "- `sq_alpha_*` applies calibration-based SmoothQuant-style scale migration using input-channel activation maxima from the same synthetic calibration set.",
            "- These are fake-quant output-drift probes. They do not yet include ATM or OHB.",
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
    parser.add_argument("--activation-scale-modes", default="absmax,p999")
    parser.add_argument("--activation-percentile", type=float, default=0.999)
    parser.add_argument("--smoothing-alphas", default="none")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=260203)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("toy_quantvla/results/phase3_fake_quant_forward.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/phase3_fake_quant_forward.md"),
    )
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
    activation_scales = [v.strip() for v in args.activation_scale_modes.split(",") if v.strip()]
    smoothing_alphas = parse_smoothing_alphas(args.smoothing_alphas)

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

    teacher_actions: list[dict[str, Any]] = []
    teacher_started = time.time()
    for item in observations:
        set_seed(int(item["seed"]))
        teacher_actions.append(policy.get_action(item["obs"]))
    teacher_seconds = time.time() - teacher_started

    smoothing_groups = set().union(*(config_groups(config) for config in configs))
    smoothing_groups.discard("dit_attention_excluded")
    calibration_started = time.time()
    activation_max = collect_activation_max(policy, observations, smoothing_groups)
    calibration_seconds = time.time() - calibration_started
    smooth_scales_by_alpha = {
        alpha: make_smooth_scales(policy.model, activation_max, alpha)
        for alpha in smoothing_alphas
        if alpha is not None
    }

    config_results = []
    for smooth_alpha in smoothing_alphas:
        smooth_scales = smooth_scales_by_alpha.get(smooth_alpha) if smooth_alpha is not None else None
        for activation_scale in activation_scales:
            for config in configs:
                groups = config_groups(config)
                originals, records = patch_modules(
                    policy.model,
                    groups,
                    weight_bits=args.weight_bits,
                    activation_bits=args.activation_bits,
                    activation_scale=activation_scale,
                    activation_percentile=args.activation_percentile,
                    smooth_scales=smooth_scales,
                    smooth_alpha=smooth_alpha,
                )
                rows = []
                started = time.time()
                try:
                    for idx, item in enumerate(observations):
                        set_seed(int(item["seed"]))
                        student = policy.get_action(item["obs"])
                        metrics = compare_actions(teacher_actions[idx], student)
                        metrics.update({"variant": item["variant"], "seed": item["seed"]})
                        rows.append(metrics)
                finally:
                    restore_modules(policy.model, originals)
                record_summary = aggregate_records(records)
                config_results.append(
                    {
                        "config": config,
                        "smoothing": smooth_label(smooth_alpha),
                        "smooth_alpha": smooth_alpha,
                        "activation_scale": activation_scale,
                        "groups": sorted(groups),
                        "quantized_modules": len(originals),
                        "seconds": time.time() - started,
                        "summary": aggregate_metrics(rows),
                        "per_observation": rows,
                        "quant_records": record_summary,
                    }
                )

    result = {
        "model_path": str(args.model_path),
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "synthetic_variants": [item["variant"] for item in observations],
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "activation_percentile": args.activation_percentile,
        "smoothing_labels": [smooth_label(alpha) for alpha in smoothing_alphas],
        "base_seed": args.base_seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "load_seconds": load_seconds,
        "teacher_seconds": teacher_seconds,
        "calibration_seconds": calibration_seconds,
        "calibration_modules": len(activation_max),
        "config_results": config_results,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        result["cuda_memory_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
        result["cuda_memory_reserved_mib"] = torch.cuda.memory_reserved() / 1024**2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "load_seconds": result["load_seconds"],
        "teacher_seconds": result["teacher_seconds"],
        "calibration_seconds": result["calibration_seconds"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "config_results": [
            {
                "config": row["config"],
                "smoothing": row["smoothing"],
                "activation_scale": row["activation_scale"],
                "quantized_modules": row["quantized_modules"],
                "nmse_mean": row["summary"]["nmse"]["mean"],
                "relative_rmse_mean": row["summary"]["relative_rmse"]["mean"],
                "cosine_mean": row["summary"]["cosine"]["mean"],
                "max_abs_diff": row["summary"]["max_abs_diff"]["max"],
            }
            for row in config_results
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
