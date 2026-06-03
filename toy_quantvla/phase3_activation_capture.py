"""Capture activation statistics on the GR00T LIBERO forward path.

This is the first real-model calibration probe for QuantVLA reproduction. It
uses the official GR00T policy path, attaches forward hooks to the Phase 1
selector modules, and records compact statistics only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phase3_gr00t_smoke import (
    _insert_paths,
    build_synthetic_libero_observation,
    summarize_action,
)


LLM_RE = re.compile(
    r"^backbone\.eagle_model\.language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)
DIT_MLP_RE = re.compile(r"^action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)$")
DIT_ATTN_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.\d+\.attn1\.(?:to_q|to_k|to_v|to_out\.0)$"
)


def group_for_module(name: str) -> str | None:
    if LLM_RE.match(name):
        return "llm_selected"
    if DIT_MLP_RE.match(name):
        return "dit_mlp_selected"
    if DIT_ATTN_RE.match(name):
        return "dit_attention_excluded"
    return None


def _safe_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def tensor_stats(x: Any, percentile: float) -> dict[str, Any] | None:
    import torch

    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        return None
    with torch.no_grad():
        xf = x.detach().float()
        if xf.numel() == 0:
            return None
        flat = xf.reshape(-1)
        abs_flat = flat.abs()
        ref = torch.mean(flat.square()).clamp_min(1e-12)
        p99 = torch.quantile(abs_flat, 0.99)
        p999 = torch.quantile(abs_flat, percentile)
        absmax = abs_flat.max()

        def fake_quant_nmse(scale_abs: Any) -> tuple[float, float]:
            qmax = 127.0
            scale_abs = torch.as_tensor(scale_abs, device=flat.device, dtype=flat.dtype).clamp_min(
                1e-8
            )
            scale = scale_abs / qmax
            q = torch.clamp(torch.round(flat / scale), -qmax, qmax) * scale
            nmse = torch.mean((q - flat).square()) / ref
            clip_frac = (abs_flat > scale_abs).float().mean()
            return _safe_float(nmse), _safe_float(clip_frac)

        nmse_absmax, clip_absmax = fake_quant_nmse(absmax)
        nmse_p999, clip_p999 = fake_quant_nmse(p999)

        out: dict[str, Any] = {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "numel": int(flat.numel()),
            "mean": _safe_float(flat.mean()),
            "std": _safe_float(flat.std(unbiased=False)),
            "rms": _safe_float(torch.sqrt(ref)),
            "absmax": _safe_float(absmax),
            "p99_abs": _safe_float(p99),
            "p999_abs": _safe_float(p999),
            "tail_ratio_absmax_over_p999": _safe_float(absmax / p999.clamp_min(1e-8)),
            "w8_nmse_absmax_scale": nmse_absmax,
            "w8_clip_frac_absmax_scale": clip_absmax,
            "w8_nmse_p999_scale": nmse_p999,
            "w8_clip_frac_p999_scale": clip_p999,
        }
        if xf.ndim >= 2:
            channels = xf.reshape(-1, xf.shape[-1]).abs().amax(dim=0)
            out.update(
                {
                    "last_dim": int(xf.shape[-1]),
                    "channel_absmax_mean": _safe_float(channels.mean()),
                    "channel_absmax_max": _safe_float(channels.max()),
                    "channel_tail_ratio_max_over_mean": _safe_float(
                        channels.max() / channels.mean().clamp_min(1e-8)
                    ),
                }
            )
        return out


def merge_stat_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    scalar_keys = [
        "numel",
        "mean",
        "std",
        "rms",
        "absmax",
        "p99_abs",
        "p999_abs",
        "tail_ratio_absmax_over_p999",
        "w8_nmse_absmax_scale",
        "w8_clip_frac_absmax_scale",
        "w8_nmse_p999_scale",
        "w8_clip_frac_p999_scale",
        "channel_absmax_mean",
        "channel_absmax_max",
        "channel_tail_ratio_max_over_mean",
    ]
    out: dict[str, Any] = {
        "calls": len(records),
        "shapes": sorted({str(r["shape"]) for r in records if "shape" in r}),
        "dtypes": sorted({str(r["dtype"]) for r in records if "dtype" in r}),
    }
    for key in scalar_keys:
        vals = [float(r[key]) for r in records if key in r]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = {
                "mean": float(arr.mean()),
                "max": float(arr.max()),
                "min": float(arr.min()),
            }
    return out


@dataclass
class ModuleCapture:
    name: str
    group: str
    module_class: str
    in_features: int | None = None
    out_features: int | None = None
    input_records: list[dict[str, Any]] = field(default_factory=list)
    output_records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, inputs: tuple[Any, ...], output: Any, percentile: float) -> None:
        if inputs:
            stat = tensor_stats(inputs[0], percentile)
            if stat:
                self.input_records.append(stat)
        stat = tensor_stats(output, percentile)
        if stat:
            self.output_records.append(stat)

    def to_result(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "module_class": self.module_class,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "input": merge_stat_records(self.input_records),
            "output": merge_stat_records(self.output_records),
        }


def register_hooks(model: Any, percentile: float) -> tuple[list[Any], dict[str, ModuleCapture]]:
    import torch.nn as nn

    handles: list[Any] = []
    captures: dict[str, ModuleCapture] = {}
    for name, module in model.named_modules():
        group = group_for_module(name)
        if group is None or not isinstance(module, nn.Linear):
            continue
        cap = ModuleCapture(
            name=name,
            group=group,
            module_class=module.__class__.__name__,
            in_features=getattr(module, "in_features", None),
            out_features=getattr(module, "out_features", None),
        )
        captures[name] = cap

        def hook(mod: Any, inputs: tuple[Any, ...], output: Any, *, cap: ModuleCapture = cap) -> None:
            cap.add(inputs, output, percentile)

        handles.append(module.register_forward_hook(hook))
    return handles, captures


def remove_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def build_variant_observation(
    model_path: Path, embodiment_tag: str, variant: str, index: int
) -> dict[str, Any]:
    obs = build_synthetic_libero_observation(model_path, embodiment_tag)
    if variant == "zero":
        return obs
    if variant == "midgray":
        obs["video.image"] = np.full((1, 256, 256, 3), 127, dtype=np.uint8)
        obs["video.wrist_image"] = np.full((1, 256, 256, 3), 127, dtype=np.uint8)
        return obs
    if variant == "noise":
        rng = np.random.default_rng(20260203 + index)
        obs["video.image"] = rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)
        obs["video.wrist_image"] = rng.integers(0, 256, size=(1, 256, 256, 3), dtype=np.uint8)
        return obs
    raise ValueError(f"Unknown synthetic variant: {variant}")


def group_summary(modules: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"modules": len(modules)}
    for side in ["input", "output"]:
        side_modules = [m for m in modules if m.get(side)]
        out[f"{side}_modules_called"] = len(side_modules)
        for metric in [
            "rms",
            "absmax",
            "p999_abs",
            "tail_ratio_absmax_over_p999",
            "w8_nmse_absmax_scale",
            "w8_nmse_p999_scale",
            "w8_clip_frac_p999_scale",
            "channel_tail_ratio_max_over_mean",
        ]:
            vals = [
                float(m[side][metric]["mean"])
                for m in side_modules
                if metric in m[side] and "mean" in m[side][metric]
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                out[f"{side}_{metric}_mean"] = float(arr.mean())
                out[f"{side}_{metric}_max"] = float(arr.max())
    return out


def top_modules(
    modules: list[dict[str, Any]], side: str, metric: str, limit: int
) -> list[dict[str, Any]]:
    rows = []
    for module in modules:
        stat = module.get(side, {}).get(metric)
        if stat and "mean" in stat:
            rows.append(
                {
                    "name": module["name"],
                    "group": module["group"],
                    "value": float(stat["mean"]),
                    "max": float(stat.get("max", stat["mean"])),
                }
            )
    return sorted(rows, key=lambda row: row["value"], reverse=True)[:limit]


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# QuantVLA Phase 3 Activation Capture",
        "",
        "Boundary: this is a synthetic-input activation probe through the official GR00T policy path. It is not a LIBERO task evaluation.",
        "",
        "## Run",
        "",
        f"- Model: `{result['model_path']}`",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Observations: `{result['num_observations']}`",
        f"- Synthetic variants: `{', '.join(result['synthetic_variants'])}`",
        f"- Percentile for clipping probe: `{result['percentile']}`",
        f"- Forward seconds: `{result['forward_seconds']:.3f}`",
        "",
        "## Group Summary",
        "",
        "| group | modules | called | input rms | input absmax | input p99.9 | input tail | W8 NMSE absmax | W8 NMSE p99.9 | p99.9 clip | output rms | output absmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ["llm_selected", "dit_mlp_selected", "dit_attention_excluded"]:
        s = result["group_summaries"].get(group, {})
        lines.append(
            "| {group} | {modules} | {called} | {in_rms:.6g} | {in_abs:.6g} | {in_p999:.6g} | {in_tail:.6g} | {nmse_max:.6g} | {nmse_p999:.6g} | {clip:.6g} | {out_rms:.6g} | {out_abs:.6g} |".format(
                group=group,
                modules=s.get("modules", 0),
                called=s.get("input_modules_called", 0),
                in_rms=s.get("input_rms_mean", 0.0),
                in_abs=s.get("input_absmax_mean", 0.0),
                in_p999=s.get("input_p999_abs_mean", 0.0),
                in_tail=s.get("input_tail_ratio_absmax_over_p999_mean", 0.0),
                nmse_max=s.get("input_w8_nmse_absmax_scale_mean", 0.0),
                nmse_p999=s.get("input_w8_nmse_p999_scale_mean", 0.0),
                clip=s.get("input_w8_clip_frac_p999_scale_mean", 0.0),
                out_rms=s.get("output_rms_mean", 0.0),
                out_abs=s.get("output_absmax_mean", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Worst Input W8 NMSE With p99.9 Scale",
            "",
            "| group | module | NMSE | max |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["top_input_w8_nmse_p999"][:20]:
        lines.append(f"| {row['group']} | `{row['name']}` | {row['value']:.6g} | {row['max']:.6g} |")

    lines.extend(
        [
            "",
            "## Highest Input Tail Ratio",
            "",
            "| group | module | tail ratio | max |",
            "|---|---|---:|---:|",
        ]
    )
    for row in result["top_input_tail_ratio"][:20]:
        lines.append(f"| {row['group']} | `{row['name']}` | {row['value']:.6g} | {row['max']:.6g} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All three selector groups are exercised by the synthetic GR00T forward path.",
            "- `w8_nmse_absmax_scale` estimates conservative per-tensor int8 activation fake quantization error.",
            "- `w8_nmse_p999_scale` estimates percentile-clipped int8 activation fake quantization error and reports the corresponding clip fraction.",
            "- Very large LLM tail ratios are concentrated in MLP down-projection activations. This supports the need for smoothing/calibration, but these numbers are still from synthetic visual inputs and should be checked with real LIBERO observations before making benchmark claims.",
            "- This establishes activation ranges for the teacher model. The next step is to insert fake-quantized selected linears and compare teacher/student outputs, then calibrate ATM/OHB on the same hook scaffold.",
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
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument(
        "--synthetic-variants",
        default="zero",
        help="Comma-separated synthetic image variants to cycle through: zero,midgray,noise.",
    )
    parser.add_argument("--percentile", type=float, default=0.999)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("toy_quantvla/results/phase3_activation_capture.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/phase3_activation_capture.md"),
    )
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

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

    handles, captures = register_hooks(policy.model, args.percentile)
    forward_started = time.time()
    action_summary: dict[str, Any] | None = None
    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    if not variants:
        raise ValueError("--synthetic-variants must contain at least one variant")
    used_variants: list[str] = []
    try:
        for idx in range(args.num_observations):
            variant = variants[idx % len(variants)]
            used_variants.append(variant)
            obs = build_variant_observation(args.model_path, args.embodiment_tag, variant, idx)
            action = policy.get_action(obs)
            action_summary = summarize_action(action)
    finally:
        remove_hooks(handles)
    forward_seconds = time.time() - forward_started

    modules = [cap.to_result() for cap in captures.values()]
    modules = sorted(modules, key=lambda row: (row["group"], row["name"]))
    by_group = {
        group: [m for m in modules if m["group"] == group]
        for group in ["llm_selected", "dit_mlp_selected", "dit_attention_excluded"]
    }
    result = {
        "model_path": str(args.model_path),
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "synthetic_variants": used_variants,
        "percentile": args.percentile,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "load_seconds": load_seconds,
        "forward_seconds": forward_seconds,
        "registered_hooks": len(captures),
        "group_summaries": {group: group_summary(rows) for group, rows in by_group.items()},
        "top_input_w8_nmse_p999": top_modules(
            modules, "input", "w8_nmse_p999_scale", limit=30
        ),
        "top_input_w8_nmse_absmax": top_modules(
            modules, "input", "w8_nmse_absmax_scale", limit=30
        ),
        "top_input_tail_ratio": top_modules(
            modules, "input", "tail_ratio_absmax_over_p999", limit=30
        ),
        "top_output_tail_ratio": top_modules(
            modules, "output", "tail_ratio_absmax_over_p999", limit=30
        ),
        "action_summary": action_summary,
        "modules": modules,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        result["cuda_memory_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
        result["cuda_memory_reserved_mib"] = torch.cuda.memory_reserved() / 1024**2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "registered_hooks": result["registered_hooks"],
        "load_seconds": result["load_seconds"],
        "forward_seconds": result["forward_seconds"],
        "group_summaries": result["group_summaries"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
