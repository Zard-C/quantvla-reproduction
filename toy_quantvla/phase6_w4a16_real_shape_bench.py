"""Benchmark W4A16 Triton Linear on real selected GR00T module shapes.

The script loads a GR00T policy, captures runtime input shapes for selected
Linear modules, and benchmarks representative real weights one shape at a time.
It is intentionally offline: no simulator server is started.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase3_activation_capture import build_variant_observation, group_for_module
from phase3_fake_quant_forward import config_groups, set_seed
from phase3_gr00t_smoke import _insert_paths
from triton_w4a16 import (
    cosine_similarity,
    dequantize_w4_weight,
    pack_weight_w4_per_row,
    relative_rmse,
    w4a16_linear_triton,
)


DEFAULT_CONFIGS = (
    "16x32x64x4",
    "16x64x64x4",
    "32x32x64x4",
    "32x64x64x4",
    "32x64x128x4",
    "32x128x64x4",
    "32x128x128x4",
    "64x64x64x4",
    "64x64x128x4",
)


@dataclass(frozen=True)
class KernelConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int

    @property
    def label(self) -> str:
        return f"bm{self.block_m}_bn{self.block_n}_bk{self.block_k}_w{self.num_warps}"


def parse_kernel_configs(value: str) -> list[KernelConfig]:
    configs: list[KernelConfig] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.lower().split("x")
        if len(parts) != 4:
            raise ValueError(f"kernel config must be BLOCK_MxBLOCK_NxBLOCK_KxWARPS, got {raw!r}")
        bm, bn, bk, nw = (int(part) for part in parts)
        configs.append(KernelConfig(bm, bn, bk, nw))
    if not configs:
        raise ValueError("at least one kernel config is required")
    return configs


def synchronize(device: Any) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def time_ms(fn: Any, device: Any, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def collect_selected_modules(model: Any, groups: set[str]) -> dict[str, Any]:
    import torch.nn as nn

    selected: dict[str, Any] = {}
    for name, module in model.named_modules():
        group = group_for_module(name)
        if group in groups and isinstance(module, nn.Linear):
            selected[name] = module
    return selected


def capture_runtime_m(
    policy: Any,
    selected: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    import torch

    captured: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            x = inputs[0]
            if x.ndim == 0 or x.shape[-1] != module.in_features:
                return
            m = int(x.numel() // module.in_features)
            item = captured.setdefault(
                name,
                {
                    "input_shapes": [],
                    "m_values": [],
                    "dtype": str(x.dtype),
                },
            )
            item["input_shapes"].append(list(x.shape))
            item["m_values"].append(m)

        return hook

    try:
        for name, module in selected.items():
            handles.append(module.register_forward_hook(make_hook(name)))
        for item in observations:
            set_seed(int(item["seed"]))
            policy.get_action(item["obs"])
    finally:
        for handle in handles:
            handle.remove()
    return captured


def representative_cases(
    selected: dict[str, Any],
    captured: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, int, int, int]] = set()
    cases: list[dict[str, Any]] = []
    for name, module in selected.items():
        cap = captured.get(name)
        if not cap:
            continue
        group = group_for_module(name) or "unknown"
        m = int(max(cap["m_values"]))
        key = (group, int(module.in_features), int(module.out_features), m)
        if key in seen:
            continue
        seen.add(key)
        cases.append(
            {
                "name": name,
                "group": group,
                "m": m,
                "k": int(module.in_features),
                "n": int(module.out_features),
                "input_shapes": cap["input_shapes"][:4],
                "module": module,
            }
        )
    return sorted(cases, key=lambda x: (x["group"], x["k"], x["n"], x["m"], x["name"]))


def benchmark_case(
    case: dict[str, Any],
    *,
    device: str,
    repeats: int,
    warmup: int,
    kernel_configs: list[KernelConfig],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    module = case["module"]
    weight = module.weight.detach().to(device=device, dtype=torch.float16).contiguous()
    bias = module.bias.detach().to(device=device, dtype=torch.float16).contiguous() if module.bias is not None else None
    torch.manual_seed(20260606 + case["m"] + 3 * case["k"] + 17 * case["n"])
    x = (torch.randn((case["m"], case["k"]), device=device, dtype=torch.float16) / (case["k"] ** 0.5)).contiguous()
    packed = pack_weight_w4_per_row(weight.detach().cpu())
    qweight = packed.qweight.to(device)
    scale = packed.scale.to(device)
    w_dequant = dequantize_w4_weight(qweight, scale, case["k"], dtype=torch.float16)
    ref = F.linear(x, w_dequant, bias)

    config_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in kernel_configs:
        try:
            out = w4a16_linear_triton(
                x,
                qweight,
                scale,
                bias,
                case["k"],
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
            )
            row = {
                "config": cfg.label,
                "max_abs_diff": float((out - ref).detach().float().abs().max().item()),
                "relative_rmse": relative_rmse(ref, out),
                "cosine": cosine_similarity(ref, out),
                "w4a16_ms": time_ms(
                    lambda: w4a16_linear_triton(
                        x,
                        qweight,
                        scale,
                        bias,
                        case["k"],
                        block_m=cfg.block_m,
                        block_n=cfg.block_n,
                        block_k=cfg.block_k,
                        num_warps=cfg.num_warps,
                    ),
                    device,
                    repeats,
                    warmup,
                ),
                "error": None,
            }
        except Exception as exc:
            row = {
                "config": cfg.label,
                "error": repr(exc),
            }
        config_rows.append(row)
        if row.get("error") is None and (best is None or row["w4a16_ms"] < best["w4a16_ms"]):
            best = row

    torch_ms = time_ms(lambda: F.linear(x, weight, bias), device, repeats, warmup)
    dense_ref_ms = time_ms(lambda: F.linear(x, w_dequant, bias), device, repeats, warmup)
    out_case = {
        "name": case["name"],
        "group": case["group"],
        "m": case["m"],
        "k": case["k"],
        "n": case["n"],
        "input_shapes": case["input_shapes"],
        "fp16_weight_bytes": packed.fp16_weight_bytes(),
        "packed_weight_bytes": packed.packed_weight_bytes(),
        "scale_bytes": packed.scale_bytes(),
        "packed_plus_scale_ratio_vs_fp16": packed.compression_ratio_vs_fp16(),
        "torch_fp16_ms": torch_ms,
        "dense_dequant_ref_ms": dense_ref_ms,
        "configs": config_rows,
        "best": best,
    }
    if best is not None:
        out_case["speedup_vs_torch_fp16"] = torch_ms / best["w4a16_ms"]
    return out_case


def strip_modules(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = dict(row)
        item.pop("module", None)
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--config", default="llm_dit_mlp")
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260606)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--kernel-configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase6_w4a16_real_shape_bench.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    variants = [v.strip() for v in args.synthetic_variants.split(",") if v.strip()]
    kernel_configs = parse_kernel_configs(args.kernel_configs)
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

    groups = config_groups(args.config)
    selected = collect_selected_modules(policy.model, groups)
    observations = [
        {
            "variant": variants[idx % len(variants)],
            "seed": args.base_seed + idx,
            "obs": build_variant_observation(args.model_path, args.embodiment_tag, variants[idx % len(variants)], idx),
        }
        for idx in range(args.num_observations)
    ]
    captured = capture_runtime_m(policy, selected, observations)
    cases = representative_cases(selected, captured)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    rows = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}/{len(cases)}] {case['group']} M={case['m']} K={case['k']} N={case['n']} {case['name']}",
            flush=True,
        )
        rows.append(
            benchmark_case(
                case,
                device=args.device,
                repeats=args.repeats,
                warmup=args.warmup,
                kernel_configs=kernel_configs,
            )
        )

    by_group: dict[str, Any] = {}
    for row in rows:
        group_rows = by_group.setdefault(row["group"], [])
        group_rows.append(row)
    summary = {
        group: {
            "cases": len(group_rows),
            "best_speedup_vs_torch_fp16_mean": float(np.mean([r.get("speedup_vs_torch_fp16", 0.0) for r in group_rows])),
            "packed_ratio_mean": float(np.mean([r["packed_plus_scale_ratio_vs_fp16"] for r in group_rows])),
        }
        for group, group_rows in by_group.items()
    }

    result = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "config": args.config,
        "groups": sorted(groups),
        "selected_modules": len(selected),
        "captured_modules": len(captured),
        "representative_cases": len(cases),
        "denoising_steps": args.denoising_steps,
        "num_observations": args.num_observations,
        "load_seconds": load_seconds,
        "kernel_configs": [cfg.label for cfg in kernel_configs],
        "case_descriptors": strip_modules(cases),
        "summary": summary,
        "cases": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "output_json": str(args.output_json)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

