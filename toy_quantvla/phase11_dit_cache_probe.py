"""Offline DiT denoising-step cache probe for GR00T LIBERO.

This script does not alter model behavior. It attaches hooks to
`action_head.model.transformer_blocks.*`, runs real/synthetic observations
through `Gr00tPolicy.get_action`, and measures how block inputs, outputs, and
residuals drift across denoising steps.

The goal is to identify safe candidates for a Cache-DiT-style acceleration
path before implementing closed-loop cache/skip logic.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase8_cutlass_blockscaled_fp4_real_activation_bench import (
    build_observations,
    cuda_memory,
    reset_cuda_peak,
    synchronize,
)
from timing_utils import summarize_float


DIT_BLOCK_RE = re.compile(r"^action_head\.model\.transformer_blocks\.(\d+)$")
EPS = 1e-12


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def maybe_first_floating_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value) and torch.is_floating_point(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = maybe_first_floating_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = maybe_first_floating_tensor(item)
            if found is not None:
                return found
    return None


def tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    x = tensor.detach().float()
    if x.numel() == 0:
        return None
    flat = x.reshape(-1)
    rms = torch.sqrt(torch.mean(flat.square()).clamp_min(EPS))
    abs_flat = flat.abs()
    return {
        "shape": [int(v) for v in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "rms": float(rms.item()),
        "absmax": float(abs_flat.max().item()),
        "p99_abs": float(torch.quantile(abs_flat, 0.99).item()),
    }


def detach_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().float().cpu().contiguous()


def compare_tensors(current: torch.Tensor | None, previous: torch.Tensor | None) -> dict[str, Any] | None:
    if current is None or previous is None or tuple(current.shape) != tuple(previous.shape):
        return None
    c = current.reshape(-1).float()
    p = previous.reshape(-1).float()
    diff = c - p
    mse = torch.mean(diff.square())
    ref = torch.mean(p.square()).clamp_min(EPS)
    c_norm = torch.linalg.vector_norm(c)
    p_norm = torch.linalg.vector_norm(p)
    denom = (c_norm * p_norm).clamp_min(EPS)
    l1_ref = torch.mean(p.abs()).clamp_min(EPS)
    return {
        "mse": float(mse.item()),
        "relative_rmse": float(torch.sqrt(mse / ref).item()),
        "relative_l1": float((torch.mean(diff.abs()) / l1_ref).item()),
        "cosine": float((torch.dot(c, p) / denom).item()),
        "max_abs_diff": float(diff.abs().max().item()),
        "previous_rms": float(torch.sqrt(ref).item()),
        "current_rms": float(torch.sqrt(torch.mean(c.square()).clamp_min(EPS)).item()),
    }


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0}
    vals = sorted(float(v) for v in values)
    p50 = vals[int(round((len(vals) - 1) * 0.50))]
    p90 = vals[int(round((len(vals) - 1) * 0.90))]
    return {
        "count": len(vals),
        "mean": float(sum(vals) / len(vals)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
        "p50": float(p50),
        "p90": float(p90),
    }


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"calls": len(rows)}
    metric_paths = [
        ("input_delta_vs_prev_step", "relative_rmse"),
        ("input_delta_vs_prev_step", "relative_l1"),
        ("input_delta_vs_prev_step", "cosine"),
        ("output_delta_vs_prev_step", "relative_rmse"),
        ("output_delta_vs_prev_step", "relative_l1"),
        ("output_delta_vs_prev_step", "cosine"),
        ("residual_delta_vs_prev_step", "relative_rmse"),
        ("residual_delta_vs_prev_step", "relative_l1"),
        ("residual_delta_vs_prev_step", "cosine"),
        ("residual_stats", "rms"),
        ("output_stats", "rms"),
        ("input_stats", "rms"),
    ]
    for section, key in metric_paths:
        vals = [
            float(row[section][key])
            for row in rows
            if isinstance(row.get(section), dict) and key in row[section]
        ]
        if vals:
            out[f"{section}.{key}"] = summarize_values(vals)
    return out


def collect_dit_blocks(model: Any) -> list[tuple[int, str, Any]]:
    blocks: list[tuple[int, str, Any]] = []
    for name, module in model.named_modules():
        match = DIT_BLOCK_RE.match(name)
        if match:
            blocks.append((int(match.group(1)), name, module))
    return sorted(blocks, key=lambda item: item[0])


class DitStepProbe:
    def __init__(self, blocks: list[tuple[int, str, Any]], *, max_denoising_steps: int | None):
        self.blocks = blocks
        self.max_denoising_steps = max_denoising_steps
        self.handles: list[Any] = []
        self.rows: list[dict[str, Any]] = []
        self.current_obs: dict[str, Any] = {}
        self.block_call_counts: dict[int, int] = {}
        self.prev_input_by_block: dict[int, torch.Tensor] = {}
        self.prev_output_by_block: dict[int, torch.Tensor] = {}
        self.prev_residual_by_block: dict[int, torch.Tensor] = {}

    def start_observation(self, item: dict[str, Any], observation_index: int) -> None:
        self.current_obs = {
            "observation_index": int(observation_index),
            "seed": int(item["seed"]),
        }
        for key in ("dataset_index", "variant", "repeat_index"):
            if key in item:
                value = item[key]
                self.current_obs[key] = int(value) if isinstance(value, int) else value
        self.block_call_counts = {}
        self.prev_input_by_block = {}
        self.prev_output_by_block = {}
        self.prev_residual_by_block = {}

    def _make_hook(self, block_index: int, block_name: str) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            call_ordinal = int(self.block_call_counts.get(block_index, 0))
            self.block_call_counts[block_index] = call_ordinal + 1
            if self.max_denoising_steps is not None and call_ordinal >= self.max_denoising_steps:
                return

            input_tensor = maybe_first_floating_tensor(inputs[0] if inputs else None)
            output_tensor = maybe_first_floating_tensor(output)
            residual_tensor: torch.Tensor | None = None
            if input_tensor is not None and output_tensor is not None and tuple(input_tensor.shape) == tuple(output_tensor.shape):
                residual_tensor = output_tensor.detach().float() - input_tensor.detach().float()

            input_cpu = detach_cpu(input_tensor)
            output_cpu = detach_cpu(output_tensor)
            residual_cpu = detach_cpu(residual_tensor)
            row = {
                **self.current_obs,
                "block_index": int(block_index),
                "block_name": block_name,
                "denoise_index": int(call_ordinal),
                "input_stats": tensor_stats(input_tensor),
                "output_stats": tensor_stats(output_tensor),
                "residual_stats": tensor_stats(residual_tensor),
                "input_delta_vs_prev_step": compare_tensors(input_cpu, self.prev_input_by_block.get(block_index)),
                "output_delta_vs_prev_step": compare_tensors(output_cpu, self.prev_output_by_block.get(block_index)),
                "residual_delta_vs_prev_step": compare_tensors(
                    residual_cpu,
                    self.prev_residual_by_block.get(block_index),
                ),
            }
            self.rows.append(row)
            if input_cpu is not None:
                self.prev_input_by_block[block_index] = input_cpu
            if output_cpu is not None:
                self.prev_output_by_block[block_index] = output_cpu
            if residual_cpu is not None:
                self.prev_residual_by_block[block_index] = residual_cpu

        return hook

    def install(self) -> None:
        for block_index, block_name, module in self.blocks:
            self.handles.append(module.register_forward_hook(self._make_hook(block_index, block_name)))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def summarize_rows(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    by_transition: dict[str, list[dict[str, Any]]] = {}
    by_block_transition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        block_key = str(row["block_index"])
        transition_key = f"{max(0, int(row['denoise_index']) - 1)}->{int(row['denoise_index'])}"
        by_block.setdefault(block_key, []).append(row)
        if int(row["denoise_index"]) > 0:
            by_transition.setdefault(transition_key, []).append(row)
            by_block_transition.setdefault(f"b{block_key}:{transition_key}", []).append(row)

    candidates: list[dict[str, Any]] = []
    for key, group_rows in by_block_transition.items():
        summary = summarize_group(group_rows)
        output_rel = summary.get("output_delta_vs_prev_step.relative_rmse", {})
        residual_rel = summary.get("residual_delta_vs_prev_step.relative_rmse", {})
        output_cos = summary.get("output_delta_vs_prev_step.cosine", {})
        candidates.append(
            {
                "key": key,
                "calls": len(group_rows),
                "output_relative_rmse_mean": float(output_rel.get("mean", 0.0)),
                "output_relative_l1_mean": float(
                    summary.get("output_delta_vs_prev_step.relative_l1", {}).get("mean", 0.0)
                ),
                "output_cosine_mean": float(output_cos.get("mean", 0.0)),
                "residual_relative_rmse_mean": float(residual_rel.get("mean", 0.0)),
                "residual_relative_l1_mean": float(
                    summary.get("residual_delta_vs_prev_step.relative_l1", {}).get("mean", 0.0)
                ),
                "summary": summary,
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["output_relative_rmse_mean"]),
            -float(row["output_cosine_mean"]),
            float(row["residual_relative_rmse_mean"]),
        )
    )

    return {
        "rows": len(rows),
        "by_block": {key: summarize_group(value) for key, value in sorted(by_block.items(), key=lambda kv: int(kv[0]))},
        "by_transition": dict(sorted((key, summarize_group(value)) for key, value in by_transition.items())),
        "cache_candidate_ranking": candidates[: int(top_k)],
        "block_transition_summary": {key: summarize_group(value) for key, value in sorted(by_block_transition.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--num-observations", type=int, default=4)
    parser.add_argument("--indices", default="115,215,315,415")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260610)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--include-raw-rows", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase11_dit_cache_probe.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    observations, observation_meta = build_observations(args, data_config)

    load_started = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    synchronize(args.device)
    load_seconds = time.perf_counter() - load_started

    blocks = collect_dit_blocks(policy.model)
    if not blocks:
        raise RuntimeError("No DiT transformer blocks matched action_head.model.transformer_blocks.*")

    probe = DitStepProbe(blocks, max_denoising_steps=args.denoising_steps)
    request_seconds: list[float] = []
    reset_cuda_peak(args.device)
    probe.install()
    try:
        with torch.no_grad():
            for observation_index, item in enumerate(observations):
                probe.start_observation(item, observation_index)
                set_seed(int(item["seed"]))
                started = time.perf_counter()
                policy.get_action(item["obs"])
                synchronize(args.device)
                request_seconds.append(time.perf_counter() - started)
    finally:
        probe.remove()

    summary = summarize_rows(probe.rows, top_k=args.top_k)
    block_call_counts: dict[str, dict[str, int]] = {}
    for row in probe.rows:
        obs_key = str(row["observation_index"])
        block_key = str(row["block_index"])
        block_call_counts.setdefault(obs_key, {})
        block_call_counts[obs_key][block_key] = block_call_counts[obs_key].get(block_key, 0) + 1

    payload: dict[str, Any] = {
        "boundary": "Offline GR00T DiT denoising-step cache probe; no model behavior changes",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "observation_meta": observation_meta,
        "observations": len(observations),
        "load_seconds": float(load_seconds),
        "request_seconds": summarize_float(request_seconds),
        "memory": cuda_memory(args.device),
        "dit_blocks": [{"index": int(index), "name": name, "class": module.__class__.__name__} for index, name, module in blocks],
        "block_call_counts": block_call_counts,
        "summary": summary,
    }
    if args.include_raw_rows:
        payload["raw_rows"] = probe.rows

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "observations": payload["observations"],
                "dit_blocks": len(blocks),
                "probe_rows": len(probe.rows),
                "request_seconds": payload["request_seconds"],
                "top_cache_candidates": summary["cache_candidate_ranking"][:10],
                "output_json": str(args.output_json),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
