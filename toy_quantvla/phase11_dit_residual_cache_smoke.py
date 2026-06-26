"""Offline static residual-cache smoke for GR00T DiT blocks.

This is the first behavioral intervention after `phase11_dit_cache_probe.py`.
Selected DiT transformer blocks are wrapped so chosen denoising steps can skip
the block forward and reuse the previous full-compute residual:

    y_t ~= x_t + (y_prev_full - x_prev_full)

The script compares cached actions against the unmodified FP16 teacher under
matched denoising seeds. It is an offline drift/timing diagnostic, not a
closed-loop LIBERO success benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from phase3_fake_quant_forward import action_to_vector, compare_actions, set_seed, set_submodule
from phase3_gr00t_smoke import _insert_paths
from phase8_cutlass_blockscaled_fp4_real_activation_bench import (
    build_observations,
    cuda_memory,
    reset_cuda_peak,
    synchronize,
)
from phase11_dit_cache_probe import collect_dit_blocks, maybe_first_floating_tensor
from timing_utils import summarize_float


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def parse_int_set(value: str | None) -> set[int]:
    if not value:
        return set()
    out: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            out.update(range(start, end + step, step))
        else:
            out.add(int(item))
    return out


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if vals.size == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    return {"mean": float(vals.mean()), "max": float(vals.max()), "min": float(vals.min())}


def aggregate_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["mse", "nmse", "rmse", "relative_rmse", "cosine", "max_abs_diff", "teacher_rms", "student_rms"]
    out: dict[str, Any] = {"observations": len(rows)}
    for key in keys:
        out[key] = summarize_metric(rows, key)
    return out


class ResidualCacheBlock(nn.Module):
    """Wrap one DiT block with static denoising-step residual reuse."""

    def __init__(self, base: nn.Module, *, block_index: int, block_name: str, cache_steps: set[int]):
        super().__init__()
        self.base = base
        self.block_index = int(block_index)
        self.block_name = block_name
        self.cache_steps = set(int(step) for step in cache_steps)
        self.call_index = 0
        self.prev_residual: torch.Tensor | None = None
        self.output_was_tensor: bool | None = None
        self.calls = 0
        self.full_compute_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_hit_steps: dict[int, int] = {}
        self.full_compute_steps: dict[int, int] = {}

    def reset_request(self) -> None:
        self.call_index = 0
        self.prev_residual = None
        self.output_was_tensor = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        step = int(self.call_index)
        self.call_index += 1
        self.calls += 1
        x = maybe_first_floating_tensor(args[0] if args else kwargs)
        can_skip = (
            step in self.cache_steps
            and x is not None
            and self.prev_residual is not None
            and tuple(x.shape) == tuple(self.prev_residual.shape)
            and self.output_was_tensor is True
        )
        if can_skip:
            self.cache_hits += 1
            self.cache_hit_steps[step] = self.cache_hit_steps.get(step, 0) + 1
            return x + self.prev_residual.to(device=x.device, dtype=x.dtype)

        if step in self.cache_steps:
            self.cache_misses += 1
        y = self.base(*args, **kwargs)
        out = maybe_first_floating_tensor(y)
        self.output_was_tensor = torch.is_tensor(y)
        if x is not None and out is not None and tuple(x.shape) == tuple(out.shape):
            self.prev_residual = (out.detach() - x.detach()).detach()
        self.full_compute_calls += 1
        self.full_compute_steps[step] = self.full_compute_steps.get(step, 0) + 1
        return y

    def to_result(self) -> dict[str, Any]:
        return {
            "block_index": self.block_index,
            "block_name": self.block_name,
            "cache_steps": sorted(int(step) for step in self.cache_steps),
            "calls": int(self.calls),
            "full_compute_calls": int(self.full_compute_calls),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_hit_rate": float(self.cache_hits / self.calls) if self.calls else 0.0,
            "cache_hit_steps": {str(k): int(v) for k, v in sorted(self.cache_hit_steps.items())},
            "full_compute_steps": {str(k): int(v) for k, v in sorted(self.full_compute_steps.items())},
        }


def patch_residual_cache_blocks(model: Any, *, cache_blocks: set[int], cache_steps: set[int]) -> dict[str, ResidualCacheBlock]:
    wrappers: dict[str, ResidualCacheBlock] = {}
    for block_index, block_name, module in collect_dit_blocks(model):
        if block_index not in cache_blocks:
            continue
        wrapper = ResidualCacheBlock(module, block_index=block_index, block_name=block_name, cache_steps=cache_steps)
        set_submodule(model, block_name, wrapper)
        wrappers[block_name] = wrapper
    return wrappers


def run_actions(
    policy: Any,
    observations: list[dict[str, Any]],
    *,
    device: str,
    wrappers: dict[str, ResidualCacheBlock] | None = None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any], dict[str, int]]:
    actions: list[dict[str, Any]] = []
    request_seconds: list[float] = []
    reset_cuda_peak(device)
    started_total = time.perf_counter()
    with torch.no_grad():
        for item in observations:
            if wrappers:
                for wrapper in wrappers.values():
                    wrapper.reset_request()
            set_seed(int(item["seed"]))
            started = time.perf_counter()
            actions.append(policy.get_action(item["obs"]))
            synchronize(device)
            request_seconds.append(time.perf_counter() - started)
    total_seconds = time.perf_counter() - started_total
    return actions, total_seconds, summarize_float(request_seconds), cuda_memory(device)


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
    parser.add_argument("--cache-blocks", default="12,15")
    parser.add_argument("--cache-steps", default="1,3,5")
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase11_dit_residual_cache_smoke.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    observations, observation_meta = build_observations(args, data_config)
    cache_blocks = parse_int_set(args.cache_blocks)
    cache_steps = parse_int_set(args.cache_steps)

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

    warmup_runs = max(0, int(args.warmup_repeats))
    warmup_seconds = 0.0
    warmup_request_seconds: list[dict[str, Any]] = []
    for _ in range(warmup_runs):
        _warmup_actions, seconds, request_summary, _memory = run_actions(
            policy,
            observations,
            device=args.device,
        )
        warmup_seconds += seconds
        warmup_request_seconds.append(request_summary)

    teacher_actions, teacher_seconds, teacher_request_seconds, teacher_memory = run_actions(
        policy,
        observations,
        device=args.device,
    )
    patch_started = time.perf_counter()
    wrappers = patch_residual_cache_blocks(policy.model, cache_blocks=cache_blocks, cache_steps=cache_steps)
    synchronize(args.device)
    patch_seconds = time.perf_counter() - patch_started
    if not wrappers:
        raise RuntimeError(f"No DiT blocks patched for cache_blocks={sorted(cache_blocks)}")

    student_actions, student_seconds, student_request_seconds, student_memory = run_actions(
        policy,
        observations,
        device=args.device,
        wrappers=wrappers,
    )

    comparisons: list[dict[str, Any]] = []
    for item, teacher, student in zip(observations, teacher_actions, student_actions, strict=True):
        row = compare_actions(teacher, student)
        row["seed"] = int(item["seed"])
        if "dataset_index" in item:
            row["dataset_index"] = int(item["dataset_index"])
        row["action_size"] = int(action_to_vector(teacher).size)
        comparisons.append(row)

    wrapper_results = {name: wrapper.to_result() for name, wrapper in wrappers.items()}
    cache_hits = sum(int(row["cache_hits"]) for row in wrapper_results.values())
    full_compute_calls = sum(int(row["full_compute_calls"]) for row in wrapper_results.values())
    patched_calls = sum(int(row["calls"]) for row in wrapper_results.values())
    all_dit_block_calls = len(collect_dit_blocks(policy.model)) * int(args.denoising_steps) * len(observations)

    payload = {
        "boundary": "Offline static residual-cache action drift smoke; selected DiT blocks skip configured denoising steps",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "observation_meta": observation_meta,
        "observations": len(observations),
        "cache_blocks": sorted(int(v) for v in cache_blocks),
        "cache_steps": sorted(int(v) for v in cache_steps),
        "warmup_repeats": int(warmup_runs),
        "warmup_seconds": float(warmup_seconds),
        "warmup_request_seconds": warmup_request_seconds,
        "load_seconds": float(load_seconds),
        "patch_seconds": float(patch_seconds),
        "teacher_seconds": float(teacher_seconds),
        "student_seconds": float(student_seconds),
        "student_over_teacher_time": float(student_seconds / teacher_seconds) if teacher_seconds else None,
        "teacher_request_seconds": teacher_request_seconds,
        "student_request_seconds": student_request_seconds,
        "teacher_memory": teacher_memory,
        "student_memory": student_memory,
        "cache_stats": {
            "patched_blocks": len(wrappers),
            "patched_block_calls": int(patched_calls),
            "full_compute_calls": int(full_compute_calls),
            "cache_hits": int(cache_hits),
            "patched_block_hit_rate": float(cache_hits / patched_calls) if patched_calls else 0.0,
            "all_dit_block_calls": int(all_dit_block_calls),
            "all_dit_block_skip_rate": float(cache_hits / all_dit_block_calls) if all_dit_block_calls else 0.0,
        },
        "wrapper_results": wrapper_results,
        "comparisons": comparisons,
        "summary": aggregate_comparisons(comparisons),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "cache_blocks": payload["cache_blocks"],
                "cache_steps": payload["cache_steps"],
                "cache_stats": payload["cache_stats"],
                "timing": {
                    "teacher_seconds": teacher_seconds,
                    "student_seconds": student_seconds,
                    "student_over_teacher_time": payload["student_over_teacher_time"],
                    "teacher_request_seconds": teacher_request_seconds,
                    "student_request_seconds": student_request_seconds,
                },
                "summary": payload["summary"],
                "output_json": str(args.output_json),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
