"""Low-intrusion torch.profiler probe for FP16 GR00T get_action.

This script is not a rollout benchmark. It runs a few fixed observations
through the official FP16 policy and asks torch.profiler where time is spent,
with special attention to hidden D2H/scalar sync events.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
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


SYNC_PATTERNS = (
    "cudaMemcpy",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::copy_",
    "aten::_to_copy",
    "aten::to",
)


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def expanded_observations(observations: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for repeat in range(max(1, int(repeats))):
        for item in observations:
            copied = dict(item)
            copied["seed"] = int(item["seed"]) + repeat * 10_000
            copied["repeat_index"] = int(repeat)
            out.append(copied)
    return out


def event_to_row(event: Any) -> dict[str, Any]:
    self_device_time_total = getattr(
        event,
        "self_device_time_total",
        getattr(event, "self_cuda_time_total", 0.0),
    )
    device_time_total = getattr(
        event,
        "device_time_total",
        getattr(event, "cuda_time_total", 0.0),
    )
    return {
        "key": str(getattr(event, "key", "")),
        "count": int(getattr(event, "count", 0)),
        "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
        "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        "self_cuda_time_total_us": float(self_device_time_total),
        "cuda_time_total_us": float(device_time_total),
        "cpu_memory_usage": int(getattr(event, "cpu_memory_usage", 0)),
        "cuda_memory_usage": int(getattr(event, "cuda_memory_usage", 0)),
    }


def top_events(events: Any, *, sort_key: str, limit: int) -> list[dict[str, Any]]:
    rows = [event_to_row(event) for event in events]
    rows.sort(key=lambda row: float(row.get(sort_key, 0.0)), reverse=True)
    return rows[: int(limit)]


def matching_events(events: Any, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in events:
        key = str(getattr(event, "key", ""))
        if any(pattern in key for pattern in patterns):
            rows.append(event_to_row(event))
    rows.sort(
        key=lambda row: (
            float(row["self_cuda_time_total_us"]) + float(row["self_cpu_time_total_us"]),
            int(row["count"]),
        ),
        reverse=True,
    )
    return rows


def run_requests(policy: Any, observations: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    latencies: list[float] = []
    reset_cuda_peak(device)
    for item in observations:
        set_seed(int(item["seed"]))
        started = time.perf_counter()
        with torch.inference_mode():
            policy.get_action(item["obs"])
        synchronize(device)
        latencies.append(time.perf_counter() - started)
    return {
        "request_seconds": summarize_float(latencies),
        "memory": cuda_memory(device),
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
    parser.add_argument("--num-observations", type=int, default=1)
    parser.add_argument("--indices", default="115")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--base-seed", type=int, default=20260619)
    parser.add_argument("--warmup-repeats", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase19_torch_profiler_probe.json"))
    parser.add_argument("--cpu-table", type=Path, default=Path("toy_quantvla/results/phase19_torch_profiler_cpu_table.txt"))
    parser.add_argument("--cuda-table", type=Path, default=Path("toy_quantvla/results/phase19_torch_profiler_cuda_table.txt"))
    parser.add_argument("--chrome-trace", type=Path, default=Path("toy_quantvla/results/phase19_torch_profiler_trace.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    from gr00t.model.policy import Gr00tPolicy
    from torch.profiler import ProfilerActivity, profile

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    observations, observation_meta = build_observations(args, data_config)
    warmup_observations = expanded_observations(observations, args.warmup_repeats)
    profile_observations = expanded_observations(observations, args.profile_repeats)

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

    warmup = run_requests(policy, warmup_observations, device=args.device)
    profile_latencies: list[float] = []
    reset_cuda_peak(args.device)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=bool(args.record_shapes),
        profile_memory=bool(args.profile_memory),
        with_stack=bool(args.with_stack),
    ) as prof:
        for item in profile_observations:
            set_seed(int(item["seed"]))
            started = time.perf_counter()
            with torch.inference_mode():
                policy.get_action(item["obs"])
            synchronize(args.device)
            profile_latencies.append(time.perf_counter() - started)
            prof.step()

    events = prof.key_averages()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.cpu_table.parent.mkdir(parents=True, exist_ok=True)
    args.cuda_table.parent.mkdir(parents=True, exist_ok=True)
    args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.chrome_trace))
    cpu_table = events.table(sort_by="self_cpu_time_total", row_limit=int(args.top_k))
    cuda_table = events.table(sort_by="self_cuda_time_total", row_limit=int(args.top_k))
    args.cpu_table.write_text(cpu_table, encoding="utf-8")
    args.cuda_table.write_text(cuda_table, encoding="utf-8")

    payload = {
        "boundary": "Low-intrusion torch.profiler fixed-observation FP16 get_action probe",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "observation_meta": observation_meta,
        "warmup_requests": len(warmup_observations),
        "profile_requests": len(profile_observations),
        "load_seconds": float(load_seconds),
        "warmup": warmup,
        "profile_request_seconds": summarize_float(profile_latencies),
        "profile_memory": cuda_memory(args.device),
        "activities": [activity.name for activity in activities],
        "record_shapes": bool(args.record_shapes),
        "profile_memory_enabled": bool(args.profile_memory),
        "with_stack": bool(args.with_stack),
        "top_self_cpu_events": top_events(events, sort_key="self_cpu_time_total_us", limit=args.top_k),
        "top_self_cuda_events": top_events(events, sort_key="self_cuda_time_total_us", limit=args.top_k),
        "sync_or_d2h_candidate_events": matching_events(events, SYNC_PATTERNS),
        "cpu_table": str(args.cpu_table),
        "cuda_table": str(args.cuda_table),
        "chrome_trace": str(args.chrome_trace),
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "profile_requests": payload["profile_requests"],
                "profile_request_seconds": payload["profile_request_seconds"],
                "top_self_cuda_events": payload["top_self_cuda_events"][:10],
                "sync_or_d2h_candidate_events": payload["sync_or_d2h_candidate_events"][:20],
                "output_json": str(args.output_json),
                "chrome_trace": str(args.chrome_trace),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
