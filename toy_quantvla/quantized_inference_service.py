"""Quantized GR00T inference service for Phase 5 LIBERO rollout.

This wrapper mirrors NVIDIA's official `scripts/inference_service.py`, but
patches the loaded `Gr00tPolicy` with the fake W4A8 QuantVLA student path used
in Phase 3/4. It intentionally keeps official GR00T source files untouched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from phase3_atm_ohb_forward import (
    collect_attention_stats,
    compute_atm_ohb_scales,
    install_attention_processors,
    scale_summary,
    select_scales,
)
from phase3_fake_quant_forward import config_groups, patch_modules, set_seed
from phase3_gr00t_smoke import _insert_paths
from phase4_real_data_validation import load_observations, parse_indices, sample_indices


DEFAULT_CALIBRATION_INDICES = (
    "115,462,632,1063,1273,1482,1823,2034,"
    "2406,2536,3053,3198,3492,3824,3980,4299"
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


def write_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_calibration_observations(args: argparse.Namespace, data_config: Any) -> list[dict[str, Any]]:
    from gr00t.data.dataset import LeRobotSingleDataset

    dataset = LeRobotSingleDataset(
        dataset_path=str(args.dataset_path),
        modality_configs=data_config.modality_config(),
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
    )
    calibration_indices = sample_indices(
        len(dataset),
        num_observations=args.num_calibration_observations,
        start_index=args.calibration_start_index,
        stride=args.calibration_sample_stride,
        explicit_indices=parse_indices(args.calibration_indices),
    )
    observations = load_observations(dataset, calibration_indices, base_seed=args.calibration_base_seed)
    for item, index in zip(observations, calibration_indices, strict=True):
        item["dataset_index"] = int(index)
    return observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"),
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token")
    parser.add_argument("--config", default="llm_dit_mlp")
    parser.add_argument("--mode", choices=["none", "identity", "atm", "ohb", "atm_ohb"], default="atm_ohb")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--activation-scale", default="absmax")
    parser.add_argument("--activation-percentile", type=float, default=0.999)
    parser.add_argument("--log-clamp", type=float, default=0.3)
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--calibration-indices", default=DEFAULT_CALIBRATION_INDICES)
    parser.add_argument("--num-calibration-observations", type=int, default=16)
    parser.add_argument("--calibration-start-index", type=int, default=0)
    parser.add_argument("--calibration-sample-stride", type=int, default=100)
    parser.add_argument("--calibration-base-seed", type=int, default=360204)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase5_quantized_server_prepare.json"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()

    result: dict[str, Any] = {
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "config": args.config,
        "mode": args.mode,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "activation_scale": args.activation_scale,
        "activation_percentile": args.activation_percentile,
        "log_clamp": args.log_clamp,
        "video_backend": args.video_backend,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
    }

    started = time.time()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    result["model_load_seconds"] = time.time() - started

    calibration_started = time.time()
    calibration_observations = build_calibration_observations(args, data_config)
    result["calibration_dataset_indices"] = [
        int(item["dataset_index"]) for item in calibration_observations
    ]
    result["calibration_base_seed"] = args.calibration_base_seed
    result["calibration_load_seconds"] = time.time() - calibration_started

    groups = config_groups(args.config)
    teacher_stats = {}
    teacher_attention_seconds = 0.0
    if args.mode != "none":
        attention_started = time.time()
        teacher_stats = collect_attention_stats(policy, calibration_observations)
        teacher_attention_seconds = time.time() - attention_started
    result["teacher_attention_calibration_seconds"] = teacher_attention_seconds

    patch_started = time.time()
    _originals, records = patch_modules(
        policy.model,
        groups,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        activation_scale=args.activation_scale,
        activation_percentile=args.activation_percentile,
    )
    result["quantized_modules"] = len(records)
    result["quantized_groups"] = sorted(groups)
    result["patch_seconds"] = time.time() - patch_started

    scales = {}
    student_attention_seconds = 0.0
    if args.mode != "none":
        attention_started = time.time()
        student_stats = collect_attention_stats(policy, calibration_observations)
        student_attention_seconds = time.time() - attention_started
        scales = compute_atm_ohb_scales(teacher_stats, student_stats, log_clamp=args.log_clamp)
        install_attention_processors(policy.model, mode="apply", scales=select_scales(scales, args.mode))
    result["student_attention_calibration_seconds"] = student_attention_seconds
    result["scale_summary"] = scale_summary(scales) if scales else None
    result["prepare_seconds"] = time.time() - started

    if args.prepare_only:
        write_json(args.output_json, result)
        print(json.dumps(result, indent=2))
        return

    write_json(args.output_json, result)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting QuantVLA student server on port {args.port}", flush=True)
    set_seed(args.calibration_base_seed)
    server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
