"""Real LeRobot validation-set action-drift probe for GR00T QuantVLA.

This script replaces the Phase 3 synthetic observations with samples from a
LeRobot LIBERO dataset. It keeps the experiment offline: compare FP16 teacher
and W4A8 fake-quant student action chunks under matched denoising seeds.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase3_atm_ohb_forward import (
    collect_attention_stats,
    compute_atm_ohb_scales,
    install_attention_processors,
    restore_attention_processors,
    scale_summary,
    select_scales,
)
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


def load_object(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"Expected import spec 'module:object', got {spec!r}")
    module_name, object_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_smoothing_alpha(value: str) -> float | None:
    raw = value.strip().lower()
    if raw in {"none", "off", "naive"}:
        return None
    return float(value)


def parse_indices(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def sample_indices(
    dataset_len: int,
    *,
    num_observations: int,
    start_index: int,
    stride: int,
    explicit_indices: list[int] | None,
) -> list[int]:
    if explicit_indices is not None:
        indices = explicit_indices
    else:
        indices = [start_index + i * stride for i in range(num_observations)]
    out: list[int] = []
    seen: set[int] = set()
    for index in indices:
        if index < 0 or index >= dataset_len:
            raise IndexError(f"Dataset index {index} outside [0, {dataset_len})")
        if index not in seen:
            out.append(index)
            seen.add(index)
    return out


def to_numpy_or_list(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return list(value)
    return np.asarray(value)


def split_policy_observation(sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    obs: dict[str, Any] = {}
    action: dict[str, Any] = {}
    for key, value in sample.items():
        converted = to_numpy_or_list(value)
        if key.startswith("action."):
            action[key] = converted
        else:
            obs[key] = converted

    lang_key = "annotation.human.action.task_description"
    if lang_key in obs:
        value = obs[lang_key]
        if isinstance(value, np.ndarray):
            obs[lang_key] = value.reshape(-1).tolist()
        elif not isinstance(value, list):
            obs[lang_key] = [str(value)]
    return obs, action


def load_observations(
    dataset: Any,
    indices: list[int],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for offset, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        obs, action = split_policy_observation(sample)
        observations.append(
            {
                "dataset_index": int(dataset_index),
                "seed": int(base_seed + offset),
                "obs": obs,
                "ground_truth_action": action,
            }
        )
    return observations


def compare_to_ground_truth(
    teacher_actions: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for teacher, item in zip(teacher_actions, observations, strict=True):
        labels = item.get("ground_truth_action") or {}
        if not labels:
            return None
        metrics = compare_actions(labels, teacher)
        metrics.update({"dataset_index": item["dataset_index"], "seed": item["seed"]})
        rows.append(metrics)
    return aggregate_metrics(rows)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# QuantVLA Phase 4 Real-Data Validation",
        "",
        "Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.",
        "",
        "## Run",
        "",
        f"- Dataset: `{result['dataset_path']}`",
        f"- Dataset length: `{result['dataset_length']}`",
        f"- Evaluation dataset indices: `{', '.join(str(i) for i in result['dataset_indices'])}`",
        f"- Calibration dataset indices: `{', '.join(str(i) for i in result['calibration_dataset_indices'])}`",
        f"- Calibration/evaluation overlap: `{', '.join(str(i) for i in result['calibration_eval_overlap']) or 'none'}`",
        f"- Model: `{result['model_path']}`",
        f"- Data config: `{result['data_config']}`",
        f"- Video backend: `{result['video_backend']}`",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Smoothing: `{result['smoothing']}`",
        f"- Activation scale modes: `{', '.join(result['activation_scale_modes'])}`",
        f"- ATM/OHB modes: `{', '.join(result['atm_ohb_modes'])}`",
        "",
        "## Teacher/Student Drift",
        "",
        "| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["config_results"]:
        s = row["summary"]
        lines.append(
            "| {config} | {mode} | {scale} | {modules} | {nmse:.6g} | {nmse_max:.6g} | {rrmse:.6g} | {cos:.6g} | {mad:.6g} |".format(
                config=row["config"],
                mode=row["mode"],
                scale=row["activation_scale"],
                modules=row["quantized_modules"],
                nmse=s["nmse"]["mean"],
                nmse_max=s["nmse"]["max"],
                rrmse=s["relative_rmse"]["mean"],
                cos=s["cosine"]["mean"],
                mad=s["max_abs_diff"]["max"],
            )
        )

    gt_summary = result.get("teacher_vs_ground_truth")
    if gt_summary is not None:
        lines.extend(
            [
                "",
                "## Teacher vs Demonstration",
                "",
                "This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.",
                "",
                f"- NMSE mean: `{gt_summary['nmse']['mean']:.6g}`",
                f"- Relative RMSE mean: `{gt_summary['relative_rmse']['mean']:.6g}`",
                f"- Cosine mean: `{gt_summary['cosine']['mean']:.6g}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.",
            "- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.",
            "- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on calibration samples.",
            "- `identity` installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`; it measures processor replacement drift without ATM/OHB rescaling.",
            "- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument(
        "--data-config",
        default="examples.Libero.custom_data_config:LiberoDataConfig",
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--num-observations", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--indices")
    parser.add_argument("--num-calibration-observations", type=int)
    parser.add_argument("--calibration-start-index", type=int)
    parser.add_argument("--calibration-sample-stride", type=int)
    parser.add_argument("--calibration-indices")
    parser.add_argument("--calibration-base-seed", type=int)
    parser.add_argument("--configs", default="llm_only,dit_mlp_only,llm_dit_mlp")
    parser.add_argument("--activation-scale-modes", default="absmax")
    parser.add_argument("--activation-percentile", type=float, default=0.999)
    parser.add_argument("--smoothing-alpha", default="none")
    parser.add_argument("--atm-ohb-modes", default="none,ohb,atm,atm_ohb")
    parser.add_argument("--log-clamp", type=float, default=0.3)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=260203)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("toy_quantvla/results/phase4_real_data_validation.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/phase4_real_data_validation.md"),
    )
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()
    configs = parse_csv(args.configs)
    activation_scales = parse_csv(args.activation_scale_modes)
    atm_ohb_modes = parse_csv(args.atm_ohb_modes)
    smoothing_alpha = parse_smoothing_alpha(args.smoothing_alpha)

    if "none" not in atm_ohb_modes:
        atm_ohb_modes.insert(0, "none")

    dataset_started = time.time()
    dataset = LeRobotSingleDataset(
        dataset_path=str(args.dataset_path),
        modality_configs=data_config.modality_config(),
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
    )
    dataset_seconds = time.time() - dataset_started

    indices = sample_indices(
        len(dataset),
        num_observations=args.num_observations,
        start_index=args.start_index,
        stride=args.sample_stride,
        explicit_indices=parse_indices(args.indices),
    )
    observations = load_observations(dataset, indices, base_seed=args.base_seed)

    calibration_requested = any(
        value is not None
        for value in [
            args.num_calibration_observations,
            args.calibration_start_index,
            args.calibration_sample_stride,
            args.calibration_indices,
            args.calibration_base_seed,
        ]
    )
    if calibration_requested:
        calibration_indices = sample_indices(
            len(dataset),
            num_observations=args.num_calibration_observations or args.num_observations,
            start_index=(
                args.calibration_start_index
                if args.calibration_start_index is not None
                else args.start_index
            ),
            stride=(
                args.calibration_sample_stride
                if args.calibration_sample_stride is not None
                else args.sample_stride
            ),
            explicit_indices=parse_indices(args.calibration_indices),
        )
        calibration_base_seed = (
            args.calibration_base_seed
            if args.calibration_base_seed is not None
            else args.base_seed + 100000
        )
        calibration_observations = load_observations(
            dataset, calibration_indices, base_seed=calibration_base_seed
        )
    else:
        calibration_indices = list(indices)
        calibration_base_seed = args.base_seed
        calibration_observations = observations

    load_started = time.time()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    model_load_seconds = time.time() - load_started

    teacher_actions: list[dict[str, Any]] = []
    teacher_started = time.time()
    for item in observations:
        set_seed(int(item["seed"]))
        teacher_actions.append(policy.get_action(item["obs"]))
    teacher_seconds = time.time() - teacher_started

    need_attention_processor = any(mode != "none" for mode in atm_ohb_modes)
    attention_teacher_calibration_seconds = 0.0
    teacher_stats = {}
    if need_attention_processor:
        attention_started = time.time()
        teacher_stats = collect_attention_stats(policy, calibration_observations)
        attention_teacher_calibration_seconds = time.time() - attention_started

    smoothing_groups = set().union(*(config_groups(config) for config in configs))
    smoothing_groups.discard("dit_attention_excluded")
    smooth_scales = None
    calibration_seconds = 0.0
    if smoothing_alpha is not None:
        calibration_started = time.time()
        activation_max = collect_activation_max(policy, calibration_observations, smoothing_groups)
        smooth_scales = make_smooth_scales(policy.model, activation_max, smoothing_alpha)
        calibration_seconds = time.time() - calibration_started

    config_results: list[dict[str, Any]] = []
    for activation_scale in activation_scales:
        for config in configs:
            groups = config_groups(config)
            originals, _records = patch_modules(
                policy.model,
                groups,
                weight_bits=args.weight_bits,
                activation_bits=args.activation_bits,
                activation_scale=activation_scale,
                activation_percentile=args.activation_percentile,
                smooth_scales=smooth_scales,
                smooth_alpha=smoothing_alpha,
            )
            try:
                scales: dict[str, dict[str, float]] = {}
                student_attention_calibration_seconds = 0.0
                if need_attention_processor:
                    attention_started = time.time()
                    student_stats = collect_attention_stats(policy, calibration_observations)
                    student_attention_calibration_seconds = time.time() - attention_started
                    scales = compute_atm_ohb_scales(
                        teacher_stats, student_stats, log_clamp=args.log_clamp
                    )
                for mode in atm_ohb_modes:
                    attn_originals: dict[str, Any] = {}
                    if mode != "none":
                        attn_originals, _ = install_attention_processors(
                            policy.model, mode="apply", scales=select_scales(scales, mode)
                        )
                    rows: list[dict[str, Any]] = []
                    started = time.time()
                    try:
                        for idx, item in enumerate(observations):
                            set_seed(int(item["seed"]))
                            student = policy.get_action(item["obs"])
                            metrics = compare_actions(teacher_actions[idx], student)
                            metrics.update(
                                {
                                    "dataset_index": item["dataset_index"],
                                    "seed": item["seed"],
                                }
                            )
                            rows.append(metrics)
                    finally:
                        if attn_originals:
                            restore_attention_processors(policy.model, attn_originals)
                    config_results.append(
                        {
                            "config": config,
                            "mode": mode,
                            "activation_scale": activation_scale,
                            "groups": sorted(groups),
                            "quantized_modules": len(originals),
                            "seconds": time.time() - started,
                            "student_attention_calibration_seconds": student_attention_calibration_seconds,
                            "summary": aggregate_metrics(rows),
                            "per_observation": rows,
                            "scale_summary": scale_summary(scales) if scales else None,
                            "scales": scales,
                        }
                    )
            finally:
                restore_modules(policy.model, originals)

    result = {
        "dataset_path": str(args.dataset_path),
        "dataset_length": len(dataset),
        "dataset_indices": indices,
        "calibration_dataset_indices": calibration_indices,
        "calibration_uses_evaluation_observations": not calibration_requested,
        "calibration_eval_overlap": sorted(set(indices).intersection(calibration_indices)),
        "dataset_seconds": dataset_seconds,
        "model_path": str(args.model_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "video_backend": args.video_backend,
        "denoising_steps": args.denoising_steps,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "activation_percentile": args.activation_percentile,
        "activation_scale_modes": activation_scales,
        "configs": configs,
        "atm_ohb_modes": atm_ohb_modes,
        "log_clamp": args.log_clamp,
        "smoothing": "none" if smoothing_alpha is None else f"sq_alpha_{smoothing_alpha:g}",
        "base_seed": args.base_seed,
        "calibration_base_seed": calibration_base_seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "model_load_seconds": model_load_seconds,
        "teacher_seconds": teacher_seconds,
        "attention_teacher_calibration_seconds": attention_teacher_calibration_seconds,
        "calibration_seconds": calibration_seconds,
        "teacher_vs_ground_truth": compare_to_ground_truth(teacher_actions, observations),
        "config_results": config_results,
    }
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        result["cuda_memory_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
        result["cuda_memory_reserved_mib"] = torch.cuda.memory_reserved() / 1024**2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "dataset_length": result["dataset_length"],
        "dataset_indices": result["dataset_indices"],
        "calibration_dataset_indices": result["calibration_dataset_indices"],
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "config_results": [
            {
                "config": row["config"],
                "mode": row["mode"],
                "activation_scale": row["activation_scale"],
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
