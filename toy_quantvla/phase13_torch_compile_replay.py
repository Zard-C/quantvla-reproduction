"""Fixed-observation replay diagnostics for torch.compile speed experiments.

This script isolates RNG and numerical drift from closed-loop simulator
trajectory changes.  It runs GR00T on a fixed set of real observations:

1. eager/eager with matched seeds as an A/A determinism check;
2. eager/compiled with the same observations and seeds.

The result is an offline action drift and latency report.  It is not a LIBERO
success-rate benchmark.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from phase3_fake_quant_forward import action_to_vector, aggregate_metrics, compare_actions, set_seed
from phase3_gr00t_smoke import _insert_paths
from phase4_real_data_validation import load_observations, parse_indices, sample_indices
from timing_utils import summarize_float


TORCH_COMPILE_TARGETS = (
    "none",
    "backbone",
    "action_head_model",
    "backbone_action_head_model",
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


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def cuda_memory(device: str) -> dict[str, int]:
    import torch

    if not (torch.device(device).type == "cuda" and torch.cuda.is_available()):
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def maybe_override_task_description(observations: list[dict[str, Any]], description: str | None) -> None:
    if not description:
        return
    for item in observations:
        obs = dict(item["obs"])
        obs["annotation.human.action.task_description"] = [description]
        item["obs"] = obs


def action_summary(action: dict[str, Any]) -> dict[str, Any]:
    vector = action_to_vector(action)
    return {
        "numel": int(vector.size),
        "mean": float(vector.mean()),
        "std": float(vector.std()),
        "min": float(vector.min()),
        "max": float(vector.max()),
    }


def run_one_action(policy: Any, obs: dict[str, Any], *, seed: int, device: str) -> tuple[dict[str, Any], float]:
    set_seed(int(seed))
    started = time.perf_counter()
    action = policy.get_action(obs)
    synchronize(device)
    seconds = time.perf_counter() - started
    return action, float(seconds)


def compile_policy_targets(policy: Any, args: argparse.Namespace, torch_module: Any) -> dict[str, Any]:
    target = str(args.torch_compile_target)
    if target == "none":
        return {"enabled": False, "target": target, "compiled_modules": []}

    compile_kwargs: dict[str, Any] = {
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
    }
    if args.torch_compile_dynamic is not None:
        compile_kwargs["dynamic"] = bool(args.torch_compile_dynamic)

    compiled_modules: list[str] = []
    started = time.perf_counter()
    if target in {"backbone", "backbone_action_head_model"}:
        policy.model.backbone = torch_module.compile(policy.model.backbone, **compile_kwargs)
        compiled_modules.append("policy.model.backbone")
    if target in {"action_head_model", "backbone_action_head_model"}:
        policy.model.action_head.model = torch_module.compile(
            policy.model.action_head.model,
            **compile_kwargs,
        )
        compiled_modules.append("policy.model.action_head.model")

    return {
        "enabled": True,
        "target": target,
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
        "dynamic": args.torch_compile_dynamic,
        "compiled_modules": compiled_modules,
        "wrap_seconds": time.perf_counter() - started,
    }


def run_eager_aa(policy: Any, observations: list[dict[str, Any]], *, device: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    teacher_actions: list[dict[str, Any]] = []
    first_seconds: list[float] = []
    second_seconds: list[float] = []

    for item in observations:
        action_a, seconds_a = run_one_action(
            policy,
            item["obs"],
            seed=int(item["seed"]),
            device=device,
        )
        action_b, seconds_b = run_one_action(
            policy,
            item["obs"],
            seed=int(item["seed"]),
            device=device,
        )
        metrics = compare_actions(action_a, action_b)
        row = {
            "dataset_index": int(item["dataset_index"]),
            "seed": int(item["seed"]),
            "first_seconds": float(seconds_a),
            "second_seconds": float(seconds_b),
            "action_size": int(action_to_vector(action_a).size),
            "first_action_summary": action_summary(action_a),
            **metrics,
        }
        rows.append(row)
        teacher_actions.append(action_a)
        first_seconds.append(float(seconds_a))
        second_seconds.append(float(seconds_b))

    return teacher_actions, {
        "metrics": aggregate_metrics(rows),
        "first_get_action_seconds": summarize_float(first_seconds),
        "second_get_action_seconds": summarize_float(second_seconds),
        "per_observation": rows,
    }


def run_compiled_replay(
    policy: Any,
    observations: list[dict[str, Any]],
    teacher_actions: list[dict[str, Any]],
    *,
    device: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seconds: list[float] = []

    for item, teacher_action in zip(observations, teacher_actions, strict=True):
        student_action, action_seconds = run_one_action(
            policy,
            item["obs"],
            seed=int(item["seed"]),
            device=device,
        )
        metrics = compare_actions(teacher_action, student_action)
        row = {
            "dataset_index": int(item["dataset_index"]),
            "seed": int(item["seed"]),
            "compiled_seconds": float(action_seconds),
            "action_size": int(action_to_vector(teacher_action).size),
            **metrics,
        }
        rows.append(row)
        seconds.append(float(action_seconds))

    return {
        "metrics": aggregate_metrics(rows),
        "compiled_get_action_seconds": summarize_float(seconds),
        "per_observation": rows,
    }


def run_compile_prewarm(policy: Any, observations: list[dict[str, Any]], *, device: str) -> dict[str, Any]:
    seconds: list[float] = []
    for item in observations:
        _action, action_seconds = run_one_action(
            policy,
            item["obs"],
            seed=int(item["seed"]),
            device=device,
        )
        seconds.append(float(action_seconds))
    return {"get_action_seconds": summarize_float(seconds), "memory": cuda_memory(device)}


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aa = result["eager_aa"]["metrics"]
    ab = result["compiled_replay"]["metrics"]
    eager = result["eager_aa"]["second_get_action_seconds"]
    compiled = result["compiled_replay"]["compiled_get_action_seconds"]
    lines = [
        "# Phase 13 torch.compile Fixed-Observation Replay",
        "",
        "Boundary: offline replay on fixed real observations. This isolates RNG and numerical drift from closed-loop simulator trajectory changes.",
        "",
        "## Run",
        "",
        f"- Dataset: `{result['dataset_path']}`",
        f"- Dataset indices: `{', '.join(str(i) for i in result['dataset_indices'])}`",
        f"- Base seed: `{result['base_seed']}`",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Compile target: `{result['torch_compile']['target']}`",
        f"- Compile backend: `{result['torch_compile'].get('backend', 'none')}`",
        f"- Compile mode: `{result['torch_compile'].get('mode', 'none')}`",
        "",
        "## Determinism Check",
        "",
        "| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |",
        "|---|---:|---:|---:|---:|",
        "| eager vs eager, same seed | {rrmse_mean:.6g} | {rrmse_max:.6g} | {cos_mean:.9g} | {max_abs:.6g} |".format(
            rrmse_mean=aa["relative_rmse"]["mean"],
            rrmse_max=aa["relative_rmse"]["max"],
            cos_mean=aa["cosine"]["mean"],
            max_abs=aa["max_abs_diff"]["max"],
        ),
        "",
        "## Compile Drift",
        "",
        "| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |",
        "|---|---:|---:|---:|---:|",
        "| eager vs compiled, same observation and seed | {rrmse_mean:.6g} | {rrmse_max:.6g} | {cos_mean:.9g} | {max_abs:.6g} |".format(
            rrmse_mean=ab["relative_rmse"]["mean"],
            rrmse_max=ab["relative_rmse"]["max"],
            cos_mean=ab["cosine"]["mean"],
            max_abs=ab["max_abs_diff"]["max"],
        ),
        "",
        "## Latency",
        "",
        "| path | count | mean | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|---:|",
        "| eager warm | {count} | {mean:.6g}s | {p50:.6g}s | {p90:.6g}s | {p99:.6g}s |".format(
            **eager
        ),
        "| compiled warm | {count} | {mean:.6g}s | {p50:.6g}s | {p90:.6g}s | {p99:.6g}s |".format(
            **compiled
        ),
        "",
        "## Notes",
        "",
        "- If eager/eager is near zero, matched request seeds are working.",
        "- Any eager/compiled drift is a compile backend numerical effect, not simulator feedback.",
        "- Rollout success-rate comparisons should be interpreted only after this fixed-observation gate.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-observations", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--indices", default="115,462,632,1063")
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--override-task-description")
    parser.add_argument("--torch-compile-target", choices=TORCH_COMPILE_TARGETS, default="action_head_model")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", choices=["true", "false"])
    parser.add_argument("--compile-prewarm", choices=["none", "same_observations"], default="same_observations")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase13_torch_compile_replay.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase13_torch_compile_replay.md"))
    args = parser.parse_args()
    if args.torch_compile_dynamic is not None:
        args.torch_compile_dynamic = args.torch_compile_dynamic == "true"

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from gr00t.data.dataset import LeRobotSingleDataset
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()

    dataset_started = time.perf_counter()
    dataset = LeRobotSingleDataset(
        dataset_path=str(args.dataset_path),
        modality_configs=data_config.modality_config(),
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
    )
    dataset_seconds = time.perf_counter() - dataset_started
    indices = sample_indices(
        len(dataset),
        num_observations=args.num_observations,
        start_index=args.start_index,
        stride=args.sample_stride,
        explicit_indices=parse_indices(args.indices),
    )
    observations = load_observations(dataset, indices, base_seed=args.base_seed)
    maybe_override_task_description(observations, args.override_task_description)

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
    model_load_seconds = time.perf_counter() - load_started

    teacher_actions, eager_aa = run_eager_aa(policy, observations, device=args.device)
    compile_info = compile_policy_targets(policy, args, torch)
    if args.compile_prewarm == "same_observations":
        compile_prewarm = run_compile_prewarm(policy, observations, device=args.device)
    else:
        compile_prewarm = {"get_action_seconds": summarize_float([]), "memory": cuda_memory(args.device)}
    compiled_replay = run_compiled_replay(
        policy,
        observations,
        teacher_actions,
        device=args.device,
    )

    result = {
        "boundary": "Fixed-observation torch.compile replay; not a simulator rollout",
        "model_path": str(args.model_path),
        "dataset_path": str(args.dataset_path),
        "dataset_length": int(len(dataset)),
        "dataset_indices": [int(index) for index in indices],
        "dataset_seconds": float(dataset_seconds),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "video_backend": args.video_backend,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "base_seed": int(args.base_seed),
        "override_task_description": args.override_task_description,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "model_load_seconds": float(model_load_seconds),
        "model_load_memory": cuda_memory(args.device),
        "torch_compile": compile_info,
        "compile_prewarm": compile_prewarm,
        "eager_aa": eager_aa,
        "compiled_replay": compiled_replay,
        "final_memory": cuda_memory(args.device),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "dataset_indices": result["dataset_indices"],
        "torch_compile": result["torch_compile"],
        "eager_aa": {
            "relative_rmse_mean": eager_aa["metrics"]["relative_rmse"]["mean"],
            "relative_rmse_max": eager_aa["metrics"]["relative_rmse"]["max"],
            "cosine_mean": eager_aa["metrics"]["cosine"]["mean"],
            "max_abs_diff": eager_aa["metrics"]["max_abs_diff"]["max"],
            "second_get_action_seconds": eager_aa["second_get_action_seconds"],
        },
        "compiled_replay": {
            "relative_rmse_mean": compiled_replay["metrics"]["relative_rmse"]["mean"],
            "relative_rmse_max": compiled_replay["metrics"]["relative_rmse"]["max"],
            "cosine_mean": compiled_replay["metrics"]["cosine"]["mean"],
            "max_abs_diff": compiled_replay["metrics"]["max_abs_diff"]["max"],
            "compiled_get_action_seconds": compiled_replay["compiled_get_action_seconds"],
        },
        "compile_prewarm": result["compile_prewarm"]["get_action_seconds"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
