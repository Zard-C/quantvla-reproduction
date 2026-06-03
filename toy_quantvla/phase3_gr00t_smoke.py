"""Phase 3 GR00T loading and synthetic LIBERO smoke test.

This script intentionally tests the official GR00T policy path with a local
checkpoint, but keeps the input synthetic and tiny. It is not an evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _insert_paths(isaac_root: Path, compat_stubs: Path | None) -> None:
    if compat_stubs is not None:
        sys.path.insert(0, str(compat_stubs))
    sys.path.insert(0, str(isaac_root))


def _read_metadata(model_path: Path, embodiment_tag: str) -> dict[str, Any]:
    metadata_path = model_path / "experiment_cfg" / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if embodiment_tag not in metadata:
        raise KeyError(f"{embodiment_tag!r} not found in {metadata_path}")
    return metadata[embodiment_tag]


def _mean_array(stats: dict[str, Any], group: str, key: str) -> np.ndarray:
    values = stats[group][key]["mean"]
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def build_synthetic_libero_observation(model_path: Path, embodiment_tag: str) -> dict[str, Any]:
    metadata = _read_metadata(model_path, embodiment_tag)
    stats = metadata["statistics"]
    obs: dict[str, Any] = {
        "video.image": np.zeros((1, 256, 256, 3), dtype=np.uint8),
        "video.wrist_image": np.zeros((1, 256, 256, 3), dtype=np.uint8),
        "annotation.human.action.task_description": ["pick up the object"],
    }
    for key in ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]:
        obs[f"state.{key}"] = _mean_array(stats, "state", key)
    return obs


def summarize_action(action: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in action.items():
        arr = np.asarray(value)
        summary[key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--mode", choices=["import", "load", "forward"], default="load")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _insert_paths(args.isaac_root, args.compat_stubs)

    started = time.time()
    import torch
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    result: dict[str, Any] = {
        "mode": args.mode,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "model_path": str(args.model_path),
        "embodiment_tag": args.embodiment_tag,
        "import_seconds": time.time() - started,
    }

    if args.mode == "import":
        print(json.dumps(result, indent=2))
        if args.output_json:
            args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return

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
    result.update(
        {
            "load_seconds": time.time() - load_started,
            "policy_denoising_steps": int(policy.denoising_steps),
            "action_horizon": int(policy.model.action_head.config.action_horizon),
            "action_dim": int(policy.model.action_head.config.action_dim),
            "video_delta_indices": policy.video_delta_indices.tolist(),
            "state_delta_indices": (
                policy.state_delta_indices.tolist()
                if policy.state_delta_indices is not None
                else None
            ),
        }
    )
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        result["cuda_memory_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
        result["cuda_memory_reserved_mib"] = torch.cuda.memory_reserved() / 1024**2

    if args.mode == "forward":
        obs = build_synthetic_libero_observation(args.model_path, args.embodiment_tag)
        forward_started = time.time()
        action = policy.get_action(obs)
        result["forward_seconds"] = time.time() - forward_started
        result["action_summary"] = summarize_action(action)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            result["cuda_memory_allocated_after_forward_mib"] = (
                torch.cuda.memory_allocated() / 1024**2
            )
            result["cuda_memory_reserved_after_forward_mib"] = torch.cuda.memory_reserved() / 1024**2

    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
