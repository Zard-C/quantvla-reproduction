"""Official FP16 GR00T inference service with server-side timing output."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from fp16_linear_profiler import (
    module_results,
    parse_name_contains,
    patch_timed_fp16_modules,
    reset_module_stats,
)
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from timing_utils import TimedPolicyWrapper


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--api-token")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_fp16_timed_server_prepare.json"))
    parser.add_argument("--server-latency-json", type=Path, help="Optional server-side get_action latency JSON.")
    parser.add_argument("--server-latency-flush-every", type=int, default=0)
    parser.add_argument("--profile-linear-modules", action="store_true")
    parser.add_argument("--profile-scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--profile-max-modules", type=int, default=0)
    parser.add_argument("--profile-name-contains")
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
        "boundary": "Official FP16 GR00T inference server with server-side get_action timing",
        "model_path": str(args.model_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "profile_linear_modules": bool(args.profile_linear_modules),
        "profile_scope": args.profile_scope,
        "profile_scope_description": scope_description(args.profile_scope),
        "profile_max_modules": int(args.profile_max_modules),
        "profile_name_contains": parse_name_contains(args.profile_name_contains),
    }

    started = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    result["model_load_seconds"] = time.perf_counter() - started
    result["prepare_seconds"] = result["model_load_seconds"]

    profiled_modules = {}
    if args.profile_linear_modules:
        patch_started = time.perf_counter()
        records, profiled_modules = patch_timed_fp16_modules(
            policy.model,
            args.profile_scope,
            max_modules=args.profile_max_modules,
            name_contains=parse_name_contains(args.profile_name_contains),
            profile=True,
        )
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        result["profile_patch_seconds"] = time.perf_counter() - patch_started
        result["profiled_modules"] = len(records)
        result["profile_patch_records"] = records
        result["profile_module_results_after_prepare"] = module_results(profiled_modules)
        reset_module_stats(profiled_modules)

    write_json(args.output_json, result)

    if args.prepare_only:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting timed FP16 server on port {args.port}", flush=True)
    set_seed(20260608)
    timed_policy = TimedPolicyWrapper(
        policy,
        output_json=args.server_latency_json,
        label="fp16_official",
        flush_every=args.server_latency_flush_every,
        extra_summary=(lambda: {"profile_module_results": module_results(profiled_modules)})
        if args.profile_linear_modules
        else None,
    )
    server = RobotInferenceServer(timed_policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
