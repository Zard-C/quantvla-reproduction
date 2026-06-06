"""GR00T inference service patched with CUTLASS dense Linear modules."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

from cutlass_dense_linear import CutlassDenseLinearCache
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase8_cutlass_dense_forward_smoke import patch_cutlass_dense_modules, parse_tile, summarize_patch


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
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--tile-shape-mnk", type=parse_tile, default=(64, 128, 64))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase8_cutlass_dense_server_prepare.json"))
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
        "boundary": "CUTLASS dense fp16/bf16 inference server; validates backend op integration, not FP4 quantization",
        "model_path": str(args.model_path),
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "denoising_steps": args.denoising_steps,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "cutlass_root": str(args.cutlass_root),
        "tile_shape_mnk": list(args.tile_shape_mnk),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
    }

    CutlassDenseLinearCache.clear()
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

    patch_started = time.time()
    records = patch_cutlass_dense_modules(
        policy.model,
        args.scope,
        cutlass_root=args.cutlass_root,
        tile_shape_mnk=args.tile_shape_mnk,
    )
    result["patch_seconds"] = time.time() - patch_started
    result["patched_modules"] = len(records)
    result["patch_summary"] = summarize_patch(records)
    result["prepare_seconds"] = time.time() - started
    write_json(args.output_json, result)

    if args.prepare_only:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting CUTLASS dense server on port {args.port}", flush=True)
    set_seed(20260606)
    server = RobotInferenceServer(policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
