"""Inspect GR00T action-head module names for compile-scope experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from phase3_gr00t_smoke import _insert_paths


def module_depth(name: str) -> int:
    return 0 if not name else name.count(".") + 1


def include_module(name: str, cls_name: str, max_depth: int) -> bool:
    lowered = f"{name} {cls_name}".lower()
    if module_depth(name) <= max_depth:
        return True
    return any(token in lowered for token in ["block", "mlp", "attn", "attention", "transformer", "dit"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import importlib
    import torch
    from gr00t.model.policy import Gr00tPolicy

    module_name, object_name = args.data_config.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for part in object_name.split("."):
        obj = getattr(obj, part)
    data_config = obj()
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

    action_head = policy.model.action_head
    model = action_head.model
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if include_module(name, cls_name, args.max_depth):
            rows.append(
                {
                    "name": name,
                    "class": cls_name,
                    "depth": module_depth(name),
                    "children": len(list(module.children())),
                    "parameters": int(sum(p.numel() for p in module.parameters(recurse=False))),
                }
            )

    result = {
        "action_head_class": type(action_head).__name__,
        "action_head_model_class": type(model).__name__,
        "rows": rows,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
