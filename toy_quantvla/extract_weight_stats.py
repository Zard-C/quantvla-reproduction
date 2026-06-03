from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from distributions import WEIGHT_STATS_SCHEMA, summarize_weight


def iter_state_dict(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported checkpoint object type: {type(obj)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump small JSON weight statistics for a checkpoint without running inference."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("toy_quantvla/results/weight_stats.json"))
    parser.add_argument("--max-layers", type=int, default=64)
    parser.add_argument("--name-filter", type=str, default="")
    args = parser.parse_args()

    state = iter_state_dict(args.checkpoint)
    layers = []
    for name, tensor in state.items():
        if args.name_filter and args.name_filter not in name:
            continue
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            continue
        entry = {"name": name}
        entry.update(summarize_weight(tensor))
        layers.append(entry)
        if len(layers) >= args.max_layers:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"layers": layers}, indent=2), encoding="utf-8")
    schema_path = args.output.parent / "weight_stats_schema.json"
    schema_path.write_text(json.dumps(WEIGHT_STATS_SCHEMA, indent=2), encoding="utf-8")
    print(f"wrote {args.output} with {len(layers)} layer summaries")


if __name__ == "__main__":
    main()
