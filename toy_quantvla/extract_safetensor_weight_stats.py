from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from distributions import WEIGHT_STATS_SCHEMA, summarize_weight

MODEL_DIR = Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain")
INDEX = MODEL_DIR / "model.safetensors.index.json"
OUT = Path("toy_quantvla/results/weight_stats.json")

LLM_RE = re.compile(r"^backbone\.eagle_model\.language_model\.model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$")
DIT_MLP_RE = re.compile(r"^action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)\.weight$")
DIT_ATTN_RE = re.compile(r"^action_head\.model\.transformer_blocks\.\d+\.attn1\.(?:to_q|to_k|to_v|to_out\.0)\.weight$")

GROUPS = [
    ("llm_selected", LLM_RE),
    ("dit_mlp_selected", DIT_MLP_RE),
    ("dit_attention_excluded", DIT_ATTN_RE),
]


def group_for(name: str) -> str | None:
    for group, pattern in GROUPS:
        if pattern.match(name):
            return group
    return None


def main() -> None:
    index = json.loads(INDEX.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    selected = {name: group_for(name) for name in weight_map}
    selected = {name: group for name, group in selected.items() if group is not None}

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in selected:
        by_file[weight_map[name]].append(name)

    layers = []
    group_counts = defaultdict(int)
    for shard, names in sorted(by_file.items()):
        shard_path = MODEL_DIR / shard
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name)
                if not torch.is_tensor(tensor) or tensor.ndim != 2:
                    continue
                group = selected[name]
                entry = {"name": name, "group": group, "shard": shard}
                entry.update(summarize_weight(tensor))
                layers.append(entry)
                group_counts[group] += 1

    group_summaries = {}
    for group in sorted(group_counts):
        items = [x for x in layers if x["group"] == group]
        for key in ["std", "max_abs", "p99_abs", "p99_9_abs", "tail_ratio_max_over_p99_9", "per_output_channel_max_abs_mean", "per_output_channel_max_abs_p99"]:
            vals = torch.tensor([float(x[key]) for x in items])
            group_summaries.setdefault(group, {})[key] = {
                "mean": float(vals.mean().item()),
                "median": float(vals.median().item()),
                "max": float(vals.max().item()),
            }

    output = {
        "model_dir": str(MODEL_DIR),
        "index": str(INDEX),
        "schema": WEIGHT_STATS_SCHEMA,
        "group_counts": dict(group_counts),
        "group_summaries": group_summaries,
        "layers": layers,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT), "group_counts": dict(group_counts)}, indent=2))
    for group, summary in group_summaries.items():
        print(group, "tail_ratio_mean", summary["tail_ratio_max_over_p99_9"]["mean"], "max_abs_max", summary["max_abs"]["max"])


if __name__ == "__main__":
    main()
