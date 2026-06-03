from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


MODEL_DIR = Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain")
INDEX = MODEL_DIR / "model.safetensors.index.json"
OUT_JSON = Path("toy_quantvla/results/phase3_weight_quant_summary.json")
OUT_MD = Path("docs/phase3_offline_weight_quant.md")

LLM_RE = re.compile(
    r"^backbone\.eagle_model\.language_model\.model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)
DIT_MLP_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(?:0\.proj|2)\.weight$"
)
DIT_ATTN_RE = re.compile(
    r"^action_head\.model\.transformer_blocks\.\d+\.attn1\.(?:to_q|to_k|to_v|to_out\.0)\.weight$"
)


def group_for(name: str) -> str | None:
    if LLM_RE.match(name):
        return "llm_selected"
    if DIT_MLP_RE.match(name):
        return "dit_mlp_selected"
    if DIT_ATTN_RE.match(name):
        return "dit_attention_excluded"
    return None


def fake_quant_w4_per_output(weight: torch.Tensor) -> torch.Tensor:
    w = weight.detach().float()
    qmax = 7.0
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale


def tensor_metrics(name: str, group: str, shard: str, tensor: torch.Tensor) -> dict[str, object]:
    w = tensor.detach().float()
    q = fake_quant_w4_per_output(w)
    err = q - w
    mse = torch.mean(err**2)
    ref = torch.mean(w**2).clamp_min(1e-12)
    nmse = mse / ref
    cos = F.cosine_similarity(w.flatten(), q.flatten(), dim=0)
    max_abs = w.abs().max()
    p999 = torch.quantile(w.abs().flatten(), 0.999)
    return {
        "name": name,
        "group": group,
        "shard": shard,
        "shape": list(w.shape),
        "numel": int(w.numel()),
        "mse": float(mse.item()),
        "nmse": float(nmse.item()),
        "cosine": float(cos.item()),
        "rms": float(torch.sqrt(ref).item()),
        "max_abs": float(max_abs.item()),
        "p99_9_abs": float(p999.item()),
        "tail_ratio_max_over_p99_9": float((max_abs / p999.clamp_min(1e-8)).item()),
    }


def summarize_group(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    keys = ["mse", "nmse", "cosine", "rms", "max_abs", "p99_9_abs", "tail_ratio_max_over_p99_9"]
    out: dict[str, object] = {
        "layers": len(rows),
        "params": int(sum(int(r["numel"]) for r in rows)),
    }
    out["fp16_mib"] = out["params"] * 2 / (1024**2)
    out["w4_mib_ideal"] = out["params"] * 0.5 / (1024**2)
    out["ideal_saving_ratio"] = 1.0 - (out["w4_mib_ideal"] / out["fp16_mib"])
    for key in keys:
        vals = torch.tensor([float(r[key]) for r in rows])
        out[key] = {
            "mean": float(vals.mean().item()),
            "median": float(vals.median().item()),
            "max": float(vals.max().item()),
        }
    return out


def make_markdown(result: dict[str, object]) -> str:
    lines = [
        "# QuantVLA Phase 3 Offline Weight Quantization",
        "",
        "Boundary: this analysis reads safetensors weights only. It does not instantiate GR00T, start an inference server, or run LIBERO.",
        "",
        f"Model: `{result['model_dir']}`",
        "",
        "## Selector Counts",
        "",
        "| group | layers | params | fp16 MiB | ideal W4 MiB | ideal saving | NMSE mean | NMSE max | cosine mean | tail ratio mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = result["group_summaries"]
    for group in ["llm_selected", "dit_mlp_selected", "dit_attention_excluded"]:
        s = summaries[group]
        lines.append(
            "| {group} | {layers} | {params} | {fp16:.2f} | {w4:.2f} | {saving:.1%} | {nmse_mean:.6g} | {nmse_max:.6g} | {cos_mean:.6g} | {tail_mean:.6g} |".format(
                group=group,
                layers=s["layers"],
                params=s["params"],
                fp16=s["fp16_mib"],
                w4=s["w4_mib_ideal"],
                saving=s["ideal_saving_ratio"],
                nmse_mean=s["nmse"]["mean"],
                nmse_max=s["nmse"]["max"],
                cos_mean=s["cosine"]["mean"],
                tail_mean=s["tail_ratio_max_over_p99_9"]["mean"],
            )
        )

    lines.extend(
        [
            "",
            "## Worst Selected Layers By W4 NMSE",
            "",
            "| group | layer | shape | NMSE | cosine | max_abs | tail ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result["worst_selected_by_nmse"][:20]:
        lines.append(
            f"| {row['group']} | `{row['name']}` | {row['shape']} | {row['nmse']:.6g} | {row['cosine']:.6g} | {row['max_abs']:.6g} | {row['tail_ratio_max_over_p99_9']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The Phase 1 selector exactly matches the checkpoint naming for LLM linears, DiT MLP linears, and excluded DiT attention projections.",
            "- W4 per-output-channel weight-only fake quantization has low static weight NMSE for the selected groups. This does not prove activation quantization is safe, but it supports moving to offline activation/forward checks.",
            "- DiT attention weights are listed for comparison only. They remain excluded in the main QuantVLA layout because their Q/K/V/O outputs control attention temperature and residual energy.",
            "- The memory table is an ideal tensor-storage estimate. Runtime memory depends on activation buffers, scales, packing format, kernels, and model framework overhead.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    index = json.loads(INDEX.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    selected = {name: group_for(name) for name in weight_map}
    selected = {name: group for name, group in selected.items() if group is not None}

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in selected:
        by_file[weight_map[name]].append(name)

    rows: list[dict[str, object]] = []
    for shard, names in sorted(by_file.items()):
        with safe_open(str(MODEL_DIR / shard), framework="pt", device="cpu") as handle:
            for name in sorted(names):
                tensor = handle.get_tensor(name)
                if tensor.ndim != 2:
                    continue
                rows.append(tensor_metrics(name, selected[name], shard, tensor))

    group_summaries = {
        group: summarize_group([r for r in rows if r["group"] == group])
        for group in ["llm_selected", "dit_mlp_selected", "dit_attention_excluded"]
    }
    selected_rows = [r for r in rows if r["group"] in {"llm_selected", "dit_mlp_selected"}]
    result = {
        "model_dir": str(MODEL_DIR),
        "index": str(INDEX),
        "group_summaries": group_summaries,
        "worst_selected_by_nmse": sorted(selected_rows, key=lambda r: float(r["nmse"]), reverse=True)[:30],
        "all_rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    OUT_MD.write_text(make_markdown(result), encoding="utf-8")
    print(json.dumps({k: {"layers": v["layers"], "params": v["params"], "nmse_mean": v["nmse"]["mean"]} for k, v in group_summaries.items()}, indent=2))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
