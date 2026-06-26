"""CUDA graph replay probe for the GR00T DiT action-head model.

This is an offline fixed-observation diagnostic, not a LIBERO rollout.  It
keeps the eager model code path, captures only the tensor-only DiT forward used
inside the denoising loop, and compares latency plus action drift against eager
with matched request seeds.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

from lossless_cache_patches import install_lossless_cache_patches, lossless_cache_stats
from phase3_fake_quant_forward import aggregate_metrics, action_to_vector, compare_actions, set_seed
from phase3_gr00t_smoke import _insert_paths
from phase8_cutlass_blockscaled_fp4_real_activation_bench import build_observations
from timing_utils import summarize_float


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


def tensor_spec(value: Any) -> dict[str, Any]:
    import torch

    if not torch.is_tensor(value):
        return {"kind": type(value).__name__}
    return {
        "kind": "tensor",
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }


class CudaGraphForwardState:
    """Callable installed as a module.forward replacement."""

    def __init__(self, original_forward: Any, *, device: str, max_captures: int) -> None:
        self.original_forward = original_forward
        self.device = device
        self.max_captures = int(max_captures)
        self.graphs: dict[str, Any] = {}
        self.kwarg_keys: tuple[str, ...] = ()
        self.num_positional_args = 0
        self.flat_specs: list[dict[str, Any]] = []
        self.flat_specs_by_key: dict[str, list[dict[str, Any]]] = {}
        self.capture_seconds = 0.0
        self.capture_seconds_by_key: dict[str, float] = {}
        self.capture_count = 0
        self.graph_calls = 0
        self.eager_fallback_calls = 0
        self.mismatch_calls = 0
        self.failed = False
        self.error: str | None = None

    def _flatten(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(args) + tuple(kwargs[key] for key in self.kwarg_keys)

    def _validate_sample(self, flat: tuple[Any, ...]) -> None:
        import torch

        for idx, value in enumerate(flat):
            if not torch.is_tensor(value):
                raise TypeError(f"CUDA graph input {idx} is not a tensor: {type(value).__name__}")
            if value.device.type != "cuda":
                raise ValueError(f"CUDA graph input {idx} is on {value.device}, expected cuda")

    def _cuda_autocast_cache_disabled(self, torch_module: Any) -> Any:
        if torch_module.device(self.device).type != "cuda":
            return nullcontext()
        try:
            enabled = bool(torch_module.is_autocast_enabled("cuda"))
        except TypeError:
            enabled = bool(torch_module.is_autocast_cuda_enabled())
        if hasattr(torch_module, "amp"):
            return torch_module.amp.autocast("cuda", enabled=enabled, cache_enabled=False)
        return torch_module.cuda.amp.autocast(enabled=enabled, cache_enabled=False)

    def _same_flat_specs(self, flat: tuple[Any, ...]) -> bool:
        if len(flat) != len(self.flat_specs):
            return False
        return all(tensor_spec(value) == spec for value, spec in zip(flat, self.flat_specs))

    def _spec_key(self, specs: list[dict[str, Any]]) -> str:
        return json.dumps(specs, sort_keys=True)

    def _capture(self, args: tuple[Any, ...], kwargs: dict[str, Any], spec_key: str) -> None:
        import torch

        if not torch.cuda.is_available() or torch.device(self.device).type != "cuda":
            raise RuntimeError("CUDA graph replay requires a CUDA device")
        flat = self._flatten(args, kwargs)
        self._validate_sample(flat)

        def flat_forward(*flat_args: Any) -> Any:
            positional = flat_args[: self.num_positional_args]
            keyword_values = flat_args[self.num_positional_args :]
            rebuilt_kwargs = dict(zip(self.kwarg_keys, keyword_values))
            return self.original_forward(*positional, **rebuilt_kwargs)

        synchronize(self.device)
        started = time.perf_counter()
        with self._cuda_autocast_cache_disabled(torch):
            self.graphs[spec_key] = torch.cuda.make_graphed_callables(flat_forward, flat)
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        self.capture_seconds += elapsed
        self.capture_seconds_by_key[spec_key] = float(elapsed)
        self.capture_count += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.failed:
            self.eager_fallback_calls += 1
            return self.original_forward(*args, **kwargs)

        if not self.graphs:
            self.num_positional_args = len(args)
            self.kwarg_keys = tuple(kwargs.keys())
        elif len(args) != self.num_positional_args or tuple(kwargs.keys()) != self.kwarg_keys:
            self.mismatch_calls += 1
            self.eager_fallback_calls += 1
            return self.original_forward(*args, **kwargs)

        flat = self._flatten(args, kwargs)
        specs = [tensor_spec(value) for value in flat]
        spec_key = self._spec_key(specs)
        if spec_key not in self.graphs:
            if self.graphs:
                self.mismatch_calls += 1
            if len(self.graphs) >= self.max_captures:
                self.eager_fallback_calls += 1
                return self.original_forward(*args, **kwargs)
            try:
                self.flat_specs_by_key[spec_key] = specs
                if not self.flat_specs:
                    self.flat_specs = specs
                self._capture(args, kwargs, spec_key)
            except Exception as exc:  # noqa: BLE001 - probe must record the failure and continue.
                self.failed = True
                self.error = f"{type(exc).__name__}: {exc}"
                self.eager_fallback_calls += 1
                return self.original_forward(*args, **kwargs)

        self.graph_calls += 1
        import torch

        with self._cuda_autocast_cache_disabled(torch):
            return self.graphs[spec_key](*flat)

    def summary(self) -> dict[str, Any]:
        return {
            "max_captures": int(self.max_captures),
            "graph_cache_size": int(len(self.graphs)),
            "capture_count": int(self.capture_count),
            "capture_seconds": float(self.capture_seconds),
            "capture_seconds_by_key": self.capture_seconds_by_key,
            "graph_calls": int(self.graph_calls),
            "eager_fallback_calls": int(self.eager_fallback_calls),
            "mismatch_calls": int(self.mismatch_calls),
            "failed": bool(self.failed),
            "error": self.error,
            "num_positional_args": int(self.num_positional_args),
            "kwarg_keys": list(self.kwarg_keys),
            "flat_specs": self.flat_specs,
            "flat_specs_by_key": self.flat_specs_by_key,
        }


def install_cuda_graph_dit_forward(policy: Any, *, device: str, max_captures: int) -> CudaGraphForwardState:
    module = policy.model.action_head.model
    state = CudaGraphForwardState(module.forward, device=device, max_captures=max_captures)
    module.forward = state
    return state


def make_cases(observations: list[dict[str, Any]], *, repeats: int, repeat_seed_stride: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for obs_index, item in enumerate(observations):
            base_seed = int(item.get("seed", 0))
            case_seed = base_seed + repeat * repeat_seed_stride
            cases.append(
                {
                    "case_index": len(cases),
                    "repeat": int(repeat),
                    "observation_list_index": int(obs_index),
                    "dataset_index": item.get("dataset_index"),
                    "seed": int(case_seed),
                    "obs": item["obs"],
                }
            )
    return cases


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
    return action, float(time.perf_counter() - started)


def run_cases(
    policy: Any,
    cases: list[dict[str, Any]],
    *,
    device: str,
    teacher_actions: list[dict[str, Any]] | None = None,
    keep_actions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    seconds: list[float] = []
    rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases):
        action, elapsed = run_one_action(policy, case["obs"], seed=int(case["seed"]), device=device)
        if keep_actions:
            actions.append(action)
        seconds.append(elapsed)
        row: dict[str, Any] = {
            "case_index": int(case["case_index"]),
            "repeat": int(case["repeat"]),
            "observation_list_index": int(case["observation_list_index"]),
            "dataset_index": case["dataset_index"],
            "seed": int(case["seed"]),
            "get_action_seconds": float(elapsed),
            "action_size": int(action_to_vector(action).size),
        }
        if idx == 0:
            row["action_summary"] = action_summary(action)
        if teacher_actions is not None:
            row.update(compare_actions(teacher_actions[idx], action))
        rows.append(row)

    report: dict[str, Any] = {
        "get_action_seconds": summarize_float(seconds),
        "per_case": rows,
    }
    if teacher_actions is not None:
        report["metrics"] = aggregate_metrics(rows)
    return actions, report


def run_prewarm(policy: Any, cases: list[dict[str, Any]], *, count: int, device: str) -> dict[str, Any]:
    selected = cases[: max(0, min(count, len(cases)))]
    _actions, report = run_cases(policy, selected, device=device, keep_actions=False)
    return report


def metric_value(report: dict[str, Any], key: str, stat: str) -> float:
    return float(report.get("metrics", {}).get(key, {}).get(stat, 0.0))


def write_markdown(result: dict[str, Any], path: Path) -> None:
    eager = result["eager_replay"]["get_action_seconds"]
    graph = result["cuda_graph_replay"]["get_action_seconds"]
    speedup = eager["p50"] / graph["p50"] if graph["p50"] else 0.0
    graph_info = result["cuda_graph"]
    graph_metrics = result["cuda_graph_replay"].get("metrics", {})
    aa_metrics = result["eager_aa"].get("metrics", {})
    lines = [
        "# CUDA Graph DiT Probe",
        "",
        "Boundary: fixed-observation offline replay. This does not run the LIBERO simulator.",
        "",
        "## Setup",
        "",
        f"- Dataset indices: `{', '.join(str(i) for i in result['observation_meta'].get('dataset_indices', []))}`",
        f"- Cases: `{len(result['cases'])}` from `{result['num_observations']}` observations x `{result['num_repeats']}` repeats",
        f"- Denoising steps: `{result['denoising_steps']}`",
        f"- Capture target: `policy.model.action_head.model.forward`",
        f"- Prepare-input pruning: `{result['lossless_cache']['request']['prepare_input_pruning']}`",
        f"- Action-head static cache: `{result['lossless_cache']['request']['action_head_static_cache']}`",
        f"- CUDA graph failed: `{graph_info['failed']}`",
        f"- CUDA graph error: `{graph_info['error']}`",
        f"- CUDA graph cache size: `{graph_info['graph_cache_size']}` / `{graph_info['max_captures']}`",
        "",
        "## Latency",
        "",
        "| path | count | mean | p50 | p90 | p99 |",
        "|---|---:|---:|---:|---:|---:|",
        "| eager | {count} | {mean:.6g}s | {p50:.6g}s | {p90:.6g}s | {p99:.6g}s |".format(
            **eager
        ),
        "| cuda graph DiT | {count} | {mean:.6g}s | {p50:.6g}s | {p90:.6g}s | {p99:.6g}s |".format(
            **graph
        ),
        f"| p50 speedup |  |  | `{speedup:.3f}x` |  |  |",
        "",
        "## Drift",
        "",
        "| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |",
        "|---|---:|---:|---:|---:|",
        "| eager A/A, same seed | {rrmse_mean:.6g} | {rrmse_max:.6g} | {cos_mean:.9g} | {max_abs:.6g} |".format(
            rrmse_mean=aa_metrics.get("relative_rmse", {}).get("mean", 0.0),
            rrmse_max=aa_metrics.get("relative_rmse", {}).get("max", 0.0),
            cos_mean=aa_metrics.get("cosine", {}).get("mean", 0.0),
            max_abs=aa_metrics.get("max_abs_diff", {}).get("max", 0.0),
        ),
        "| eager vs cuda graph, same seed | {rrmse_mean:.6g} | {rrmse_max:.6g} | {cos_mean:.9g} | {max_abs:.6g} |".format(
            rrmse_mean=graph_metrics.get("relative_rmse", {}).get("mean", 0.0),
            rrmse_max=graph_metrics.get("relative_rmse", {}).get("max", 0.0),
            cos_mean=graph_metrics.get("cosine", {}).get("mean", 0.0),
            max_abs=graph_metrics.get("max_abs_diff", {}).get("max", 0.0),
        ),
        "",
        "## Capture Counters",
        "",
        f"- Capture count: `{graph_info['capture_count']}`",
        f"- Capture seconds: `{graph_info['capture_seconds']:.6g}`",
        f"- Graph calls: `{graph_info['graph_calls']}`",
        f"- Eager fallback calls: `{graph_info['eager_fallback_calls']}`",
        f"- Shape mismatch calls: `{graph_info['mismatch_calls']}`",
        "",
        "## Interpretation",
        "",
        "- This probes whether CUDA graph replay can reduce launch/Python overhead while preserving the eager DiT kernels.",
        "- If drift is near the eager A/A row and p50 improves, the next step is an opt-in inference-service switch plus a small matched rollout.",
        "- If graph capture fails or falls back, we should keep torch.compile/block-island as the engineering route and avoid rollout tests for this path.",
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
    parser.add_argument("--observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--num-observations", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=100)
    parser.add_argument("--indices", default="115,462,632,1063")
    parser.add_argument("--synthetic-variants", default="nominal")
    parser.add_argument("--base-seed", type=int, default=20260614)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--repeat-seed-stride", type=int, default=10000)
    parser.add_argument("--graph-prewarm-actions", type=int, default=4)
    parser.add_argument("--cuda-graph-max-captures", type=int, default=8)
    parser.add_argument("--lossless-cache-prepare-input-pruning", action="store_true")
    parser.add_argument("--lossless-cache-action-head-static", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase14_cuda_graph_dit_probe.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase14_cuda_graph_dit_probe.md"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    import torch
    from gr00t.model.policy import Gr00tPolicy

    data_config_cls = load_object(args.data_config)
    data_config = data_config_cls()

    observations, observation_meta = build_observations(args, data_config)
    cases = make_cases(
        observations,
        repeats=args.num_repeats,
        repeat_seed_stride=args.repeat_seed_stride,
    )

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
    lossless_cache = install_lossless_cache_patches(
        policy,
        prepare_input_pruning=bool(args.lossless_cache_prepare_input_pruning),
        action_head_static_cache=bool(args.lossless_cache_action_head_static),
    )

    eager_actions, eager_replay = run_cases(policy, cases, device=args.device)
    _aa_actions, eager_aa = run_cases(
        policy,
        cases,
        device=args.device,
        teacher_actions=eager_actions,
        keep_actions=False,
    )

    graph_state = install_cuda_graph_dit_forward(
        policy,
        device=args.device,
        max_captures=args.cuda_graph_max_captures,
    )
    graph_prewarm = run_prewarm(
        policy,
        cases,
        count=args.graph_prewarm_actions,
        device=args.device,
    )
    _graph_actions, cuda_graph_replay = run_cases(
        policy,
        cases,
        device=args.device,
        teacher_actions=eager_actions,
        keep_actions=False,
    )

    result = {
        "boundary": "Fixed-observation CUDA graph replay for action_head.model.forward; not a simulator rollout",
        "model_path": str(args.model_path),
        "observation_meta": observation_meta,
        "data_config": args.data_config,
        "embodiment_tag": args.embodiment_tag,
        "video_backend": args.video_backend,
        "device": args.device,
        "denoising_steps": int(args.denoising_steps),
        "num_observations": int(args.num_observations),
        "num_repeats": int(args.num_repeats),
        "repeat_seed_stride": int(args.repeat_seed_stride),
        "cases": [
            {key: value for key, value in case.items() if key != "obs"}
            for case in cases
        ],
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "model_load_seconds": float(model_load_seconds),
        "model_load_memory": cuda_memory(args.device),
        "lossless_cache": {
            "request": {
                "prepare_input_pruning": bool(args.lossless_cache_prepare_input_pruning),
                "action_head_static_cache": bool(args.lossless_cache_action_head_static),
            },
            "patch_result": lossless_cache,
        },
        "eager_replay": eager_replay,
        "eager_aa": eager_aa,
        "cuda_graph": graph_state.summary(),
        "cuda_graph_prewarm": graph_prewarm,
        "cuda_graph_replay": cuda_graph_replay,
        "lossless_cache_stats": lossless_cache_stats(policy),
        "final_memory": cuda_memory(args.device),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)

    compact = {
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "eager_get_action_seconds": eager_replay["get_action_seconds"],
        "cuda_graph_get_action_seconds": cuda_graph_replay["get_action_seconds"],
        "lossless_cache": result["lossless_cache"],
        "lossless_cache_stats": result["lossless_cache_stats"],
        "cuda_graph": result["cuda_graph"],
        "eager_aa": {
            "relative_rmse_mean": metric_value(eager_aa, "relative_rmse", "mean"),
            "relative_rmse_max": metric_value(eager_aa, "relative_rmse", "max"),
            "cosine_mean": metric_value(eager_aa, "cosine", "mean"),
            "max_abs_diff": metric_value(eager_aa, "max_abs_diff", "max"),
        },
        "cuda_graph_vs_eager": {
            "relative_rmse_mean": metric_value(cuda_graph_replay, "relative_rmse", "mean"),
            "relative_rmse_max": metric_value(cuda_graph_replay, "relative_rmse", "max"),
            "cosine_mean": metric_value(cuda_graph_replay, "cosine", "mean"),
            "max_abs_diff": metric_value(cuda_graph_replay, "max_abs_diff", "max"),
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
