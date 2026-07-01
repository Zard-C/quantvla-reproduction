"""N1.7 GR00T policy server with timing and tactic-switch support."""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Callable

import numpy as np


TORCH_COMPILE_TARGETS = (
    "none",
    "action_head_model",
    "action_head_model_blocks_0_3_eager",
    "action_head_model_blocks_0_7_eager",
    "action_head_model_blocks_0_15_eager",
    "action_head_model_blocks_8_15_eager",
    "action_head_model_blocks_16_31_eager",
    "action_head_model_blocks_0_31_eager",
)


def summarize_float(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def write_json(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def insert_isaac_root(isaac_root: Path) -> None:
    root = str(isaac_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def cuda_memory(torch_module: Any, device: str) -> dict[str, int]:
    if not torch_module.cuda.is_available() or torch_module.device(device).type != "cuda":
        return {"allocated_mib": 0, "reserved_mib": 0, "max_allocated_mib": 0, "max_reserved_mib": 0}
    return {
        "allocated_mib": int(torch_module.cuda.memory_allocated(device) // (1024 * 1024)),
        "reserved_mib": int(torch_module.cuda.memory_reserved(device) // (1024 * 1024)),
        "max_allocated_mib": int(torch_module.cuda.max_memory_allocated(device) // (1024 * 1024)),
        "max_reserved_mib": int(torch_module.cuda.max_memory_reserved(device) // (1024 * 1024)),
    }


def get_submodule(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if part.isdigit() and hasattr(obj, "__getitem__"):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def set_submodule(root: Any, path: str, value: Any) -> None:
    parent_path, _, leaf = path.rpartition(".")
    parent = get_submodule(root, parent_path) if parent_path else root
    if leaf.isdigit() and hasattr(parent, "__setitem__"):
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def action_head_block_count(model: Any) -> int:
    blocks = get_submodule(model, "action_head.model.transformer_blocks")
    return len(blocks)


def _block_paths(start: int, end: int, *, num_blocks: int) -> list[str]:
    if start < 0 or end < start or end >= num_blocks:
        raise ValueError(f"Invalid block range {start}-{end} for DiT with {num_blocks} blocks")
    return [f"action_head.model.transformer_blocks.{idx}" for idx in range(start, end + 1)]


def compile_module_paths_for_target(target: str) -> list[str]:
    if target == "none":
        return []
    if target == "action_head_model":
        return ["action_head.model"]
    if re.fullmatch(r"action_head_model_blocks_\d+_\d+_eager", target):
        return ["action_head.model"]
    raise ValueError(f"Unsupported N1.7 compile target: {target!r}")


def eager_island_paths_for_target(target: str, *, num_blocks: int) -> list[str]:
    match = re.fullmatch(r"action_head_model_blocks_(\d+)_(\d+)_eager", target)
    if not match:
        return []
    return _block_paths(int(match.group(1)), int(match.group(2)), num_blocks=num_blocks)


def compile_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
    }
    if args.torch_compile_dynamic is not None:
        kwargs["dynamic"] = bool(args.torch_compile_dynamic)
    return kwargs


def install_eager_islands(model: Any, paths: list[str], torch_module: Any) -> list[str]:
    installed: list[str] = []
    for path in paths:
        module = get_submodule(model, path)
        if getattr(module, "_phase36_n17_eager_island", False):
            installed.append(f"model.{path}")
            continue
        module.forward = torch_module._dynamo.disable(module.forward)
        module._phase36_n17_eager_island = True
        installed.append(f"model.{path}")
    return installed


def compile_policy_targets(model: Any, args: argparse.Namespace, torch_module: Any, *, num_blocks: int) -> dict[str, Any]:
    target = str(args.torch_compile_target)
    paths = compile_module_paths_for_target(target)
    if not paths:
        return {"enabled": False, "target": target, "compiled_modules": []}

    eager_island_modules = install_eager_islands(
        model,
        eager_island_paths_for_target(target, num_blocks=num_blocks),
        torch_module,
    )
    kwargs = compile_kwargs_from_args(args)
    started = time.perf_counter()
    compiled_modules: list[str] = []
    for path in paths:
        module = get_submodule(model, path)
        set_submodule(model, path, torch_module.compile(module, **kwargs))
        compiled_modules.append(f"model.{path}")
    return {
        "enabled": True,
        "target": target,
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
        "dynamic": args.torch_compile_dynamic,
        "compiled_modules": compiled_modules,
        "eager_island_modules": eager_island_modules,
        "wrap_seconds": time.perf_counter() - started,
    }


class CompileTargetSwitcher:
    """Switch N1.7 modules between eager and compiled implementations."""

    def __init__(self, model: Any, args: argparse.Namespace, torch_module: Any, *, num_blocks: int):
        self.model = model
        self.target = str(args.torch_compile_target)
        self.paths = compile_module_paths_for_target(self.target)
        self.eager_island_modules = install_eager_islands(
            model,
            eager_island_paths_for_target(self.target, num_blocks=num_blocks),
            torch_module,
        )
        self.kwargs = compile_kwargs_from_args(args)
        self.eager_modules = {path: get_submodule(model, path) for path in self.paths}
        self.compiled_modules = {
            path: torch_module.compile(module, **self.kwargs)
            for path, module in self.eager_modules.items()
        }

    def use_eager(self) -> None:
        for path, module in self.eager_modules.items():
            set_submodule(self.model, path, module)

    def use_compiled(self) -> None:
        for path, module in self.compiled_modules.items():
            set_submodule(self.model, path, module)

    def info(self, args: argparse.Namespace) -> dict[str, Any]:
        if not self.paths:
            return {"enabled": False, "target": self.target, "compiled_modules": []}
        return {
            "enabled": True,
            "target": self.target,
            "backend": args.torch_compile_backend,
            "mode": args.torch_compile_mode,
            "fullgraph": bool(args.torch_compile_fullgraph),
            "dynamic": args.torch_compile_dynamic,
            "compiled_modules": [f"model.{path}" for path in self.paths],
            "eager_island_modules": self.eager_island_modules,
        }


class StepWindowCompileFallbackPolicy:
    """Use eager modules for request-index windows and compiled modules elsewhere."""

    def __init__(self, policy: Any, switcher: CompileTargetSwitcher, *, step_start: int, step_end: int):
        self._policy = policy
        self._switcher = switcher
        self._step_start = int(step_start)
        self._step_end = int(step_end)
        self._request_index = 0
        self._compiled_requests = 0
        self._eager_requests = 0
        self._last_mode = None
        self._switcher.use_compiled()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        step = self._request_index
        self._request_index += 1
        use_eager = self._step_start <= step < self._step_end
        if use_eager:
            self._switcher.use_eager()
            self._eager_requests += 1
            self._last_mode = "eager"
        else:
            self._switcher.use_compiled()
            self._compiled_requests += 1
            self._last_mode = "compiled"
        return self._policy.get_action(observation, options)

    def summary(self) -> dict[str, Any]:
        total = self._compiled_requests + self._eager_requests
        return {
            "enabled": True,
            "step_start": self._step_start,
            "step_end": self._step_end,
            "request_index": int(self._request_index),
            "compiled_requests": int(self._compiled_requests),
            "eager_requests": int(self._eager_requests),
            "eager_fraction": float(self._eager_requests / total) if total else 0.0,
            "last_mode": self._last_mode,
        }


class TimedN17Policy:
    """Proxy that records server-side get_action latency and optional request traces."""

    def __init__(
        self,
        policy: Any,
        *,
        output_json: Path | None,
        label: str,
        flush_every: int,
        extra_summary: Callable[[], dict[str, Any]],
        request_trace_jsonl: Path | None,
        cuda_sync_device: str | None,
        torch_module: Any,
    ):
        self._policy = policy
        self._output_json = output_json
        self._label = label
        self._flush_every = int(flush_every)
        self._extra_summary = extra_summary
        self._request_trace_jsonl = request_trace_jsonl
        self._cuda_sync_device = cuda_sync_device
        self._torch = torch_module
        self._latencies: list[float] = []
        self._write_count = 0
        self._trace_file = None
        if request_trace_jsonl is not None:
            request_trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = request_trace_jsonl.open("a", encoding="utf-8")
            atexit.register(self.close_trace)
        if output_json is not None:
            atexit.register(self.write_summary)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self._handle_signal)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.write_summary()
        self.close_trace()
        raise SystemExit(128 + int(signum))

    def get_action(self, observation: dict[str, Any], options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        request_index = len(self._latencies) + 1
        wall_started = time.time()
        started = time.perf_counter()
        out = self._policy.get_action(observation, options)
        if self._cuda_sync_device is not None:
            self._torch.cuda.synchronize(self._cuda_sync_device)
        seconds = time.perf_counter() - started
        self._latencies.append(float(seconds))
        self._write_trace(request_index, wall_started, float(seconds))
        if self._flush_every > 0 and len(self._latencies) % self._flush_every == 0:
            self.write_summary()
        return out

    def _write_trace(self, request_index: int, wall_started: float, seconds: float) -> None:
        if self._trace_file is None:
            return
        row = {
            "label": self._label,
            "request_index": int(request_index),
            "wall_start_unix": float(wall_started),
            "wall_end_unix": float(time.time()),
            "get_action_seconds": float(seconds),
        }
        self._trace_file.write(json.dumps(row, sort_keys=True) + "\n")
        self._trace_file.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "label": self._label,
            "writes": int(self._write_count),
            "get_action_seconds": summarize_float(self._latencies),
            "extra": self._extra_summary(),
        }

    def write_summary(self) -> None:
        if self._output_json is None:
            return
        payload = self.summary()
        payload["writes"] = int(self._write_count + 1)
        write_json(self._output_json, payload)
        self._write_count += 1

    def close_trace(self) -> None:
        if self._trace_file is not None:
            self._trace_file.close()
            self._trace_file = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/GR00T-N1.7-LIBERO/libero_10"))
    parser.add_argument("--embodiment-tag", default="LIBERO_PANDA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sim-policy-wrapper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase36_n17_timed_server_prepare.json"))
    parser.add_argument("--server-latency-json", type=Path)
    parser.add_argument("--server-latency-flush-every", type=int, default=0)
    parser.add_argument("--server-request-trace-jsonl", type=Path)
    parser.add_argument("--server-request-trace-cuda-sync", action="store_true")
    parser.add_argument("--torch-compile-target", choices=TORCH_COMPILE_TARGETS, default="none")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", choices=["true", "false"])
    parser.add_argument("--torch-compile-fallback-step-start", type=int)
    parser.add_argument("--torch-compile-fallback-step-end", type=int)
    args = parser.parse_args()
    if args.torch_compile_dynamic is not None:
        args.torch_compile_dynamic = args.torch_compile_dynamic == "true"

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    insert_isaac_root(args.isaac_root)

    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    result: dict[str, Any] = {
        "boundary": "GR00T N1.7 official policy server with timing and tactic switches",
        "isaac_root": str(args.isaac_root),
        "model_path": str(args.model_path),
        "embodiment_tag": args.embodiment_tag,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": args.device,
        "strict": bool(args.strict),
        "use_sim_policy_wrapper": bool(args.use_sim_policy_wrapper),
    }

    started = time.perf_counter()
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    gr00t_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=str(args.model_path),
        device=args.device,
        strict=args.strict,
    )
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    result["model_load_seconds"] = time.perf_counter() - started
    result["model_load_memory"] = cuda_memory(torch, args.device)
    result["action_head_num_blocks"] = action_head_block_count(gr00t_policy.model)
    result["action_head_num_inference_timesteps"] = int(gr00t_policy.model.action_head.num_inference_timesteps)
    result["action_horizon"] = int(gr00t_policy.model.action_head.action_horizon)

    compile_fallback_policy = None
    compile_info: dict[str, Any]
    if (args.torch_compile_fallback_step_start is None) != (args.torch_compile_fallback_step_end is None):
        raise ValueError("Both --torch-compile-fallback-step-start and --torch-compile-fallback-step-end are required")
    if args.torch_compile_fallback_step_start is not None:
        if args.torch_compile_target == "none":
            raise ValueError("Step-window fallback requires a non-none --torch-compile-target")
        if args.torch_compile_fallback_step_start >= args.torch_compile_fallback_step_end:
            raise ValueError("Fallback step start must be smaller than end")
        switcher = CompileTargetSwitcher(
            gr00t_policy.model,
            args,
            torch,
            num_blocks=int(result["action_head_num_blocks"]),
        )
        compile_info = switcher.info(args)
    else:
        compile_info = compile_policy_targets(
            gr00t_policy.model,
            args,
            torch,
            num_blocks=int(result["action_head_num_blocks"]),
        )
        switcher = None
    result["torch_compile"] = compile_info

    served_policy: Any = gr00t_policy
    if args.use_sim_policy_wrapper:
        served_policy = Gr00tSimPolicyWrapper(gr00t_policy)
    if args.torch_compile_fallback_step_start is not None:
        compile_fallback_policy = StepWindowCompileFallbackPolicy(
            served_policy,
            switcher,
            step_start=args.torch_compile_fallback_step_start,
            step_end=args.torch_compile_fallback_step_end,
        )
        served_policy = compile_fallback_policy
        result["torch_compile"]["step_window_fallback"] = compile_fallback_policy.summary()

    result["prepare_seconds"] = time.perf_counter() - started
    write_json(args.output_json, result)
    if args.prepare_only:
        print(json.dumps(result, indent=2), flush=True)
        return

    def extra_summary() -> dict[str, Any]:
        summary = {"server_memory": cuda_memory(torch, args.device)}
        if compile_fallback_policy is not None:
            summary["compile_step_window_fallback"] = compile_fallback_policy.summary()
        return summary

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting timed N1.7 server on port {args.port}", flush=True)
    timed_policy = TimedN17Policy(
        served_policy,
        output_json=args.server_latency_json,
        label="n17_fp16",
        flush_every=args.server_latency_flush_every,
        extra_summary=extra_summary,
        request_trace_jsonl=args.server_request_trace_jsonl,
        cuda_sync_device=args.device if args.server_request_trace_cuda_sync else None,
        torch_module=torch,
    )
    with PolicyServer(policy=timed_policy, host=args.host, port=args.port) as server:
        server.run()


if __name__ == "__main__":
    main()
