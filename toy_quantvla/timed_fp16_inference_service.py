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
from lossless_cache_patches import install_lossless_cache_patches, lossless_cache_stats
from phase13_compile_targets import (
    TORCH_COMPILE_TARGETS,
    CompileTargetSwitcher,
    compile_policy_targets,
)
from phase14_cuda_graph_dit_probe import install_cuda_graph_dit_forward
from phase3_fake_quant_forward import set_seed
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, scope_description
from phase8_cutlass_blockscaled_fp4_real_activation_bench import (
    build_observations,
    cuda_memory,
    reset_cuda_peak,
    synchronize,
)
from timing_utils import TimedPolicyWrapper, summarize_float


REQUEST_POLICY_STEP_KEY = "quantvla.policy_step"


class StepWindowCompileFallbackPolicy:
    """Use eager modules only for a policy-step window, compiled modules elsewhere."""

    def __init__(self, policy: Any, switcher: CompileTargetSwitcher, *, step_start: int, step_end: int):
        self._policy = policy
        self._switcher = switcher
        self._step_start = int(step_start)
        self._step_end = int(step_end)
        self._compiled_requests = 0
        self._eager_requests = 0
        self._last_mode = None
        self._switcher.use_compiled()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        policy_observation = observation
        raw_step = None
        if REQUEST_POLICY_STEP_KEY in observation:
            policy_observation = dict(observation)
            raw_step = policy_observation.pop(REQUEST_POLICY_STEP_KEY)
        policy_step = None if raw_step is None else int(raw_step)
        use_eager = policy_step is not None and self._step_start <= policy_step < self._step_end
        if use_eager:
            self._switcher.use_eager()
            self._eager_requests += 1
            self._last_mode = "eager"
        else:
            self._switcher.use_compiled()
            self._compiled_requests += 1
            self._last_mode = "compiled"
        return self._policy.get_action(policy_observation)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "step_start": self._step_start,
            "step_end": self._step_end,
            "compiled_requests": int(self._compiled_requests),
            "eager_requests": int(self._eager_requests),
            "last_mode": self._last_mode,
        }


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


def add_prewarm_task_descriptions(
    observations: list[dict[str, Any]],
    descriptions: list[str],
    *,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Add warmup observations with controlled task text to cover LLM sequence lengths."""

    if not descriptions:
        return observations
    if not observations:
        raise ValueError("--prewarm-task-description requires at least one base prewarm observation")

    out = list(observations)
    base = observations[0]
    base_obs = dict(base["obs"])
    for offset, description in enumerate(descriptions, start=1):
        obs = dict(base_obs)
        obs["annotation.human.action.task_description"] = [description]
        item = dict(base)
        item["obs"] = obs
        item["seed"] = int(base_seed) + offset
        item["prewarm_task_description"] = description
        out.append(item)
    return out


def build_prewarm_observations(args: argparse.Namespace, data_config: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.prewarm_observations <= 0:
        return [], {
            "observation_source": args.prewarm_observation_source,
            "prewarm_observations": 0,
        }
    prewarm_args = argparse.Namespace(
        observation_source=args.prewarm_observation_source,
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        video_backend=args.video_backend,
        num_observations=args.prewarm_observations,
        indices=args.prewarm_indices,
        start_index=args.prewarm_start_index,
        sample_stride=args.prewarm_sample_stride,
        synthetic_variants=args.synthetic_variants,
        base_seed=args.prewarm_base_seed,
    )
    return build_observations(prewarm_args, data_config)


def run_prewarm(policy: Any, observations: list[dict[str, Any]], *, device: str) -> tuple[dict[str, Any], dict[str, int]]:
    latencies: list[float] = []
    for item in observations:
        set_seed(int(item.get("seed", 0)))
        started = time.perf_counter()
        policy.get_action(item["obs"])
        synchronize(device)
        latencies.append(time.perf_counter() - started)
    return summarize_float(latencies), cuda_memory(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--dataset-path", type=Path, default=Path("/root/autodl-tmp/datasets/libero_10_subset"))
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
    parser.add_argument("--torch-compile-target", choices=TORCH_COMPILE_TARGETS, default="none")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", choices=["true", "false"])
    parser.add_argument("--torch-compile-fallback-step-start", type=int)
    parser.add_argument("--torch-compile-fallback-step-end", type=int)
    parser.add_argument(
        "--torch-compile-cudagraph-mark-step",
        action="store_true",
        help="Call torch.compiler.cudagraph_mark_step_begin before each compiled submodule invocation.",
    )
    parser.add_argument("--prewarm-observations", type=int, default=0)
    parser.add_argument("--lossless-cache-eagle-tokenizer", action="store_true")
    parser.add_argument("--lossless-cache-prepare-input-pruning", action="store_true")
    parser.add_argument("--lossless-cache-static-normalized-input", action="store_true")
    parser.add_argument("--lossless-cache-action-head-static", action="store_true")
    parser.add_argument("--cuda-graph-dit-forward", action="store_true")
    parser.add_argument("--cuda-graph-max-captures", type=int, default=8)
    parser.add_argument("--prewarm-observation-source", choices=["real", "synthetic"], default="real")
    parser.add_argument("--prewarm-indices", default="115")
    parser.add_argument("--prewarm-start-index", type=int, default=0)
    parser.add_argument("--prewarm-sample-stride", type=int, default=100)
    parser.add_argument("--prewarm-base-seed", type=int, default=20260613)
    parser.add_argument("--prewarm-task-description", action="append", default=[])
    parser.add_argument("--video-backend", default="torchvision_av")
    parser.add_argument("--synthetic-variants", default="zero,midgray,noise")
    parser.add_argument("--server-request-trace-jsonl", type=Path, help="Optional per-request server latency JSONL.")
    parser.add_argument("--server-request-trace-min-seconds", type=float, default=0.0)
    parser.add_argument("--server-request-trace-cuda-sync", action="store_true")
    args = parser.parse_args()
    if args.torch_compile_dynamic is not None:
        args.torch_compile_dynamic = args.torch_compile_dynamic == "true"
    if args.cuda_graph_dit_forward and args.lossless_cache_action_head_static:
        raise ValueError(
            "--cuda-graph-dit-forward must not be combined with "
            "--lossless-cache-action-head-static; Phase22 measured nonzero action drift for that pair."
        )

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
        "dataset_path": str(args.dataset_path),
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
        "torch_compile": {
            "target": args.torch_compile_target,
            "backend": args.torch_compile_backend,
            "mode": args.torch_compile_mode,
            "fullgraph": bool(args.torch_compile_fullgraph),
            "dynamic": args.torch_compile_dynamic,
            "cudagraph_mark_step": bool(args.torch_compile_cudagraph_mark_step),
            "step_window_fallback": {
                "start": args.torch_compile_fallback_step_start,
                "end": args.torch_compile_fallback_step_end,
            },
        },
        "lossless_cache": {
            "eagle_tokenizer_cache": bool(args.lossless_cache_eagle_tokenizer),
            "prepare_input_pruning": bool(args.lossless_cache_prepare_input_pruning),
            "static_normalized_input_cache": bool(args.lossless_cache_static_normalized_input),
            "action_head_static_cache": bool(args.lossless_cache_action_head_static),
        },
        "cuda_graph": {
            "dit_forward": bool(args.cuda_graph_dit_forward),
            "max_captures": int(args.cuda_graph_max_captures),
        },
        "prewarm_observations": int(args.prewarm_observations),
        "prewarm_observation_source": args.prewarm_observation_source,
        "prewarm_indices": args.prewarm_indices,
        "video_backend": args.video_backend,
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
    result["model_load_memory"] = cuda_memory(args.device)

    compile_fallback_policy = None
    if (args.torch_compile_fallback_step_start is None) != (args.torch_compile_fallback_step_end is None):
        raise ValueError("Both --torch-compile-fallback-step-start and --torch-compile-fallback-step-end are required")
    if args.torch_compile_fallback_step_start is not None:
        if args.torch_compile_target == "none":
            raise ValueError("Step-window fallback requires a non-none --torch-compile-target")
        if args.torch_compile_fallback_step_start >= args.torch_compile_fallback_step_end:
            raise ValueError("Fallback step start must be smaller than end")
        switcher = CompileTargetSwitcher(policy, args, torch)
        compile_fallback_policy = StepWindowCompileFallbackPolicy(
            policy,
            switcher,
            step_start=args.torch_compile_fallback_step_start,
            step_end=args.torch_compile_fallback_step_end,
        )
        result["torch_compile"] = switcher.info(args)
        result["torch_compile"]["step_window_fallback"] = compile_fallback_policy.summary()
    else:
        result["torch_compile"] = compile_policy_targets(policy, args, torch)
    result["lossless_cache"] = install_lossless_cache_patches(
        policy,
        eagle_tokenizer_cache=bool(args.lossless_cache_eagle_tokenizer),
        prepare_input_pruning=bool(args.lossless_cache_prepare_input_pruning),
        static_normalized_input_cache=bool(args.lossless_cache_static_normalized_input),
        action_head_static_cache=bool(args.lossless_cache_action_head_static),
    )
    cuda_graph_state = None
    if args.cuda_graph_dit_forward:
        cuda_graph_state = install_cuda_graph_dit_forward(
            policy,
            device=args.device,
            max_captures=args.cuda_graph_max_captures,
        )
        result["cuda_graph"] = cuda_graph_state.summary()
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

    prewarm_started = time.perf_counter()
    observations, observation_meta = build_prewarm_observations(args, data_config)
    observations = add_prewarm_task_descriptions(
        observations,
        args.prewarm_task_description,
        base_seed=args.prewarm_base_seed,
    )
    if args.prewarm_task_description:
        observation_meta["prewarm_task_descriptions"] = list(args.prewarm_task_description)
        observation_meta["prewarm_observations_with_task_descriptions"] = len(observations)
    result["prewarm_observation_meta"] = observation_meta
    if observations:
        result["prewarm_get_action_seconds"], result["prewarm_memory"] = run_prewarm(
            policy,
            observations,
            device=args.device,
        )
    else:
        result["prewarm_get_action_seconds"] = summarize_float([])
        result["prewarm_memory"] = cuda_memory(args.device)
    result["prewarm_total_seconds"] = time.perf_counter() - prewarm_started
    if args.profile_linear_modules:
        result["profile_module_results_after_prewarm"] = module_results(profiled_modules)
        reset_module_stats(profiled_modules)
    if cuda_graph_state is not None:
        result["cuda_graph_after_prewarm"] = cuda_graph_state.summary()

    result["prepare_seconds"] = time.perf_counter() - started
    write_json(args.output_json, result)

    if args.prepare_only:
        print(json.dumps(result, indent=2))
        return

    print(json.dumps(result, indent=2), flush=True)
    print(f"Starting timed FP16 server on port {args.port}", flush=True)
    set_seed(20260608)

    reset_cuda_peak(args.device)

    def server_extra_summary() -> dict[str, Any]:
        summary: dict[str, Any] = {
            "server_memory": cuda_memory(args.device),
            "lossless_cache_stats": lossless_cache_stats(policy),
        }
        if compile_fallback_policy is not None:
            summary["compile_step_window_fallback"] = compile_fallback_policy.summary()
        if cuda_graph_state is not None:
            summary["cuda_graph"] = cuda_graph_state.summary()
        if args.profile_linear_modules:
            summary["profile_module_results"] = module_results(profiled_modules)
        return summary

    served_policy = compile_fallback_policy if compile_fallback_policy is not None else policy
    timed_policy = TimedPolicyWrapper(
        served_policy,
        output_json=args.server_latency_json,
        label="fp16_official",
        flush_every=args.server_latency_flush_every,
        extra_summary=server_extra_summary,
        request_trace_jsonl=args.server_request_trace_jsonl,
        request_trace_min_seconds=args.server_request_trace_min_seconds,
        cuda_sync_device=args.device if args.server_request_trace_cuda_sync else None,
    )
    server = RobotInferenceServer(timed_policy, port=args.port, api_token=args.api_token)
    server.run()


if __name__ == "__main__":
    main()
