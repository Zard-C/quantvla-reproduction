"""Focused replay around a suspected torch.compile action spike.

This probe keeps the simulator on the eager FP16 trajectory, jumps to a narrow
policy-step window, and repeatedly evaluates eager/compiled actions on exactly
the same live observations.  It answers whether a spike is deterministic,
seed-sensitive, or localized to a small observation neighborhood.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from phase13_compile_targets import TORCH_COMPILE_TARGETS, CompileTargetSwitcher
from phase13_torch_compile_online_drift import (
    ACTION_KEYS,
    CompileDriftPolicy,
    as_float_list,
    compare_libero_actions,
    deterministic_policy_seed,
    max_steps_for_suite,
    synchronize_if_cuda,
)
from phase3_fake_quant_forward import compare_actions, set_seed
from phase3_gr00t_smoke import _insert_paths


CONTINUOUS_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
EPS = 1e-8


def parse_case_list(case_list: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in case_list.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected case item 'task_id:init_index', got {item!r}")
        task_id, init_index = item.split(":", 1)
        cases.append((int(task_id), int(init_index)))
    if not cases:
        raise ValueError("--case-list must include at least one task:init pair")
    return cases


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(percentile(values, 0.50)),
        "p90": float(percentile(values, 0.90)),
        "p99": float(percentile(values, 0.99)),
    }


def raw_action_value(action_chunk: dict[str, Any], key: str) -> float:
    return float(np.atleast_1d(action_chunk[f"action.{key}"][0])[0])


def raw_action_diff(eager_chunk: dict[str, Any], compiled_chunk: dict[str, Any]) -> dict[str, float]:
    return {
        key: raw_action_value(compiled_chunk, key) - raw_action_value(eager_chunk, key)
        for key in ACTION_KEYS
    }


def continuous_l2(diffs: dict[str, float]) -> float:
    return float(np.linalg.norm([diffs[key] for key in CONTINUOUS_KEYS]))


def continuous_max_abs(diffs: dict[str, float]) -> float:
    return float(max(abs(diffs[key]) for key in CONTINUOUS_KEYS))


def max_dim(diffs: dict[str, float]) -> str:
    return max(CONTINUOUS_KEYS, key=lambda key: abs(diffs[key]))


def summarize_focus_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    cont_max = [float(record["continuous_max_abs"]) for record in records]
    cont_l2 = [float(record["continuous_l2"]) for record in records]
    eager_seconds = [float(record["timing"]["eager_get_action_seconds"]) for record in records]
    compiled_seconds = [float(record["timing"]["compiled_get_action_seconds"]) for record in records]
    ratios = [
        float(record["timing"]["compiled_get_action_seconds"]) / max(float(record["timing"]["eager_get_action_seconds"]), EPS)
        for record in records
    ]

    repeat_spans: list[float] = []
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed_offset"]), []).append(record)
    for seed_records in by_seed.values():
        if len(seed_records) < 2:
            continue
        for key in CONTINUOUS_KEYS:
            values = [float(record["raw_diff"][key]) for record in seed_records]
            repeat_spans.append(max(values) - min(values))

    top_record = None
    if records:
        top_record = max(records, key=lambda record: float(record["continuous_max_abs"]))

    return {
        "samples": len(records),
        "continuous_max_abs": summarize(cont_max),
        "continuous_l2": summarize(cont_l2),
        "eager_get_action_seconds": summarize(eager_seconds),
        "compiled_get_action_seconds": summarize(compiled_seconds),
        "compiled_over_eager_time": summarize(ratios),
        "same_seed_repeat_span": summarize([abs(value) for value in repeat_spans]),
        "top_record": top_record,
    }


def focus_steps_from_args(args: argparse.Namespace) -> list[int]:
    if args.focus_policy_steps:
        return sorted(set(int(step) for step in args.focus_policy_steps))
    if args.focus_end < args.focus_start:
        raise ValueError("--focus-end must be >= --focus-start")
    return list(range(int(args.focus_start), int(args.focus_end) + 1))


def evaluate_focus_observation(
    local_policy: CompileDriftPolicy,
    processed_obs: dict[str, Any],
    *,
    task_id: int,
    init_index: int,
    policy_step: int,
    base_seed: int,
    seed_offsets: list[int],
    repeats: int,
    warmup_repeats: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    nominal_seed = deterministic_policy_seed(base_seed, task_id, init_index, policy_step)
    for _ in range(warmup_repeats):
        local_policy.get_eager_compiled(processed_obs, nominal_seed)

    for seed_offset in seed_offsets:
        seed = nominal_seed + int(seed_offset)
        for repeat in range(repeats):
            eager_chunk, compiled_chunk, timing = local_policy.get_eager_compiled(processed_obs, seed)
            eager_action = local_policy.convert_to_libero_action(eager_chunk)
            compiled_action = local_policy.convert_to_libero_action(compiled_chunk)
            diffs = raw_action_diff(eager_chunk, compiled_chunk)
            records.append(
                {
                    "policy_step": int(policy_step),
                    "nominal_seed": int(nominal_seed),
                    "denoising_seed": int(seed),
                    "seed_offset": int(seed_offset),
                    "repeat": int(repeat),
                    "raw_diff": diffs,
                    "continuous_l2": continuous_l2(diffs),
                    "continuous_max_abs": continuous_max_abs(diffs),
                    "max_dim": max_dim(diffs),
                    "eager_raw_action": local_policy.trace_raw_action(eager_chunk),
                    "compiled_raw_action": local_policy.trace_raw_action(compiled_chunk),
                    "eager_libero_action": as_float_list(eager_action),
                    "compiled_libero_action": as_float_list(compiled_action),
                    "raw_action_metrics": compare_actions(eager_chunk, compiled_chunk),
                    "libero_action_metrics": compare_libero_actions(eager_action, compiled_action),
                    "timing": timing,
                }
            )
    return records


def drive_eager_action(
    policy: Any,
    local_policy: CompileDriftPolicy,
    target_switcher: CompileTargetSwitcher,
    processed_obs: dict[str, Any],
    *,
    seed: int,
    device: str,
) -> np.ndarray:
    target_switcher.use_eager()
    set_seed(seed)
    synchronize_if_cuda(device)
    with torch.no_grad():
        eager_chunk = policy.get_action(processed_obs)
    synchronize_if_cuda(device)
    return local_policy.convert_to_libero_action(eager_chunk)


def run_focused_replay(args: argparse.Namespace) -> dict[str, Any]:
    from examples.Libero.custom_data_config import LiberoDataConfig
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env
    from gr00t.model.policy import Gr00tPolicy

    focus_steps = focus_steps_from_args(args)
    max_focus_step = max(focus_steps)
    max_policy_steps = args.max_policy_steps or (max_focus_step + 1)

    data_config = LiberoDataConfig()
    load_started = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    synchronize_if_cuda(args.device)
    model_load_seconds = time.perf_counter() - load_started

    compile_started = time.perf_counter()
    target_switcher = CompileTargetSwitcher(policy, args, torch)
    compile_wrap_seconds = time.perf_counter() - compile_started
    local_policy = CompileDriftPolicy(
        policy,
        headless=args.headless,
        device=args.device,
        target_switcher=target_switcher,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    cases = parse_case_list(args.case_list)
    cases_by_task: dict[int, list[int]] = {}
    for task_id, init_index in cases:
        cases_by_task.setdefault(task_id, []).append(init_index)

    result: dict[str, Any] = {
        "boundary": "focused eager-controlled same-observation replay around suspected compile drift spike",
        "model_path": str(args.model_path),
        "task_suite_name": args.task_suite_name,
        "case_list": args.case_list,
        "focus_policy_steps": focus_steps,
        "seed_offsets": [int(value) for value in args.seed_offsets],
        "repeats": int(args.repeats),
        "warmup_repeats": int(args.warmup_repeats),
        "denoising_steps": int(args.denoising_steps),
        "num_steps_wait": int(args.num_steps_wait),
        "max_policy_steps": int(max_policy_steps),
        "base_seed": int(args.base_seed),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "device": args.device,
        "model_load_seconds": float(model_load_seconds),
        "torch_compile": target_switcher.info(args) | {"wrap_seconds": float(compile_wrap_seconds)},
        "case_summaries": [],
        "trace_files": [],
    }

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({k: v for k, v in result.items() if k not in {"case_summaries", "trace_files"}}, indent=2) + "\n")
        for task_id in tqdm.tqdm(sorted(cases_by_task), desc="tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)
            for init_index in tqdm.tqdm(sorted(set(cases_by_task[task_id])), desc=f"task {task_id}"):
                if init_index >= len(initial_states):
                    raise IndexError(f"Task {task_id} requested init {init_index}, but only {len(initial_states)} states are available")

                env.reset()
                obs = env.set_init_state(initial_states[init_index])
                t = 0
                done = False
                reward = 0.0
                exception_message = None
                focus_payloads: list[dict[str, Any]] = []
                focus_step_summaries: list[dict[str, Any]] = []
                started = time.perf_counter()
                max_total_steps = max_policy_steps + args.num_steps_wait
                suite_limit = max_steps_for_suite(args.task_suite_name)
                if max_policy_steps > suite_limit:
                    raise ValueError(f"Requested max policy steps {max_policy_steps}, but suite limit is {suite_limit}")

                log_file.write(f"\nTask: {task_description}\nTask id: {task_id}\nInit index: {init_index}\n")
                while t < max_total_steps:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        policy_step = t - args.num_steps_wait
                        processed_obs = local_policy.process_observation(obs, task.language)
                        seed = deterministic_policy_seed(args.base_seed, task_id, init_index, policy_step)
                        pre_state = {
                            "robot0_eef_pos": as_float_list(obs.get("robot0_eef_pos", [])),
                            "robot0_eef_quat": as_float_list(obs.get("robot0_eef_quat", [])),
                            "robot0_gripper_qpos": as_float_list(obs.get("robot0_gripper_qpos", [])),
                        }

                        if policy_step in focus_steps:
                            records = evaluate_focus_observation(
                                local_policy,
                                processed_obs,
                                task_id=task_id,
                                init_index=init_index,
                                policy_step=policy_step,
                                base_seed=args.base_seed,
                                seed_offsets=[int(value) for value in args.seed_offsets],
                                repeats=int(args.repeats),
                                warmup_repeats=int(args.warmup_repeats),
                            )
                            step_summary = summarize_focus_records(records)
                            step_payload = {
                                "policy_step": int(policy_step),
                                "pre_state": pre_state,
                                "summary": step_summary,
                                "records": records,
                            }
                            focus_payloads.append(step_payload)
                            focus_step_summaries.append({k: v for k, v in step_payload.items() if k != "records"})
                            log_file.write(
                                "focus step {step}: max={maxv:.6g}, top_dim={dim}\n".format(
                                    step=policy_step,
                                    maxv=step_summary["continuous_max_abs"]["max"],
                                    dim=(step_summary["top_record"] or {}).get("max_dim", ""),
                                )
                            )
                            log_file.flush()

                        eager_action = drive_eager_action(
                            policy,
                            local_policy,
                            target_switcher,
                            processed_obs,
                            seed=seed,
                            device=args.device,
                        )
                        obs, reward, done, info = env.step(eager_action.tolist())
                        if done:
                            break
                        t += 1
                    except Exception as exc:
                        exception_message = str(exc)
                        log_file.write(f"Caught exception: {exception_message}\n")
                        break

                all_records = [record for step in focus_payloads for record in step["records"]]
                case_summary = {
                    "task_id": int(task_id),
                    "task_description": task_description,
                    "init_index": int(init_index),
                    "success_before_window_end": bool(done),
                    "exception": exception_message,
                    "drive_policy_steps": int(max(0, t - args.num_steps_wait)),
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "focus_steps_observed": [int(step["policy_step"]) for step in focus_payloads],
                    "focus_record_summary": summarize_focus_records(all_records),
                    "focus_step_summaries": focus_step_summaries,
                }
                trace_payload = {
                    **case_summary,
                    "task_suite_name": args.task_suite_name,
                    "torch_compile": result["torch_compile"],
                    "focus_steps": focus_payloads,
                }
                safe_task = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in task_description)[:80]
                trace_path = args.trace_dir / f"task={task_id:02d}--init={init_index:02d}--task={safe_task}.json"
                trace_path.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")
                result["trace_files"].append(str(trace_path))
                result["case_summaries"].append(case_summary)

    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 16.5 Step-focused Replay",
        "",
        "Boundary: eager-controlled same-observation replay. The simulator stays on the eager FP16 trajectory; focused observations are evaluated repeatedly with eager and compiled paths under matched denoising seeds.",
        "",
        "## Run",
        "",
        f"- Cases: `{result['case_list']}`",
        f"- Focus policy steps: `{result['focus_policy_steps']}`",
        f"- Seed offsets: `{result['seed_offsets']}`",
        f"- Repeats per seed: `{result['repeats']}`",
        f"- Warmup repeats per focused observation: `{result['warmup_repeats']}`",
        f"- Compile target: `{result['torch_compile']['target']}`",
        f"- Compile mode: `{result['torch_compile'].get('mode')}`",
        "",
        "## Focus Steps",
        "",
        "| case | policy step | samples | cont max max | cont max p99 | repeat span max | top dim | top seed offset | compiled p50 | eager p50 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for case in result["case_summaries"]:
        case_name = f"task{case['task_id']}:init{case['init_index']}"
        for step in case["focus_step_summaries"]:
            summary = step["summary"]
            top = summary.get("top_record") or {}
            lines.append(
                "| {case} | {step} | {samples} | {cmax:.6g} | {p99:.6g} | {span:.6g} | `{dim}` | {offset} | {compiled:.1f} ms | {eager:.1f} ms |".format(
                    case=case_name,
                    step=step["policy_step"],
                    samples=summary["samples"],
                    cmax=summary["continuous_max_abs"]["max"],
                    p99=summary["continuous_max_abs"]["p99"],
                    span=summary["same_seed_repeat_span"]["max"],
                    dim=top.get("max_dim", ""),
                    offset=top.get("seed_offset", ""),
                    compiled=1000.0 * summary["compiled_get_action_seconds"]["p50"],
                    eager=1000.0 * summary["eager_get_action_seconds"]["p50"],
                )
            )
    lines.extend(
        [
            "",
            "## Reading This",
            "",
            "- If the same step repeatedly owns the max drift, the spike is observation-local rather than random noise.",
            "- If `same_seed_repeat_span` is near zero, repeated calls are deterministic under fixed seed; seed changes then explain sensitivity.",
            "- This probe is not a closed-loop success benchmark; it is a numerical-behavior diagnostic before rollout expansion.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--case-list", default="6:8")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--focus-start", type=int, default=248)
    parser.add_argument("--focus-end", type=int, default=258)
    parser.add_argument("--focus-policy-steps", type=int, nargs="+")
    parser.add_argument("--seed-offsets", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--torch-compile-target", choices=TORCH_COMPILE_TARGETS, default="action_head_model")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", choices=["true", "false"])
    parser.add_argument("--torch-compile-cudagraph-mark-step", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/phase16_step_focused_replay"))
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/logs/phase16_step_focused_replay.log"))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase16_step_focused_replay.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase16_step_focused_replay.md"))
    args = parser.parse_args()
    if args.torch_compile_dynamic is not None:
        args.torch_compile_dynamic = args.torch_compile_dynamic == "true"

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    result = run_focused_replay(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "cases": len(result["case_summaries"]),
                "target": result["torch_compile"]["target"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
