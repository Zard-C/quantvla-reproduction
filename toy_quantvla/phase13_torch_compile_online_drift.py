"""Online action drift probe for torch.compile GR00T action head.

The environment is stepped with the eager FP16 policy.  At every live
observation, the script evaluates both eager and compiled action-head paths
under the same denoising seed, records the action drift, and then applies the
eager action to LIBERO.  This isolates same-observation compile drift from
closed-loop simulator feedback.
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from phase3_fake_quant_forward import compare_actions, set_seed, set_submodule
from phase3_gr00t_smoke import _insert_paths
from timing_utils import summarize_float


ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
TORCH_COMPILE_TARGETS = ("action_head_model",)
EPS = 1e-8


def synchronize_if_cuda(device: str) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def show_obs_images_cv2(new_obs: dict[str, Any]) -> None:
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


def parse_case_list(case_list: str) -> list[tuple[int, int]]:
    out = []
    for item in case_list.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected case item 'task_id:init_index', got {item!r}")
        task_id, init_index = item.split(":", 1)
        out.append((int(task_id), int(init_index)))
    if not out:
        raise ValueError("--case-list must include at least one task:init pair")
    return out


def as_float_list(value: Any) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 600
    if task_suite_name == "libero_10":
        return 1000
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def deterministic_policy_seed(base_seed: int, task_id: int, init_index: int, policy_step: int) -> int:
    return int(base_seed) + int(task_id) * 100_000 + int(init_index) * 1_000 + int(policy_step)


def summarize_floats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min())}


def compare_libero_actions(eager: np.ndarray, compiled: np.ndarray) -> dict[str, Any]:
    diff = compiled.astype(np.float64) - eager.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
        "eager_l2": float(np.linalg.norm(eager.astype(np.float64))),
        "compiled_l2": float(np.linalg.norm(compiled.astype(np.float64))),
        "per_dim_diff": diff.tolist(),
    }


def aggregate_step_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        empty = {"mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "policy_steps": 0,
            "raw_relative_rmse": empty,
            "raw_cosine": empty,
            "raw_max_abs_diff": empty,
            "libero_action_rmse": empty,
            "libero_action_max_abs_diff": empty,
            "eager_get_action_seconds": empty,
            "compiled_get_action_seconds": empty,
            "compiled_over_eager_time": empty,
        }
    raw_rrmse = [float(step["raw_action_metrics"]["relative_rmse"]) for step in steps]
    raw_cos = [float(step["raw_action_metrics"]["cosine"]) for step in steps]
    raw_max = [float(step["raw_action_metrics"]["max_abs_diff"]) for step in steps]
    libero_rmse = [float(step["libero_action_metrics"]["rmse"]) for step in steps]
    libero_max = [float(step["libero_action_metrics"]["max_abs_diff"]) for step in steps]
    eager_seconds = [float(step["timing"]["eager_get_action_seconds"]) for step in steps]
    compiled_seconds = [float(step["timing"]["compiled_get_action_seconds"]) for step in steps]
    compiled_over_eager = [float(step["timing"]["compiled_over_eager_time"]) for step in steps]
    return {
        "policy_steps": len(steps),
        "raw_relative_rmse": summarize_floats(raw_rrmse),
        "raw_cosine": summarize_floats(raw_cos),
        "raw_max_abs_diff": summarize_floats(raw_max),
        "libero_action_rmse": summarize_floats(libero_rmse),
        "libero_action_max_abs_diff": summarize_floats(libero_max),
        "eager_get_action_seconds": summarize_floats(eager_seconds),
        "compiled_get_action_seconds": summarize_floats(compiled_seconds),
        "compiled_over_eager_time": summarize_floats(compiled_over_eager),
    }


def window_summary(steps: list[dict[str, Any]], windows: list[int]) -> list[dict[str, Any]]:
    rows = []
    for end in windows:
        subset = steps[: min(int(end), len(steps))]
        metrics = aggregate_step_metrics(subset)
        rows.append({"first_steps": int(end), "actual_steps": len(subset), "metrics": metrics})
    return rows


class CompileDriftPolicy:
    def __init__(
        self,
        policy: Any,
        *,
        headless: bool,
        device: str,
        compiled_action_head_model: Any,
        eager_action_head_model: Any,
    ):
        self.policy = policy
        self.headless = headless
        self.device = device
        self.compiled_action_head_model = compiled_action_head_model
        self.eager_action_head_model = eager_action_head_model

    def process_observation(self, obs: dict[str, Any], lang: str) -> dict[str, Any]:
        from examples.Libero.eval.utils import get_libero_image, quat2axisangle

        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]], dtype=np.float32),
            "state.y": np.array([[xyz[1]]], dtype=np.float32),
            "state.z": np.array([[xyz[2]]], dtype=np.float32),
            "state.roll": np.array([[rpy[0]]], dtype=np.float32),
            "state.pitch": np.array([[rpy[1]]], dtype=np.float32),
            "state.yaw": np.array([[rpy[2]]], dtype=np.float32),
            "state.gripper": np.expand_dims(np.asarray(gripper, dtype=np.float32), axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def get_eager_compiled(self, processed_obs: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        set_submodule(self.policy.model, "action_head.model", self.eager_action_head_model)
        set_seed(seed)
        synchronize_if_cuda(self.device)
        eager_started = time.perf_counter()
        with torch.no_grad():
            eager = self.policy.get_action(processed_obs)
        synchronize_if_cuda(self.device)
        eager_seconds = time.perf_counter() - eager_started

        set_submodule(self.policy.model, "action_head.model", self.compiled_action_head_model)
        set_seed(seed)
        synchronize_if_cuda(self.device)
        compiled_started = time.perf_counter()
        with torch.no_grad():
            compiled = self.policy.get_action(processed_obs)
        synchronize_if_cuda(self.device)
        compiled_seconds = time.perf_counter() - compiled_started

        set_submodule(self.policy.model, "action_head.model", self.eager_action_head_model)
        return eager, compiled, {
            "eager_get_action_seconds": float(eager_seconds),
            "compiled_get_action_seconds": float(compiled_seconds),
            "compiled_over_eager_time": float(compiled_seconds / max(eager_seconds, EPS)),
        }

    def convert_to_libero_action(self, action_chunk: dict[str, Any], idx: int = 0) -> np.ndarray:
        from examples.Libero.eval.utils import normalize_gripper_action

        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in ACTION_KEYS
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array

    def trace_raw_action(self, action_chunk: dict[str, Any], idx: int = 0) -> dict[str, float]:
        return {
            f"action.{key}": float(np.atleast_1d(action_chunk[f"action.{key}"][idx])[0])
            for key in ACTION_KEYS
        }


def write_episode(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 13 torch.compile Online Drift",
        "",
        "Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.",
        "",
        "## Run",
        "",
        f"- Cases: `{result['case_list']}`",
        f"- Episodes: `{result['episodes']}`",
        f"- Eager successes: `{result['eager_successes']}/{result['episodes']}`",
        f"- Compile target: `{result['torch_compile']['target']}`",
        f"- Compile mode: `{result['torch_compile']['mode']}`",
        "",
        "## Episodes",
        "",
        "| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["episode_summaries"]:
        metrics = row["drift_summary"]
        lines.append(
            "| {task} | {init} | {success} | {steps} | {rrmse:.6g} | {rawmax:.6g} | {lrmse:.6g} | {ratio:.4g} |".format(
                task=row["task_id"],
                init=row["init_index"],
                success=str(row["success"]),
                steps=metrics["policy_steps"],
                rrmse=metrics["raw_relative_rmse"]["mean"],
                rawmax=metrics["raw_max_abs_diff"]["max"],
                lrmse=metrics["libero_action_rmse"]["mean"],
                ratio=metrics["compiled_over_eager_time"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.",
            "- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_online_drift(args: argparse.Namespace) -> dict[str, Any]:
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image, save_rollout_video
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

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

    eager_action_head_model = policy.model.action_head.model
    compile_kwargs: dict[str, Any] = {
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
    }
    if args.torch_compile_dynamic is not None:
        compile_kwargs["dynamic"] = args.torch_compile_dynamic == "true"
    compile_started = time.perf_counter()
    compiled_action_head_model = torch.compile(eager_action_head_model, **compile_kwargs)
    compile_wrap_seconds = time.perf_counter() - compile_started

    local_policy = CompileDriftPolicy(
        policy,
        headless=args.headless,
        device=args.device,
        compiled_action_head_model=compiled_action_head_model,
        eager_action_head_model=eager_action_head_model,
    )

    cases = parse_case_list(args.case_list)
    cases_by_task: dict[int, list[int]] = {}
    for task_id, init_index in cases:
        cases_by_task.setdefault(task_id, []).append(init_index)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()

    result: dict[str, Any] = {
        "boundary": "online eager-controlled torch.compile drift diagnostic, not compiled closed-loop benchmark",
        "model_path": str(args.model_path),
        "task_suite_name": args.task_suite_name,
        "case_list": args.case_list,
        "denoising_steps": args.denoising_steps,
        "num_steps_wait": args.num_steps_wait,
        "max_policy_steps": args.max_policy_steps,
        "base_seed": args.base_seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "device": args.device,
        "model_load_seconds": float(model_load_seconds),
        "torch_compile": {
            "target": args.torch_compile_target,
            "backend": args.torch_compile_backend,
            "mode": args.torch_compile_mode,
            "fullgraph": bool(args.torch_compile_fullgraph),
            "dynamic": args.torch_compile_dynamic,
            "wrap_seconds": float(compile_wrap_seconds),
        },
        "episode_summaries": [],
        "episode_trace_files": [],
    }

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps({k: v for k, v in result.items() if k not in {"episode_summaries", "episode_trace_files"}}, indent=2) + "\n")
        total_successes = 0
        total_episodes = 0
        for task_id in tqdm.tqdm(sorted(cases_by_task)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=256)
            for init_index in tqdm.tqdm(sorted(set(cases_by_task[task_id]))):
                if init_index >= len(initial_states):
                    raise IndexError(f"Task {task_id} requested init {init_index}, but only {len(initial_states)} states are available")

                env.reset()
                obs = env.set_init_state(initial_states[init_index])
                max_policy_steps = args.max_policy_steps or max_steps_for_suite(args.task_suite_name)
                max_total_steps = max_policy_steps + args.num_steps_wait
                top_view: list[Any] = []
                wrist_view: list[Any] = []
                trace_steps: list[dict[str, Any]] = []
                exception_message = None
                done = False
                reward = 0.0
                t = 0
                started = time.perf_counter()

                print(f"\nTask {task_id}: {task_description}")
                print(f"Init index: {init_index}")
                log_file.write(f"\nTask: {task_description}\nTask id: {task_id}\nInit index: {init_index}\n")
                while t < max_total_steps:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        img, wrist_img = get_libero_image(obs)
                        top_view.append(img)
                        wrist_view.append(wrist_img)
                        processed_obs = local_policy.process_observation(obs, task.language)
                        policy_step = t - args.num_steps_wait
                        seed = deterministic_policy_seed(args.base_seed, task_id, init_index, policy_step)
                        pre_eef_pos = as_float_list(obs.get("robot0_eef_pos", []))
                        pre_eef_quat = as_float_list(obs.get("robot0_eef_quat", []))
                        pre_gripper_qpos = as_float_list(obs.get("robot0_gripper_qpos", []))

                        eager_chunk, compiled_chunk, timing = local_policy.get_eager_compiled(processed_obs, seed)
                        eager_action = local_policy.convert_to_libero_action(eager_chunk)
                        compiled_action = local_policy.convert_to_libero_action(compiled_chunk)
                        raw_metrics = compare_actions(eager_chunk, compiled_chunk)
                        libero_metrics = compare_libero_actions(eager_action, compiled_action)

                        obs, reward, done, info = env.step(eager_action.tolist())
                        trace_steps.append(
                            {
                                "step": int(t),
                                "policy_step": int(policy_step),
                                "denoising_seed": int(seed),
                                "pre_robot0_eef_pos": pre_eef_pos,
                                "pre_robot0_eef_quat": pre_eef_quat,
                                "pre_robot0_gripper_qpos": pre_gripper_qpos,
                                "post_robot0_eef_pos": as_float_list(obs.get("robot0_eef_pos", [])),
                                "post_robot0_eef_quat": as_float_list(obs.get("robot0_eef_quat", [])),
                                "post_robot0_gripper_qpos": as_float_list(obs.get("robot0_gripper_qpos", [])),
                                "eager_raw_action": local_policy.trace_raw_action(eager_chunk),
                                "compiled_raw_action": local_policy.trace_raw_action(compiled_chunk),
                                "eager_libero_action": as_float_list(eager_action),
                                "compiled_libero_action": as_float_list(compiled_action),
                                "raw_action_metrics": raw_metrics,
                                "libero_action_metrics": libero_metrics,
                                "timing": timing,
                                "reward": float(reward),
                                "done": bool(done),
                            }
                        )
                        if done:
                            break
                        t += 1
                    except Exception as exc:
                        exception_message = str(exc)
                        print(f"Caught exception: {exc}")
                        log_file.write(f"Caught exception: {exc}\n")
                        break

                total_episodes += 1
                if done:
                    total_successes += 1
                drift_summary = aggregate_step_metrics(trace_steps)
                episode_payload = {
                    "task_suite_name": args.task_suite_name,
                    "task_id": task_id,
                    "task_description": task_description,
                    "init_index": init_index,
                    "success": bool(done),
                    "exception": exception_message,
                    "num_steps": len(trace_steps),
                    "elapsed_seconds": time.perf_counter() - started,
                    "torch_compile": result["torch_compile"],
                    "drift_summary": drift_summary,
                    "window_summaries": window_summary(trace_steps, args.summary_windows),
                    "steps": trace_steps,
                }
                safe_task = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in task_description)[:80]
                trace_path = args.trace_dir / (
                    f"task={task_id:02d}--init={init_index:02d}--success={bool(done)}--task={safe_task}.json"
                )
                write_episode(trace_path, episode_payload)
                result["episode_trace_files"].append(str(trace_path))
                result["episode_summaries"].append({k: v for k, v in episode_payload.items() if k != "steps"})

                if not args.no_video:
                    save_rollout_video(
                        top_view,
                        wrist_view,
                        total_episodes,
                        success=done,
                        task_description=task_description,
                        log_file=log_file,
                    )

                print(f"Success: {done}")
                print(pprint.pformat(drift_summary))
                log_file.write(f"Success: {done}\n")
                log_file.write(pprint.pformat(drift_summary) + "\n")
                log_file.flush()

        result["episodes"] = total_episodes
        result["eager_successes"] = total_successes
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--case-list", default="4:6,6:0")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260613)
    parser.add_argument("--torch-compile-target", choices=TORCH_COMPILE_TARGETS, default="action_head_model")
    parser.add_argument("--torch-compile-backend", default="inductor")
    parser.add_argument("--torch-compile-mode", default="reduce-overhead")
    parser.add_argument("--torch-compile-fullgraph", action="store_true")
    parser.add_argument("--torch-compile-dynamic", choices=["true", "false"])
    parser.add_argument("--summary-windows", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100, 150, 200, 250])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/phase13_torch_compile_online_drift"))
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/logs/phase13_torch_compile_online_drift.log"))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase13_torch_compile_online_drift.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase13_torch_compile_online_drift.md"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    result = run_online_drift(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, args.output_md)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "episodes": result["episodes"],
                "eager_successes": result["eager_successes"],
                "compile_mode": result["torch_compile"]["mode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
