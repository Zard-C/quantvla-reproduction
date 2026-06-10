"""LIBERO eval wrapper that runs a disjoint initial-state range.

The official GR00T LIBERO script uses `initial_states[episode_idx]`, so
`--num-trials-per-task 5` evaluates init indices 0..4. This wrapper keeps the
same policy client, action conversion, max-step logic, and rollout video writer,
but evaluates explicit init indices such as 5..14 for generalization checks
without repeating the accepted Phase 5 baseline inits.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pprint
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from timing_utils import REQUEST_SEED_KEY, summarize_breakdowns, summarize_float


def insert_paths(isaac_root: Path) -> None:
    sys.path.insert(0, str(isaac_root))


def summarize_obs(obs_dict):
    summary = {}
    for key, value in obs_dict.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"shape": tuple(value.shape), "dtype": value.dtype, "device": value.device}
        elif isinstance(value, np.ndarray):
            summary[key] = {"shape": value.shape, "dtype": value.dtype}
        else:
            summary[key] = type(value).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


class GR00TPolicy:
    """GR00T policy wrapper for LIBERO environments."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(self, host="localhost", port=5555, headless=False, gripper_oracle_port=None):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.gripper_oracle_policy = (
            ExternalRobotInferenceClient(host=host, port=gripper_oracle_port)
            if gripper_oracle_port is not None
            else None
        )
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

    def get_action(self, observation_dict, lang: str):
        action_array, _trace, _timing = self.get_action_with_trace(observation_dict, lang)
        return action_array

    def get_action_with_trace(self, observation_dict, lang: str, request_seed: int | None = None):
        started = time.perf_counter()
        obs_dict = self._process_observation(observation_dict, lang, request_seed=request_seed)
        preprocess_seconds = time.perf_counter() - started
        remote_started = time.perf_counter()
        action_chunk = self.policy.get_action(obs_dict)
        primary_remote_get_action_seconds = time.perf_counter() - remote_started
        gripper_oracle_remote_get_action_seconds = 0.0
        primary_gripper_action = None
        oracle_gripper_action = None
        if self.gripper_oracle_policy is not None:
            oracle_started = time.perf_counter()
            gripper_oracle_chunk = self.gripper_oracle_policy.get_action(obs_dict)
            gripper_oracle_remote_get_action_seconds = time.perf_counter() - oracle_started
            primary_gripper_action = copy.deepcopy(action_chunk["action.gripper"])
            oracle_gripper_action = copy.deepcopy(gripper_oracle_chunk["action.gripper"])
            action_chunk = {
                key: value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
                for key, value in action_chunk.items()
            }
            action_chunk["action.gripper"] = (
                gripper_oracle_chunk["action.gripper"].copy()
                if hasattr(gripper_oracle_chunk["action.gripper"], "copy")
                else copy.deepcopy(gripper_oracle_chunk["action.gripper"])
            )
        remote_get_action_seconds = (
            primary_remote_get_action_seconds + gripper_oracle_remote_get_action_seconds
        )
        postprocess_started = time.perf_counter()
        action_array = self._convert_to_libero_action(action_chunk, 0)
        action_trace = self._trace_action(action_chunk, 0)
        if self.gripper_oracle_policy is not None:
            action_trace["primary.action.gripper"] = float(
                np.asarray(primary_gripper_action).reshape(-1)[0]
            )
            action_trace["gripper_oracle.action.gripper"] = float(
                np.asarray(oracle_gripper_action).reshape(-1)[0]
            )
        postprocess_seconds = time.perf_counter() - postprocess_started
        timing = {
            "preprocess_seconds": float(preprocess_seconds),
            "remote_get_action_seconds": float(remote_get_action_seconds),
            "primary_remote_get_action_seconds": float(primary_remote_get_action_seconds),
            "gripper_oracle_remote_get_action_seconds": float(
                gripper_oracle_remote_get_action_seconds
            ),
            "postprocess_seconds": float(postprocess_seconds),
            "policy_total_seconds": float(time.perf_counter() - started),
        }
        return action_array, action_trace, timing

    def _process_observation(self, obs, lang: str, request_seed: int | None = None):
        from examples.Libero.eval.utils import get_libero_image, quat2axisangle

        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        if request_seed is not None:
            new_obs[REQUEST_SEED_KEY] = int(request_seed)
        if not self.headless:
            show_obs_images_cv2(new_obs)
        return new_obs

    def _convert_to_libero_action(self, action_chunk: dict[str, np.array], idx: int = 0) -> np.ndarray:
        from examples.Libero.eval.utils import normalize_gripper_action

        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array

    def _trace_action(self, action_chunk: dict[str, np.array], idx: int = 0) -> dict[str, float]:
        return {
            f"action.{key}": float(np.atleast_1d(action_chunk[f"action.{key}"][idx])[0])
            for key in self.action_keys
        }


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


def parse_case_list(case_list: str | None) -> list[tuple[int, int]] | None:
    if not case_list:
        return None
    pairs = []
    for item in case_list.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected case item 'task_id:init_index', got {item!r}")
        task_id, init_index = item.split(":", 1)
        pairs.append((int(task_id), int(init_index)))
    return pairs


def as_float_list(value) -> list[float]:
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def deterministic_policy_seed(base_seed: int, task_id: int, init_index: int, policy_step: int) -> int:
    return int(base_seed) + int(task_id) * 100_000 + int(init_index) * 1_000 + int(policy_step)


def write_episode_trace(
    trace_dir: Path | None,
    *,
    task_suite_name: str,
    task_id: int,
    task_description: str,
    init_index: int,
    episode_index: int,
    success: bool,
    exception: str | None,
    steps: list[dict],
) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_task = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in task_description)[:80]
    path = trace_dir / (
        f"task={task_id:02d}--init={init_index:02d}--episode={episode_index:03d}"
        f"--success={success}--task={safe_task}.json"
    )
    payload = {
        "task_suite_name": task_suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "init_index": init_index,
        "episode_index": episode_index,
        "success": bool(success),
        "exception": exception,
        "num_steps": len(steps),
        "steps": steps,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def eval_libero(args: argparse.Namespace) -> None:
    from examples.Libero.eval.utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        save_rollout_video,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    init_indices = list(range(args.init_start, args.init_start + args.num_inits))
    requested_cases = parse_case_list(args.case_list)
    cases_by_task: dict[int, list[int]] | None = None
    if requested_cases is not None:
        cases_by_task = {}
        for task_id, init_index in requested_cases:
            cases_by_task.setdefault(task_id, []).append(init_index)
        for task_id in cases_by_task:
            cases_by_task[task_id] = sorted(set(cases_by_task[task_id]))

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        print(f"Task suite: {args.task_suite_name}")
        print(f"Init indices: {init_indices}")
        if requested_cases is not None:
            print(f"Case list: {requested_cases}")
        log_file.write(f"Task suite: {args.task_suite_name}\n")
        log_file.write(f"Init indices: {init_indices}\n")
        if requested_cases is not None:
            log_file.write(f"Case list: {requested_cases}\n")
        if args.deterministic_policy_seeds:
            seed_line = (
                f"Deterministic policy seeds: base={args.policy_seed_base}, "
                "formula=base+task_id*100000+init_index*1000+policy_step"
            )
            print(seed_line)
            log_file.write(seed_line + "\n")

        total_episodes, total_successes = 0, 0
        all_policy_latencies: list[float] = []
        all_policy_breakdowns: list[dict[str, float]] = []
        episode_latency_rows: list[dict] = []
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            if cases_by_task is not None and task_id not in cases_by_task:
                continue
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            task_init_indices = cases_by_task[task_id] if cases_by_task is not None else init_indices
            max_init = max(task_init_indices)
            if max_init >= len(initial_states):
                raise IndexError(
                    f"Task {task_id} requested init {max_init}, "
                    f"but only {len(initial_states)} states are available"
                )

            env, task_description = get_libero_env(task, resolution=256)
            gr00t_policy = GR00TPolicy(
                host="localhost",
                port=args.port,
                headless=args.headless,
                gripper_oracle_port=args.gripper_oracle_port,
            )

            task_episodes, task_successes = 0, 0
            for init_index in tqdm.tqdm(task_init_indices):
                print(f"\nTask: {task_description}")
                print(f"Init index: {init_index}")
                log_file.write(f"\nTask: {task_description}\n")
                log_file.write(f"Task id: {task_id}\n")
                log_file.write(f"Init index: {init_index}\n")

                env.reset()
                obs = env.set_init_state(initial_states[init_index])

                done = False
                t = 0
                top_view = []
                wrist_view = []
                max_steps = max_steps_for_suite(args.task_suite_name)
                trace_steps: list[dict] = []
                episode_policy_latencies: list[float] = []
                episode_policy_breakdowns: list[dict[str, float]] = []
                exception_message = None

                print(f"Starting episode {task_episodes + 1}...")
                log_file.write(f"Starting episode {task_episodes + 1}...\n")
                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action())
                            t += 1
                            continue

                        img, wrist_img = get_libero_image(obs)
                        top_view.append(img)
                        wrist_view.append(wrist_img)

                        pre_eef_pos = as_float_list(obs.get("robot0_eef_pos", []))
                        pre_eef_quat = as_float_list(obs.get("robot0_eef_quat", []))
                        pre_gripper_qpos = as_float_list(obs.get("robot0_gripper_qpos", []))
                        policy_step = int(t - args.num_steps_wait)
                        request_seed = (
                            deterministic_policy_seed(
                                args.policy_seed_base,
                                task_id,
                                init_index,
                                policy_step,
                            )
                            if args.deterministic_policy_seeds
                            else None
                        )
                        action, raw_action_trace, policy_timing = gr00t_policy.get_action_with_trace(
                            obs,
                            task.language,
                            request_seed=request_seed,
                        )
                        policy_latency_seconds = float(policy_timing["policy_total_seconds"])
                        episode_policy_latencies.append(policy_latency_seconds)
                        all_policy_latencies.append(policy_latency_seconds)
                        episode_policy_breakdowns.append(policy_timing)
                        all_policy_breakdowns.append(policy_timing)
                        obs, reward, done, info = env.step(action.tolist())
                        trace_steps.append(
                            {
                                "step": int(t),
                                "policy_step": policy_step,
                                "request_seed": request_seed,
                                "pre_robot0_eef_pos": pre_eef_pos,
                                "pre_robot0_eef_quat": pre_eef_quat,
                                "pre_robot0_gripper_qpos": pre_gripper_qpos,
                                "post_robot0_eef_pos": as_float_list(obs.get("robot0_eef_pos", [])),
                                "post_robot0_eef_quat": as_float_list(obs.get("robot0_eef_quat", [])),
                                "post_robot0_gripper_qpos": as_float_list(obs.get("robot0_gripper_qpos", [])),
                                "raw_action": raw_action_trace,
                                "libero_action": as_float_list(action),
                                "policy_latency_seconds": float(policy_latency_seconds),
                                "policy_timing_seconds": policy_timing,
                                "reward": float(reward),
                                "done": bool(done),
                            }
                        )
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as exc:
                        exception_message = str(exc)
                        print(f"Caught exception: {exc}")
                        log_file.write(f"Caught exception: {exc}\n")
                        break

                task_episodes += 1
                total_episodes += 1
                episode_latency_summary = summarize_float(episode_policy_latencies)
                episode_breakdown_summary = summarize_breakdowns(episode_policy_breakdowns)
                episode_latency_rows.append(
                    {
                        "episode_index": int(total_episodes),
                        "task_id": int(task_id),
                        "init_index": int(init_index),
                        "success": bool(done),
                        "deterministic_policy_seeds": bool(args.deterministic_policy_seeds),
                        "policy_seed_base": int(args.policy_seed_base),
                        "policy_latency_seconds": episode_latency_summary,
                        "policy_breakdown_seconds": episode_breakdown_summary,
                    }
                )
                write_episode_trace(
                    args.trace_dir,
                    task_suite_name=args.task_suite_name,
                    task_id=task_id,
                    task_description=task_description,
                    init_index=init_index,
                    episode_index=total_episodes,
                    success=done,
                    exception=exception_message,
                    steps=trace_steps,
                )

                save_rollout_video(
                    top_view,
                    wrist_view,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                )

                print(f"Success: {done}")
                print(f"Policy latency seconds: {episode_latency_summary}")
                print(f"Policy breakdown seconds: {episode_breakdown_summary}")
                print(f"# episodes completed so far: {total_episodes}")
                print(
                    f"# successes: {total_successes} "
                    f"({total_successes / total_episodes * 100:.1f}%)"
                )
                log_file.write(f"Success: {done}\n")
                log_file.write(f"Policy latency seconds: {episode_latency_summary}\n")
                log_file.write(f"Policy breakdown seconds: {episode_breakdown_summary}\n")
                log_file.write(f"# episodes completed so far: {total_episodes}\n")
                log_file.write(
                    f"# successes: {total_successes} "
                    f"({total_successes / total_episodes * 100:.1f}%)\n"
                )
                log_file.flush()

            print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
            print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
            log_file.write(
                f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
            )
            log_file.write(
                f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
            )
            log_file.flush()

        final_latency_summary = summarize_float(all_policy_latencies)
        final_breakdown_summary = summarize_breakdowns(all_policy_breakdowns)
        print(f"Final policy latency seconds: {final_latency_summary}")
        print(f"Final policy breakdown seconds: {final_breakdown_summary}")
        log_file.write(f"Final policy latency seconds: {final_latency_summary}\n")
        log_file.write(f"Final policy breakdown seconds: {final_breakdown_summary}\n")
        log_file.flush()
        if args.latency_json is not None:
            args.latency_json.parent.mkdir(parents=True, exist_ok=True)
            args.latency_json.write_text(
                json.dumps(
                    {
                        "task_suite_name": args.task_suite_name,
                        "port": args.port,
                        "total_episodes": int(total_episodes),
                        "total_successes": int(total_successes),
                        "success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
                        "deterministic_policy_seeds": bool(args.deterministic_policy_seeds),
                        "policy_seed_base": int(args.policy_seed_base),
                        "policy_latency_seconds": final_latency_summary,
                        "policy_breakdown_seconds": final_breakdown_summary,
                        "episodes": episode_latency_rows,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--init-start", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=10)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--gripper-oracle-port",
        type=int,
        help="Optional second inference server; use its same-observation gripper channel only.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--case-list",
        help="Comma-separated task:init pairs, for example '8:7,8:9,4:10'.",
    )
    parser.add_argument("--trace-dir", type=Path, help="Optional directory for per-episode JSON traces.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/tmp/logs/libero_eval_libero_10_init5_14.log"),
    )
    parser.add_argument("--latency-json", type=Path, help="Optional JSON output for policy request latency statistics.")
    parser.add_argument("--deterministic-policy-seeds", action="store_true")
    parser.add_argument("--policy-seed-base", type=int, default=20260609)
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    insert_paths(args.isaac_root)
    eval_libero(args)


if __name__ == "__main__":
    main()
