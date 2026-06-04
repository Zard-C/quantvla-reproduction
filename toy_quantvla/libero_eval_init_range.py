"""LIBERO eval wrapper that runs a disjoint initial-state range.

The official GR00T LIBERO script uses `initial_states[episode_idx]`, so
`--num-trials-per-task 5` evaluates init indices 0..4. This wrapper keeps the
same policy client, action conversion, max-step logic, and rollout video writer,
but evaluates explicit init indices such as 5..14 for generalization checks
without repeating the accepted Phase 5 baseline inits.
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from libero.libero import benchmark


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

    def __init__(self, host="localhost", port=5555, headless=False):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

    def get_action(self, observation_dict, lang: str):
        obs_dict = self._process_observation(observation_dict, lang)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, 0)

    def _process_observation(self, obs, lang: str):
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

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("w", encoding="utf-8") as log_file:
        print(f"Task suite: {args.task_suite_name}")
        print(f"Init indices: {init_indices}")
        log_file.write(f"Task suite: {args.task_suite_name}\n")
        log_file.write(f"Init indices: {init_indices}\n")

        total_episodes, total_successes = 0, 0
        for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            max_init = max(init_indices)
            if max_init >= len(initial_states):
                raise IndexError(
                    f"Task {task_id} requested init {max_init}, "
                    f"but only {len(initial_states)} states are available"
                )

            env, task_description = get_libero_env(task, resolution=256)
            gr00t_policy = GR00TPolicy(host="localhost", port=args.port, headless=args.headless)

            task_episodes, task_successes = 0, 0
            for init_index in tqdm.tqdm(init_indices):
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

                        action = gr00t_policy.get_action(obs, task.language)
                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as exc:
                        print(f"Caught exception: {exc}")
                        log_file.write(f"Caught exception: {exc}\n")
                        break

                task_episodes += 1
                total_episodes += 1

                save_rollout_video(
                    top_view,
                    wrist_view,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    log_file=log_file,
                )

                print(f"Success: {done}")
                print(f"# episodes completed so far: {total_episodes}")
                print(
                    f"# successes: {total_successes} "
                    f"({total_successes / total_episodes * 100:.1f}%)"
                )
                log_file.write(f"Success: {done}\n")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--init-start", type=int, default=5)
    parser.add_argument("--num-inits", type=int, default=10)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/tmp/logs/libero_eval_libero_10_init5_14.log"),
    )
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    insert_paths(args.isaac_root)
    eval_libero(args)


if __name__ == "__main__":
    main()
