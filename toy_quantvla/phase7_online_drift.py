"""Online action-drift probe for Blackwell FP4-like QuantVLA.

This script runs a LIBERO environment with the FP16 teacher in control.  At each
live observation it evaluates both teacher and student actions under the same
denoising seed, records their drift, and then steps the environment with the
teacher action.  It is an online drift diagnostic, not a success-rate benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import pprint
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from phase3_activation_capture import build_variant_observation
from phase3_atm_ohb_forward import (
    ATMOHBProcessor,
    collect_attention_stats,
    compute_atm_ohb_scales,
    install_attention_processors,
    scale_summary,
    select_scales,
)
from phase3_fake_quant_forward import compare_actions, set_seed, set_submodule
from phase3_gr00t_smoke import _insert_paths
from phase6_w4a16_scopes import SCOPE_CHOICES, include_module_for_scope, module_family, scope_description


EPS = 1e-8
FP4_E2M1_ABS_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@dataclass
class FP4RuntimeState:
    enabled: bool = False


@dataclass
class FP4Record:
    module: str
    family: str
    calls: int = 0
    input_rel_rmse: list[float] = field(default_factory=list)
    weight_rel_rmse: list[float] = field(default_factory=list)

    def add(self, x: Any, x_q: Any, weight: Any, weight_q: Any) -> None:
        self.calls += 1
        self.input_rel_rmse.append(relative_rmse_tensor(x, x_q))
        self.weight_rel_rmse.append(relative_rmse_tensor(weight, weight_q))

    def to_result(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "family": self.family,
            "calls": self.calls,
            "input_relative_rmse": summarize_floats(self.input_rel_rmse),
            "weight_relative_rmse": summarize_floats(self.weight_rel_rmse),
        }


def summarize_floats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "max": 0.0, "min": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min())}


def relative_rmse_tensor(x: Any, y: Any) -> float:
    xf = x.detach().float()
    yf = y.detach().float()
    return float(torch.sqrt(torch.mean((yf - xf).square()) / torch.mean(xf.square()).clamp_min(EPS)).item())


def synchronize_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def fp4_e2m1_codebook(device: Any, dtype: Any) -> Any:
    vals = torch.tensor(FP4_E2M1_ABS_VALUES, device=device, dtype=torch.float32)
    vals = torch.cat([-vals.flip(0)[:-1], vals], dim=0)
    return vals.to(dtype=dtype)


def fake_quant_blockscaled_fp4(
    x: Any,
    *,
    block_size: int = 32,
    block_dim: int = -1,
) -> Any:
    """Approximate Blackwell block-scaled FP4.

    The codebook approximates Float4E2M1 finite values and the scale is absmax
    per block divided by the largest representable magnitude.  This is a
    numerical proxy for online drift, not a bit-exact CUTLASS packing path.
    """

    if not torch.is_tensor(x) or not torch.is_floating_point(x):
        return x
    if x.numel() == 0:
        return x

    orig_dtype = x.dtype
    block_dim = block_dim % x.ndim
    moved = x.detach().float().movedim(block_dim, -1).contiguous()
    last = moved.shape[-1]
    pad = (-last) % block_size
    if pad:
        moved = torch.nn.functional.pad(moved, (0, pad))
    blocks = moved.reshape(*moved.shape[:-1], moved.shape[-1] // block_size, block_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / 6.0
    normalized = blocks / scale
    codebook = fp4_e2m1_codebook(normalized.device, normalized.dtype)
    nearest = torch.argmin((normalized.unsqueeze(-1) - codebook).abs(), dim=-1)
    quantized = codebook[nearest] * scale
    quantized = quantized.reshape(*moved.shape)
    if pad:
        quantized = quantized[..., :last]
    return quantized.movedim(-1, block_dim).to(dtype=orig_dtype)


class FP4LikeLinear:
    """Module wrapper with a runtime switch for teacher/student evaluation."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        import torch.nn as nn
        import torch.nn.functional as F

        class _FP4LikeLinear(nn.Module):
            def __init__(
                self,
                base: nn.Linear,
                name: str,
                family: str,
                state: FP4RuntimeState,
                record: FP4Record,
                block_size: int,
            ):
                super().__init__()
                self.base = base
                self.name = name
                self.family = family
                self.state = state
                self.record = record
                self.block_size = block_size
                self.in_features = base.in_features
                self.out_features = base.out_features

            @property
            def weight(self) -> Any:
                return self.base.weight

            @property
            def bias(self) -> Any:
                return self.base.bias

            def forward(self, x: Any) -> Any:
                if not self.state.enabled:
                    return self.base(x)
                x_q = fake_quant_blockscaled_fp4(x, block_size=self.block_size, block_dim=-1)
                w_q = fake_quant_blockscaled_fp4(self.base.weight, block_size=self.block_size, block_dim=1)
                self.record.add(x, x_q, self.base.weight, w_q)
                return F.linear(x_q, w_q.to(dtype=self.base.weight.dtype), self.base.bias)

            def extra_repr(self) -> str:
                return (
                    f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
                    f"family={self.family}, fp4_like_block={self.block_size}"
                )

        return _FP4LikeLinear(*args, **kwargs)


class StudentOnlyATMOHBProcessor(ATMOHBProcessor):
    """Apply ATM/OHB scales only while the FP4-like student path is enabled."""

    def __init__(
        self,
        *,
        name: str,
        state: FP4RuntimeState,
        alpha: float | None = None,
        beta: float | None = None,
    ):
        super().__init__(name=name, alpha=alpha, beta=beta)
        self.state = state
        self.student_alpha = alpha
        self.student_beta = beta

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        old_alpha, old_beta = self.alpha, self.beta
        if self.state.enabled:
            self.alpha, self.beta = self.student_alpha, self.student_beta
        else:
            self.alpha, self.beta = None, None
        try:
            return super().__call__(*args, **kwargs)
        finally:
            self.alpha, self.beta = old_alpha, old_beta


def patch_fp4_like_modules(model: Any, scope: str, block_size: int) -> tuple[FP4RuntimeState, dict[str, Any], dict[str, FP4Record]]:
    import torch.nn as nn

    state = FP4RuntimeState(enabled=False)
    originals: dict[str, Any] = {}
    records: dict[str, FP4Record] = {}
    for name, module in list(model.named_modules()):
        family = module_family(name)
        if family is None or not include_module_for_scope(name, scope) or not isinstance(module, nn.Linear):
            continue
        record = FP4Record(module=name, family=family)
        originals[name] = module
        records[name] = record
        set_submodule(
            model,
            name,
            FP4LikeLinear(module, name=name, family=family, state=state, record=record, block_size=block_size),
        )
    return state, originals, records


def get_submodule(root: Any, name: str) -> Any:
    current = root
    for part in name.split("."):
        current = current[int(part)] if part.isdigit() else getattr(current, part)
    return current


def install_student_only_attention_processors(
    model: Any,
    *,
    state: FP4RuntimeState,
    mode: str,
    scales: dict[str, dict[str, float]],
) -> dict[str, Any]:
    originals, _stats = install_attention_processors(model, mode="collect")
    selected = select_scales(scales, mode)
    for name in list(originals):
        module = get_submodule(model, name)
        scale = selected.get(name, {})
        module.set_processor(
            StudentOnlyATMOHBProcessor(
                name=name,
                state=state,
                alpha=scale.get("alpha"),
                beta=scale.get("beta"),
            )
        )
    return originals


def build_synthetic_calibration_observations(
    *,
    model_path: Path,
    embodiment_tag: str,
    variants: list[str],
    num_observations: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    return [
        {
            "variant": variants[idx % len(variants)],
            "seed": base_seed + idx,
            "obs": build_variant_observation(
                model_path,
                embodiment_tag,
                variants[idx % len(variants)],
                idx,
            ),
        }
        for idx in range(num_observations)
    ]


def collect_attention_stats_with_state(
    policy: Any,
    observations: list[dict[str, Any]],
    state: FP4RuntimeState,
    *,
    enabled: bool,
) -> Any:
    old_enabled = state.enabled
    state.enabled = enabled
    try:
        return collect_attention_stats(policy, observations)
    finally:
        state.enabled = old_enabled


def aggregate_records(records: dict[str, FP4Record]) -> dict[str, Any]:
    rows = [record.to_result() for record in records.values()]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    family_summary = {}
    for family, family_rows in by_family.items():
        family_summary[family] = {
            "modules": len(family_rows),
            "calls": int(sum(row["calls"] for row in family_rows)),
            "input_relative_rmse_mean": float(np.mean([row["input_relative_rmse"]["mean"] for row in family_rows])),
            "weight_relative_rmse_mean": float(np.mean([row["weight_relative_rmse"]["mean"] for row in family_rows])),
        }
    return {"modules": rows, "family_summary": family_summary}


def show_obs_images_cv2(new_obs: dict[str, Any]) -> None:
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


class LocalGR00TPolicy:
    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    def __init__(self, policy: Any, *, headless: bool, state: FP4RuntimeState, device: str):
        self.policy = policy
        self.headless = headless
        self.state = state
        self.device = device

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

    def get_teacher_student(self, processed_obs: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
        self.state.enabled = False
        set_seed(seed)
        synchronize_if_cuda(self.device)
        teacher_started = time.perf_counter()
        teacher = self.policy.get_action(processed_obs)
        synchronize_if_cuda(self.device)
        teacher_seconds = time.perf_counter() - teacher_started
        self.state.enabled = True
        set_seed(seed)
        synchronize_if_cuda(self.device)
        student_started = time.perf_counter()
        student = self.policy.get_action(processed_obs)
        synchronize_if_cuda(self.device)
        student_seconds = time.perf_counter() - student_started
        self.state.enabled = False
        return teacher, student, {
            "teacher_get_action_seconds": float(teacher_seconds),
            "student_get_action_seconds": float(student_seconds),
            "student_over_teacher_time": float(student_seconds / max(teacher_seconds, EPS)),
        }

    def convert_to_libero_action(self, action_chunk: dict[str, Any], idx: int = 0) -> np.ndarray:
        from examples.Libero.eval.utils import normalize_gripper_action

        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array

    def trace_raw_action(self, action_chunk: dict[str, Any], idx: int = 0) -> dict[str, float]:
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


def compare_libero_actions(teacher: np.ndarray, student: np.ndarray) -> dict[str, Any]:
    diff = student.astype(np.float64) - teacher.astype(np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "l2": float(np.linalg.norm(diff)),
        "teacher_l2": float(np.linalg.norm(teacher.astype(np.float64))),
        "student_l2": float(np.linalg.norm(student.astype(np.float64))),
        "per_dim_diff": diff.tolist(),
    }


def aggregate_step_metrics(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        empty = {"mean": 0.0, "max": 0.0, "min": 0.0}
        return {
            "policy_steps": 0,
            "raw_relative_rmse": empty,
            "raw_cosine": empty,
            "libero_action_rmse": empty,
            "libero_action_max_abs_diff": empty,
            "teacher_get_action_seconds": empty,
            "student_get_action_seconds": empty,
            "student_over_teacher_time": empty,
        }
    raw_rrmse = [float(step["raw_action_metrics"]["relative_rmse"]) for step in steps]
    raw_cos = [float(step["raw_action_metrics"]["cosine"]) for step in steps]
    libero_rmse = [float(step["libero_action_metrics"]["rmse"]) for step in steps]
    libero_max = [float(step["libero_action_metrics"]["max_abs_diff"]) for step in steps]
    teacher_seconds = [float(step["timing"]["teacher_get_action_seconds"]) for step in steps]
    student_seconds = [float(step["timing"]["student_get_action_seconds"]) for step in steps]
    student_over_teacher = [float(step["timing"]["student_over_teacher_time"]) for step in steps]
    return {
        "policy_steps": len(steps),
        "raw_relative_rmse": summarize_floats(raw_rrmse),
        "raw_cosine": summarize_floats(raw_cos),
        "libero_action_rmse": summarize_floats(libero_rmse),
        "libero_action_max_abs_diff": summarize_floats(libero_max),
        "teacher_get_action_seconds": summarize_floats(teacher_seconds),
        "student_get_action_seconds": summarize_floats(student_seconds),
        "student_over_teacher_time": summarize_floats(student_over_teacher),
    }


def write_episode(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase 7 Online Drift Smoke",
        "",
        "Boundary: this is an online teacher-controlled drift diagnostic. The FP16 teacher steps the LIBERO environment; the FP4-like student is evaluated on the same live observations under matched denoising seeds. This is not a quantized closed-loop success-rate benchmark.",
        "",
        "## Run",
        "",
        f"- Scope: `{result['scope']}` ({result['scope_description']})",
        f"- Mode: `{result['mode']}`",
        f"- Quant format: `{result['quant_format']}`",
        f"- FP4 block size: `{result['fp4_block_size']}`",
        f"- Cases: `{result['case_list']}`",
        f"- Episodes: `{result['episodes']}`",
        f"- Teacher successes: `{result['teacher_successes']}/{result['episodes']}`",
        "",
        "## Episodes",
        "",
        "| task | init | success | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff | student/teacher time | exception |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["episode_summaries"]:
        metrics = row["drift_summary"]
        lines.append(
            "| {task} | {init} | {success} | {steps} | {rrmse:.6g} | {cos:.6g} | {lrmse:.6g} | {maxdiff:.6g} | {ratio:.4g} | {exc} |".format(
                task=row["task_id"],
                init=row["init_index"],
                success=str(row["success"]),
                steps=metrics["policy_steps"],
                rrmse=metrics["raw_relative_rmse"]["mean"],
                cos=metrics["raw_cosine"]["mean"],
                lrmse=metrics["libero_action_rmse"]["mean"],
                maxdiff=metrics["libero_action_max_abs_diff"]["max"],
                ratio=metrics["student_over_teacher_time"]["mean"],
                exc=row.get("exception") or "",
            )
        )
    if result.get("attention_scale_summary"):
        sc = result["attention_scale_summary"]
        lines.extend(
            [
                "",
                "## ATM/OHB Scale Summary",
                "",
                f"- Attention processors: `{result['attention_processors']}`",
                f"- Calibration observations: `{result['num_calibration_observations']}`",
                f"- Alpha mean/min/max: `{sc['alpha']['mean']:.6g}` / `{sc['alpha']['min']:.6g}` / `{sc['alpha']['max']:.6g}`",
                f"- Beta mean/min/max: `{sc['beta']['mean']:.6g}` / `{sc['beta']['min']:.6g}` / `{sc['beta']['max']:.6g}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Same-observation drift is measured before the environment step, so the teacher and student see exactly the same simulator state.",
            "- The environment executes only the teacher action. This isolates numerical action drift from closed-loop trajectory divergence.",
            "- A follow-up paired closed-loop run is only meaningful after this same-observation drift is acceptably small.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_online_drift(args: argparse.Namespace) -> dict[str, Any]:
    from examples.Libero.eval.utils import get_libero_dummy_action, get_libero_env, get_libero_image, save_rollout_video
    from examples.Libero.custom_data_config import LiberoDataConfig
    from gr00t.model.policy import Gr00tPolicy

    data_config = LiberoDataConfig()
    load_started = time.time()
    policy = Gr00tPolicy(
        model_path=str(args.model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    model_load_seconds = time.time() - load_started

    state, _originals, records = patch_fp4_like_modules(policy.model, args.scope, args.fp4_block_size)
    attention_originals: dict[str, Any] = {}
    attention_scales: dict[str, dict[str, float]] = {}
    attention_scale_summary = None
    calibration_seconds = 0.0
    if args.mode != "none":
        calibration_started = time.time()
        calibration_variants = [item.strip() for item in args.calibration_variants.split(",") if item.strip()]
        calibration_observations = build_synthetic_calibration_observations(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            variants=calibration_variants,
            num_observations=args.num_calibration_observations,
            base_seed=args.calibration_base_seed,
        )
        teacher_stats = collect_attention_stats_with_state(
            policy, calibration_observations, state, enabled=False
        )
        student_stats = collect_attention_stats_with_state(
            policy, calibration_observations, state, enabled=True
        )
        attention_scales = compute_atm_ohb_scales(
            teacher_stats, student_stats, log_clamp=args.log_clamp
        )
        attention_scale_summary = scale_summary(attention_scales)
        attention_originals = install_student_only_attention_processors(
            policy.model,
            state=state,
            mode=args.mode,
            scales=attention_scales,
        )
        calibration_seconds = time.time() - calibration_started

    local_policy = LocalGR00TPolicy(policy, headless=args.headless, state=state, device=args.device)
    cases = parse_case_list(args.case_list)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    cases_by_task: dict[int, list[int]] = {}
    for task_id, init_index in cases:
        cases_by_task.setdefault(task_id, []).append(init_index)

    result: dict[str, Any] = {
        "boundary": "online teacher-controlled drift diagnostic, not quantized closed-loop benchmark",
        "model_path": str(args.model_path),
        "task_suite_name": args.task_suite_name,
        "case_list": args.case_list,
        "scope": args.scope,
        "scope_description": scope_description(args.scope),
        "mode": args.mode,
        "quant_format": "fp4_e2m1_blockscaled_fake",
        "fp4_block_size": args.fp4_block_size,
        "calibration_source": "synthetic" if args.mode != "none" else None,
        "num_calibration_observations": args.num_calibration_observations if args.mode != "none" else 0,
        "calibration_variants": args.calibration_variants if args.mode != "none" else None,
        "calibration_base_seed": args.calibration_base_seed if args.mode != "none" else None,
        "log_clamp": args.log_clamp if args.mode != "none" else None,
        "attention_scale_summary": attention_scale_summary,
        "attention_scales": attention_scales,
        "attention_processors": len(attention_originals),
        "calibration_seconds": calibration_seconds,
        "denoising_steps": args.denoising_steps,
        "num_steps_wait": args.num_steps_wait,
        "max_policy_steps": args.max_policy_steps,
        "matched_denoising_seed": True,
        "base_seed": args.base_seed,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
        "device": args.device,
        "model_load_seconds": model_load_seconds,
        "patched_modules": len(records),
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
                started = time.time()

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
                        seed = args.base_seed + task_id * 100000 + init_index * 1000 + policy_step
                        pre_eef_pos = as_float_list(obs.get("robot0_eef_pos", []))
                        pre_eef_quat = as_float_list(obs.get("robot0_eef_quat", []))
                        pre_gripper_qpos = as_float_list(obs.get("robot0_gripper_qpos", []))

                        teacher_chunk, student_chunk, timing = local_policy.get_teacher_student(processed_obs, seed)
                        teacher_action = local_policy.convert_to_libero_action(teacher_chunk)
                        student_action = local_policy.convert_to_libero_action(student_chunk)
                        raw_metrics = compare_actions(teacher_chunk, student_chunk)
                        libero_metrics = compare_libero_actions(teacher_action, student_action)

                        obs, reward, done, info = env.step(teacher_action.tolist())
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
                                "teacher_raw_action": local_policy.trace_raw_action(teacher_chunk),
                                "student_raw_action": local_policy.trace_raw_action(student_chunk),
                                "teacher_libero_action": as_float_list(teacher_action),
                                "student_libero_action": as_float_list(student_action),
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
                    "elapsed_seconds": time.time() - started,
                    "scope": args.scope,
                    "mode": args.mode,
                    "quant_format": result["quant_format"],
                    "fp4_block_size": args.fp4_block_size,
                    "drift_summary": drift_summary,
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
        result["teacher_successes"] = total_successes
        result["quant_records"] = aggregate_records(records)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-root", type=Path, default=Path("/root/autodl-tmp/Isaac-GR00T-n1.5"))
    parser.add_argument("--model-path", type=Path, default=Path("/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain"))
    parser.add_argument("--compat-stubs", type=Path)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--case-list", default="8:7,8:9,4:10,0:3")
    parser.add_argument("--scope", choices=SCOPE_CHOICES, default="dit_mlp_only")
    parser.add_argument("--mode", choices=["none", "identity", "atm", "ohb", "atm_ohb"], default="none")
    parser.add_argument("--fp4-block-size", type=int, default=32)
    parser.add_argument("--num-calibration-observations", type=int, default=3)
    parser.add_argument("--calibration-variants", default="zero,midgray,noise")
    parser.add_argument("--calibration-base-seed", type=int, default=470104)
    parser.add_argument("--log-clamp", type=float, default=0.3)
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-policy-steps", type=int, default=80)
    parser.add_argument("--base-seed", type=int, default=470204)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("/tmp/quantvla_phase7_online_drift"))
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/logs/phase7_online_drift.log"))
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase7_online_drift.json"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/phase7_online_drift.md"))
    args = parser.parse_args()

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    _insert_paths(args.isaac_root, args.compat_stubs)

    result = run_online_drift(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(result, args.output_md)
    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "episodes": result["episodes"],
        "teacher_successes": result["teacher_successes"],
        "patched_modules": result["patched_modules"],
        "mode": result["mode"],
        "attention_processors": result["attention_processors"],
    }, indent=2))


if __name__ == "__main__":
    main()
