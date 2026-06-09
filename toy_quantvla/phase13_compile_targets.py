"""Shared torch.compile target helpers for Phase 13 experiments."""

from __future__ import annotations

import time
from typing import Any

from phase3_fake_quant_forward import get_submodule, set_submodule


DIT_BLOCKS = tuple(range(16))

TORCH_COMPILE_TARGETS = (
    "none",
    "backbone",
    "action_head_model",
    "action_head_model_ff_8_15_eager",
    "action_head_model_blocks_8_15_eager",
    "backbone_action_head_model",
    "action_head_dit_blocks_all",
    "action_head_dit_blocks_0_7",
    "action_head_dit_blocks_8_15",
    "action_head_dit_attn_all",
    "action_head_dit_attn_0_7",
    "action_head_dit_attn_8_15",
    "action_head_dit_ff_all",
    "action_head_dit_ff_0_7",
    "action_head_dit_ff_8_15",
)


def _block_indices(start: int, end: int) -> tuple[int, ...]:
    return tuple(range(start, end + 1))


def _dit_paths(kind: str, indices: tuple[int, ...]) -> list[str]:
    if kind == "block":
        return [f"action_head.model.transformer_blocks.{idx}" for idx in indices]
    if kind == "attn":
        return [f"action_head.model.transformer_blocks.{idx}.attn1" for idx in indices]
    if kind == "ff":
        return [f"action_head.model.transformer_blocks.{idx}.ff" for idx in indices]
    raise ValueError(f"Unknown DiT compile kind: {kind}")


def compile_module_paths_for_target(target: str) -> list[str]:
    """Return policy.model-relative module paths for a compile target."""

    if target == "none":
        return []
    if target == "backbone":
        return ["backbone"]
    if target == "action_head_model":
        return ["action_head.model"]
    if target == "action_head_model_ff_8_15_eager":
        return ["action_head.model"]
    if target == "action_head_model_blocks_8_15_eager":
        return ["action_head.model"]
    if target == "backbone_action_head_model":
        return ["backbone", "action_head.model"]
    if target == "action_head_dit_blocks_all":
        return _dit_paths("block", DIT_BLOCKS)
    if target == "action_head_dit_blocks_0_7":
        return _dit_paths("block", _block_indices(0, 7))
    if target == "action_head_dit_blocks_8_15":
        return _dit_paths("block", _block_indices(8, 15))
    if target == "action_head_dit_attn_all":
        return _dit_paths("attn", DIT_BLOCKS)
    if target == "action_head_dit_attn_0_7":
        return _dit_paths("attn", _block_indices(0, 7))
    if target == "action_head_dit_attn_8_15":
        return _dit_paths("attn", _block_indices(8, 15))
    if target == "action_head_dit_ff_all":
        return _dit_paths("ff", DIT_BLOCKS)
    if target == "action_head_dit_ff_0_7":
        return _dit_paths("ff", _block_indices(0, 7))
    if target == "action_head_dit_ff_8_15":
        return _dit_paths("ff", _block_indices(8, 15))
    raise ValueError(f"Unknown torch compile target: {target!r}")


def eager_island_paths_for_target(target: str) -> list[str]:
    """Return policy.model-relative paths that should stay outside Dynamo graphs."""

    if target == "action_head_model_ff_8_15_eager":
        return _dit_paths("ff", _block_indices(8, 15))
    if target == "action_head_model_blocks_8_15_eager":
        return _dit_paths("block", _block_indices(8, 15))
    return []


def compile_kwargs_from_args(args: Any) -> dict[str, Any]:
    compile_kwargs: dict[str, Any] = {
        "backend": args.torch_compile_backend,
        "mode": args.torch_compile_mode,
        "fullgraph": bool(args.torch_compile_fullgraph),
    }
    if args.torch_compile_dynamic is not None:
        compile_kwargs["dynamic"] = bool(args.torch_compile_dynamic)
    return compile_kwargs


def compile_policy_targets(policy: Any, args: Any, torch_module: Any) -> dict[str, Any]:
    """Install torch.compile wrappers on selected policy.model submodules."""

    target = str(args.torch_compile_target)
    paths = compile_module_paths_for_target(target)
    eager_island_paths = eager_island_paths_for_target(target)
    if not paths:
        return {"enabled": False, "target": target, "compiled_modules": []}

    compile_kwargs = compile_kwargs_from_args(args)
    compiled_modules: list[str] = []
    eager_island_modules = install_eager_islands(policy, eager_island_paths, torch_module)
    started = time.perf_counter()
    for path in paths:
        module = get_submodule(policy.model, path)
        set_submodule(policy.model, path, torch_module.compile(module, **compile_kwargs))
        compiled_modules.append(f"policy.model.{path}")

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


def install_eager_islands(policy: Any, paths: list[str], torch_module: Any) -> list[str]:
    """Mark selected module forwards as torch._dynamo.disable eager islands."""

    installed: list[str] = []
    for path in paths:
        module = get_submodule(policy.model, path)
        if getattr(module, "_phase13_eager_island", False):
            installed.append(f"policy.model.{path}")
            continue
        module.forward = torch_module._dynamo.disable(module.forward)
        module._phase13_eager_island = True
        installed.append(f"policy.model.{path}")
    return installed


class CompileTargetSwitcher:
    """Switch a policy between eager and compiled modules for A/B replay."""

    def __init__(self, policy: Any, args: Any, torch_module: Any):
        self.policy = policy
        self.target = str(args.torch_compile_target)
        self.paths = compile_module_paths_for_target(self.target)
        self.eager_island_paths = eager_island_paths_for_target(self.target)
        self.compile_kwargs = compile_kwargs_from_args(args)
        self.eager_island_modules = install_eager_islands(
            policy,
            self.eager_island_paths,
            torch_module,
        )
        self.eager_modules = {path: get_submodule(policy.model, path) for path in self.paths}
        self.compiled_modules = {
            path: torch_module.compile(module, **self.compile_kwargs)
            for path, module in self.eager_modules.items()
        }

    def use_eager(self) -> None:
        for path, module in self.eager_modules.items():
            set_submodule(self.policy.model, path, module)

    def use_compiled(self) -> None:
        for path, module in self.compiled_modules.items():
            set_submodule(self.policy.model, path, module)

    def info(self, args: Any) -> dict[str, Any]:
        if not self.paths:
            return {"enabled": False, "target": self.target, "compiled_modules": []}
        return {
            "enabled": True,
            "target": self.target,
            "backend": args.torch_compile_backend,
            "mode": args.torch_compile_mode,
            "fullgraph": bool(args.torch_compile_fullgraph),
            "dynamic": args.torch_compile_dynamic,
            "compiled_modules": [f"policy.model.{path}" for path in self.paths],
            "eager_island_modules": self.eager_island_modules,
        }
