"""Low-intrusion FP16 Linear module timing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from phase3_fake_quant_forward import set_submodule
from phase6_w4a16_scopes import include_module_for_scope, module_family
from timing_utils import summarize_float


@dataclass
class TimedFP16LinearStats:
    calls: int = 0
    forward_seconds: list[float] = field(default_factory=list)

    def add_forward(self, seconds: float) -> None:
        self.forward_seconds.append(float(seconds))
        self.calls += 1

    def to_result(self) -> dict[str, Any]:
        return {
            "calls": int(self.calls),
            "forward_seconds": summarize_float(self.forward_seconds),
        }


class TimedFP16Linear(nn.Module):
    """Wrapper around an existing ``nn.Linear`` that records forward latency."""

    def __init__(self, module: nn.Linear, *, profile: bool = True):
        super().__init__()
        self.linear = module
        self.profile = bool(profile)
        self.stats = TimedFP16LinearStats()
        self.in_features = int(module.in_features)
        self.out_features = int(module.out_features)
        self.bias = module.bias is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        out = self.linear(x)
        if self.profile and x.is_cuda:
            torch.cuda.synchronize(x.device)
        self.stats.add_forward(time.perf_counter() - started)
        return out

    def reset_runtime_stats(self) -> None:
        self.stats = TimedFP16LinearStats()

    def to_result(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "profile": self.profile,
            "stats": self.stats.to_result(),
        }

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, bias={self.bias}, profile={self.profile}"


def parse_name_contains(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def patch_timed_fp16_modules(
    model: Any,
    scope: str,
    *,
    max_modules: int,
    name_contains: list[str],
    profile: bool = True,
) -> tuple[dict[str, Any], dict[str, TimedFP16Linear]]:
    records: dict[str, Any] = {}
    patched_modules: dict[str, TimedFP16Linear] = {}
    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in list(model.named_modules()):
        family = module_family(name)
        if family is None or not include_module_for_scope(name, scope) or not isinstance(module, nn.Linear):
            continue
        if name_contains and not any(token in name for token in name_contains):
            continue
        targets.append((name, module, family))
        if max_modules > 0 and len(targets) >= max_modules:
            break

    for name, module, family in targets:
        wrapped = TimedFP16Linear(module, profile=profile)
        records[name] = {
            "module": name,
            "family": family,
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
            "bias": module.bias is not None,
            "original_weight_dtype": str(module.weight.dtype),
        }
        set_submodule(model, name, wrapped)
        patched_modules[name] = wrapped
    return records, patched_modules


def module_results(modules: dict[str, TimedFP16Linear]) -> dict[str, Any]:
    return {name: module.to_result() for name, module in modules.items()}


def reset_module_stats(modules: dict[str, TimedFP16Linear]) -> None:
    for module in modules.values():
        module.reset_runtime_stats()
