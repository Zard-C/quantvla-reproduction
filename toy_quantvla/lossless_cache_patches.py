"""Low-risk semantic-preserving cache patches for GR00T inference.

These patches avoid torch.compile and do not change model weights or dtypes.
They target repeated inference-time tensor construction and unnecessary device
copies observed in the FP16 profiler.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import tree
from transformers.feature_extraction_utils import BatchFeature


def _device_key(device: torch.device | str) -> tuple[str, int | None]:
    dev = torch.device(device)
    return dev.type, dev.index


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(iter(module.parameters())).device


def _to_model_device_and_dtype(model: Any, x: Any) -> Any:
    if not torch.is_tensor(x):
        return x
    device = _module_device(model)
    if torch.is_floating_point(x):
        return x.to(device=device, dtype=model.action_head.dtype)
    return x.to(device=device)


def install_prepare_input_pruning(model: Any) -> dict[str, Any]:
    """Patch GR00T_N1_5.prepare_input to move only keys each branch uses.

    The stock implementation passes the full normalized batch to both the
    backbone and action head, then recursively moves both copies to CUDA.  For
    inference, the backbone only consumes ``eagle_*`` tensors and the action head
    only consumes ``state`` and ``embodiment_id`` in ``get_action``.
    """

    if getattr(model, "_quantvla_prepare_input_pruning", False):
        return {"enabled": True, "already_installed": True}

    original_prepare_input = model.prepare_input
    stats = {
        "calls": 0,
        "backbone_keys": [],
        "action_keys": [],
        "moved_backbone_values": 0,
        "moved_action_values": 0,
    }

    def move_tree(branch: dict[str, Any]) -> BatchFeature:
        moved = tree.map_structure(lambda x: _to_model_device_and_dtype(model, x), branch)
        return BatchFeature(data=moved)

    def pruned_prepare_input(self: Any, inputs: dict[str, Any]) -> tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_source = {
            key: value
            for key, value in inputs.items()
            if key.startswith("eagle_")
        }
        action_source = {
            key: inputs[key]
            for key in ("state", "embodiment_id")
            if key in inputs
        }
        stats["calls"] += 1
        stats["backbone_keys"] = sorted(backbone_source)
        stats["action_keys"] = sorted(action_source)
        stats["moved_backbone_values"] += len(backbone_source)
        stats["moved_action_values"] += len(action_source)
        return move_tree(backbone_source), move_tree(action_source)

    model._quantvla_original_prepare_input = original_prepare_input
    model._quantvla_prepare_input_pruning_stats = stats
    model.prepare_input = MethodType(pruned_prepare_input, model)
    model._quantvla_prepare_input_pruning = True
    return {
        "enabled": True,
        "already_installed": False,
        "description": "Move only eagle_* tensors to backbone and state/embodiment_id to action head.",
    }


def install_action_head_static_cache(action_head: Any) -> dict[str, Any]:
    """Patch action_head.get_action to cache fixed per-step tensors."""

    if getattr(action_head, "_quantvla_static_cache", False):
        return {"enabled": True, "already_installed": True}

    original_get_action = action_head.get_action
    cache: dict[str, dict[Any, torch.Tensor]] = {
        "timesteps": {},
        "position_embeddings": {},
        "future_tokens": {},
    }
    stats = {
        "calls": 0,
        "timestep_hits": 0,
        "timestep_misses": 0,
        "position_embedding_hits": 0,
        "position_embedding_misses": 0,
        "future_token_hits": 0,
        "future_token_misses": 0,
    }

    def cached_timesteps(self: Any, batch_size: int, t_discretized: int, device: torch.device) -> torch.Tensor:
        key = (int(batch_size), int(t_discretized), _device_key(device))
        tensor = cache["timesteps"].get(key)
        if tensor is None:
            tensor = torch.full(size=(batch_size,), fill_value=int(t_discretized), device=device)
            cache["timesteps"][key] = tensor
            stats["timestep_misses"] += 1
        else:
            stats["timestep_hits"] += 1
        return tensor

    def cached_position_embedding(self: Any, seq_len: int, device: torch.device) -> torch.Tensor:
        weight = self.position_embedding.weight
        key = (int(seq_len), _device_key(device), weight.dtype, int(weight.data_ptr()))
        tensor = cache["position_embeddings"].get(key)
        if tensor is None:
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            tensor = self.position_embedding(pos_ids).unsqueeze(0)
            cache["position_embeddings"][key] = tensor
            stats["position_embedding_misses"] += 1
        else:
            stats["position_embedding_hits"] += 1
        return tensor

    def cached_future_tokens(self: Any, batch_size: int) -> torch.Tensor:
        weight = self.future_tokens.weight
        key = (int(batch_size), _device_key(weight.device), weight.dtype, int(weight.data_ptr()))
        tensor = cache["future_tokens"].get(key)
        if tensor is None:
            tensor = weight.unsqueeze(0).expand(batch_size, -1, -1)
            cache["future_tokens"][key] = tensor
            stats["future_token_misses"] += 1
        else:
            stats["future_token_hits"] += 1
        return tensor

    @torch.no_grad()
    def cached_get_action(self: Any, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        stats["calls"] += 1
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id
        state_features = self.state_encoder(action_input.state, embodiment_id)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        future_tokens = cached_future_tokens(self, batch_size)

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = cached_timesteps(self, batch_size, t_discretized, device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_embs = cached_position_embedding(self, action_features.shape[1], device)
                action_features = action_features + pos_embs

            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity
        return BatchFeature(data={"action_pred": actions})

    action_head._quantvla_original_get_action = original_get_action
    action_head._quantvla_static_cache_storage = cache
    action_head._quantvla_static_cache_stats = stats
    action_head.get_action = MethodType(cached_get_action, action_head)
    action_head._quantvla_static_cache = True
    return {
        "enabled": True,
        "already_installed": False,
        "description": "Cache denoising timestep tensors, action position embeddings, and expanded future tokens.",
    }


def lossless_cache_stats(policy_or_model: Any) -> dict[str, Any]:
    model = getattr(policy_or_model, "model", policy_or_model)
    action_head = getattr(model, "action_head", None)
    out: dict[str, Any] = {}
    if hasattr(model, "_quantvla_prepare_input_pruning_stats"):
        out["prepare_input_pruning"] = dict(model._quantvla_prepare_input_pruning_stats)
    if action_head is not None and hasattr(action_head, "_quantvla_static_cache_stats"):
        out["action_head_static_cache"] = dict(action_head._quantvla_static_cache_stats)
        out["action_head_static_cache_sizes"] = {
            key: len(value)
            for key, value in getattr(action_head, "_quantvla_static_cache_storage", {}).items()
        }
    return out


def install_lossless_cache_patches(
    policy: Any,
    *,
    prepare_input_pruning: bool = False,
    action_head_static_cache: bool = False,
) -> dict[str, Any]:
    """Install selected patches on a loaded Gr00tPolicy."""

    result: dict[str, Any] = {
        "prepare_input_pruning": {"enabled": False},
        "action_head_static_cache": {"enabled": False},
    }
    if prepare_input_pruning:
        result["prepare_input_pruning"] = install_prepare_input_pruning(policy.model)
    if action_head_static_cache:
        result["action_head_static_cache"] = install_action_head_static_cache(policy.model.action_head)
    return result

