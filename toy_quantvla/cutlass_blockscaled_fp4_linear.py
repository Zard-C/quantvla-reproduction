"""CUTLASS SM120 blockscaled FP4 Linear wrapper.

This is a bridge implementation.  It uses the proven Phase 8 helper packing
path to make a drop-in `nn.Linear` replacement, so we can measure end-to-end
GR00T `get_action` behavior before writing a production activation packer.

The wrapper is intentionally honest about its boundary:
- weights are packed once during module construction;
- activations are packed every forward and cached per runtime M when possible;
- GEMM compilation is cached per runtime shape and can be shared across modules;
- this is still a bridge, not a final production kernel package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn

from blockscaled_fp4_packer import (
    fill_triton_blockscaled_fp4_operand,
    make_empty_triton_blockscaled_fp4_operand,
    make_torch_blockscaled_fp4_operand,
    make_triton_blockscaled_fp4_operand,
)
from phase8_cutlass_blockscaled_fp4_smoke import (
    load_blockscaled_example,
    make_blockscaled_fp4_operand,
)


DEFAULT_CUTLASS_ROOT = Path("/root/autodl-tmp/cutlass")
DEFAULT_TILE_SHAPE_MNK = (128, 128, 128)
DEFAULT_EPI_TILE = (64, 32)


@dataclass
class CutlassBlockscaledFP4Stats:
    calls: int = 0
    forward_seconds: list[float] = field(default_factory=list)
    input_prepare_seconds: list[float] = field(default_factory=list)
    activation_pack_seconds: list[float] = field(default_factory=list)
    output_prepare_seconds: list[float] = field(default_factory=list)
    compile_lookup_seconds: list[float] = field(default_factory=list)
    compile_seconds: list[float] = field(default_factory=list)
    gemm_seconds: list[float] = field(default_factory=list)
    finalize_seconds: list[float] = field(default_factory=list)
    local_compile_cache_hits: int = 0
    shared_compile_cache_hits: int = 0

    def add_forward(self, seconds: float) -> None:
        self.forward_seconds.append(float(seconds))

    def add_input_prepare(self, seconds: float) -> None:
        self.input_prepare_seconds.append(float(seconds))

    def add_pack(self, seconds: float) -> None:
        self.activation_pack_seconds.append(float(seconds))

    def add_output_prepare(self, seconds: float) -> None:
        self.output_prepare_seconds.append(float(seconds))

    def add_compile_lookup(self, seconds: float) -> None:
        self.compile_lookup_seconds.append(float(seconds))

    def add_compile(self, seconds: float) -> None:
        self.compile_seconds.append(float(seconds))

    def add_local_compile_cache_hit(self) -> None:
        self.local_compile_cache_hits += 1

    def add_shared_compile_cache_hit(self) -> None:
        self.shared_compile_cache_hits += 1

    def add_gemm(self, seconds: float) -> None:
        self.gemm_seconds.append(float(seconds))

    def add_finalize(self, seconds: float) -> None:
        self.finalize_seconds.append(float(seconds))

    @staticmethod
    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "max": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0}
        sorted_values = sorted(values)
        p50_index = int(round((len(sorted_values) - 1) * 0.50))
        p90_index = int(round((len(sorted_values) - 1) * 0.90))
        return {
            "count": len(values),
            "mean": float(sum(values) / len(values)),
            "max": float(max(values)),
            "min": float(min(values)),
            "p50": float(sorted_values[p50_index]),
            "p90": float(sorted_values[p90_index]),
        }

    def to_result(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "forward_seconds": self._summary(self.forward_seconds),
            "input_prepare_seconds": self._summary(self.input_prepare_seconds),
            "activation_pack_seconds": self._summary(self.activation_pack_seconds),
            "output_prepare_seconds": self._summary(self.output_prepare_seconds),
            "compile_lookup_seconds": self._summary(self.compile_lookup_seconds),
            "compile_seconds": self._summary(self.compile_seconds),
            "compile_cache_hits": {
                "local": int(self.local_compile_cache_hits),
                "shared": int(self.shared_compile_cache_hits),
            },
            "gemm_seconds": self._summary(self.gemm_seconds),
            "finalize_seconds": self._summary(self.finalize_seconds),
        }


class CutlassBlockscaledFP4Context:
    _cache: dict[tuple[str, str], dict[str, Any]] = {}

    @classmethod
    def get(cls, cutlass_root: Path, sf_dtype_name: str) -> dict[str, Any]:
        key = (str(cutlass_root), sf_dtype_name)
        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cutlass.cute.runtime import from_dlpack

        blockscaled = load_blockscaled_example(cutlass_root)
        cached = {
            "cutlass": cutlass,
            "cute": cute,
            "cutlass_torch": cutlass_torch,
            "from_dlpack": from_dlpack,
            "blockscaled": blockscaled,
            "sf_dtype": getattr(cutlass, sf_dtype_name),
            "stream": cutlass_torch.default_stream(),
        }
        cls._cache[key] = cached
        return cached


class CutlassBlockscaledFP4Linear(nn.Module):
    """Bridge `nn.Linear` replacement using CUTLASS SM120 blockscaled FP4."""

    _shared_compiled_by_shape: dict[tuple[Any, ...], dict[str, Any]] = {}

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        cutlass_root: Path = DEFAULT_CUTLASS_ROOT,
        sf_vec_size: int = 16,
        sf_dtype: str = "Float8E4M3FN",
        tile_shape_mnk: tuple[int, int, int] = DEFAULT_TILE_SHAPE_MNK,
        epi_tile: tuple[int, int] = DEFAULT_EPI_TILE,
        pack_backend: str = "helper",
        share_compile_cache: bool = True,
        profile: bool = False,
        fallback: bool = False,
    ):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.cutlass_root = Path(cutlass_root)
        self.sf_vec_size = int(sf_vec_size)
        self.sf_dtype_name = str(sf_dtype)
        self.tile_shape_mnk = tuple(int(v) for v in tile_shape_mnk)
        self.epi_tile = tuple(int(v) for v in epi_tile)
        self.pack_backend = str(pack_backend)
        if self.pack_backend not in {"helper", "torch", "triton"}:
            raise ValueError(f"unsupported pack_backend={self.pack_backend!r}; expected helper, torch, or triton")
        if self.pack_backend in {"torch", "triton"} and self.sf_dtype_name != "Float8E4M3FN":
            raise ValueError(f"{self.pack_backend} pack_backend currently supports only Float8E4M3FN scales")
        self.share_compile_cache = bool(share_compile_cache)
        self.profile = bool(profile)
        self.fallback = bool(fallback)
        self.stats = CutlassBlockscaledFP4Stats()
        self._compiled_by_m: dict[int, dict[str, Any]] = {}
        self._triton_activation_pack_by_m: dict[int, dict[str, Any]] = {}

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().contiguous())

        device = weight.device if weight.is_cuda else torch.device("cuda")
        weight_3d = weight.detach().to(device=device, dtype=torch.float32).contiguous().view(
            self.out_features, self.in_features, 1
        )
        started = time.perf_counter()
        packed = self._pack_operand(weight_3d, cache_activation=False)
        torch.cuda.synchronize(device)
        self.weight_pack_seconds = time.perf_counter() - started
        self.weight_fp4_tensor = packed["fp4_tensor"]
        self.weight_scale_tensor = packed["scale_tensor"]
        self.register_buffer("weight_fp4_storage", packed["fp4_storage"])
        self.register_buffer("weight_scale_storage", packed["scale_storage"])

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        cutlass_root: Path = DEFAULT_CUTLASS_ROOT,
        sf_vec_size: int = 16,
        sf_dtype: str = "Float8E4M3FN",
        tile_shape_mnk: tuple[int, int, int] = DEFAULT_TILE_SHAPE_MNK,
        epi_tile: tuple[int, int] = DEFAULT_EPI_TILE,
        pack_backend: str = "helper",
        share_compile_cache: bool = True,
        profile: bool = False,
        fallback: bool = False,
    ) -> "CutlassBlockscaledFP4Linear":
        return cls(
            module.weight.detach(),
            module.bias.detach() if module.bias is not None else None,
            cutlass_root=cutlass_root,
            sf_vec_size=sf_vec_size,
            sf_dtype=sf_dtype,
            tile_shape_mnk=tile_shape_mnk,
            epi_tile=epi_tile,
            pack_backend=pack_backend,
            share_compile_cache=share_compile_cache,
            profile=profile,
            fallback=fallback,
        )

    def _compile_cache_key(self, m: int) -> tuple[Any, ...]:
        return (
            str(self.cutlass_root),
            self.sf_dtype_name,
            int(self.sf_vec_size),
            tuple(int(v) for v in self.tile_shape_mnk),
            tuple(int(v) for v in self.epi_tile),
            int(m),
            int(self.in_features),
            int(self.out_features),
            "Float16Output",
        )

    def _pack_operand(self, x_mkl: torch.Tensor, *, cache_activation: bool) -> dict[str, Any]:
        ctx = CutlassBlockscaledFP4Context.get(self.cutlass_root, self.sf_dtype_name)
        if self.pack_backend == "triton":
            if cache_activation:
                m = int(x_mkl.shape[0])
                cached = self._triton_activation_pack_by_m.get(m)
                if cached is None:
                    cached = make_empty_triton_blockscaled_fp4_operand(
                        x_mkl,
                        fp4_dtype=ctx["cutlass"].Float4E2M1FN,
                        sf_dtype=ctx["sf_dtype"],
                        sf_vec_size=self.sf_vec_size,
                        cutlass_torch=ctx["cutlass_torch"],
                    )
                    self._triton_activation_pack_by_m[m] = cached
                fill_triton_blockscaled_fp4_operand(
                    x_mkl,
                    fp4_storage=cached["fp4_storage"],
                    scale_storage=cached["scale_storage"],
                    sf_vec_size=self.sf_vec_size,
                    zero_storage=False,
                )
                return cached
            return make_triton_blockscaled_fp4_operand(
                x_mkl,
                fp4_dtype=ctx["cutlass"].Float4E2M1FN,
                sf_dtype=ctx["sf_dtype"],
                sf_vec_size=self.sf_vec_size,
                cutlass_torch=ctx["cutlass_torch"],
            )
        if self.pack_backend == "torch":
            return make_torch_blockscaled_fp4_operand(
                x_mkl,
                fp4_dtype=ctx["cutlass"].Float4E2M1FN,
                sf_dtype=ctx["sf_dtype"],
                sf_vec_size=self.sf_vec_size,
                cutlass_torch=ctx["cutlass_torch"],
            )
        return make_blockscaled_fp4_operand(
            x_mkl,
            fp4_dtype=ctx["cutlass"].Float4E2M1FN,
            sf_dtype=ctx["sf_dtype"],
            sf_vec_size=self.sf_vec_size,
            blockscaled_module=ctx["blockscaled"],
            cutlass_torch=ctx["cutlass_torch"],
            from_dlpack=ctx["from_dlpack"],
        )

    def _compile_for_m(self, m: int, activation_pack: dict[str, Any], output_tensor: Any) -> dict[str, Any]:
        cached = self._compiled_by_m.get(m)
        if cached is not None:
            self.stats.add_local_compile_cache_hit()
            return cached
        shared_key = self._compile_cache_key(m)
        if self.share_compile_cache:
            cached = self._shared_compiled_by_shape.get(shared_key)
            if cached is not None:
                self.stats.add_shared_compile_cache_hit()
                self._compiled_by_m[m] = cached
                return cached
        ctx = CutlassBlockscaledFP4Context.get(self.cutlass_root, self.sf_dtype_name)
        gemm = ctx["blockscaled"].Sm120BlockScaledGemmKernel(
            ctx["cutlass"].Float32,
            self.sf_vec_size,
            self.tile_shape_mnk,
            self.epi_tile,
        )
        max_active_clusters = ctx["cutlass"].utils.HardwareInfo().get_max_active_clusters(1)
        started = time.perf_counter()
        compiled = ctx["cute"].compile(
            gemm,
            activation_pack["fp4_tensor"],
            self.weight_fp4_tensor,
            activation_pack["scale_tensor"],
            self.weight_scale_tensor,
            output_tensor,
            max_active_clusters,
            ctx["stream"],
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        self.stats.add_compile(seconds)
        cached = {"compiled": compiled, "stream": ctx["stream"], "shared_key": shared_key}
        self._compiled_by_m[m] = cached
        if self.share_compile_cache:
            self._shared_compiled_by_shape[shared_key] = cached
        return cached

    def reset_runtime_stats(self) -> None:
        self.stats = CutlassBlockscaledFP4Stats()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forward_started = time.perf_counter()
        if self.fallback or not x.is_cuda:
            raise RuntimeError("CutlassBlockscaledFP4Linear bridge requires CUDA input and fallback=False")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")

        ctx = CutlassBlockscaledFP4Context.get(self.cutlass_root, self.sf_dtype_name)
        orig_shape = tuple(x.shape[:-1])
        input_started = time.perf_counter()
        x_3d = x.reshape(-1, self.in_features).float().contiguous().view(-1, self.in_features, 1)
        if self.profile:
            torch.cuda.synchronize(x.device)
            self.stats.add_input_prepare(time.perf_counter() - input_started)
        m = int(x_3d.shape[0])

        pack_started = time.perf_counter()
        activation_pack = self._pack_operand(x_3d, cache_activation=True)
        torch.cuda.synchronize(x.device)
        self.stats.add_pack(time.perf_counter() - pack_started)

        output_started = time.perf_counter()
        c_ref = torch.empty((m, self.out_features, 1), device=x.device, dtype=torch.float32)
        output_tensor, output_storage = ctx["cutlass_torch"].cute_tensor_like(
            c_ref,
            ctx["cutlass"].Float16,
            is_dynamic_layout=True,
            assumed_align=16,
        )
        output_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=1)
        if self.profile:
            torch.cuda.synchronize(x.device)
            self.stats.add_output_prepare(time.perf_counter() - output_started)

        compile_lookup_started = time.perf_counter()
        entry = self._compile_for_m(m, activation_pack, output_tensor)
        if self.profile:
            torch.cuda.synchronize(x.device)
            self.stats.add_compile_lookup(time.perf_counter() - compile_lookup_started)

        gemm_started = time.perf_counter()
        entry["compiled"](
            activation_pack["fp4_tensor"],
            self.weight_fp4_tensor,
            activation_pack["scale_tensor"],
            self.weight_scale_tensor,
            output_tensor,
            entry["stream"],
        )
        torch.cuda.synchronize(x.device)
        self.stats.add_gemm(time.perf_counter() - gemm_started)
        self.stats.calls += 1

        finalize_started = time.perf_counter()
        out = output_storage.view(m, self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(device=x.device, dtype=out.dtype)
        out = out.reshape(*orig_shape, self.out_features).to(dtype=x.dtype)
        if self.profile:
            torch.cuda.synchronize(x.device)
            self.stats.add_finalize(time.perf_counter() - finalize_started)
            self.stats.add_forward(time.perf_counter() - forward_started)
        return out

    def to_result(self) -> dict[str, Any]:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "sf_vec_size": self.sf_vec_size,
            "sf_dtype": self.sf_dtype_name,
            "pack_backend": self.pack_backend,
            "share_compile_cache": self.share_compile_cache,
            "tile_shape_mnk": list(self.tile_shape_mnk),
            "epi_tile": list(self.epi_tile),
            "weight_pack_seconds": float(self.weight_pack_seconds),
            "compiled_m_values": sorted(int(m) for m in self._compiled_by_m),
            "stats": self.stats.to_result(),
            "weight_storage_bytes": {
                "fp4_storage_bytes": int(self.weight_fp4_storage.numel() * self.weight_fp4_storage.element_size()),
                "scale_storage_bytes": int(self.weight_scale_storage.numel() * self.weight_scale_storage.element_size()),
            },
        }

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
            f"sf_vec_size={self.sf_vec_size}, sf_dtype={self.sf_dtype_name}, pack_backend={self.pack_backend}"
        )
