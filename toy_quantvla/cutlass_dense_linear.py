"""CUTLASS SM120 dense GEMM Linear wrapper.

This module is a Phase 8 engineering bridge: it verifies that a PyTorch
``nn.Linear`` can be replaced by a CUTLASS CuTe callable op that accepts real
``torch.Tensor`` inputs.  It is intentionally dense fp16/bf16 first.  The FP4
blockscaled packing path can reuse the same module replacement and benchmarking
scaffold once real activation/weight packing is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CUTLASS_ROOT = Path("/root/autodl-tmp/cutlass")
DEFAULT_TILE_SHAPE_MNK = (64, 128, 64)


def _torch_dtype_to_cutlass(dtype: torch.dtype) -> Any:
    import cutlass

    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    raise TypeError(f"CUTLASS dense wrapper supports fp16/bf16, got {dtype}")


def _load_sm120_dense_kernel(cutlass_root: Path) -> Any:
    example = (
        cutlass_root
        / "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py"
    )
    if not example.exists():
        raise FileNotFoundError(f"CUTLASS dense GEMM example not found: {example}")
    spec = importlib.util.spec_from_file_location("quantvla_cutlass_dense_gemm", example)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CUTLASS dense GEMM example: {example}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sm120GemmKernel


class CutlassDenseLinearCache:
    """Shape-level cache for compiled CUTLASS dense GEMM functions."""

    _kernel_cls: Any | None = None
    _compiled: dict[tuple[Any, ...], dict[str, Any]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._compiled.clear()

    @classmethod
    def stats(cls) -> dict[str, Any]:
        return {
            "compiled_shapes": len(cls._compiled),
            "keys": [
                {
                    "m": int(key[0]),
                    "k": int(key[1]),
                    "n": int(key[2]),
                    "tile_shape_mnk": [int(key[3]), int(key[4]), int(key[5])],
                    "dtype": str(key[6]),
                }
                for key in cls._compiled
            ],
            "compile_seconds": {
                str(key): float(value["compile_seconds"]) for key, value in cls._compiled.items()
            },
        }

    @classmethod
    def get(
        cls,
        *,
        m: int,
        k: int,
        n: int,
        dtype: torch.dtype,
        device: torch.device,
        cutlass_root: Path,
        tile_shape_mnk: tuple[int, int, int],
    ) -> dict[str, Any]:
        if device.type != "cuda":
            raise RuntimeError("CUTLASS dense op requires CUDA tensors")
        if cls._kernel_cls is None:
            cls._kernel_cls = _load_sm120_dense_kernel(cutlass_root)

        key = (int(m), int(k), int(n), *tuple(int(v) for v in tile_shape_mnk), str(dtype))
        cached = cls._compiled.get(key)
        if cached is not None:
            return cached

        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        from cutlass.cute.runtime import from_dlpack

        _torch_dtype_to_cutlass(dtype)
        a = torch.empty((m, k, 1), device=device, dtype=dtype).contiguous()
        b = torch.empty((n, k, 1), device=device, dtype=dtype).contiguous()
        c = torch.empty((m, n, 1), device=device, dtype=dtype).contiguous()
        a_cute = from_dlpack(a, enable_tvm_ffi=True).mark_layout_dynamic()
        b_cute = from_dlpack(b, enable_tvm_ffi=True).mark_layout_dynamic()
        c_cute = from_dlpack(c, enable_tvm_ffi=True).mark_layout_dynamic()
        gemm = cls._kernel_cls(cutlass.Float32, tile_shape_mnk)
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
        stream = cutlass_torch.default_stream()

        started = time.perf_counter()
        compiled = cute.compile(
            gemm,
            a_cute,
            b_cute,
            c_cute,
            max_active_clusters,
            stream,
            options="--enable-tvm-ffi",
        )
        torch.cuda.synchronize(device)
        cached = {
            "compiled": compiled,
            "stream": stream,
            "compile_seconds": time.perf_counter() - started,
        }
        cls._compiled[key] = cached
        return cached


class CutlassDenseLinear(nn.Module):
    """Drop-in inference wrapper for ``nn.Linear`` using CUTLASS dense GEMM."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        cutlass_root: Path = DEFAULT_CUTLASS_ROOT,
        tile_shape_mnk: tuple[int, int, int] = DEFAULT_TILE_SHAPE_MNK,
        fallback: bool = True,
    ):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be 2D, got shape {tuple(weight.shape)}")
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.cutlass_root = Path(cutlass_root)
        self.tile_shape_mnk = tuple(int(v) for v in tile_shape_mnk)
        self.fallback = bool(fallback)
        self.register_buffer("weight", weight.detach().contiguous())
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().contiguous())

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        cutlass_root: Path = DEFAULT_CUTLASS_ROOT,
        tile_shape_mnk: tuple[int, int, int] = DEFAULT_TILE_SHAPE_MNK,
        dtype: torch.dtype | None = None,
        fallback: bool = True,
    ) -> "CutlassDenseLinear":
        weight = module.weight.detach()
        bias = module.bias.detach() if module.bias is not None else None
        if dtype is not None:
            weight = weight.to(dtype=dtype)
            bias = bias.to(dtype=dtype) if bias is not None else None
        return cls(
            weight,
            bias,
            cutlass_root=cutlass_root,
            tile_shape_mnk=tile_shape_mnk,
            fallback=fallback,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.fallback
            and (not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16))
        ):
            return F.linear(x, self.weight.to(device=x.device, dtype=x.dtype), self.bias)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")

        orig_shape = tuple(x.shape[:-1])
        x_2d = x.reshape(-1, self.in_features).contiguous()
        m = int(x_2d.shape[0])
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        out = torch.empty((m, self.out_features, 1), device=x.device, dtype=x.dtype)

        entry = CutlassDenseLinearCache.get(
            m=m,
            k=self.in_features,
            n=self.out_features,
            dtype=x.dtype,
            device=x.device,
            cutlass_root=self.cutlass_root,
            tile_shape_mnk=self.tile_shape_mnk,
        )
        a = x_2d.view(m, self.in_features, 1)
        b = weight.view(self.out_features, self.in_features, 1)
        entry["compiled"](a, b, out, entry["stream"])
        out_2d = out.view(m, self.out_features)
        if bias is not None:
            out_2d = out_2d + bias
        return out_2d.reshape(*orig_shape, self.out_features)

    def extra_repr(self) -> str:
        return (
            f"{self.in_features}, {self.out_features}, bias={self.bias is not None}, "
            f"tile_shape_mnk={self.tile_shape_mnk}"
        )
