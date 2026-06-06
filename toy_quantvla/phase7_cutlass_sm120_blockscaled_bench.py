"""Benchmark CUTLASS CuTe SM120 block-scaled FP4 GEMM on real GR00T shapes.

This script intentionally treats CUTLASS as an external checkout.  It invokes
the official Blackwell GeForce CuTe DSL example and parses its timing output, so
we can keep this repository small while making the benchmark reproducible.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShapeCase:
    family: str
    m: int
    k: int
    n: int
    torch_fp16_ms: float | None = None
    bnb_nf4_ms: float | None = None

    @property
    def mnkl(self) -> str:
        return f"{self.m},{self.n},{self.k},1"


REAL_SHAPES = [
    ShapeCase("DiT MLP", 49, 1536, 6144, 0.0153, 0.1070),
    ShapeCase("DiT MLP", 49, 6144, 1536, 0.0313, 0.1052),
    ShapeCase("LLM attn", 551, 2048, 1024, 0.0167, 0.0985),
    ShapeCase("LLM attn", 551, 2048, 2048, 0.0270, 0.0996),
    ShapeCase("LLM MLP", 551, 2048, 6144, 0.0762, 0.1016),
    ShapeCase("LLM MLP", 551, 6144, 2048, 0.0722, 0.1003),
]


EXEC_TIME_RE = re.compile(r"Execution time:\s*([0-9.]+)\s*microseconds per iteration")
GFLOPS_RE = re.compile(r"GFLOPS:\s*([0-9.]+)")


def parse_csv_tuple(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid comma-separated tuple: {raw}") from exc


def run_cutlass_case(
    *,
    python_bin: Path,
    cutlass_root: Path,
    shape: ShapeCase,
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    sf_vec_size: int,
    sf_dtype: str,
    warmup: int,
    iterations: int,
    timeout: int,
) -> dict[str, Any]:
    example = cutlass_root / "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py"
    cmd = [
        str(python_bin),
        str(example),
        f"--mnkl={shape.mnkl}",
        "--tile_shape_mnk=" + ",".join(str(v) for v in tile_shape_mnk),
        "--epi_tile=" + ",".join(str(v) for v in epi_tile),
        "--a_dtype=Float4E2M1FN",
        "--b_dtype=Float4E2M1FN",
        f"--sf_dtype={sf_dtype}",
        f"--sf_vec_size={sf_vec_size}",
        "--c_dtype=Float16",
        "--acc_dtype=Float32",
        f"--warmup_iterations={warmup}",
        f"--iterations={iterations}",
        "--tolerance=1e-1",
        "--skip_ref_check",
    ]

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cutlass_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - started
    output = proc.stdout
    time_match = EXEC_TIME_RE.search(output)
    gflops_match = GFLOPS_RE.search(output)

    row: dict[str, Any] = {
        **asdict(shape),
        "mnkl": shape.mnkl,
        "tile_shape_mnk": tile_shape_mnk,
        "epi_tile": epi_tile,
        "sf_vec_size": sf_vec_size,
        "sf_dtype": sf_dtype,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "raw_output_tail": "\n".join(output.strip().splitlines()[-12:]),
    }
    if proc.returncode == 0 and time_match:
        cutlass_ms = float(time_match.group(1)) / 1000.0
        row.update(
            {
                "cutlass_fp4_ms": cutlass_ms,
                "gflops": float(gflops_match.group(1)) if gflops_match else None,
                "speedup_vs_torch_fp16": shape.torch_fp16_ms / cutlass_ms if shape.torch_fp16_ms else None,
                "speedup_vs_bnb_nf4": shape.bnb_nf4_ms / cutlass_ms if shape.bnb_nf4_ms else None,
                "error": None,
            }
        )
    else:
        row.update(
            {
                "cutlass_fp4_ms": None,
                "gflops": None,
                "speedup_vs_torch_fp16": None,
                "speedup_vs_bnb_nf4": None,
                "error": output.strip().splitlines()[-1] if output.strip() else "no output",
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", type=Path, default=Path("/root/autodl-tmp/envs/gr00t-libero-py310/bin/python"))
    parser.add_argument("--cutlass-root", type=Path, default=Path("/root/autodl-tmp/cutlass"))
    parser.add_argument("--tile-shapes", default="128,128,128;128,128,256")
    parser.add_argument("--epi-tiles", default="128,128;64,32")
    parser.add_argument("--sf-vec-sizes", default="32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=Path("toy_quantvla/results/phase7_cutlass_sm120_blockscaled_bench.json"))
    args = parser.parse_args()

    tile_shapes = [parse_csv_tuple(item) for item in args.tile_shapes.split(";") if item.strip()]
    epi_tiles = [parse_csv_tuple(item) for item in args.epi_tiles.split(";") if item.strip()]
    sf_vec_sizes = [int(item.strip()) for item in args.sf_vec_sizes.split(",") if item.strip()]
    shapes = REAL_SHAPES[: args.max_cases] if args.max_cases > 0 else REAL_SHAPES

    rows = []
    total = len(shapes) * len(tile_shapes) * len(epi_tiles) * len(sf_vec_sizes)
    idx = 0
    for shape in shapes:
        for tile_shape in tile_shapes:
            for epi_tile in epi_tiles:
                for sf_vec_size in sf_vec_sizes:
                    idx += 1
                    sf_dtype = "Float8E8M0FNU" if sf_vec_size == 32 else "Float8E4M3FN"
                    print(
                        f"[{idx}/{total}] {shape.family} M={shape.m} K={shape.k} N={shape.n} "
                        f"tile={tile_shape} epi={epi_tile} sf_vec={sf_vec_size}",
                        flush=True,
                    )
                    row = run_cutlass_case(
                        python_bin=args.python_bin,
                        cutlass_root=args.cutlass_root,
                        shape=shape,
                        tile_shape_mnk=tile_shape,
                        epi_tile=epi_tile,
                        sf_vec_size=sf_vec_size,
                        sf_dtype=sf_dtype,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        timeout=args.timeout,
                    )
                    if row["error"]:
                        print(f"  ERROR: {row['error']}", flush=True)
                    else:
                        print(
                            f"  {row['cutlass_fp4_ms']:.6f} ms, "
                            f"{row['speedup_vs_torch_fp16']:.3f}x vs torch fp16, "
                            f"{row['speedup_vs_bnb_nf4']:.3f}x vs bnb nf4",
                            flush=True,
                        )
                    rows.append(row)

    best_by_shape = []
    for shape in shapes:
        candidates = [
            row
            for row in rows
            if row["m"] == shape.m and row["k"] == shape.k and row["n"] == shape.n and row["error"] is None
        ]
        best = min(candidates, key=lambda row: row["cutlass_fp4_ms"]) if candidates else None
        best_by_shape.append(best)

    result = {
        "backend": "CUTLASS CuTe SM120 Blackwell GeForce blockscaled FP4",
        "cutlass_root": str(args.cutlass_root),
        "python_bin": str(args.python_bin),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tile_shapes": tile_shapes,
        "epi_tiles": epi_tiles,
        "sf_vec_sizes": sf_vec_sizes,
        "rows": rows,
        "best_by_shape": best_by_shape,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "valid_rows": sum(row["error"] is None for row in rows)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
