"""Small timing helpers for QuantVLA server/client latency probes."""

from __future__ import annotations

import atexit
import json
import random
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

REQUEST_SEED_KEY = "quantvla.request_seed"


def summarize_float(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def summarize_breakdowns(rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = sorted({key for row in rows for key in row})
    return {key: summarize_float([float(row.get(key, 0.0)) for row in rows]) for key in keys}


def _coerce_seed(value: Any) -> int:
    if isinstance(value, np.ndarray):
        value = value.reshape(-1)[0]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed % (2**63 - 1))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed % (2**63 - 1))
    except Exception:
        return


def _strip_private_request_fields(observation: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    if REQUEST_SEED_KEY not in observation:
        return observation, None
    policy_observation = dict(observation)
    request_seed = _coerce_seed(policy_observation.pop(REQUEST_SEED_KEY))
    return policy_observation, request_seed


class TimedPolicyWrapper:
    """Proxy that records server-side `get_action` latency and writes JSON on exit."""

    def __init__(
        self,
        policy: Any,
        *,
        output_json: Path | None,
        label: str,
        flush_every: int = 0,
        extra_summary: Callable[[], dict[str, Any]] | None = None,
        request_trace_jsonl: Path | None = None,
        request_trace_min_seconds: float = 0.0,
        request_extra: Callable[[int, float], dict[str, Any]] | None = None,
        cuda_sync_device: str | None = None,
    ):
        self._policy = policy
        self._output_json = output_json
        self._label = label
        self._flush_every = int(flush_every)
        self._extra_summary = extra_summary
        self._request_trace_jsonl = request_trace_jsonl
        self._request_trace_min_seconds = float(request_trace_min_seconds)
        self._request_extra = request_extra
        self._cuda_sync_device = cuda_sync_device
        self._latencies: list[float] = []
        self._write_count = 0
        self._trace_file = None
        if self._request_trace_jsonl is not None:
            self._request_trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = self._request_trace_jsonl.open("a", encoding="utf-8")
            atexit.register(self.close_trace)
        if self._output_json is not None:
            atexit.register(self.write_summary)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self._handle_signal)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.write_summary()
        self.close_trace()
        raise SystemExit(128 + int(signum))

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        request_index = len(self._latencies) + 1
        wall_started = time.time()
        policy_observation, request_seed = _strip_private_request_fields(observation)
        if request_seed is not None:
            _seed_everything(request_seed)
        started = time.perf_counter()
        out = self._policy.get_action(policy_observation)
        sync_seconds = None
        if self._cuda_sync_device is not None:
            import torch

            sync_started = time.perf_counter()
            torch.cuda.synchronize(self._cuda_sync_device)
            sync_seconds = time.perf_counter() - sync_started
        seconds = time.perf_counter() - started
        self._latencies.append(float(seconds))
        self._write_request_trace(
            request_index=request_index,
            wall_started=wall_started,
            seconds=float(seconds),
            sync_seconds=None if sync_seconds is None else float(sync_seconds),
            request_seed=request_seed,
        )
        if self._flush_every > 0 and len(self._latencies) % self._flush_every == 0:
            self.write_summary()
        return out

    def _write_request_trace(
        self,
        *,
        request_index: int,
        wall_started: float,
        seconds: float,
        sync_seconds: float | None,
        request_seed: int | None,
    ) -> None:
        if self._trace_file is None:
            return
        if seconds < self._request_trace_min_seconds:
            return
        row: dict[str, Any] = {
            "label": self._label,
            "request_index": int(request_index),
            "wall_start_unix": float(wall_started),
            "wall_end_unix": float(time.time()),
            "get_action_seconds": float(seconds),
        }
        if sync_seconds is not None:
            row["cuda_sync_seconds_after_get_action"] = float(sync_seconds)
        if request_seed is not None:
            row["request_seed"] = int(request_seed)
        if self._request_extra is not None:
            row["extra"] = self._request_extra(int(request_index), float(seconds))
        self._trace_file.write(json.dumps(row, sort_keys=True) + "\n")
        self._trace_file.flush()

    def summary(self) -> dict[str, Any]:
        payload = {
            "label": self._label,
            "writes": int(self._write_count),
            "get_action_seconds": summarize_float(self._latencies),
        }
        if self._extra_summary is not None:
            payload["extra"] = self._extra_summary()
        return payload

    def write_summary(self) -> None:
        if self._output_json is None:
            return
        self._output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = self.summary()
        payload["writes"] = int(self._write_count + 1)
        self._output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._write_count += 1

    def close_trace(self) -> None:
        if self._trace_file is None:
            return
        self._trace_file.close()
        self._trace_file = None
