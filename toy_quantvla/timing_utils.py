"""Small timing helpers for QuantVLA server/client latency probes."""

from __future__ import annotations

import atexit
import json
import signal
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


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
    ):
        self._policy = policy
        self._output_json = output_json
        self._label = label
        self._flush_every = int(flush_every)
        self._extra_summary = extra_summary
        self._latencies: list[float] = []
        self._write_count = 0
        if self._output_json is not None:
            atexit.register(self.write_summary)
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, self._handle_signal)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.write_summary()
        raise SystemExit(128 + int(signum))

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        out = self._policy.get_action(observation)
        seconds = time.perf_counter() - started
        self._latencies.append(float(seconds))
        if self._flush_every > 0 and len(self._latencies) % self._flush_every == 0:
            self.write_summary()
        return out

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
