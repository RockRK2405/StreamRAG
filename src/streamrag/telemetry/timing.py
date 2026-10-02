"""Monotonic timing helpers (perf_counter) and percentile summaries."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np


class Stopwatch:
    """``with Stopwatch() as sw: ...`` then ``sw.ms``. Also usable for named laps."""

    def __init__(self) -> None:
        self.t0 = 0.0
        self.ms = 0.0
        self.laps: dict[str, float] = {}
        self._lap_t = 0.0

    def __enter__(self) -> "Stopwatch":
        self.t0 = self._lap_t = time.perf_counter()
        return self

    def lap(self, name: str) -> float:
        now = time.perf_counter()
        dt = (now - self._lap_t) * 1000.0
        self.laps[name] = self.laps.get(name, 0.0) + dt
        self._lap_t = now
        return dt

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def percentile_summary(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"n": 0}
    return {"n": int(arr.size), "p50": round(float(np.percentile(arr, 50)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3), "mean": round(float(arr.mean()), 3),
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3)}
