"""Explains the realtime vs virtual retrieval wall-time gap (Phase 4 report §16).

Measures the same hybrid retrieve() call on the fixture index (TEST FIXTURE ONLY, not a benchmark result):
back-to-back vs after a 300 ms idle pause, on the main thread and via asyncio.to_thread (as AsyncExecutor does).

Usage: .venv/bin/python research/phase4/latency_idle_probe.py --index-root /tmp/idx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from streamrag.config.settings import load_config
from streamrag.retrieval.store import build_index
from streamrag.streaming.factory import build_stack

REPO = Path(__file__).resolve().parents[2]
QUERIES = ["how high should the wicks be trimmed", "fog signal during a storm", "where are ladders stored overnight"]


def summary(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"n": len(xs), "p50": round(statistics.median(xs), 3), "p95": round(xs[int(0.95 * (len(xs) - 1))], 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", type=Path, required=True)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--idle-ms", type=float, default=300.0)
    a = ap.parse_args()
    cfg = load_config(REPO / "configs" / "default.yaml",
                      {"paths.corpus": str(REPO / "tests/fixtures/corpus"), "paths.index_root": str(a.index_root),
                       "telemetry.log_level": "ERROR"}, base_dir=REPO)
    svc = build_stack(cfg, build_index(cfg).path).service

    def call(q: str) -> float:
        t = time.perf_counter()
        svc.retrieve(q)
        return (time.perf_counter() - t) * 1000.0

    for q in QUERIES:  # warm-up
        call(q)
    idle_s = a.idle_ms / 1000.0

    async def threaded(idle: float) -> list[float]:
        out = []
        for i in range(a.n):
            if idle:
                await asyncio.sleep(idle)
            out.append(await asyncio.to_thread(call, QUERIES[i % 3]))
        return out

    res = {"main_back_to_back": summary([call(QUERIES[i % 3]) for i in range(a.n)]),
           "thread_back_to_back": summary(asyncio.run(threaded(0.0)))}
    idle = []
    for i in range(a.n):
        time.sleep(idle_s)
        idle.append(call(QUERIES[i % 3]))
    res[f"main_after_{int(a.idle_ms)}ms_idle"] = summary(idle)
    res[f"thread_after_{int(a.idle_ms)}ms_idle"] = summary(asyncio.run(threaded(idle_s)))
    res["label"] = "TEST FIXTURE ONLY - NOT A BENCHMARK RESULT"
    out = REPO / "research/phase4/results/latency_idle_probe.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
