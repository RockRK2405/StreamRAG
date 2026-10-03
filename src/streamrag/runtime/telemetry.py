"""TelemetryManager (docs/runtime/10): runtime metrics on top of the event log.

* task metrics from the scheduler: queue wait, execution time, status per task type and pool;
* queue depth samples per pool, backpressure and coalescing counts, cancellations, retries, failures;
* event-loop health: a monitor task sleeps a fixed interval and records how late it wakes up (the loop lag). A
  blocking call on the loop shows up directly as lag;
* per-turn latency milestones, from the moment the turn's first input was accepted (wall clock):
  time_to_first_event, time_to_first_evidence (first RETRIEVAL_PARTIAL / RETRIEVAL_COMPLETED with evidence),
  time_to_first_answer (first ANSWER_COMPLETED, draft or final), time_to_validated_answer (first ANSWER_FINALIZED
  with status VALIDATED_FINAL - emitted when a validated final is released, by the runtime and by the Phase 7
  pipeline alike), total (last event of the turn).
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections import Counter

from streamrag.models.events import EventType as E


def pct(values: list[float]) -> dict:
    v = sorted(x for x in values if x is not None)
    if not v:
        return {"n": 0, "p50": None, "p95": None, "max": None, "mean": None}

    def q(p: float) -> float:
        k = (len(v) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)
    return {"n": len(v), "p50": round(q(0.5), 3), "p95": round(q(0.95), 3), "max": round(v[-1], 3),
            "mean": round(statistics.fmean(v), 3)}


class LoopLagMonitor:
    def __init__(self, interval_ms: float = 5.0) -> None:
        self.interval = interval_ms / 1000.0
        self.lags_ms: list[float] = []
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        try:
            while True:
                t = time.perf_counter()
                await asyncio.sleep(self.interval)
                self.lags_ms.append(max(0.0, (time.perf_counter() - t - self.interval) * 1000.0))
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None


def turn_milestones(events, first_input_wall_ms: dict[str, float]) -> dict[str, dict]:
    """Per utterance: ms from its first accepted input to each milestone (wall clock of the run)."""
    out: dict[str, dict] = {}
    for e in events:
        uid = e.utterance_id
        if uid is None or uid not in first_input_wall_ms:
            continue
        m = out.setdefault(uid, {"first_event": None, "first_evidence": None, "first_answer": None,
                                 "validated_answer": None, "utterance_end": None, "last_event": None})
        dt = round(e.t_wall_ms - first_input_wall_ms[uid], 3)
        m["last_event"] = dt
        if m["first_event"] is None:
            m["first_event"] = dt
        if m["first_evidence"] is None and (
                (e.type == E.RETRIEVAL_PARTIAL and e.payload.get("hits")) or
                (e.type == E.RETRIEVAL_COMPLETED and e.payload.get("evidence_ids"))):
            m["first_evidence"] = dt
        if m["first_answer"] is None and e.type == E.ANSWER_COMPLETED:
            m["first_answer"] = dt
        if m["validated_answer"] is None and e.type == E.ANSWER_FINALIZED and \
                e.payload.get("status") == "VALIDATED_FINAL":            # emitted by both pipelines at release
            m["validated_answer"] = dt
        if m["utterance_end"] is None and e.type == E.UTTERANCE_FINALIZED:
            m["utterance_end"] = dt
    return out


class TelemetryManager:
    def __init__(self) -> None:
        self.loop_lag = LoopLagMonitor()
        self.counters: Counter = Counter()

    def scheduler_summary(self, scheduler) -> dict:
        by_type: dict[str, dict] = {}
        for m in scheduler.metrics:
            d = by_type.setdefault(m["task_type"], {"statuses": Counter(), "queue_wait_ms": [], "exec_ms": []})
            d["statuses"][m["status"]] += 1
            if m.get("queue_wait_ms") is not None:
                d["queue_wait_ms"].append(m["queue_wait_ms"])
            if m.get("exec_ms") is not None and m["status"] != "LATE_RETURN":
                d["exec_ms"].append(m["exec_ms"])
        total = len([m for m in scheduler.metrics if m["status"] != "LATE_RETURN"])
        st = scheduler.stats
        return {
            "tasks": total,
            "by_type": {k: {"statuses": dict(v["statuses"]), "queue_wait_ms": pct(v["queue_wait_ms"]),
                            "exec_ms": pct(v["exec_ms"])} for k, v in sorted(by_type.items())},
            "cancellation_rate": round(st["CANCELLED"] / total, 4) if total else None,
            "retry_rate": round(st["retries"] / total, 4) if total else None,
            "failure_rate": round((st["FAILED"] + st["TIMED_OUT"]) / total, 4) if total else None,
            "stats": dict(st),
            "max_depth": {n: {"pending": p.max_pending, "running": p.max_running}
                          for n, p in scheduler.pools.items()},
        }
