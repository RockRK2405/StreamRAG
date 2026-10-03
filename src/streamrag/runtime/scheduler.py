"""TaskScheduler, worker pools and runners (docs/runtime/03, 04).

* **Pools.** Each resource class (``retrieval``, ``cpu``, ``llm``) has a bounded number of running slots and a
  bounded pending queue. Slots are held until the worker actually returns - a timed-out task whose thread is still
  finishing keeps its slot ("zombie"), so a timeout can never create more threads than configured.
* **Priority with aging.** The next task of a pool is the one with the smallest ``priority - waited / aging_ms``
  (ties: FIFO). A low-priority task therefore gains one level per ``aging_ms`` of waiting and is never starved.
* **Deadlines.** A task whose deadline passed while pending never starts (TIMED_OUT); a running one is reported
  TIMED_OUT at its deadline and its token is cancelled so the worker stops at the next checkpoint.
* **Retries.** Transient failures are resubmitted after a bounded exponential backoff (RetryManager), while the
  deadline allows; the attempt count is part of the task.
* **Idempotency.** A task with an ``idempotency_key`` already pending or running is not started twice - the new
  caller is attached to the running one; a completed result is reused (identical retrievals in one session).
* **Runners.** ``VirtualRunner`` computes the task when it starts and delivers it after a modelled latency on the
  virtual clock (deterministic replay). ``ThreadRunner`` runs it on the pool's thread pool and delivers the result
  on the asyncio loop (real measurements).
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from streamrag.runtime.cancellation import CancellationManager, CancellationToken, TaskCancelled
from streamrag.runtime.retry import RetryManager
from streamrag.runtime.tasks import TERMINAL, Task, TaskResult, TaskStatus, TaskType

class WorkContext:
    """Handed to every worker: its task, its cancellation token, and mode-aware waiting. ``delay`` is a real
    interruptible wait in realtime mode and extra modelled latency on the virtual clock."""

    def __init__(self, task: Task, token: CancellationToken, virtual: bool) -> None:
        self.task, self.token, self.virtual = task, token, virtual

    def checkpoint(self) -> None:
        self.token.raise_if_cancelled()

    def delay(self, ms: float) -> None:
        if self.virtual:
            self.task.meta["sim_extra_ms"] = self.task.meta.get("sim_extra_ms", 0.0) + ms
            return
        if self.token.wait(ms / 1000.0):
            raise TaskCancelled(self.token.reason or "cancelled")


WorkFn = Callable[[WorkContext], Any]
ResultFn = Callable[[TaskResult], None]
PRIORITY_COMPLETION = 0                     # same ordering constant as the Phase 4 schedulers


class QueueFull(Exception):
    pass


@dataclass
class _Entry:
    task: Task
    fn: WorkFn
    callbacks: list[ResultFn]
    token: CancellationToken
    seq: int
    enqueued_ms: float
    started_wall: float | None = None
    zombie: bool = False                     # reported TIMED_OUT, worker still returning
    delivered: bool = False


@dataclass
class Pool:
    name: str
    capacity: int
    queue_capacity: int
    pending: list[_Entry] = field(default_factory=list)
    running: dict[str, _Entry] = field(default_factory=dict)
    max_pending: int = 0
    max_running: int = 0

    @property
    def busy_slots(self) -> int:
        return len(self.running)


class VirtualRunner:
    """Deterministic: run now (real computation), deliver at now + modelled latency on the virtual clock."""

    mode = "virtual"

    def __init__(self, clock, sim_latency_ms) -> None:
        self.clock, self.sim = clock, sim_latency_ms

    def latency_ms(self, task: Task) -> float:
        if "sim_ms" in task.meta:
            return float(task.meta["sim_ms"])
        name = {TaskType.RETRIEVAL: "dense", TaskType.ANALYTICS: "assemble",
                TaskType.ANSWER_EXTRACTIVE: "draft"}.get(task.task_type, task.task_type.value)
        return float(getattr(self.sim, name, 0.0))

    def start(self, entry: _Entry, done: Callable[[_Entry, Any, BaseException | None, float, float], None]) -> None:
        t0 = time.perf_counter()
        res, err = None, None
        try:
            res = entry.fn(WorkContext(entry.task, entry.token, True))
        except BaseException as exc:          # noqa: BLE001 - delivered as a structured task failure
            err = exc
        wall = (time.perf_counter() - t0) * 1000.0
        lat = self.latency_ms(entry.task) + float(entry.task.meta.pop("sim_extra_ms", 0.0))
        self.clock.call_at(self.clock.now_ms() + lat, PRIORITY_COMPLETION, done, entry, res, err, wall, lat)

    def close(self) -> None:
        return None


class ThreadRunner:
    """Realtime: one bounded ThreadPoolExecutor per pool; results come back on the asyncio loop."""

    mode = "realtime"

    def __init__(self, loop: asyncio.AbstractEventLoop, clock, pool_sizes: dict[str, int]) -> None:
        self.loop, self.clock = loop, clock
        self.executors = {name: ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"streamrag-rt-{name}")
                          for name, n in pool_sizes.items()}

    def start(self, entry: _Entry, done) -> None:
        def call():
            t0 = time.perf_counter()
            try:
                return entry.fn(WorkContext(entry.task, entry.token, False)), None, (time.perf_counter() - t0) * 1000.0
            except BaseException as exc:      # noqa: BLE001
                return None, exc, (time.perf_counter() - t0) * 1000.0

        fut = self.loop.run_in_executor(self.executors[entry.task.pool], call)

        def finished(f: asyncio.Future) -> None:
            if f.cancelled():
                done(entry, None, TaskCancelled("executor shut down"), 0.0, 0.0)
                return
            res, err, wall = f.result()
            done(entry, res, err, wall, wall)

        fut.add_done_callback(finished)

    def close(self, wait: bool = True) -> None:
        for ex in self.executors.values():
            ex.shutdown(wait=wait, cancel_futures=True)


class TaskScheduler:
    def __init__(self, rcfg, clock, runner, cancellation: CancellationManager, retry: RetryManager,
                 on_event: Callable[[Task, str, dict], None] | None = None) -> None:
        self.cfg, self.clock, self.runner = rcfg, clock, runner
        self.cancellation, self.retry = cancellation, retry
        self.on_event = on_event or (lambda task, kind, payload: None)
        q = rcfg.queues
        self.pools = {"retrieval": Pool("retrieval", rcfg.max_concurrent_retrievals, q.retrieval),
                      "cpu": Pool("cpu", rcfg.max_concurrent_cpu, q.cpu),
                      "llm": Pool("llm", rcfg.max_concurrent_llm_calls, q.llm)}
        self._seq = itertools.count()
        self._ids = itertools.count(1)
        self.tasks: dict[str, Task] = {}
        self._live_by_key: dict[str, _Entry] = {}
        self._done_by_key: dict[str, TaskResult] = {}
        self.accepting = True
        self.stats: Counter = Counter()
        self.metrics: list[dict] = []            # one record per terminal task (queue wait, execution, status)
        self.depth_samples: list[tuple[float, dict[str, int]]] = []

    # ------------------------------------------------------------------ public API
    def new_task_id(self) -> str:
        return f"T{next(self._ids)}"

    def submit(self, task: Task, fn: WorkFn, on_result: ResultFn) -> Task:
        if not self.accepting:
            self._deliver_now(task, TaskStatus.CANCELLED, None, "runtime shutting down", [on_result])
            return task
        key = task.idempotency_key
        if key is not None:
            live = self._live_by_key.get(key)
            if live is not None and live.task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                live.callbacks.append(on_result)
                self.stats["deduplicated"] += 1
                self.on_event(task, "deduplicated", {"joined_task_id": live.task.task_id})
                return live.task
            done = self._done_by_key.get(key)
            if done is not None:
                self.stats["result_reused"] += 1
                self.tasks[task.task_id] = task
                self.on_event(task, "reused", {"from_task_id": done.task_id})
                self._deliver_now(task, TaskStatus.COMPLETED, done.result, None, [on_result],
                                  {"reused_from": done.task_id})
                return task
        pool = self.pools[task.pool]
        if len(pool.pending) >= pool.queue_capacity:
            victim = max(pool.pending, key=lambda e: (e.task.priority, e.seq))
            if victim.task.priority > task.priority:          # shed the least important pending task
                pool.pending.remove(victim)
                self.stats["shed"] += 1
                self._finish(victim, TaskStatus.CANCELLED, None, "shed_by_backpressure", 0.0)
            else:
                self.stats["rejected"] += 1
                self.tasks[task.task_id] = task
                self.on_event(task, "rejected", {"reason": "queue_full", "pool": pool.name,
                                                 "depth": len(pool.pending)})
                self._deliver_now(task, TaskStatus.FAILED, None, "queue_full", [on_result])
                return task
        token = self.cancellation.task_token(task.task_id, task.session_id)
        entry = _Entry(task, fn, [on_result], token, next(self._seq), self.clock.now_ms())
        self.tasks[task.task_id] = task
        if key is not None:
            self._live_by_key[key] = entry
        pool.pending.append(entry)
        pool.max_pending = max(pool.max_pending, len(pool.pending))
        self.stats["submitted"] += 1
        self.on_event(task, "scheduled", {"pool": pool.name, "depth": len(pool.pending)})
        self._sample()
        self._pump(pool)
        return task

    def cancel(self, task_id: str, reason: str) -> bool:
        """Pending: removed and reported CANCELLED now. Running: its token is cancelled (cooperative) and the
        result, when the worker returns, is reported CANCELLED and never applied."""
        for pool in self.pools.values():
            for e in pool.pending:
                if e.task.task_id == task_id:
                    pool.pending.remove(e)
                    self.cancellation.cancel_task(task_id, reason)
                    self._finish(e, TaskStatus.CANCELLED, None, reason, 0.0)
                    self._pump(pool)
                    return True
            e = pool.running.get(task_id)
            if e is not None and not e.zombie:
                return self.cancellation.cancel_task(task_id, reason)
        return False

    def cancel_where(self, pred: Callable[[Task], bool], reason: str) -> list[str]:
        ids = [e.task.task_id for p in self.pools.values() for e in list(p.pending) + list(p.running.values())
               if pred(e.task) and not e.zombie]
        return [t for t in ids if self.cancel(t, reason)]

    def busy(self, pred: Callable[[Task], bool] | None = None) -> bool:
        """Pending or running work (zombies excluded: their result is already reported)."""
        for p in self.pools.values():
            for e in p.pending:
                if pred is None or pred(e.task):
                    return True
            for e in p.running.values():
                if not e.zombie and (pred is None or pred(e.task)):
                    return True
        return False

    def zombies(self) -> int:
        return sum(1 for p in self.pools.values() for e in p.running.values() if e.zombie)

    def depths(self) -> dict[str, dict[str, int]]:
        return {n: {"pending": len(p.pending), "running": len(p.running)} for n, p in self.pools.items()}

    def stop_accepting(self) -> None:
        self.accepting = False

    # ------------------------------------------------------------------ internals
    def _sample(self) -> None:
        self.depth_samples.append((self.clock.now_ms(), {n: len(p.pending) for n, p in self.pools.items()}))

    def _effective(self, e: _Entry, now: float) -> tuple[float, int]:
        return e.task.priority - (now - e.enqueued_ms) / self.cfg.priorities.aging_ms, e.seq

    def _pump(self, pool: Pool) -> None:
        while pool.pending and len(pool.running) < pool.capacity:
            now = self.clock.now_ms()
            e = min(pool.pending, key=lambda x: self._effective(x, now))
            pool.pending.remove(e)
            if e.token.is_cancelled():
                self._finish(e, TaskStatus.CANCELLED, None, e.token.reason or "cancelled", 0.0)
                continue
            if e.task.deadline_ms is not None and now >= e.task.deadline_ms:
                self._finish(e, TaskStatus.TIMED_OUT, None, "deadline_expired_while_pending", 0.0)
                continue
            e.task.status = TaskStatus.RUNNING
            e.task.started_at_ms = now
            pool.running[e.task.task_id] = e
            pool.max_running = max(pool.max_running, len(pool.running))
            self.on_event(e.task, "started", {"pool": pool.name, "queue_wait_ms": round(now - e.enqueued_ms, 3),
                                              "attempt": e.task.attempt})
            if e.task.deadline_ms is not None and self.runner.mode == "realtime":
                delay = max(0.0, (e.task.deadline_ms - now) / 1000.0)
                self.runner.loop.call_later(delay, self._deadline_hit, e)
            elif e.task.deadline_ms is not None:
                self.clock.call_at(e.task.deadline_ms, PRIORITY_COMPLETION, self._deadline_hit, e)
            e.started_wall = time.perf_counter()
            self.runner.start(e, self._on_done)
        self._sample()

    def _deadline_hit(self, e: _Entry) -> None:
        if e.delivered or e.task.task_id not in self.pools[e.task.pool].running:
            return
        e.zombie = True                                     # slot stays taken until the worker returns
        self.cancellation.cancel_task(e.task.task_id, "deadline")
        self._finish(e, TaskStatus.TIMED_OUT, None, "deadline_exceeded", 0.0, release=False)

    def _on_done(self, e: _Entry, res, err, wall_ms: float, latency_ms: float) -> None:
        pool = self.pools[e.task.pool]
        try:
            self._complete(e, pool, res, err, wall_ms)
        finally:
            self._pump(pool)

    def _complete(self, e: _Entry, pool: Pool, res, err, wall_ms: float) -> None:
        pool.running.pop(e.task.task_id, None)
        if e.delivered:                                     # already reported (timed out / cancelled late)
            self.stats["late_results_dropped"] += 1
            self.metrics_late(e, wall_ms)
            self.cancellation.finished(e.task.task_id)
            return
        if e.token.is_cancelled() or isinstance(err, TaskCancelled):
            self._finish(e, TaskStatus.CANCELLED, None, e.token.reason or str(err), wall_ms)
        elif err is not None:
            if self.retry.should_retry(err, e.task.attempt) and self._retry_fits(e):
                self._retry(e, err, wall_ms)
            else:
                self._finish(e, TaskStatus.FAILED, None, f"{err.__class__.__name__}: {err}", wall_ms)
        else:
            self._finish(e, TaskStatus.COMPLETED, res, None, wall_ms)

    def metrics_late(self, e: _Entry, wall_ms: float) -> None:
        self.metrics.append({"task_id": e.task.task_id, "task_type": e.task.task_type.value, "status": "LATE_RETURN",
                             "exec_ms": round(wall_ms, 3), "session_id": e.task.session_id})

    def _retry_fits(self, e: _Entry) -> bool:
        if e.task.deadline_ms is None:
            return True
        return self.clock.now_ms() + self.retry.cfg.initial_delay_ms < e.task.deadline_ms

    def _retry(self, e: _Entry, err: BaseException, wall_ms: float) -> None:
        delay = self.retry.delay_ms(e.task.attempt)
        e.task.attempt += 1
        e.task.status = TaskStatus.PENDING
        self.stats["retries"] += 1
        self.on_event(e.task, "retried", {"attempt": e.task.attempt, "error": f"{err.__class__.__name__}: {err}",
                                          "backoff_ms": round(delay, 3), "wall": {"exec_ms": round(wall_ms, 3)}})
        self.clock.call_at(self.clock.now_ms() + delay, PRIORITY_COMPLETION, self._requeue, e)

    def _requeue(self, e: _Entry) -> None:
        pool = self.pools[e.task.pool]
        if e.token.is_cancelled():
            self._finish(e, TaskStatus.CANCELLED, None, e.token.reason or "cancelled", 0.0)
            return
        e.enqueued_ms, e.seq = self.clock.now_ms(), next(self._seq)
        pool.pending.append(e)
        self._pump(pool)

    def _finish(self, e: _Entry, status: TaskStatus, res, error: str | None, wall_ms: float,
                release: bool = True) -> None:
        if e.delivered:
            return
        e.delivered = True
        t = e.task
        t.status, t.completed_at_ms = status, self.clock.now_ms()
        if t.idempotency_key is not None and self._live_by_key.get(t.idempotency_key) is e:
            del self._live_by_key[t.idempotency_key]
        meta = {"queue_wait_ms": None if t.started_at_ms is None else round(t.started_at_ms - e.enqueued_ms, 3),
                "exec_ms": round(wall_ms, 3), "attempts": t.attempt + 1}
        result = TaskResult(t.task_id, t.task_type, t.session_id, t.state_version, t.epoch, t.correlation_id,
                            status, res, error, meta)
        if status == TaskStatus.COMPLETED and t.idempotency_key is not None:
            self._done_by_key[t.idempotency_key] = result
        self.stats[status.value] += 1
        self.metrics.append({"task_id": t.task_id, "task_type": t.task_type.value, "status": status.value,
                             "session_id": t.session_id, "priority": t.priority, **meta,
                             "virtual_ms": None if t.started_at_ms is None else round(t.completed_at_ms - t.started_at_ms, 3)})
        self.on_event(t, status.value.lower(), {"error": error, "queue_wait_ms": meta["queue_wait_ms"],
                                                "attempts": meta["attempts"], "wall": {"exec_ms": meta["exec_ms"]}})
        if release:
            self.cancellation.finished(t.task_id)
        for cb in e.callbacks:
            self._safe(cb, result)

    def _safe(self, cb, result: TaskResult) -> None:
        """A failing result handler must not stall the pool or the loop (failure isolation): it is reported as a
        task event and the scheduler carries on."""
        try:
            cb(result)
        except Exception as exc:              # noqa: BLE001
            self.stats["callback_errors"] += 1
            task = self.tasks.get(result.task_id)
            if task is not None:
                self.on_event(task, "callback_error", {"error": f"{exc.__class__.__name__}: {exc}"})

    def _deliver_now(self, task: Task, status: TaskStatus, res, error: str | None, callbacks: list[ResultFn],
                     extra: dict | None = None) -> None:
        task.status, task.completed_at_ms = status, self.clock.now_ms()
        result = TaskResult(task.task_id, task.task_type, task.session_id, task.state_version, task.epoch,
                            task.correlation_id, status, res, error, {"queue_wait_ms": 0.0, "exec_ms": 0.0,
                                                                      "attempts": 0, **(extra or {})})
        self.stats[status.value] += 1
        for cb in callbacks:
            self.clock.call_at(self.clock.now_ms(), PRIORITY_COMPLETION, self._safe, cb, result)

    def forget_session(self, session_id: str) -> None:
        """Drop cached results of a session (reset): a reused result must never cross an epoch."""
        self._done_by_key = {k: v for k, v in self._done_by_key.items() if v.session_id != session_id}

    def all_terminal(self) -> bool:
        return all(t.status in TERMINAL for t in self.tasks.values())
