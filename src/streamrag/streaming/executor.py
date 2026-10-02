"""Retrieval executors: run Phase 3 retrieval without blocking the transcript stream.

Both executors keep a FIFO queue with ``max_concurrency`` running slots. A *queued* job can be cancelled (it
never runs); a *running* job always completes (retrieval is CPU work in a worker thread and cannot be safely
interrupted; its result is kept and marked stale if superseded - see docs/streaming/06).

* ``VirtualExecutor``  deterministic: the retrieval is computed when it starts (real evidence), and its
  completion is delivered at ``start + sim_retrieval_latency_ms`` on the virtual clock. The measured wall
  latency is reported under ``wall`` (not used for ordering).
* ``AsyncExecutor``    realtime: ``asyncio.to_thread(retrieve)`` with a timeout; completion is delivered on the
  event loop when the thread finishes.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from streamrag.errors import RetrieverTimeoutError
from streamrag.models.evidence import EvidenceSet
from streamrag.models.retrieval import RetrievalOptions
from streamrag.streaming.clock import PRIORITY_COMPLETION, RealtimeScheduler, VirtualScheduler


class RetrievalBackend(Protocol):
    def retrieve(self, query: str, options: RetrievalOptions | None = None) -> EvidenceSet: ...


@dataclass
class Job:
    query_id: str
    query_text: str
    options: RetrievalOptions
    on_start: Callable[[str], None]
    on_done: Callable[[str, EvidenceSet | None, Exception | None, float], None]
    cancelled: bool = False


class _QueueMixin:
    def _init_queue(self, max_concurrency: int) -> None:
        self.queue: deque[Job] = deque()
        self.running = 0
        self.max_concurrency = max_concurrency

    def submit(self, job: Job) -> None:
        self.queue.append(job)
        self._pump()

    def cancel_queued(self, query_id: str) -> bool:
        for job in list(self.queue):
            if job.query_id == query_id:
                job.cancelled = True
                self.queue.remove(job)
                return True
        return False

    def is_queued(self, query_id: str) -> bool:
        return any(j.query_id == query_id for j in self.queue)

    def busy(self) -> bool:
        return self.running > 0 or bool(self.queue)

    def _pump(self) -> None:
        while self.running < self.max_concurrency and self.queue:
            job = self.queue.popleft()
            if job.cancelled:
                continue
            self.running += 1
            self._start(job)

    def _start(self, job: Job) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class VirtualExecutor(_QueueMixin):
    def __init__(self, scheduler: VirtualScheduler, backend: RetrievalBackend, latency_ms: float,
                 max_concurrency: int = 2) -> None:
        self.sched, self.backend, self.latency_ms = scheduler, backend, latency_ms
        self._init_queue(max_concurrency)

    def _start(self, job: Job) -> None:
        job.on_start(job.query_id)
        t = time.perf_counter()
        es, err = None, None
        try:
            es = self.backend.retrieve(job.query_text, job.options)
        except Exception as exc:   # noqa: BLE001 - surfaced as a structured failure event
            err = exc
        wall = (time.perf_counter() - t) * 1000.0
        self.sched.call_at(self.sched.now_ms() + self.latency_ms, PRIORITY_COMPLETION, self._finish, job, es, err, wall)

    def _finish(self, job: Job, es, err, wall: float) -> None:
        self.running -= 1
        job.on_done(job.query_id, es, err, wall)
        self._pump()


class AsyncExecutor(_QueueMixin):
    def __init__(self, scheduler: RealtimeScheduler, backend: RetrievalBackend, timeout_ms: float,
                 max_concurrency: int = 2) -> None:
        self.sched, self.backend, self.timeout_s = scheduler, backend, timeout_ms / 1000.0
        self._init_queue(max_concurrency)
        self.tasks: set[asyncio.Task] = set()

    def _start(self, job: Job) -> None:
        task = self.sched.loop.create_task(self._run(job))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(self, job: Job) -> None:
        job.on_start(job.query_id)
        t = time.perf_counter()
        es, err = None, None
        try:
            es = await asyncio.wait_for(asyncio.to_thread(self.backend.retrieve, job.query_text, job.options),
                                        timeout=self.timeout_s)
        except asyncio.TimeoutError:
            err = RetrieverTimeoutError(f"retrieval exceeded {self.timeout_s * 1000:.0f} ms (thread left to finish)")
        except Exception as exc:   # noqa: BLE001
            err = exc
        wall = (time.perf_counter() - t) * 1000.0
        self.running -= 1
        job.on_done(job.query_id, es, err, wall)
        self._pump()
