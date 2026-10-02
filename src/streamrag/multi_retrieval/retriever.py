"""Per-intent retrieval through the Phase 3 RetrievalService (docs/multi_intent/05).

There is no separate retrieval implementation: every intent query goes through ``RetrievalService`` (BM25 + dense
+ RRF + dedup [+ rerank]) with identical options, so retrieval behaviour is consistent across intents.

Dispatch strategies (measured, not assumed - research/phase5/latency):
  sequential   one query after another (baseline)
  parallel     one worker thread per query, at most ``max_concurrent_retrievals`` at a time
  batched      one ``retrieve_batch`` call: all query embeddings in one ONNX call, then per-query search

Budget: at most ``max_queries`` queries are dispatched, in priority order; the rest are recorded as skipped with
the reason (never silently discarded).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

from streamrag.models.evidence import EvidenceSet
from streamrag.models.intents import IntentQuery
from streamrag.models.retrieval import RetrievalOptions, RetrievalRequest

Dispatch = Literal["sequential", "parallel", "batched"]


@dataclass
class QueryOutcome:
    intent_id: str
    query: IntentQuery
    evidence: EvidenceSet | None
    error: str | None
    start_ms: float                       # relative to the batch start
    end_ms: float

    @property
    def latency_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class MultiRetrievalResult:
    dispatch: Dispatch
    outcomes: list[QueryOutcome]
    skipped: list[tuple[str, str]] = field(default_factory=list)   # (intent_id, reason)
    total_ms: float = 0.0

    @property
    def critical_path_ms(self) -> float:
        """Slowest single query: the lower bound of any parallel schedule."""
        return max((o.latency_ms for o in self.outcomes), default=0.0)

    @property
    def max_concurrency(self) -> int:
        """Maximum number of queries actually running at the same time (from measured intervals); a batched call
        is one call, so it counts as 1."""
        if self.dispatch == "batched":
            return 1 if self.outcomes else 0
        ev = sorted([(o.start_ms, 1) for o in self.outcomes] + [(o.end_ms, -1) for o in self.outcomes],
                    key=lambda x: (x[0], x[1]))
        cur = best = 0
        for _, d in ev:
            cur += d
            best = max(best, cur)
        return best

    def by_intent(self) -> dict[str, QueryOutcome]:
        return {o.intent_id: o for o in self.outcomes}


class MultiQueryRetriever:
    def __init__(self, service, options: RetrievalOptions | None = None, max_concurrent: int = 3,
                 max_queries: int = 10) -> None:
        self.service, self.options = service, options or RetrievalOptions()
        self.max_concurrent, self.max_queries = max_concurrent, max_queries
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="streamrag-multi")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def retrieve(self, queries: list[IntentQuery], dispatch: Dispatch = "parallel",
                 priority: dict[str, int] | None = None) -> MultiRetrievalResult:
        order = sorted(queries, key=lambda q: ((priority or {}).get(q.intent_id, 0),))
        run, skipped = order[: self.max_queries], [(q.intent_id, f"max_queries={self.max_queries}")
                                                    for q in order[self.max_queries:]]
        t0 = time.perf_counter()

        def one(q: IntentQuery) -> QueryOutcome:
            s = (time.perf_counter() - t0) * 1000.0
            es, err = None, None
            try:
                es = self.service.retrieve(RetrievalRequest(query=q.text, options=self.options))
            except Exception as exc:    # noqa: BLE001 - surfaced per intent, the other intents still complete
                err = f"{exc.__class__.__name__}: {exc}"
            return QueryOutcome(q.intent_id, q, es, err, s, (time.perf_counter() - t0) * 1000.0)

        if dispatch == "sequential":
            outcomes = [one(q) for q in run]
        elif dispatch == "parallel":
            outcomes = list(self._pool.map(one, run))
        else:
            s = (time.perf_counter() - t0) * 1000.0
            try:
                sets = self.service.retrieve_batch([RetrievalRequest(query=q.text, options=self.options) for q in run])
                e = (time.perf_counter() - t0) * 1000.0
                outcomes = [QueryOutcome(q.intent_id, q, es, None, s, e) for q, es in zip(run, sets)]
            except Exception as exc:    # noqa: BLE001
                e = (time.perf_counter() - t0) * 1000.0
                outcomes = [QueryOutcome(q.intent_id, q, None, f"{exc.__class__.__name__}: {exc}", s, e) for q in run]
        return MultiRetrievalResult(dispatch, outcomes, skipped, round((time.perf_counter() - t0) * 1000.0, 4))
