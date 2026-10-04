# 11 · Budget management (Phase 9)

Code: `AdaptiveBudget` (config), `RetrievalBudgetState`, `controller.py`; Phase 8 integration in
`runtime/aggregator.py` (`_submit_adaptive`).

| Budget | Default | Enforced by |
|---|---|---|
| `max_queries` | 5 searches per need | loop (room check before a round), QUERY_LIMIT |
| `max_results` | 40 (sum of k) | QUERY_LIMIT |
| `max_iterations` | 3 rounds | ITERATION_LIMIT (1 when `iterative: false` or the latency budget is tight) |
| `max_latency_ms` | 1500 | LATENCY_LIMIT: stop before a round if spent + mean measured search time exceeds it |
| `max_parallel_tasks` | 2 | searches per round |
| `max_hops` | 2 | bridge hops |

**Phase 8 runtime.** An adaptive need is one RETRIEVAL task on the retrieval pool; its latency budget is
min(`max_latency_ms`, the task's remaining deadline from Phase 8 deadline propagation). Below `tight_latency_ms`
(150 ms) the router takes the fast path: LEXICAL (no embedding), one round, no rerank. The task's cancellation token
is checked between searches (cooperative); injected faults (`FaultInjector`) apply to every dense search of the
controller. Buffered adaptive events are emitted on the loop when the result is committed; a superseded run's events
are never emitted (its task is reported cancelled / stale by Phase 8).

**Resource exhaustion.** Query text is truncated to 2000 characters and at most 8 constraints are analysed; caches are
bounded LRUs; metadata values are whitelisted and ≤ 120 characters (`corpus/metadata.py`).
