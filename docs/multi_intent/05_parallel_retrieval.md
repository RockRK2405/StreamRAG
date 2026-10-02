# 05: Parallel Retrieval

**Code:** `multi_retrieval/retriever.py` (`MultiQueryRetriever`, offline/batch API), `multi_retrieval/coordinator.py` (streaming), Phase 4 executors
**Config:** `multi_intent.max_concurrent_retrievals`, `max_queries_per_utterance`, `max_queries_per_intent`, `max_candidates_per_intent`, `dispatch`

## One retrieval implementation

- Every intent query goes through the Phase 3 `RetrievalService` (BM25 + dense, RRF, dedup, optional cross-encoder) with identical options.
- No second retrieval path exists. A test asserts that all three dispatch strategies return exactly what `RetrievalService.retrieve` returns.

## Dispatch strategies

| Strategy | Mechanism |
|---|---|
| `sequential` | One query after another |
| `parallel` | One worker thread per query, at most `max_concurrent_retrievals` (3) at a time. Streaming uses the Phase 4 executor with that many slots. |
| `batched` | One `retrieve_batch` call: all query embeddings in one ONNX call, then per-query search |

The choice was **measured, not assumed** (`research/phase5/results/dispatch_latency.json`, Phase 5 report §8 and §19). Configuration default: `parallel`.

## Streaming dispatch (delta retrieval)

On each controller tick whose gate is open (docs/multi_intent/08), the coordinator handles every active intent in **priority** order:

1. Build the query.
2. If the query text equals the intent's active query: **reuse** (no retrieval).
3. If an earlier utterance's completed query is a near-duplicate: **ledger hit** (reuse its evidence).
4. Apply the guards, each failure → `RETRIEVAL_SKIPPED` with its reason:
   - per-intent budget `max_queries_per_intent` (3) → `intent_budget_exhausted`;
   - utterance budget `max_queries_per_utterance` (10) → `utterance_budget_exhausted`;
   - provisional per-intent cooldown (400 ms stream time) → `intent_cooldown`.
5. Create the ledger record (`intent_id`, `intent_version`, `batch_id`) and emit `QUERY_GENERATED`.
6. Cancel a still-queued superseded version of the same intent.

All queries of one tick form a batch:
- `MULTI_QUERY_STARTED` (with `reused_intents`);
- one job per query in the executor (same-tick queries start together; tested: equal `RETRIEVAL_STARTED` times);
- `MULTI_QUERY_COMPLETED` (makespan, critical path) when the last one finishes.

## Prioritisation (when the budget or slots bind)

```
priority_score = 0.35*confidence + 0.25*specificity + 0.15*explicit + 0.10*has_constraints
               + 0.10*novelty (vs earlier session queries) + 0.05*has_dependents
```

- Ties are broken by order of mention.
- `order` (mention order) is kept separately for answer organisation.
- Intents beyond `max_intents` (4) are dropped with an explicit reason (`DropRecord`), never silently.

## Provenance

Every retrieval is tied to session, utterance, intent and query:
- the envelope fields `intent_id` / `query_id` of `RETRIEVAL_STARTED` / `RETRIEVAL_COMPLETED`;
- the ledger record.

`QueryLedger.lineage_tree(utterance)` returns `intent → query versions → evidence ids`. It is included in `TURN_COMPLETED`.
