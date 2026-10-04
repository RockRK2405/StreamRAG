# 06 · Iterative retrieval (Phase 9)

Code: `controller.py` (`AdaptiveRetrievalController.run`).

```
plan → [cache] → [session evidence: delta requirements] →
loop (bounded):  run ≤ max_parallel_tasks searches of one action family
                 merge (chunk-id dedup, RRF across searches, alternates collapsed)
                 assess (05) → actual gain → stop? (10)
                 next actions with expected gain: contradiction search | hop | relax filter | broaden retrievers |
                                                  requirement query (unmet slot's own text) | rerank | expand k
                 run the best family (targeted actions run alone: a speculative expansion next to a hop would spend
                 a search before the hop's value is known)
final evidence: supporting first, then context (if not SUFFICIENT), conflict losers / invalid / not applicable
                excluded with reasons, ≤ max_per_document per source, ≤ final_k
```

* Identical searches are never repeated (signature: queries, retrievers, k, filters).
* **Delta requirements:** with session evidence meeting some slots, the first searches target only the unmet slots.
* Each decision is a `RetrievalDecision` (inputs, reason, expected / actual gain, latency, result); each round emits
  EVIDENCE_ASSESSED, each follow-up RETRIEVAL_EXPANDED, the end RETRIEVAL_STOPPED with the `RetrievalState`.
* Evidence diversity: duplicates and near-duplicates (Phase 3 dedup alternates) are one item; at most
  `max_per_document` (3) items per document so one source cannot fill the answer, but one authoritative source is not
  replaced by weaker ones (the cap is per source, not a quota).
* `iterative: false` (ablation) = one round.
