# Memory (session state)

"Memory" in StreamRAG means **per-session conversational state**:

* it lives only in the running process;
* it is scoped to one session and destroyed when the session closes or goes idle (15 minutes in the demo server);
* nothing about the user is persisted.

The design documents are in `docs/session/`:

| topic | document |
|---|---|
| what the session holds (needs, frames, ledger, evidence, claims, answer versions) | `docs/session/01_session_state.md`, `02_session_memory.md` |
| late details, refinements, corrections | `docs/session/03_context_change_detection.md` |
| delta retrieval (only changed needs are retrieved again) | `docs/session/04_delta_planning.md` |
| evidence lifecycle (active / retained / revalidation / stale / superseded) | `docs/session/05_evidence_lifecycle.md` |
| claims and their re-validation | `docs/session/06_claim_model.md`, `07_claim_revalidation.md` |
| versioned answers and diffs | `docs/session/08_answer_versioning.md` |
| bounds | `docs/session/09_context_compression.md`, `10_memory_boundaries.md` |
| caches (semantic query cache, validated-claim cache, validity signatures) | `docs/retrieval/08_cache_strategy.md` |

## Phase 11 changes

* **Within one utterance,** evidence retrieved only by superseded partial-transcript queries and not confirmed by the
  refined query is marked STALE (`not_confirmed_by_refined_query`).
* **When the per-need query budget is used up,** the need is re-validated against its latest completed query
  (`budget_exhausted_reuse_latest`).

See `docs/architecture/14_final_architecture.md` §14.4.

## Measured effect of memory

From `PHASE_10_EVALUATION_REPORT.md` and `FINAL_BENCHMARK_RESULTS/`:

* **Reuse saves retrieval.** Session and claim reuse lowered retrieval calls on conversation turns (dev: −0.20 calls
  per turn, p = 0.031).
* **Delta retrieval** lowered them further (−0.25 to −0.33 calls per conversation turn).
* **Removing memory hurts.** In the streaming runtime, removing memory left 5 more turns without a validated answer.
* **Reuse also cost answers.** It lost 1–2 turns of answer correctness in Phase 10. Those were stale-reuse cases, which
  the Phase 11 fixes target.
