# ADR-015: Multi-Intent Decomposition, Delta Retrieval and Intent-Aware Fusion (Phase 5)

- **Status:** Accepted (Phase 5)
- **Date:** 2026-10-02
- **Refines:** ADR-004 (controller), ADR-009 (fusion), ADR-012 (events), ADR-014 (execution model)

## Context

Phase 4 had one active query per utterance. Compound utterances need:
- one query per information need;
- parallel retrieval;
- fusion that keeps every need covered;
- incremental updates while the user is still speaking.

No LLM backend is configured (ADR-007 Q2). Every result must be traceable to what was said.

## Decision

1. **Rule-first decomposer.**
   - Clause roles: request / constraint / correction / context.
   - Guarded coordination splitting; scope-ruled constraints; span-traceable anaphora and ellipsis resolution; corrections that supersede.
   - The optional LLM check (spec §9.6) sits behind deterministic validation and verbatim-span grounding, is off by default, and can only add or confirm needs.
2. **Session-scoped ids and versioned IntentSets.**
   - The tracker re-decomposes the current transcript on each open gate and reconciles it with the previous version.
   - Only added or modified intents are retrieved (delta retrieval). Superseded or removed intents keep their ledger lineage as `stale`.
3. **The controller stays the gate.** Phase 4 suppression and stability rules decide *whether* to work. Its single-query storm guards are replaced by per-intent guards: per-intent and per-utterance budgets, a per-intent cooldown, and session ledger reuse (REQ-MI-005).
4. **One retrieval implementation.** Per-intent queries go through the Phase 3 `RetrievalService`, dispatched in parallel through the Phase 4 executor (3 slots).
5. **Fusion.**
   - `intent_aware` is the default: coverage floor `min_per_intent` = 2 round-robin in priority order, then fill to `top_k` by (rank, priority), with a section cap per intent. This is ADR-009 adapted to a global item budget.
   - Cross-intent dedup by chunk id and Phase 3 near-duplicate alternates.
   - Pluggable rerank, `none` by default.
   - Conservative numeric conflict flags.
6. **Telemetry.** The envelope gains `intent_id` and `query_id`. New events: INTENT_DETECTED, INTENT_UPDATED, INTENT_SUPERSEDED, QUERY_GENERATED, MULTI_QUERY_STARTED, MULTI_QUERY_COMPLETED, EVIDENCE_DEDUPLICATED, RERANK_STARTED, RERANK_COMPLETED. INTENTS_UPDATED and EVIDENCE_FUSED, defined in the spec, are now used.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| LLM-first decomposition | No backend; latency 0.3–1.5 s per call [Phase 2 E]; opaque; would put a non-deterministic step in front of retrieval |
| Split on every noun phrase or "and" | Over-decomposition: "information and details about X", "third and fifth week", narrative nouns |
| Re-retrieve all intents on every update | Duplicate retrieval; the dev streaming run measured 0 duplicates with delta retrieval |
| Global RRF across intents | Rewards chunks shared by several intents. Measured below `intent_aware` at small budgets on the dev suite (report §11). |
| `global_score` | Tied with `intent_aware` on the dev suite. Scores are not comparable across queries, and it gives no coverage guarantee (unit test: a strong intent starves the others under concat-like ranking). |
| Batched-only dispatch | Measured (report §8). One ONNX call for all embeddings. Parallel stays the streaming default because same-tick batches are small and parallel jobs complete independently (one slow query does not hold back the others' evidence). |

## Consequences

- **Determinism:** decomposition is deterministic and sub-millisecond-to-millisecond (measured). Virtual traces replay exactly; realtime traces replay with identical behaviour.
- **Known gaps** (report §20):
  - needs without lexical overlap are not linked as refinements;
  - topic inheritance for follow-ups works only for facet-only needs ("the application process"), not for new nouns ("the opening hours");
  - wh-restarts without an auxiliary are missed;
  - pronoun resolution is blocked by any earlier content word in the same need;
  - restriction scope does not cross clause boundaries without a marker.
- **Not held out:** all dev-suite numbers are from a team-authored, fixture-domain suite. Official multi-intent cases are required (Phase 2 §22).
