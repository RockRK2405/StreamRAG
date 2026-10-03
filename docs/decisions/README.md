# Architecture Decision Records

| ADR | Decision | Status |
|---|---|---|
| [ADR-001](ADR-001-retrieval-strategy.md) | Hybrid BM25 + dense, RRF, exact in-process search, section-bounded chunks | Accepted |
| [ADR-002](ADR-002-embedding-model.md) | bge-small-en-v1.5 default, MiniLM fallback; final choice by Exp 1 | Accepted, pending experiment |
| [ADR-003](ADR-003-reranking.md) | RRF+dedup always (official minimum, K3); cross-encoder gated; rerank-on-stability | Accepted |
| [ADR-004](ADR-004-streaming-controller.md) | Rule-first per-segment controller; segment-event triggers; no suppression of information requests (K4) | Accepted |
| [ADR-005](ADR-005-session-state.md) | In-memory session actor with structured state; write-only telemetry | Accepted |
| [ADR-006](ADR-006-answer-refinement.md) | Claim-level versions; verbatim carry-over; delta-only retrieval | Accepted |
| [ADR-007](ADR-007-llm-backend-and-packaging.md) | Hosted → local → extractive fallback; container is the primary path (K2) | Accepted, primary backend pending Q2 |
| [ADR-008](ADR-008-grounding-and-citation.md) | Label-constrained citations; L0–L2 verifier; sentence-gated streaming | Accepted |
| [ADR-009](ADR-009-evidence-fusion.md) | RRF within an intent; quota round-robin across intents | Accepted |
| [ADR-010](ADR-010-state-machine-model.md) | Hierarchical statechart with concurrent regions (K9) | Accepted |
| [ADR-011](ADR-011-baseline-definition.md) | B0 = static hybrid (K1); dense-only becomes an ablation | Accepted, supersedes P1 §14.1 |
| [ADR-012](ADR-012-event-schema-and-telemetry.md) | One event bus, 16 types, guide-superset turn record | Accepted |
| [ADR-013](ADR-013-corpus-ids-chunking-runtime.md) | Phase 3: `native_or_stem` doc IDs, paragraph 180/300 chunking (provisional), ONNX runtime, batch 8 | Accepted (Phase 3) |
| [ADR-014](ADR-014-streaming-execution-model.md) | Phase 4: virtual/realtime schedulers, async executor, stale-query policy, endpoint timeout 3 s | Accepted (Phase 4) |
| [ADR-015](ADR-015-multi-intent-decomposition-and-fusion.md) | Phase 5: rule-first decomposition (validated, optional gated LLM), versioned IntentSets with delta retrieval, controller as gate, intent-aware fusion | Accepted (Phase 5) |
| [ADR-016](ADR-016-adaptive-session-rag.md) | Phase 6: versioned 4-layer session memory, change detection → delta planning, per-need evidence lifecycle, extractive claims with targeted revalidation, sectioned answer versions, topic frames, redaction at ingestion | Accepted (Phase 6) |
