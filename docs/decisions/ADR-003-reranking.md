# ADR-003: Reranking

- **Status:** Accepted (the cross-encoder is gated by Exp 3)
- **Phase 3 update:** implemented as a pluggable `Reranker` (off by default). Measured rerank cost for 20 candidates: ~95 ms (default threads) and ~167 ms (2 threads) at 10k synthetic chunks. The int8 variant was rejected (slower). Quality gain is still blocked.
- **Date:** 2026-10-02
- **Spec:** §10.2 stage 10, §20.4; correction **K3**

## Context

- The guide requires a re-rank and deduplicate step [G§2 p2]. Its roadmap names "RRF and deduplication reranker" [G§7 p5].
- Phase 1 equated reranking with an optional cross-encoder. Re-reading the guide shows that the **official minimum is RRF + dedup**.
- Phase 2 measurement of the MiniLM-L6 compute shape (`research/phase2`): 105–124 ms per 20 pairs and 150–185 ms per 30 pairs (15 vs 2 threads). Three intents at 20 candidates take ≈0.31–0.37 s serial. That fits in the guide's 0.5 s endpoint gap on the dev machine, with little margin for slower hardware.

## Decision

1. **RRF + deduplication is always on.** It meets the official requirement on its own.
2. `cross-encoder/ms-marco-MiniLM-L-6-v2` is an **optional enhancement**:
   - at most `M_rerank = 20` candidates per intent;
   - applied with **rerank-on-stability** (as soon as an intent's segment is CLOSED, even if still provisional);
   - 600 ms timeout, falling back to `rrf_dedup`;
   - `finalize_wait_ms = 150`, after which unfinished intents use RRF order.
3. The cross-encoder is adopted only if Exp 3 shows a Precision@3 / groundedness gain worth the latency.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Always-on cross-encoder at finalize | Puts ~0.3–1 s on the critical path |
| bge-reranker-base / mxbai | 3–10× the cost [E] |
| LLM reranking | Violates parsimony |
| TinyBERT-L2 | Faster but weaker. Kept as an Exp 3 arm. |

## Why selected

It meets the official requirement deterministically and cheaply, and adds precision only when it is measurably worth it. Rerank-on-stability hides the cost in speech time.

## Trade-offs

- Wasted rerank work when a provisional intent is later dropped (logged).
- MS MARCO domain mismatch (web QA vs policy text) could even hurt. Exp 3 checks this.

## Consequences

- Coverage thresholds come in two variants: cross-encoder score, or dense cosine when the cross-encoder is off (§10.3).
- Judge-hardware sensitivity is tracked as "evidence-ready slack" (§20).
