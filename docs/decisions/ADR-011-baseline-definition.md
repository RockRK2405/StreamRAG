# ADR-011: Baseline Definition (B0)

- **Status:** Accepted. Supersedes Phase 1 §14.1.
- **Date:** 2026-10-02
- **Spec:** §1.0 correction **K1**, §21, §26

## Context

- Phase 1 defined the comparison baseline B0 as *dense-only* static RAG.
- The guide's roadmap defines the baseline retrieval pipeline as **dense/sparse hybrid** [G§7 p5].
- The deliverables require a quantitative comparison "against the baseline pipeline" and list *hybrid vs dense-only* only as an **ablation** example [G§8 p6].
- This is a contradiction between Phase 1 and the official documentation. The official requirement wins.

## Decision

**B0 = a static, turn-based hybrid pipeline:**

- wait for `UTTERANCE_END`;
- one query = the full normalized utterance;
- BM25 + dense → RRF, top-k = 5;
- one LLM synthesis call with the same label-constrained prompt;
- no controller (always retrieve), no decomposition;
- restart on late detail (concatenate turns, re-run everything).

Dense-only becomes an **ablation arm** (Exp 2).

The improvement ladder:

| Rung | Adds |
|---|---|
| B0 | Hybrid static |
| B1 | + cross-encoder rerank |
| B2 | + decomposition and quota fusion |
| B3 | + streaming controller |
| B4 | + session refinement |
| B5 | + grounding verifier |

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Keep dense-only as B0 | Contradicts the guide's roadmap |
| Two baselines | Muddies the "against the baseline" comparison. Dense-only is reported as an ablation instead. |

## Why selected

It follows the official definition, and makes every rung's gain attributable to the Theme 4 features (streaming, decomposition, refinement), not to adding BM25.

## Trade-offs

- Smaller headline retrieval gains (hybrid is already in B0). This is more honest.

## Consequences

- Phase 1 Exp 1/2 wording is reinterpreted: Exp 2 = hybrid (B0) vs dense-only ablation.
- M3 builds B0 first.
