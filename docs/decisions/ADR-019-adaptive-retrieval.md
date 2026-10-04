# ADR-019: Adaptive retrieval intelligence (Phase 9)

**Status:** accepted for the dev fixture domain; **off by default** (`adaptive_retrieval.enabled: false`) until it is
validated on the real Theme 4 corpus.

## Context
Phases 3-8 run the same retrieval for every need (hybrid BM25 + dense, RRF, fixed k = 5). Phase 9 asks whether the
system can choose the minimum retrieval that yields a sufficiently grounded answer.

## Decision
1. A per-need controller (`AdaptiveRetrievalController`) analyses the need (explainable complexity classes from need
   features and corpus statistics), rewrites it (contextual form from Phase 6, bounded validated expansions), builds
   **claim requirements** (value / condition / comparison / bridge slots) and routes to one of nine strategies with a
   logged `strategy_reason`.
2. Retrieval is a bounded loop driven by a **lexical sufficiency gate** over the requirements (anchor term, key terms,
   value presence, condition, validity, applicability, conflicts) with explicit stop reasons and a heuristic expected
   gain per action — not by similarity scores.
3. Conflicts are resolved by validity window, *agreed* supersession and configured authority; unresolved conflicts get
   one targeted third-source search, then are handed on for the answer to report.
4. Caches reuse only what is still valid (per-document source signatures, entity / constraint / temporal / staleness
   checks) and log every invalidation with its reason.
5. Corpus text and metadata are untrusted: they can narrow a search (filters the *user* stated, validity) and provide
   a ≤ 6-word hop target; they never set strategy, k, budgets, scheduler behaviour or prompts.
6. Defaults that are design choices were calibrated on a set disjoint from the Phase 9 evaluation questions
   (`research/phase9/calibrate_routing.py`): simple-need fast path = LEXICAL; rerank = never.

## Consequences
* Measured on the dev fixtures (NOT REPORTABLE, see PHASE_9 report): higher ranking quality and evidence precision
  than fixed hybrid, fewer embeddings; median retrieval latency about equal, tail latency higher (hard needs get more
  searches). The latency hypothesis H1 is not supported as stated.
* The gate is lexical: paraphrases are under-recognised (extra searches, INSUFFICIENT states), and capitalisation is
  the multi-hop entity cue. Both are listed as Phase 10 prerequisites.
* A new event family and contracts (`docs/schemas/AdaptiveRetrieval*.schema.json`, `QueryAnalysis`, …).

## Alternatives considered
* LLM-based query planning / sufficiency judgement — rejected for this phase: an extra model call per need on the
  latency-critical path, and corpus text would reach a prompt that decides policy (security requirement).
* Learned routing — no labelled data at the needed scale; the rule-based router is explainable and ablatable.
* Dense-similarity sufficiency thresholds — model-specific, not calibratable on the fixture set without leakage.
