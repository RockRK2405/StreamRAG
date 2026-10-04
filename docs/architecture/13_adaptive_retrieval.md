# 13: Adaptive Retrieval (as implemented in Phase 9)

Code: `src/streamrag/adaptive/` (analyzer, rewrite, catalog, requirements, sufficiency, policy, stopping, cache,
controller, integration). Details: `docs/retrieval/01_query_analysis.md` … `11_budget_management.md`, ADR-019.
Off by default (`adaptive_retrieval.enabled`); measurements: `research/phase9/results/` (fixture, NOT REPORTABLE).

## Required architecture (brief §70)

```mermaid
flowchart TD
  Q["USER QUERY<br/>(Phase 5/6 need: contextual query + Intent + constraints)"] --> QA["Query Analyzer<br/>QueryComplexityAnalyzer: signals"]
  QA --> CD["Complexity Detector<br/>SIMPLE / MODERATE / COMPLEX / MULTI_HOP (+ reasons)"]
  CD --> CN["Intent / Claim Needs<br/>RequirementBuilder: value / condition / item slots"]
  CN --> PE["Retrieval Policy Engine<br/>RetrievalPolicySelector: strategy, k, filters, rerank, budget"]
  PE --> C["CACHE<br/>query cache / session evidence / validated claims"]
  PE --> FP["FAST PATH<br/>LEXICAL, k=3, one search"]
  PE --> H["HYBRID / FILTERED / SEMANTIC / MULTI_HOP<br/>Phase 3 stages (BM25 || dense, RRF, dedup)"]
  C --> F1["Evidence Fusion<br/>chunk dedup + RRF across searches"]
  FP --> F1
  H --> F1
  F1 --> ES{"Evidence Sufficiency<br/>SUFFICIENT / PARTIAL / INSUFFICIENT / CONTRADICTORY"}
  ES -- "YES: SUFFICIENT" --> STOP1["STOP<br/>SUFFICIENT_EVIDENCE"]
  ES -- "NO" --> RX["Retrieval Expansion<br/>expected gain per action, budget check"]
  RX --> RW["Rewrite<br/>requirement query, relax filter,<br/>broaden retrievers, larger k"]
  RX --> MH["Multi-Hop<br/>bridge: entity -> target, reference -> document"]
  RX --> RR["Rerank<br/>(policy mode) / contradiction search"]
  RW --> F2["Evidence Fusion"]
  MH --> F2
  RR --> F2
  F2 --> RE["Re-evaluate<br/>actual gain"]
  RE --> STOP2["STOP<br/>sufficient | contradiction | budget | no expected gain | no results | error | cancelled"]
  RE -. "unmet and budget left" .-> RX
```

## Inside one need (controller)

```mermaid
sequenceDiagram
  participant P as Pipeline / runtime task
  participant A as Analyzer + Rewriter
  participant S as Policy selector
  participant K as Caches
  participant R as RetrievalService stages
  participant E as Sufficiency evaluator
  P->>A: AdaptiveRequest(query, intent, constraints, session snapshot, budget, checkpoint)
  A->>S: QueryAnalysis + RewrittenQuery + EvidenceRequirements
  S-->>P: RETRIEVAL_POLICY_SELECTED (strategy_reason)
  S->>K: cache key + validity signature
  alt valid hit
    K-->>P: CACHE_HIT -> evidence, stop
  else session evidence meets all slots
    K-->>P: SESSION_REUSE, stop
  end
  loop until a stop reason
    S->>R: search(es) (lexical / dense / both, k, filters)
    R-->>E: evidence (merged, deduplicated)
    E-->>S: assessment (slots MET / UNMET / CONFLICT), excluded evidence + reasons
    S->>S: stop? else best next action (expected gain), HOP_CREATED / RETRIEVAL_EXPANDED
  end
  S-->>P: final evidence + RetrievalState + decisions + hops + buffered events (RETRIEVAL_STOPPED)
```

## Integration

```mermaid
flowchart LR
  subgraph P6["Phase 6 session (sync pipeline / StreamingSession)"]
    DP["DeltaPlanner<br/>retrieve | reuse_active | cache_hit"]
    ST["EvidenceStore<br/>(usable evidence)"]
    CH["ContextChange<br/>(correction, constraint removal)"]
  end
  subgraph P9["SessionAdaptiveRetriever (one per session)"]
    CTL["AdaptiveRetrievalController<br/>+ query cache"]
    VC["ValidatedClaimCache"]
  end
  subgraph P8["Phase 8 runtime"]
    T["RETRIEVAL task (retrieval pool)<br/>deadline -> latency budget,<br/>token -> checkpoint between searches"]
    SC["StateCoordinator commit<br/>(stale -> discarded)"]
  end
  DP -- "queries_to_create" --> CTL
  ST -- "snapshot: SessionView" --> CTL
  CH -- "invalidate (reason)" --> CTL
  CTL -- "final evidence" --> ST
  CTL -- "bridge terms" --> CS["Phase 6 claim selection"]
  GA["Phase 7 GroundedAnswer"] -- "SUPPORTED claims" --> VC
  T --> CTL
  CTL --> SC
```

| Earlier component | Phase 9 change |
|---|---|
| `RetrievalFilters` / `RetrievalService` (Phase 3) | metadata filters (whitelisted fields) and validity window (`valid_at`, `valid_to`); evidence carries document metadata |
| corpus pipeline | sanitised whitelisted front-matter metadata (`corpus/metadata.py`) |
| `AdaptivePipeline` (Phase 6) | `_execute_adaptive`: one controller run per need; Phase 6 change -> cache invalidation; Phase 7 claims -> claim cache |
| `AdaptiveSessionEngine` (Phase 6) | `retrieval_terms`: hop targets extend claim selection |
| `RuntimeRetrievalExecutor` (Phase 8) | `_submit_adaptive`: one RETRIEVAL task per need (instead of lexical / dense / assemble subtasks) |
| `claims/textcheck.py` (Phase 7) | time values include clock times ("9:00") and weekday / weekend / holiday words |
| event vocabulary | RETRIEVAL_POLICY_SELECTED, QUERY_REWRITTEN, EVIDENCE_ASSESSED, RETRIEVAL_EXPANDED, RETRIEVAL_STOPPED, CACHE_HIT, CACHE_MISS, RETRIEVAL_INVALIDATED, HOP_CREATED (+ per-search RETRIEVAL_STARTED / COMPLETED with `adaptive_search`) |
