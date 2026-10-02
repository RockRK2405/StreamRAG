# 09: Multi-Intent RAG (as implemented in Phase 5)

The diagrams match `src/streamrag/{intents,multi_retrieval,fusion}` and the Phase 4 session they plug into. Phase 3 retrieval is reused unchanged; Phase 5 ends at the `UnifiedEvidenceSet` (no answer generation).

## Pipeline

```mermaid
flowchart TD
  ST["Streaming transcript<br/>(Phase 4 chunk manager)"] --> RC["Retrieval Controller<br/>(Phase 4 rules) = gate"]
  RC -- "gate closed: suppressed / not stable / not worthy" --> NONE["no intents, no retrieval"]
  RC -- "gate open" --> IA["Intent Analyzer + Decomposer<br/>clauses, roles, guarded splits,<br/>constraints, anaphora, corrections"]
  IA --> VAL["Validation<br/>spans, duplicates, types"]
  VAL --> IS["IntentSet vN<br/>(IntentTracker: ids, versions, delta)"]
  IS --> I1["I1"]
  IS --> I2["I2"]
  IS --> I3["I3"]
  I1 --> Q1["Q1 (IntentQueryBuilder)"]
  I2 --> Q2["Q2"]
  I3 --> Q3["Q3"]
  Q1 --> R1["R1 hybrid search"]
  Q2 --> R2["R2 hybrid search"]
  Q3 --> R3["R3 hybrid search"]
  R1 --> U["Evidence union<br/>(provenance hits: intent, query, method)"]
  R2 --> U
  R3 --> U
  U --> DD["Deduplication<br/>same chunk / near-dup alternates"]
  DD --> RR["Reranker (optional, final only)<br/>intent_ce / cross_intent_dense / cross_intent_ce"]
  RR --> F["Intent-aware fusion<br/>coverage floor + fill, section cap"]
  F --> CF["Conflict check (numeric, conservative)"]
  CF --> UES["Unified EvidenceSet"]
  P3[["Phase 3 RetrievalService<br/>BM25 + dense + RRF + dedup"]] -.-> R1
  P3 -.-> R2
  P3 -.-> R3
  LG[("QueryLedger<br/>intent -> query versions -> evidence")] -.-> Q1
  LG -.-> Q2
  LG -.-> Q3
```

Notes:
- **Retrieval:** R1–R3 are the same Phase 3 service, dispatched in parallel through the Phase 4 executor (`max_concurrent_retrievals` = 3).
- **Rerank order:** the reranker runs before selection so that cross-intent relevance can move a chunk into another intent's list. With `rerank: none` the node is a pass-through.

## Incremental intent addition (delta retrieval)

```mermaid
sequenceDiagram
  participant U as User (stream)
  participant C as Controller (gate)
  participant T as IntentTracker
  participant X as Executor (Phase 3 retrieval)
  participant F as Fusion
  U->>C: "I need information about the fog signal"
  C->>T: gate open
  T-->>X: V1 = {I1} -> Q1 (I1)
  X-->>F: E(I1) -> provisional unified set
  U->>C: "... and also the lens"
  C->>T: gate open
  T-->>X: V2 = {I1, I2}: only Q2 (I2), I1 evidence reused
  X-->>F: E(I1) + E(I2)
  U->>C: "... especially during a storm"
  C->>T: gate open
  T-->>X: V3 = {I1, I2} + K1 global -> Q3 (I1 v2), Q4 (I2 v2) in parallel
  X-->>F: new evidence for both (older versions kept as stale lineage)
  U->>C: UTTERANCE_END
  C->>T: no query changed -> no retrieval
  T->>F: final fusion (+ optional rerank) -> UnifiedEvidenceSet -> TURN_COMPLETED
```

## Versions

```mermaid
flowchart LR
  V1["V1<br/>I1"] -->|"+I2 (added)"| V2["V2<br/>I1 + I2"]
  V2 -->|"+K1 global constraint<br/>(I1, I2 modified)"| V3["V3<br/>I1 + I2 + constraint"]
  V3 -->|"correction: I1 superseded by I3"| V4["V4<br/>I3 + I2<br/>(I1 kept as SUPERSEDED)"]
```
