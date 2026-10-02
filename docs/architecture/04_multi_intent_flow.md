# 04: Multi-Intent Flow

This is a design artifact (Phase 2). Normative: spec §9.

## Rule path (during speech) and gated LLM check (at finalize)

```mermaid
flowchart TD
  S["Closed / stable segments"] --> A["Intent Analyzer: act per segment<br/>INFO_REQUEST, CONSTRAINT, CONTEXT, PRESENTATION, SOCIAL, …"]
  A -->|"PRESENTATION / SOCIAL / BACKCHANNEL / META"| X["No intent (render directive or none)"]
  A -->|"INFO_REQUEST / CONTEXT+need"| C["Coordination split<br/>HEAD (NP1 and NP2 …) → sibling intents"]
  A -->|CONSTRAINT| K["Constraint attachment<br/>nearest intent, or scope=all"]
  C --> D["Dependency detection<br/>anaphora → depends_on"]
  K --> D
  D --> P["Context propagation<br/>shared slots → each intent's queries"]
  P --> NZ["Normalize: lexical_query / dense_query"]
  NZ --> G1{"Over-split?<br/>top-5 Jaccard ≥ 0.6 and cos ≥ τ_merge"}
  G1 -- yes --> M["Merge intents (logged in merged[])"]
  G1 -- no --> G2{"Ledger hit?"}
  M --> G2
  G2 -- yes --> R1["Reuse retrieval"]
  G2 -- no --> R2["Dispatch (parallel)"]

  subgraph FIN["At FINALIZING (≤1 per utterance)"]
    Q{"Gate: G-a … G-e?"} -- yes --> L["LLM structured decomposition"]
    L --> RC["Reconcile with rule intents<br/>match → keep IDs, new → retrieve (final),<br/>weak unmatched → DROPPED"]
    Q -- no --> KEEP["Keep rule intents"]
  end
  R1 --> Q
  R2 --> Q
  RC --> OUT[("Final IntentSet → INTENTS_UPDATED")]
  KEEP --> OUT
```

## Intent lifecycle

```mermaid
stateDiagram-v2
  [*] --> CANDIDATE
  CANDIDATE --> PROVISIONAL: early retrieval
  CANDIDATE --> COMMITTED: confirmed
  PROVISIONAL --> COMMITTED: survives decomposition
  CANDIDATE --> MERGED
  PROVISIONAL --> MERGED
  CANDIDATE --> DROPPED
  PROVISIONAL --> DROPPED: not in final set
  COMMITTED --> RERANKED
  RERANKED --> FUSED: finalize
  FUSED --> ANSWERED: ≥1 supported claim
  FUSED --> UNCERTAIN: not covered / unverifiable
  MERGED --> [*]
  DROPPED --> [*]
  ANSWERED --> [*]
  UNCERTAIN --> [*]
```

## Abstract example

*"I need information about X, and also tell me Y, especially if Z applies."*

| Element | Value |
|---|---|
| I1 | X (CLOSED at ", and also", so provisional retrieval) |
| I2 | Y (multi_intent) |
| K1 | condition Z → scope [I2], confidence 0.6. Ambiguous scope triggers gate G-c, and the LLM check may widen it to [I1, I2]. |
| Dispatch | I1 and I2 in parallel. Z's terms are appended to I2's lexical query. |
