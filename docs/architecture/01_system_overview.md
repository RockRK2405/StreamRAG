# 01: System Overview

This is a design artifact (Phase 2). The normative source is `PHASE_2_SYSTEM_SPECIFICATION.md` §18, §19 and §25.

## Component view

```mermaid
flowchart LR
  subgraph IN["Input (per session)"]
    RP["Replay JSONL / live text"] --> ST["Transcript Streamer"]
  end

  subgraph ACTOR["Session actor (one per session_id, in-memory, TTL)"]
    CM["Chunk Manager<br/>segments, dangling, stability"]
    IA["Intent Analyzer<br/>acts, slots, anchors"]
    RC["Retrieval Controller<br/>WAIT / RETRIEVE / NO_RETRIEVE"]
    QD["Query Decomposer<br/>intents, constraints, context"]
    LG["Query Ledger<br/>dedup / reuse / delta"]
    FU["Evidence Fusion<br/>RRF in-intent, quota RR, conflicts"]
    SY["Answer Synthesizer<br/>hosted / local / extractive"]
    GV["Grounding Validator<br/>sentence gate L0-L2"]
    CI["Citation Manager<br/>Doc_ID §Section"]
    SS["Session State Manager<br/>frames, evidence, claims, versions"]
  end

  subgraph RET["Retrieval (shared, read-only index)"]
    LX["Lexical BM25"]
    DN["Dense bge-small"]
    RR["RRF"]
    RK["Reranker<br/>cross-encoder or RRF+dedup"]
  end

  subgraph OFF["Offline indexing"]
    CORP[("corpus/")] --> IDX["Corpus Indexer"] --> CIX[("CorpusIndex<br/>read-only")]
  end

  LLM[["LLM endpoint<br/>(only permitted egress)"]]
  TM["Telemetry Manager<br/>JSONL, write-only"]
  OUT["Client stream + TURN_COMPLETED"]

  ST --> CM --> IA --> RC --> QD --> LG
  LG -->|miss| LX
  LG -->|miss| DN
  LX --> RR
  DN --> RR
  RR --> RK --> FU
  CIX -.-> LX
  CIX -.-> DN
  CIX -.-> IA
  FU --> SY --> GV --> CI --> SS --> OUT
  SY <--> LLM
  QD -. gated check .-> LLM
  SS -. snapshots .-> RC
  SS -. snapshots .-> IA
  SS -. snapshots .-> QD
  ACTOR -. events .-> TM
  RET -. events .-> TM
```

## Responsibilities at a glance

| Component | One-line responsibility | Spec |
|---|---|---|
| Transcript Streamer | Parse, validate and pace input events | §18.1 |
| Chunk Manager | Idempotent accumulation and segmentation | §18.2 |
| Intent Analyzer | Dialog act, slots, anchors, topic continuity | §18.4 |
| Retrieval Controller | Per-segment WAIT / RETRIEVE / NO_RETRIEVE | §8, §18.3 |
| Query Decomposer | Intents, constraints, context propagation, guards | §9, §18.5 |
| Query Ledger | No duplicate retrieval; reuse; delta | §13, §18.15 |
| Lexical / Dense / RRF / Reranker | Hybrid retrieval per intent | §10 |
| Evidence Fusion | One budgeted, labeled evidence set | §12 |
| Answer Synthesizer | Label-constrained generation; refine ops | §16.3, §14 |
| Grounding Validator | Sentence-gated verification and uncertainty | §16 |
| Citation Manager | Label → evidence → citation key; existence check | §16.7 |
| Session State Manager | Frames, ledger, evidence, claims, versions | §13, §17 |
| Telemetry Manager | Every event to JSONL; never read back | §6, §18.14 |

## Trust boundaries

- **Inside the container:** everything except the LLM endpoint.
- **Egress:** only the configured LLM endpoint, in hosted mode. Extractive and local modes run with networking disabled.
- **Shared across sessions:** only the read-only `CorpusIndex` and pure-function caches.
