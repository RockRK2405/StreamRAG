# 08: Streaming Engine (as implemented in Phase 4)

This diagram matches `src/streamrag/{streaming,controller,ledger,replay}`. Phase 3 retrieval is reused unchanged.

```mermaid
flowchart TD
  SRC["Transcript source<br/>simulator / JSONL / future ASR"] --> CM["TranscriptChunkManager<br/>validate, dedup, order, gaps, revisions"]
  CM --> TS[("Transcript state<br/>per utterance")]
  TS --> RC["RetrievalController<br/>acts, signals, guards"]
  TM["Timers<br/>quiet tick, endpoint timeout"] --> RC
  RC --> D{"Decision"}
  D -- WAIT --> EV
  D -- SKIP --> EV
  D -- RETRIEVE --> QB["QueryBuilder<br/>filler-free, traceable"]
  QB --> QL["QueryLedger<br/>new version Qn, supersede Qn-1"]
  QL -- "Qn-1 still queued" --> CX["Cancel queued<br/>RETRIEVAL_CANCELLED"]
  QL --> EX["Async executor<br/>FIFO, 2 slots, timeout"]
  EX -. "worker thread" .-> P3["Phase 3 RetrievalService<br/>BM25 + dense, RRF, dedup"]
  P3 -. "EvidenceSet" .-> EX
  EX --> DONE["Completion handler"]
  DONE -- "query superseded meanwhile" --> STALE["Mark stale, keep evidence with lineage"]
  DONE --> QL2["QueryLedger<br/>evidence_ids, status, timestamps"]
  STALE --> QL2
  QL2 --> TURN["TURN_COMPLETED<br/>retrieval_events, final query, evidence view, metrics"]
  CM --> EV["EventBus<br/>TelemetryEvent JSONL, write-only"]
  RC --> EV
  QL --> EV
  DONE --> EV
  TURN --> EV
  EV --> RP["ReplayEngine<br/>trace to inputs, re-run, compare"]
```

## Asynchronous flow

```mermaid
sequenceDiagram
  autonumber
  participant U as Transcript
  participant C as Controller
  participant L as Ledger
  participant R as Retrieval (thread)
  U->>C: chunk 0 "I need" (WAIT: trailing_function_word)
  U->>C: chunk 1 "about the fog signal"
  C->>L: RETRIEVE, create Q1
  L->>R: start Q1
  U->>C: chunk 2 "during a storm" (stream keeps flowing)
  C->>L: RETRIEVE, create Q2 (Q1 stale, refines)
  R-->>L: Q1 completed (stale, evidence kept)
  L->>R: start Q2
  R-->>L: Q2 completed (active)
  U->>C: UTTERANCE_END
  C-->>L: final tick: SKIP redundant (Q2 covers the final transcript)
  L-->>U: TURN_COMPLETED (lead time = end - Q1 start)
```

## Scope boundaries (Phase 4)

- **Single active query** per utterance. No multi-intent decomposition (Phase 5).
- **Output ends at `EvidenceSet`** plus `TURN_COMPLETED(answer = null)`. No answer generation, grounding validator or refinement (Phases 6–7).
- **No UI.** The CLI `streamrag stream` prints a debug stream.
