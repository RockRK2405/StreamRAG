# 12: Streaming Runtime (as implemented in Phase 8)

The diagrams match `src/streamrag/runtime/` and its integration with the Phase 4–7 session (`StreamingSession`, `MultiIntentCoordinator`, `GroundedAnswerEngine`). Details: `docs/runtime/01–12`, ADR-018.

## Required architecture (brief §78)

```mermaid
flowchart TD
  U["USER STREAM<br/>push_transcript_delta / end_utterance"] --> IQ["bounded input queue<br/>(coalescing, backpressure)"]
  IQ --> BUS["EVENT BUS<br/>RuntimeEventBus: correlation, causation,<br/>parent, state_version, output_seq"]
  BUS --> SCH["TASK SCHEDULER<br/>priority + aging, deadlines, retries,<br/>idempotency, bounded pools"]
  SCH --> IW["Intent / session lane<br/>(Phase 4-6 logic, on the loop)"]
  SCH --> RW["Retrieval workers<br/>lexical || dense subtasks<br/>(retrieval pool)"]
  SCH --> MW["Memory / state<br/>StateCoordinator<br/>(version + relevance check)"]
  IW --> ES["Evidence store<br/>(StreamingEvidenceAggregator:<br/>RETRIEVAL_PARTIAL -> assemble)"]
  RW --> ES
  MW --> ES
  ES --> CE["Claim engine<br/>(Phase 6 claims, Phase 7 claim plan)"]
  CE --> GW["Generation worker<br/>AnswerLane: drafts + finals<br/>(llm / cpu pool, snapshot isolation)"]
  GW --> VW["Validation<br/>(entailment, citations, coverage)<br/>atomic commit"]
  VW --> AS["Answer stream<br/>AnswerStreamer: ordered release,<br/>bounded subscribers"]
  AS --> OUT["USER<br/>get_events()"]
  X["Cross-cutting: cancellation tokens | deadlines | retries | backpressure |<br/>telemetry + loop-lag | state versioning | replay | fault isolation"]
  SCH -.-> X
```

## One session actor, shared bounded workers

```mermaid
flowchart LR
  subgraph LOOP["asyncio event loop (one thread)"]
    S1["RuntimeSession A<br/>StreamingSession + lane<br/>state v, epoch"]
    S2["RuntimeSession B<br/>StreamingSession + lane"]
    SCHED["TaskScheduler<br/>pending heaps per pool"]
    S1 -- "submit(task, fn)" --> SCHED
    S2 -- "submit(task, fn)" --> SCHED
  end
  subgraph POOLS["bounded thread pools"]
    R["retrieval x4<br/>BM25 / ONNX embed + search"]
    C["cpu x2<br/>assemble, drafts, extractive finals"]
    L["llm x1<br/>generation (HTTP to local model)"]
  end
  SCHED --> R
  SCHED --> C
  SCHED --> L
  R -- "TaskResult (call_soon on loop)" --> SCHED
  C -- "TaskResult" --> SCHED
  L -- "TaskResult" --> SCHED
  SCHED -- "commit if current<br/>else STALE_RESULT_DISCARDED" --> S1
  SCHED -- "commit if current" --> S2
```

## A turn with a late detail (end-to-end demo, research/phase8/results/e2e_demo)

```mermaid
sequenceDiagram
  participant U as User stream
  participant S as Session lane
  participant R as Retrieval workers
  participant A as Answer lane / LLM
  U->>S: "What are the eligibility"
  S->>R: Q1 (lexical || dense)
  U->>S: "requirements" ... "for the permit?"
  S->>R: Q2 supersedes Q1
  S-->>R: cancel Q1 (cooperative) -> RETRIEVAL_CANCELLED superseded_in_flight
  U->>S: utterance end (u1)
  U->>S: "For international" (u2: late detail)
  S->>R: Q3 supersedes Q2 -> Q2 cancelled
  S-->>S: u1 TURN_COMPLETED answer=SUPERSEDED (its need is being re-retrieved)
  U->>S: "applicants." -> Q4 supersedes Q3
  R-->>S: RETRIEVAL_PARTIAL lexical, dense -> assemble -> evidence (commit: current)
  S->>A: final for u2 (snapshot of claims / evidence)
  A-->>S: answer events + ANSWER_COMMITTED (atomic, verified, cited)
```

## Task state machine

```mermaid
stateDiagram-v2
  [*] --> PENDING: submit (bounded queue; else rejected / shed)
  PENDING --> RUNNING: slot free, best priority-aging
  PENDING --> CANCELLED: superseded / reset / shed
  PENDING --> TIMED_OUT: deadline passed before start
  RUNNING --> COMPLETED
  RUNNING --> FAILED: deterministic error / retries exhausted
  RUNNING --> PENDING: transient error -> backoff retry
  RUNNING --> CANCELLED: token cancelled, worker reached a checkpoint
  RUNNING --> TIMED_OUT: deadline (slot held until the worker returns)
  COMPLETED --> [*]: StateCoordinator: applied, or STALE_RESULT_DISCARDED
```

| Phase 4–7 component | Phase 8 change |
|---|---|
| `StreamingSession` (Phase 4) | unchanged logic; batched chunks (one decision per batch), busy hooks, cooperative cancellation hook |
| `VirtualScheduler` / `RealtimeScheduler` | reused as the runtime clock; `SessionClock` adds causal dispatch and epoch guards |
| executors (Phase 4) | replaced in the runtime by `RuntimeRetrievalExecutor` (same interface) on the `TaskScheduler` |
| `RetrievalService` (Phase 3) | split into `plan` / `search_lexical` / `search_dense` / `assemble`; `retrieve()` composes them (identical results) |
| coordinator (Phase 5/6) | `_supersede` helper (cooperative cancellation), answer hand-off to the lane |
| `GroundedAnswerEngine` (Phase 7) | checkpoint / restore, cancellation checkpoints, call context for the LLM |
| `ReplayEngine` (Phase 4) | runtime traces replayed by `replay_runtime` (virtual clock, recorded LLM outputs) |
