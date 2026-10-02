# 06: State Machine

This is a design artifact (Phase 2). Normative: spec §7 (tables, guards, failure behavior).

The machine is a hierarchical statechart:

| Level | Scope |
|---|---|
| L0 | Session |
| L1 | Turn |
| L2 | Concurrent regions inside `LISTENING` |

A flat state list cannot represent full duplex, where listening, retrieving and reranking happen at the same time (spec §7.1).

## L0: Session

```mermaid
stateDiagram-v2
  [*] --> NEW: SESSION_START or first chunk
  NEW --> ACTIVE: state allocated
  ACTIVE --> CLOSING: SESSION_END / TTL / fatal
  CLOSING --> CLOSED: turn finalized or grace timeout
  CLOSED --> [*]
```

## L1: Turn (with L2 concurrent regions inside LISTENING)

```mermaid
stateDiagram-v2
  [*] --> IDLE
  IDLE --> LISTENING: chunk of new utterance

  state LISTENING {
    [*] --> Segments
    state Segments {
      [*] --> OPEN
      OPEN --> STABLE: quiet Δt and sufficient
      STABLE --> OPEN: material change
      OPEN --> CLOSED: boundary / force-close
      STABLE --> CLOSED: boundary
      STABLE --> DISPATCHED: RETRIEVE
      CLOSED --> DISPATCHED: RETRIEVE / ledger hit
      OPEN --> IGNORED: suppressed act
      CLOSED --> IGNORED: suppressed act
      CLOSED --> MERGED: fixed phrase
    }
    --
    [*] --> Intents
    state Intents {
      [*] --> CANDIDATE
      CANDIDATE --> PROVISIONAL
      PROVISIONAL --> COMMITTED
      CANDIDATE --> COMMITTED
      COMMITTED --> RERANKED
    }
    --
    [*] --> Jobs
    state Jobs {
      [*] --> QUEUED
      QUEUED --> RUNNING
      RUNNING --> DONE
      RUNNING --> PARTIAL
      RUNNING --> FAILED
      RUNNING --> CANCELLED
    }
  }

  LISTENING --> FINALIZING: UTTERANCE_END / endpoint timeout / SESSION_END
  FINALIZING --> FINALIZING: previous turn not yet committed
  FINALIZING --> SYNTHESIZING: plan = query / refinement
  FINALIZING --> TRANSFORMING: plan = presentation / meta / social
  FINALIZING --> CLARIFYING: plan = clarification
  FINALIZING --> IDLE: plan = backchannel (no answer)
  SYNTHESIZING --> SYNTHESIZING: LLM failure → next backend
  SYNTHESIZING --> FINALIZING: correction barge-in (if cancel_on_correction)
  SYNTHESIZING --> VALIDATING
  TRANSFORMING --> VALIDATING
  CLARIFYING --> VALIDATING
  VALIDATING --> COMMITTED: ok, or unverified=true on validator failure
  COMMITTED --> FINALIZING: pending constraints → synthetic refinement
  COMMITTED --> IDLE
```

## Concurrency rule

A new utterance may enter `LISTENING` while the previous turn is `SYNTHESIZING` or `VALIDATING`, so early retrieval continues during the answer. Its `FINALIZING` waits for the previous `COMMITTED`. Commits are serialized, so the version lineage stays linear.

## Mapping of the brief's suggested states

| Suggested | Here |
|---|---|
| PARTIAL_UTTERANCE | Segment `OPEN` |
| WAITING_FOR_STABILITY | Segment `OPEN`, sufficient but not `STABLE` (WAIT `awaiting_stability`) |
| EARLY_RETRIEVAL | Intent `PROVISIONAL` + Job `RUNNING` during `LISTENING` |
| DECOMPOSING | Intent-region transitions + gated LLM check in `FINALIZING` |
| RETRIEVING | Job `RUNNING` |
| FUSING | Inside `FINALIZING` |
| ANSWERING | `SYNTHESIZING` (validated deltas streamed) |
| WAITING_FOR_LATE_DETAIL | `IDLE` with an open topic frame |
| REFINING | A plan mode through `FINALIZING → SYNTHESIZING` |
| ERROR | `ERROR` events with explicit degradation transitions (spec §23), not a state |
