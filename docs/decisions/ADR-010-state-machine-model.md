# ADR-010: State Machine Model

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §7; refinement **K9**

## Context

The brief proposed a flat list of states (LISTENING, EARLY_RETRIEVAL, DECOMPOSING, RETRIEVING, …). In full duplex these are *simultaneously* true. At 1.6 s in Example 1 the system is listening, decomposing, retrieving and reranking at once. Phase 1 implied a linear pipeline.

## Decision

A **hierarchical statechart**:

| Level | States / regions |
|---|---|
| L0 | Session: NEW, ACTIVE, CLOSING, CLOSED |
| L1 | Turn: IDLE, LISTENING, FINALIZING, SYNTHESIZING, TRANSFORMING, CLARIFYING, VALIDATING, COMMITTED |
| L2 | Concurrent regions inside LISTENING: Segment, Intent, RetrievalJob |

Rules:

- Errors are events with explicit degradation transitions, not a state.
- A new utterance may be LISTENING while the previous turn is SYNTHESIZING.
- **Commits are serialized per session**, so lineage stays linear.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Flat FSM | Loses concurrency, or explodes into a product of states |
| Pure pipeline with callbacks | No explicit place to enforce ordering and invariants |
| Actor-per-component | More concurrency than needed; harder to reason about ordering |

## Why selected

It represents full duplex faithfully, gives every transition a guard and an action that can be tested, and maps cleanly onto asyncio (one session actor plus background tasks).

## Trade-offs

- More concepts to explain. Mitigated by the diagrams in `docs/architecture/06`.

## Consequences

- The Orchestrator implements L1. The Chunk Manager, Decomposer and Retrieval own the L2 regions.
- Race tests (§23 #20) target the concurrency rule.
