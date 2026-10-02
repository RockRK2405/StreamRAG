# ADR-005: Session State

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §13, §24

## Context

- Memory must be ephemeral and session-scoped. Cross-session profiling or persistence is prohibited [G§3 p3], [D s7].
- Refinement needs structured state: a ledger, evidence provenance, claims with citations, topic frames and version lineage (spec §1.8).
- Concurrent sessions in a replay must not contaminate each other.

## Decision

- **In-memory, structured `SessionState`, owned by one asyncio actor per `session_id`.**
  - Only that actor mutates it; other components receive immutable snapshots.
  - Contents: turns, topic frames (slots), intents, query ledger, evidence store, claims, answer versions, pending constraints, counters.
  - Destroyed on `SESSION_END` or idle TTL (30 min). Bounded sizes.
  - Invariants I-1…I-5 checked after every commit.
- **Telemetry is write-only** and is never read back by the pipeline (lint-enforced).
- **Shared objects:** only the read-only CorpusIndex (arrays with `writeable=False`) and pure-function caches.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Plain chat history | Cannot track claims, ledger or lineage (G5) |
| Redis / SQLite | Persistence risk against C4; extra infrastructure |
| Global dicts with locks | Contamination and race risk |
| LLM-summarized memory | Drops citations; ungroundable |

## Why selected

It is the simplest design that supports delta retrieval, affected-claim tracking and auditable lineage, while making isolation structural rather than a matter of discipline.

## Trade-offs

- State is lost on a process crash. That is acceptable, and arguably required, under C4.
- The actor model serializes work within a session. Cross-session parallelism is unaffected.

## Consequences

- An isolation test (Category 11), a filesystem-diff test and a GC test are part of CI.
- Degraded-mode handling for invariant violations is specified in §23.
