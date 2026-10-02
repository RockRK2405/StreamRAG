# ADR-012: Event Schema and Telemetry

- **Status:** Accepted
- **Date:** 2026-10-02
- **Spec:** §5, §6

## Context

- G6 requires 100% trace coverage of timestamps, retrieval triggers, citations, answer version lineage and token cost [G§5 p5].
- The guide's output record uses `retrieval_events[{timestamp_s, query, trigger}]`, `sub_queries`, `answer`, `citations`, `uncertainty` [G§4 p4].
- Metrics must be computable from traces alone.
- Reproducibility requires deterministic output.

## Decision

1. **One event bus. Every event is telemetry.** There are 16 output event types with a common envelope `{schema_version, event_id=session:seq, type, session_id, utterance_id, seq, t_stream_s, t_wall_ms, component, payload}`. There is no separate "telemetry event".
2. **Input contract:**
   - four types: `SESSION_START`, `TRANSCRIPT_CHUNK`, `UTTERANCE_END`, `SESSION_END`;
   - utterance-relative `timestamp_s` (guide semantics);
   - `UTTERANCE_END` is a separate timed event; ASR `stability` is a separate field;
   - no gold labels or state injection.
3. **`TURN_COMPLETED`** is a strict superset of the guide's record. Its guide keys are identical.
4. pydantic v2 models are the source of truth, exported to JSON Schema (`docs/schemas/`, built in Phase 3). The telemetry writer is non-blocking JSONL, flushes on shutdown, and is never read by the pipeline.
5. Event IDs are deterministic. There are two clocks: stream time (for G2) and monotonic wall time (for latency).

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Free-form logs | Not schema-checkable |
| OpenTelemetry spans | Heavier; adds a collector; can be layered on later |
| Many fine-grained event types (e.g. CITATION_ADDED) | Duplicates information already in versioned answers |

## Why selected

The smallest vocabulary that covers every G6 field and every metric. It is directly compatible with the guide's example record.

## Trade-offs

- Payloads are rich. `verbosity` and `text_mode` settings control their size and sensitivity.

## Consequences

- M1 (contracts first) precedes all components in Phase 3.
- A G6 validator becomes a CI check.

## Amendment (Phase 4, 2026-10-02)

The event vocabulary was extended for the streaming engine. There is still one canonical vocabulary (`EventType`):

- `CONTROLLER_DECISION` is **renamed** `RETRIEVAL_DECISION` (the name used by the Phase 4 brief). It had no consumers yet.
- **Added:** `SESSION_STARTED`, `TRANSCRIPT_UPDATED`, `QUERY_UPDATED` (new query version), `RETRIEVAL_CANCELLED`, `UTTERANCE_FINALIZED`.
- **Envelope:** added `t_session_ms` (session stream clock), next to `t_stream_s` (utterance-relative) and `t_wall_ms`.
- **Rule:** non-deterministic measurements go under `payload.wall`, so replay can compare everything else exactly.
- Retrieval failures are `RETRIEVAL_COMPLETED(status=error|timeout)` plus a structured `ERROR`, not a separate event type.
