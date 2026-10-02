# 04: Query Ledger

**Code:** `ledger/ledger.py`, `ledger/models.py` (`QueryRecord`, schema in `docs/schemas/QueryRecord.schema.json`)

## Purpose

The session's record of every retrieval query version. Later phases build session refinement on it (Phase 6: delta retrieval, reuse).

## `QueryRecord` fields

| Group | Fields |
|---|---|
| Identity | `query_id` (Q1, Q2, …, session-monotonic), `session_id`, `utterance_id`, `version` (within the utterance) |
| Traceability | `query_text`, `transcript_snapshot`, `source_spans` (char spans of kept tokens), `terms` (analyzed) |
| Creation | `created_at_ms`, `trigger` (provisional/final), `trigger_chunk`, `tick`, `decision_reason` |
| Status | `pending` → `queued` → `in_flight` → `completed` \| `failed` \| `cancelled`; `retrieval_status` |
| Lineage | `supersedes`, `superseded_by`, `relation` (`initial` / `refines` / `replaces`), `lineage_root`, `stale`, `stale_at_completion` |
| Timing | `retrieval_queued_at_ms`, `retrieval_started_at_ms`, `retrieval_completed_at_ms` |
| Results | `evidence_set_id`, `evidence_ids`, `citations`, `error` |

## Versioning rules (Phase 4: single active query per utterance)

- Each `RETRIEVE` creates a new version. The previous active version gets `superseded_by` and `stale = true`.
- `relation = refines` if at least 60% of the old query's terms survive in the new one (Q2 evolves Q1); otherwise `replaces`.
- Records are never deleted. Cancelled records stay, with `status = cancelled`.

```
Q1  status: completed  stale  superseded_by: Q2  evidence: E1,E2,E3
Q2  status: completed  active (refines Q1)       evidence: E1,E4,E5
```

## Queries

| Method | Purpose |
|---|---|
| `active(utterance_id)` | Latest non-cancelled version |
| `most_similar(terms)` | Session-wide term-Jaccard (controller novelty) |
| `in_flight()`, `retrieval_count()`, `last_issued_ms()` | Storm guards (cooldown counts from the stream time the last query was issued) |
| `evidence_view(utterance_id)` | Active-query evidence first, then stale lineage evidence. Deduplicated, and each item lists every query that retrieved it. |

## Telemetry

`QUERY_UPDATED` (new version + supersession), `RETRIEVAL_CANCELLED`, the `ledger` summary in `SESSION_CLOSED`, and `evidence` in `TURN_COMPLETED`.
