# 01: Event Model, Correlation and Causation

**Code:** `runtime/events.py` (`RuntimeEventBus`, `trace_tree`, `untraceable`, `path_to_root`), `models/events.py` (`TelemetryEvent`, `EventType`)

## One envelope for everything

The runtime keeps the Phase 4 `TelemetryEvent`, with its deterministic `event_id = <session>:<seq>`. It adds five fields (brief §5):

| Field | Meaning |
|---|---|
| `correlation_id` | the turn the event belongs to: `<session>/<utterance>` (session-level events: `<session>/session`) |
| `causation_id` | the event whose handling produced this one |
| `parent_event_id` | trace-tree parent by entity lineage: session → utterance → intent → query → task → evidence / claim → answer |
| `state_version` | the session state version at emission (StateCoordinator, doc 09) |
| `output_seq` | contiguous position in the user-visible stream (only on user-visible events) |

`seq` is a total order of a session's state transitions: every emission happens on the event loop (doc 02).

## Causation rules

- **Runtime events name their cause explicitly:**
  - `TASK_SCHEDULED` ← the query event (`QUERY_GENERATED` / `QUERY_UPDATED`) or the answer request;
  - `TASK_STARTED` / `TASK_COMPLETED` ← the task's previous event;
  - `RETRIEVAL_STARTED` ← `TASK_STARTED`;
  - `RETRIEVAL_PARTIAL` / `RETRIEVAL_COMPLETED` ← the subtask's `TASK_COMPLETED`;
  - every event of a committed answer ← the generation task's `TASK_COMPLETED`.
- **Pipeline events (Phase 4–7 code) inherit the cause of the dispatch they run in.** The runtime opens a dispatch for:
  - every input batch: the first event, `CHUNK_RECEIVED`, is a root; the rest are caused by it;
  - every timer: caused by the event that armed it;
  - every committed task result.
- **Result:** `untraceable(events)` (no parent and no cause, or a dangling reference) is empty for every runtime trace (tested; demo trace: 0 of 165).

## Event types

The brief's list (§4) maps onto existing and new types:

| Brief | StreamRAG event |
|---|---|
| TRANSCRIPT_STARTED / DELTA / UPDATED / COMPLETED | `CHUNK_RECEIVED` (first chunk of an utterance) / `CHUNK_RECEIVED` / `TRANSCRIPT_UPDATED` / `UTTERANCE_FINALIZED` |
| INTENT_DETECTED / UPDATED | `INTENT_DETECTED` / `INTENT_UPDATED` (Phase 5) |
| QUERY_CREATED / SUPERSEDED | `QUERY_GENERATED` (`QUERY_UPDATED` single-query mode) / `QUERY_SUPERSEDED` (Phase 6) |
| RETRIEVAL_STARTED / PARTIAL / COMPLETED / FAILED / CANCELLED | `RETRIEVAL_STARTED` / **`RETRIEVAL_PARTIAL`** / `RETRIEVAL_COMPLETED` (`status=error` = failed) / `RETRIEVAL_CANCELLED` |
| EVIDENCE_ADDED / INVALIDATED | `EVIDENCE_FUSED`, `EVIDENCE_RETAINED`, `EVIDENCE_REVALIDATED` / `EVIDENCE_INVALIDATED` (Phase 6) |
| CLAIM_CREATED / UPDATED / INVALIDATED / VERIFIED | `CLAIM_CREATED` / `CLAIM_REVALIDATED` / `CLAIM_INVALIDATED` / `CLAIM_VERIFIED` (Phase 6–7) |
| ANSWER_STARTED / DELTA / REVISED / VALIDATED / COMPLETED | `ANSWER_STARTED` / **`ANSWER_DELTA`** (claim diff per version) / `ANSWER_REVISED` / `ANSWER_VALIDATED` / `ANSWER_COMPLETED`, plus **`ANSWER_COMMITTED`** (a version became current) |
| SESSION_UPDATED | **`SESSION_UPDATED`** (session-end input), **`SESSION_RESET`**, **`SESSION_CANCELLED`** |
| ERROR / TIMEOUT / CANCELLED | `ERROR` / **`TASK_TIMED_OUT`** / **`TASK_CANCELLED`** |

**New in Phase 8:**
- `TASK_SCHEDULED` / `STARTED` / `COMPLETED` / `FAILED` / `CANCELLED` / `TIMED_OUT` / `RETRIED` / `REJECTED`;
- `STALE_RESULT_DISCARDED`, `TRANSCRIPT_COALESCED`, `BACKPRESSURE_APPLIED`, `DEGRADED_MODE_CHANGED`;
- `RUNTIME_STARTED`, `RUNTIME_SHUTDOWN`.

**User-visible events:** session lifecycle, `UTTERANCE_FINALIZED`, the `ANSWER_*` contract, `TURN_COMPLETED`, `DEGRADED_MODE_CHANGED` and `ERROR`. Everything else is telemetry.

## Wall-clock fields

Measured wall times live under `payload.wall` (and `t_wall_ms`). Replay comparison ignores them (doc 11).
