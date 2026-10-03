# 01: Session State

**Code:** `session/models.py` (`SessionState`, `SessionStateVersion`, `SessionSnapshot`, `TranscriptEntry`, `EntityRecord`, `SessionArchive`), `session/memory.py` (`SessionMemory.get_current_state`)
**Schemas:** `docs/schemas/SessionState.schema.json`, `SessionStateVersion.schema.json`, `SessionArchive.schema.json`

A session is not a list of utterances. It is a versioned state that changes only when something retrieval-relevant changes.

## `SessionState` (canonical view)

| Field | Content | Layer |
|---|---|---|
| `session_id`, `current_utterance_id`, `session_version` | identity and the current version number | — |
| `transcript_state` | last `transcript_window` utterances, PII-redacted; older ones keep only a SHA-1 and a length | A transcript |
| `frames`, `active_frame_id` | topic frames (`T<n>`) with status `active` / `dormant` | B semantic |
| `intent_set` | the active needs of the active frame: id, version, resolved text, type, active constraints, query ids | B semantic |
| `constraints` | active session constraints (retracted ones stay in the registry, with `retracted_in`) | B semantic |
| `entities` | entities of the active frame (text, analyzed terms, first / last utterance, needs) | B semantic |
| `query_ledger`, `active_queries`, `superseded_queries` | ledger summary with Phase 6 lineage (`parent_query_id`, `derived_from_change_id`, `semantic_key`, `reused_from`) | C retrieval |
| `evidence_store` | evidence assignments by lifecycle status | C retrieval |
| `claims` | claims by status | D answer |
| `answer_state` | current `AnswerVersion` of the active frame | D answer |
| `telemetry_state` | session counters (retrievals planned, reuses, cache hits, evidence retained / revalidated, claims created / revalidated) | — |
| `configuration` | config hash, index content hash, delta scope, cache flag, full-restart flag | — |

## Versioning

`SessionStateVersion` = `version_id`, `parent_version`, `created_at_ms`, `utterance_id`, `trigger`, `changes` (context-change ids), `summary`, `state_snapshot`.

- **`trigger`** is one of: `interpretation` (context changes were planned), `evidence` (a retrieval result was applied), `answer` (an answer version was committed), `reset`, `restore`.
- **`state_snapshot`** (`SessionSnapshot`) is a compact, deterministic map from ids to statuses: frames, intents (`v<n>:<status>`), constraints, queries (status plus `:stale`), evidence assignments (`evidence@intent`), claims, and the current answer id. Replay and tests compare these snapshots.
- **No version for a no-op.** `update_session` returns `None` when the snapshot equals the previous one, so "Okay." creates no version (tested). Versions form a parent chain `1 ← 2 ← 3 …` (tested).

## Example (dev session S01, generated: `research/phase6/results/demo_trace.md`)

```
v1 interpretation  NEW_INTENT(I1)                 <- provisional "What are the rules"
v2 evidence        evidence for I1 via Q1
v3 interpretation  REFINEMENT(I1)                 <- "... for ladders"
...
v7 answer          answer A1 (initial)
v8 interpretation  CONSTRAINT_ADDITION(I1)        <- "Specifically overnight."
v9 evidence        evidence for I1 via Q4
v10 answer         answer A2 (refinement)
```
