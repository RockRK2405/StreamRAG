# 02: Session Memory

**Code:** `session/memory.py` (`SessionMemory`), `session/engine.py` (`AdaptiveSessionEngine` owns the components)

## Four layers, never mixed

| Layer | Object(s) | Holds | Does not hold |
|---|---|---|---|
| **A Transcript** | `SessionMemory.transcript` | what was said: last `transcript_window` utterances verbatim (redacted); older ones as hash + length | interpretation, evidence |
| **B Semantic** | `IntentTracker` (needs, versions, constraint registry), `FrameManager`, entity records | what the user wants: frames, needs, constraints with lifecycle, entities | evidence text |
| **C Retrieval** | `QueryLedger`, `SemanticCache`, `EvidenceStore` | what was searched and found: query lineage, cache entries, evidence records plus per-need assignments with lifecycle | claims, answers |
| **D Answer** | `ClaimGraph`, `AnswerStateManager` | what can be said: claims, claim-evidence links, transitions, answer versions | raw transcript |

`SessionMemory` is the façade: it reads the layers, versions them, snapshots them, and resets or archives them. The engine registers its own state (the interpretation it last planned from, each need's previous query, id counters, change and plan logs) as a memory *extra*, so a snapshot covers the whole session.

## API (brief §3)

| Method | Behaviour |
|---|---|
| `initialize_session()` | empty transcript / entities / versions / counters |
| `update_session(trigger, now, uid, changes, summary)` | new `SessionStateVersion` if the snapshot changed, else `None` |
| `get_current_state()` | `SessionState` (doc 01) |
| `get_intent_context(intent_id=None)` | active needs of the active frame (or one need), each with its constraints and queries |
| `get_entity_context(frame_only=True)` | entities of the active frame's needs |
| `get_relevant_evidence(intent_id)` | usable evidence of a need (ACTIVE / RETAINED) with citations |
| `create_snapshot()` | JSON of every layer + extras (insertion order is part of the state) |
| `restore_snapshot(blob)` | returns every layer to the snapshot; derived indexes (intent terms, claim keys) are rebuilt |
| `reset_session()` | clears all layers and extras; the components keep working for a new conversation |
| `archive_session(now)` | `SessionArchive`: hashes, counts, config / index hashes; no raw text |

Tested:
- **Restore:** the restored state equals the snapshot exactly, and the restored session produces the same change as before.
- **Reset:** every layer is empty after a reset, and a fresh question works afterwards.
- **Counters:** the counters stay shared between memory and engine across reset and restore (a bug found while writing the tests, fixed).

## Retrieval memory: what is kept

- **Evidence:** records (id, document, section, citation, text, first query) are never deleted. Applicability per need is a separate `EvidenceAssignment` with history (doc 05).
- **Query lineage:** each query records `parent_query_id`, the `supersedes` relation, and `derived_from_change_id`; `reused` records carry `reused_from`.
- **Cache:** key = sha1(index hash | options hash | sorted analyzed terms) (doc 04).
