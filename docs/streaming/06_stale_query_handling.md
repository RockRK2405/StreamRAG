# 06: Stale Queries and the Cancellation Policy

**Code:** `streaming/session.py` (`decide`, `_on_done`), `ledger/ledger.py`

## The situation

1. Chunk 3: "tell me about X" starts retrieval Q1.
2. Before Q1 completes, chunk 4: "specifically under Y" creates Q2.
3. Q1 is now **superseded**, but its evidence may still be useful.

## Policy

| State of the superseded query | Action | Rationale |
|---|---|---|
| **Queued** (not started) | **Cancel** (`cancel_superseded: queued_only`, default). `RETRIEVAL_CANCELLED(reason=superseded_before_start)`; it never runs. | Its work would be fully replaced by Q2. Saves a retrieval slot at zero information loss. |
| **In flight** | **Let it complete.** Mark it `stale` and keep the evidence with lineage (`stale_at_completion = true`). | Retrieval costs ~3 ms back-to-back and ~10 ms in a live stream after idle gaps (Phase 4 report §16, measured on the fixture index), so cancelling saves almost nothing. Python threads cannot be safely interrupted. The evidence often overlaps the newer query's (measured: `useful_early_rate`). |
| **Completed** before supersession | Keep. It becomes stale lineage evidence. | Evidence is never discarded. |

`cancel_superseded: never` disables queued cancellation (ablation).

## Evidence reuse

`QueryLedger.evidence_view(utterance_id)` returns the active query's evidence first, then stale lineage evidence. It is deduplicated, and each item carries `query_ids` and a `stale` flag. `TURN_COMPLETED.evidence` exposes it. Later phases (fusion, refinement) decide how to use stale evidence. Phase 4 only guarantees it is retained and labeled.

## When cancellation would become worthwhile

- When retrieval is expensive: the cross-encoder adds ~95–170 ms per query (Phase 3), and a remote index would add network time.
- Even then, cancellation would only skip *queued* work. A running job's result is still kept.

This is reflected in `sim_retrieval_latency_ms` sensitivity runs (Phase 4 report §15).

## Tests

- Stale completion keeps its evidence with lineage (CASE 7).
- A queued superseded query is cancelled and never reaches the backend.
