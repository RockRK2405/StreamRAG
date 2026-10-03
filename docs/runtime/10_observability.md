# 10: Observability

**Code:** `runtime/telemetry.py` (`TelemetryManager`, `LoopLagMonitor`, `turn_milestones`), `runtime/events.py`, `StreamingRuntime.summary()`, `streaming/printer.py`

## What is recorded (brief §49)

| Signal | Source |
|---|---|
| task latency: queue wait, execution, attempts, status | `TASK_*` events; `scheduler.metrics` |
| worker / retrieval / generation / validation latency | per task type (`summary()["scheduler"]["by_type"]`), `LLM_CALL`, answer stage timings |
| cancellations, timeouts, retries, failures, rejections, sheds | task events + counters (`cancellation_rate`, `retry_rate`, `failure_rate`) |
| backpressure, coalescing | `BACKPRESSURE_APPLIED`, `TRANSCRIPT_COALESCED`, input queue max depth |
| queue depth | sampled per pool on every submit / start; max pending / running |
| event-loop health | loop lag (5 ms probe): max, p95 |
| degraded modes, stale discards, rollbacks | `DEGRADED_MODE_CHANGED`, `STALE_RESULT_DISCARDED`, session summary |

## Turn milestones (brief §51)

Measured from the turn's first processed chunk (wall clock):
- `first_event`;
- `first_evidence`: first `RETRIEVAL_PARTIAL` / `RETRIEVAL_COMPLETED` with hits;
- `first_answer`: the first `ANSWER_COMPLETED`, draft or final;
- `validated_answer`: the first `ANSWER_FINALIZED` with VALIDATED_FINAL. Both pipelines emit this event when they release a validated final, so the Phase 7 baseline is measured identically;
- `utterance_end`, `last_event`.

`wait_after_utterance_end = validated_answer − utterance_end` is the wait a user feels.

## Trace tree (brief §50)

```
SESSION_STARTED -> CHUNK_RECEIVED (utterance) -> INTENT_DETECTED -> QUERY_GENERATED -> TASK_SCHEDULED
   -> RETRIEVAL_STARTED / RETRIEVAL_PARTIAL / RETRIEVAL_COMPLETED -> CLAIM_* ; answer events -> generation task
```

`trace_tree(events)` gives children per event, `path_to_root(events, id)` gives one lineage, and `untraceable(events)` must be empty.

## Console

`streamrag stream --runtime` prints partial results, cancellations, stale discards, backpressure, degraded modes and commits, along with the Phase 4–7 lines.
