# 07: Latency Measurement

**Code:** `streaming/metrics.py`, `streaming/events.py`, `bench/streaming.py`

## Clocks

| Field | Clock | Deterministic? |
|---|---|---|
| `t_session_ms` | Session stream clock: virtual time, or wall × `speed` in realtime | Yes, in virtual mode |
| `t_stream_s` | Utterance-relative stream time (spec §6.1; the guide's `timestamp_s`) | Yes, in virtual mode |
| `t_wall_ms` | Monotonic wall clock since run start | No |
| `payload.wall.*` | Measured costs (`controller_ms`, `measured_ms`, `stages_ms`) | No; excluded from replay comparison |
| `RETRIEVAL_DECISION.payload.t_session_ms`, ledger `created_at_ms` | **Stream time** of the decision: the time its triggering input or timer was scheduled for (`logical_now_ms`). The controller (cooldown, quiet/endpoint timers) runs on this clock. In realtime the envelope `t_session_ms` is when the decision actually happened, typically 1–2 ms later (ADR-014 §1b). | Yes, in both modes |

## Recorded timestamps (structured, never parsed from logs)

| Timestamp | Event |
|---|---|
| `chunk_received_at` | `CHUNK_RECEIVED.t_session_ms` |
| `controller_decision_at` | `RETRIEVAL_DECISION.t_session_ms` (+ `wall.controller_ms`) |
| `retrieval_started_at` | `RETRIEVAL_STARTED.t_session_ms` (+ `queue_wait_ms`) |
| `retrieval_completed_at` | `RETRIEVAL_COMPLETED.t_session_ms` (+ `latency_ms`, `wall.measured_ms`) |
| `utterance_finalized_at` | `UTTERANCE_FINALIZED.t_session_ms` |

## Metrics (per utterance, `utterance_stats`)

| Metric | Definition |
|---|---|
| `retrieval_lead_time` | `utterance_finalized_at − first retrieval_started_at`. Positive means retrieval started before the end. |
| `time_to_first_retrieval` | `first retrieval_started_at − first chunk_received_at` |
| `controller_latency` | `wall.controller_ms` per decision |
| `retrieval_latency` | `retrieval_completed_at − retrieval_started_at` (realtime), plus `wall.measured_ms` (both modes) |
| `post_final_retrieval_latency` | `max(0, final-query completion − utterance_finalized_at)` |
| `evidence_ready_slack` | `utterance_finalized_at − final-query completion`. Negative means waiting for evidence after the user stopped. |
| `retrieved_early` | `lead_time > 0` |
| `duplicate_retrievals`, `stale_completions`, `retrieval_count`, `cancelled` | Counts |

## Benchmark aggregates (`bench/streaming.py`)

| Aggregate | Definition |
|---|---|
| `early_retrieval_rate` | Over **eligible** turns: retrieval required and ≥ 2 chunks (Phase 2 definition; official definition unknown, Q4) |
| `false_retrieval_rate` | Over turns that need no retrieval. `suppression_rate = 1 − false_retrieval_rate`. |
| `missed_retrieval_rate` | Retrieval-required turns with no retrieval at all |
| `useful_early_rate` | The first early retrieval's top-5 overlaps the final query's top-5 |
| Other | Retrievals per utterance, duplicates, `evidence_ready_at_end_rate`, percentiles of lead time / TTFR / post-final latency, controller and retrieval wall latency |

## Caveat

In virtual mode, `retrieval_latency` is the modeled constant (`sim_retrieval_latency_ms`). Only realtime runs at `speed: 1.0` give wall-true stream timings.
