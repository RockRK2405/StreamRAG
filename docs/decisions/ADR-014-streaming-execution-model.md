# ADR-014: Streaming Execution Model, Stale Queries and Timeouts (Phase 4)

- **Status:** Accepted (Phase 4)
- **Date:** 2026-10-02
- **Refines:** ADR-004 (controller), ADR-010 (state machine), ADR-012 (events)

## Context

Phase 4 needed:
- incremental processing of transcript chunks;
- asynchronous retrieval that never freezes the stream;
- handling of queries that become stale while in flight;
- precise lead-time measurement;
- **deterministic replay** for debugging.

Wall-clock asyncio alone is realistic but not reproducible (completion order depends on thread timing). Pure synchronous execution is reproducible but cannot show non-blocking behavior.

## Decision

1. **One session state machine, two schedulers.**
   - `VirtualScheduler` is a deterministic discrete-event simulation (default; tests, benchmarks, replay).
   - `RealtimeScheduler` uses asyncio and the wall clock (latency measurement).

   The session logic is identical in both. Both schedulers order callbacks by (time, priority, sequence): simultaneous events run completion first, then input, then timer, then in scheduling order. The realtime scheduler keeps its own heap and arms a single loop timer at the absolute time of the earliest entry.

   This replaced one `call_later` per callback. Each delay was derived from a fresh `now`, so callbacks sharing a timestamp fired in arbitrary order. SESSION_END overtook the final UTTERANCE_END in 109 of 189 realtime dev-suite runs (found in Phase 4 validation; regression-tested).
1b. **Two clocks.**
   - Controller decisions and their timers (quiet tick, endpoint timeout, cooldown) run on **stream time**: `logical_now_ms()`, the time the triggering input or timer was scheduled for.
   - Event timestamps and latency telemetry use the actual time, `now_ms()`.
   - The cooldown counts from when the previous query was *issued* (the RETRIEVE decision), not from when its retrieval started, which includes execution and queue delay.
   - Consequence: realtime decisions depend only on the input stream, and a realtime trace replays with identical behavior in virtual mode. Before this change, a chunk arriving exactly one cooldown after a retrieval was decided differently in the two modes.
2. **Executors.** A FIFO queue with `max_concurrent_retrievals = 2`.
   - Realtime: `asyncio.to_thread(retrieve)` with a timeout.
   - Virtual: retrieval executes at its start (real evidence), and completion is delivered at `start + sim_retrieval_latency_ms` (10 ms, from Phase 3 measurements).
3. **Stale-query policy.**
   - A superseded *queued* query is cancelled.
   - A superseded *running* query completes, is marked `stale`, and keeps its evidence with lineage.

   Rationale: retrieval costs milliseconds, and threads cannot be safely interrupted.
4. **All non-deterministic measurements live in `payload.wall`** (and `t_wall_ms`). Replay compares everything else exactly.
5. **Endpoint timeout default 3,000 ms** (Phase 2 said 2,000). It is a fallback for a missing `UTTERANCE_END`. At 2,000 ms, inter-chunk gaps of 2.10 s and 2.13 s (5–6-word chunks at 2.6 words/s with jitter, `stack_height-authored`) truncated a dev-suite utterance, and later chunks were rejected.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| Realtime only | Not reproducible; flaky tests |
| Synchronous only | Hides concurrency and stale-query behavior |
| Cancel every superseded query | Loses evidence that is usually still useful, and saves ~5 ms |
| Process pool | Unnecessary at millisecond retrieval cost; adds serialization |

## Consequences

- Benchmarks are reproducible, and a saved virtual trace replays identically.
- A realtime trace replays with identical *behavior*: the same decisions, query versions, finalizations and retrieval outcomes. Its timestamps differ by real jitter.
- Virtual-mode latency is modeled. Realtime runs at speed 1.0 provide true timings, and both are reported.
- A late chunk after a *timeout* finalization is rejected, with an explicit error. Reopening utterances is deferred.
