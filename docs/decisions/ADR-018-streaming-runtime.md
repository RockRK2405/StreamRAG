# ADR-018: Streaming Runtime: Session Actors on One Event Loop, Bounded Worker Pools, Commit-Time Version Checks (Phase 8)

- **Status:** Accepted (Phase 8)
- **Date:** 2026-10-03
- **Refines:** ADR-010 (statechart with concurrent regions), ADR-016 (adaptive session), ADR-017 (grounded generation)

## Context

- **Where Phase 7 left off:** a streaming pipeline whose answer stage ran synchronously at turn end. It blocked the event loop for the whole model call (measured: up to 2.7 s, report §20). It could not cancel in-flight work, had no backpressure, and ran retrieval only as whole queries.
- **The brief asks for** a concurrent, interruptible, latency-aware, fault-tolerant runtime with deterministic replay.
- **The spec already fixed the ownership rule:** REQ-SESS-001, one actor per session; the only shared state is the read-only index and pure caches.

## Decision

1. **One asyncio event loop; one actor per session on it.**
   - All of a session's state mutations (inputs, timers, committed results) run on the loop, one at a time.
   - There are no locks on session state.
   - Every event is emitted on the loop, so a session's log is a total order.
2. **Work runs as tasks on bounded thread pools per resource class** (retrieval / cpu / llm):
   - priority with aging;
   - absolute deadlines propagated from a turn budget;
   - retries of transient failures only, with seeded jitter;
   - idempotency keys.

   Workers get immutable inputs or snapshots; they never touch session state. No process pool (no measured need).
3. **Results are committed, not applied.** The `StateCoordinator` checks the task's epoch and relevance (current query, newest draft, not cancelled) before applying. Stale results are discarded with an event. Multi-step commits snapshot the session and roll back on failure.
4. **Cooperative cancellation through token trees.** Checkpoints sit between retrieval stages, per streamed LLM chunk (closing the HTTP stream) and per verified claim. Slots are held until a worker returns, so timeouts never multiply threads.
5. **Answers through a per-session lane.** Drafts are coalesced to the latest; finals run in order with serialized commits. A correction cancels an in-flight answer (spec §7.3 barge-in). A turn superseded by a refining utterance is not answered. Snapshot isolation; atomic commit of buffered events.
6. **Backpressure.**
   - adaptive batching (one decision per batch), coalescing of replaced intermediate transcript states;
   - bounded input / task / output queues; control inputs are never dropped;
   - a full LLM queue degrades to an extractive answer on the cpu pool.
7. **Degraded modes are explicit events**, with a deterministic fallback chain. Nothing is fabricated: the last resort keeps the previous validated answer.
8. **Hybrid retrieval split into concurrent lexical / dense subtasks**, with partial results. `RetrievalService.retrieve()` composes the same stages, so results are identical.
9. **Replay from the log alone.** Inputs with arrival times, coalesced and rejected inputs, recorded LLM outputs and the fault config are all in the trace. Virtual-clock traces replay exactly; realtime traces replay their timing-independent behaviour.

## Alternatives

| Alternative | Why not chosen |
|---|---|
| A lock per session store (multi-threaded session logic) | many small locks across Phase 4–7 code; ordering of events would depend on lock acquisition; the actor model needs none |
| Generation on the loop, async HTTP | NLI verification and engine bookkeeping are CPU work (tens to hundreds of ms per answer); they would still block the loop |
| Process pool for retrieval / NLI | index and models would be duplicated per process or shared via IPC; the hot paths already release the GIL; no measured bottleneck |
| Kill / abandon timed-out threads | Python cannot kill threads; abandoning them without holding the slot would make concurrency unbounded under repeated timeouts |
| Drop the oldest delta when the queue is full | can drop an appended chunk, i.e. user content; coalescing only drops states that a later delta replaces |
| Let the first finished result win (no version check) | the stale Q1-after-Q2 race overwrote state in the test (reproduced 10 of 10 without protection) |

## Consequences

- **Measured gains** (fixture domain, one machine; report §20–24):
  - the loop is never blocked (max lag 19 ms vs 2.7 s);
  - first evidence and first (draft) answer arrive while the user speaks;
  - cancellation removes most obsolete work;
  - bounded queues keep memory flat under bursts.
- **Barely improved:** the validated-answer latency is dominated by the local model. In the comparison, the median wait after the utterance fell only by about 8 % (145 ms). Early retrieval costs extra retrieval calls (+64 % in the comparison).
- **Bounded by the ownership model:** one loop thread runs all session logic. Per-turn session work (decomposition, delta planning, fusion) is milliseconds at fixture scale, but at higher load it is the next bottleneck (measured as loop lag, report §24).
- **Known gaps** (report §25):
  - no persistence of the event log;
  - no multi-process / multi-host scale-out;
  - "user relevance" priority is not modelled;
  - barge-in depends on Phase 6's change classification.
