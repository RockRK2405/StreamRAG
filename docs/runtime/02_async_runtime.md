# 02: The Asynchronous Runtime

**Code:** `runtime/runtime.py` (`StreamingRuntime`, `RuntimeSession`, `SessionClock`), `runtime/driver.py`, CLI `stream --runtime`

## From pipeline to runtime

Phase 7 ran the turn as one synchronous chain at utterance end; generation and verification blocked the event loop (Phase 7 report §22.6). In the Phase 8 runtime:

```
push_transcript_delta -> bounded input queue (coalescing) -> session lane (Phase 4-7 session logic, on the loop)
   -> retrieval subtasks / answer tasks -> TaskScheduler -> worker pools (threads)
   -> TaskResult -> StateCoordinator (stale? apply) -> event log -> AnswerStreamer -> subscribers
```

## Actors and ownership (REQ-SESS-001)

**One actor per session.** A `RuntimeSession` owns all of its session's mutable state:
- transcript, intents, ledger, evidence and claim stores, answer state;
- the grounded answer engine.

Its logic is the Phase 4–7 `StreamingSession`, unchanged. Every mutation runs on the event loop, one at a time:
- an input batch;
- a timer;
- a committed task result.

So no lock protects session state.

**Workers never touch session state.** They receive immutable inputs: a retrieval plan, or a snapshot of what the answer engine reads (doc 09). They return a `TaskResult`.

**Shared between sessions, all read-only or internally bounded:**
- the index and retrieval service;
- the NLI model;
- the LLM backend;
- the scheduler and its pools;
- the cancellation, retry and timeout managers.

## API (brief §60)

| Call | Effect |
|---|---|
| `await runtime.start()` | realtime mode: creates the loop clock, worker pools and the loop-lag monitor (`async with` works too) |
| `start_session(id=None)` | new session (bounded by `max_concurrent_sessions`, `RuntimeCapacityError`) |
| `push_transcript_delta(sid, uid, text, stability=, replaces=)` | a delta, or a revision of an earlier chunk (ASR partial). Priority, routing and deadlines are never caller-controlled |
| `end_utterance(sid, uid)`, `end_session(sid)` | endpoint inputs |
| `get_events(sid, user_visible=True)` | async iterator over the ordered stream (bounded subscription, doc 08) |
| `events(sid)` | the full log |
| `cancel_session(sid)`, `reset_session(sid)`, `complete_session(sid)` | doc 05; complete = end and wait until closed |
| `shutdown()`, `force_shutdown()` | doc 12 |

**Virtual mode** (`mode="virtual"`) runs the same code on the deterministic clock:
- inputs take `at_ms`;
- `run()` processes everything;
- task latencies are modelled (`runtime.sim_latency_ms`).

It is used for exact replay and for deterministic orchestration tests.

## Answers off the loop (AnswerLane)

**One answering turn at a time per session** (spec §7.3). Requests are handled by the per-session `AnswerLane`:
- **Drafts** are verified extractive answers, produced after provisional evidence batches. Only the latest is kept.
- **Finals** are produced at turn end, in order, and commits are serialized.
- **Correction barge-in:** a correction detected while a previous final is generated cancels it (`cancel_on_correction`). This happens at the next streamed LLM chunk, the next answer stage or the next claim.
- **Superseded turns:** a turn whose need is now being retrieved again for a later, refining utterance is not answered (`TURN_COMPLETED.answer.status = SUPERSEDED`). Answering would wrongly claim "no evidence"; the later turn answers instead.
- **Turn payload:** `TURN_COMPLETED` carries `answer.status = PENDING` with a request id. The answer follows as `ANSWER_*` events, then `ANSWER_COMMITTED`. The session closes only after its answers committed (`busy_hooks`).

## Fast path / slow path (brief §27–28)

- **Fast path.** The Phase 6 delta planner's semantic cache (`cache_hit`, `reuse_active`) and the scheduler's idempotent-result reuse skip retrieval for a need whose evidence is current. The answer engine keeps validated sentences and sections without a model call when nothing new must be said (tested: a repeated question makes no retrieval subtask; a late constraint makes no LLM call).
- **Slow path.** Multi-intent, conflicting or insufficient evidence takes more tasks (parallel subtasks, revision attempts, validation retrievals) without blocking other sessions or the loop.
