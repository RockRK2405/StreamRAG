# Samsung PRISM Theme 4
# Phase 8 — Streaming Runtime

| | |
|---|---|
| Date | 2026-10-03 |
| Status | Phase 8 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** (unchanged). Not committed (commit on request). |
| Code | New package `src/streamrag/runtime/` (17 modules, about 2,900 lines). It covers: events, tasks, scheduler, cancellation, timeouts, retry, backpressure, state, aggregator, answers, streamer, telemetry, faults, sim, replay, driver, runtime. Extensions in Phases 3–7: `RetrievalService` stage methods; `StreamingSession` batching / busy hooks / cooperative cancellation; coordinator supersede helper and answer hand-off; engine checkpoints and cancellation checkpoints; LLM call context; thread-safe text analyzer; `TelemetryEvent` envelope fields; `RuntimeConfig`; CLI `stream --runtime`, `replay` of runtime traces. |
| Tests | **515 passing, 0 skipped**: 452 from Phases 3–7 and 63 new: `tests/runtime` 8, `events` 5, `scheduler` 7, `concurrency` 10, `cancellation` 6, `timeouts` 3, `retries` 6, `backpressure` 6, `replay` 3, `failure_injection` 8, and isolation scan +1 (`runtime`). One Phase 4 isolation assertion was widened for the read-only retrieval stage methods (§8). `compileall` passes for all 154 modules. |
| Workloads | Dev-suite questions (`eval/dev_grounded`, fixture corpora) streamed chunk by chunk. **SYNTHETIC parts, labelled in every result file:** the `SimulatedLLM` (fixed latency, deterministic output), used everywhere except the real-model runs, and injected "remote index" latencies where a benchmark needs retrieval to take time. **REAL:** retrieval (bge-small ONNX, BM25, RRF), NLI verification, and the local `qwen3:4b` in the pipeline comparison, the end-to-end demo and 7 of the 13 final-validation scenarios. |
| Output of the phase | A concurrent, interruptible, bounded, fault-tolerant runtime for the Phase 3–7 pipeline, with an ordered answer stream and replay from the event log. No UI, voice, persistence or multi-host deployment. |

**Read this first: what the numbers mean.**

1. **Synthetic is synthetic.** Load, concurrency, backpressure, cancellation and failure numbers use a simulated model (fixed latency). They measure *orchestration*, not model quality or real generation speed. Only §20's pipeline comparison, the demo (§16) and the real-LLM final-validation scenarios use the actual local model.
2. **One machine** (Apple M5 Pro, 24 GB), a 6-document fixture corpus, at most 16 sessions. **None of this is production scale**, and nothing here is reportable as official performance.
3. **One run per configuration** for the real-LLM comparison; 3–10 repetitions (medians) for the synthetic benchmarks. Run-to-run variation is visible in the per-run values in `research/phase8/results/*.json`.
4. **"NOT MEASURED" is stated where it applies.** Peak RSS is only available as a process-lifetime maximum (models included), so per-configuration memory comes from Python heap peaks (tracemalloc).

**Quality gate (brief §82):**

| Gate | Status | Where |
|---|---|---|
| Event bus; structured events; event correlation | ✅ (correlation / causation / parent / state version; 0 untraceable events in every tested trace) | §4 |
| Task model; scheduler | ✅ | §5, §6 |
| Bounded concurrency; bounded queues; backpressure; transcript coalescing | ✅ | §8, §12, §22 |
| Cancellation; stale-result protection; state version checks | ✅ (race reproduced 10/10, 0 overwrites) | §9, §13, §21 |
| Timeout propagation; retry manager; idempotency | ✅ | §10, §11 |
| Failure isolation; degraded mode | ✅ (9 injected fault types, every turn answered or honestly "not established") | §17, §23 |
| Partial results; answer streaming; output ordering | ✅ | §15, §16 |
| Replay; observability | ✅ (virtual traces exact; realtime traces behaviour-identical) | §18, §19 |
| Race-condition, failure-injection, resource-leak, session-isolation tests pass; graceful shutdown | ✅ | §13, §17, §24 |
| Load tests (where practical) | ✅ synthetic, 1–16 sessions | §24 |
| Documentation; actual benchmarks; no fabricated results | ✅ `docs/runtime/01–12`, `docs/architecture/12`, ADR-018; every number in `research/phase8/results/` | — |

---

## 1. Objective

Turn the Phase 3–7 components into a **streaming runtime**. It should process concurrently, cancel what became obsolete, protect state from late results, bound every resource, degrade visibly instead of failing silently, and remain replayable. The aim is to stream events, schedule work, run independent work in parallel, emit partial results and refine the answer, instead of "receive the full input, process sequentially, answer".

## 2. Previous Architecture

**Phase 7 already streamed input.** It used the Phase 4 dual-mode schedulers (virtual and realtime), a bounded FIFO retrieval executor and an event bus with deterministic ids and replay. Its limits for a live system:

| Limit (Phase 7) | Measured / observed |
|---|---|
| answer generation + verification ran **on the event loop** at turn end | loop blocked up to **2,670 ms** per answer with the real model (§20) |
| cancellation only of *queued* retrievals; running work completed and was marked stale | no way to stop obsolete LLM work |
| hybrid retrieval as one opaque call | no partial results; a dense failure was handled inside the call |
| no input bound, no coalescing | every ASR revision triggered full processing |
| no deadlines beyond per-call timeouts | no turn budget |
| shared Snowball stemmer used from retrieval threads | **not thread-safe** (found in Phase 8: 25 exceptions and 123 wrong stems in 300 calls on 8 threads); this affected Phase 4/5 realtime parallel retrieval |

**Reused, not duplicated:**
- the schedulers (as the runtime clock);
- the executor interface the session logic already calls;
- the event envelope and bus (extended);
- the ledger's stale marking;
- the Phase 6 snapshot / restore;
- the Phase 4 replay;
- the Phase 7 engine (with checkpoints added).

## 3. Runtime Architecture

| Brief component | Implementation |
|---|---|
| StreamingRuntime | `runtime/runtime.py`: API, sessions, clock, shutdown |
| EventBus | `RuntimeEventBus` (extends the Phase 4 bus) |
| TaskScheduler | `TaskScheduler`: priority + aging, deadlines, retries, idempotency, shedding |
| WorkQueue | bounded pending queue per pool; bounded per-session `InputQueue` |
| WorkerPool | `ThreadRunner`: one bounded thread pool per resource class (retrieval 4 / cpu 2 / llm 1); `VirtualRunner` for the virtual clock |
| CancellationManager | token tree session → task (`runtime/cancellation.py`) |
| TimeoutManager | per-type timeouts, turn budget, deadline propagation |
| RetryManager | transient-only retries, bounded exponential backoff, seeded jitter |
| BackpressureController | input batching / coalescing / rejection, pool shedding, bounded subscribers (`backpressure.py`, `streamer.py`, scheduler) |
| StateCoordinator | per-session version / epoch, relevance check, atomic commit with rollback |
| ResultAggregator | `StreamingEvidenceAggregator` (partial retrieval results → assembly) |
| AnswerStreamer | ordered release buffer, `ANSWER_DELTA`, bounded subscriptions; `AnswerLane` produces the answers |
| TelemetryManager | task metrics, queue depths, loop-lag monitor, turn milestones, trace tree |

**Ownership model** (REQ-SESS-001, ADR-018):
- One asyncio loop; one actor per session.
- Every mutation of a session's state (inputs, timers, committed task results) runs on the loop, one at a time, so there are no locks.
- Workers get immutable inputs or snapshots and return results.

Diagrams (Mermaid, validated): `docs/architecture/12_streaming_runtime.md`, covering the brief's §78 architecture, actors and pools, the late-detail sequence, and the task state machine.

## 4. Event Model

- **Envelope** (doc runtime/01): `event_id`, `type`, `session_id`, `utterance_id`, `state_version`, `t_session_ms`, `t_wall_ms`, `parent_event_id`, `correlation_id` (turn), `causation_id`, `payload`, and `output_seq` for user-visible events.
- **Causation chain:** runtime events name their cause; for example, QUERY_GENERATED → TASK_SCHEDULED → TASK_STARTED → RETRIEVAL_STARTED → RETRIEVAL_PARTIAL / COMPLETED. Answer events are caused by their generation task's TASK_COMPLETED. Pipeline events inherit the cause of their dispatch (input batch, timer, committed result).
- **Trace tree** by entity lineage: session → utterance → intent → query → task → evidence / claim → answer.
- **No untraceable events:** `untraceable()` is empty in every tested trace (end-to-end demo: 0 of 165 events).
- **Event types:** the brief's list maps onto existing and new types (doc runtime/01 table). New: `TASK_*` (8), `RETRIEVAL_PARTIAL`, `STALE_RESULT_DISCARDED`, `TRANSCRIPT_COALESCED`, `BACKPRESSURE_APPLIED`, `DEGRADED_MODE_CHANGED`, `SESSION_UPDATED`, `SESSION_RESET`, `SESSION_CANCELLED`, `RUNTIME_STARTED`, `RUNTIME_SHUTDOWN`.

## 5. Task Model

**`Task`:**
- `task_id`, `task_type`, `session_id`, `priority`, `status`;
- `created_at` / `started_at` / `completed_at`, `deadline`;
- `parent_task_id`, `correlation_id`, `causation_id`;
- `state_version`, `epoch`, `idempotency_key`, `attempt`.

**Statuses:** PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMED_OUT, SUPERSEDED.

**Types:** lexical, dense, retrieval (unsplit), assemble, generation, answer_extractive, draft, validation_retrieval, analytics.

**`TaskResult`:** task_id, state_version, epoch, correlation_id, status, result, error, metadata (queue wait, execution time, attempts). It is applied only through the StateCoordinator (§14).

## 6. Scheduler

- **Pools and slots:** separate pools (retrieval / cpu / llm) with bounded slots and bounded pending queues; a slow model cannot starve retrieval.
- **Selection:** the minimum of `priority − waited / aging_ms`. Priorities: CRITICAL 0 (final answer), HIGH 1 (retrieval for a finished utterance or a need without evidence; validation retrieval), MEDIUM 2 (early retrieval for a need with evidence; drafts), LOW 3. Aging (500 ms per level) prevents starvation (tested).
- **Query priority** uses only what is known (finished utterance, evidence present). "User relevance / answer importance" have no measurable basis yet and are not used (§25).
- **Shedding:** a full queue sheds the least important pending task, or rejects the new one (`TASK_REJECTED`). A rejected generation is answered extractively on the cpu pool (§24), never dropped.
- **Deadlines:** a pending task past its deadline never starts; a running one is reported TIMED_OUT and keeps its slot until the worker returns, so timeouts never multiply threads.
- **Failure isolation:** a failing result handler is reported as an `ERROR` and the pool keeps going (tested). This was added after a handler exception stalled a pool and hung a session in development (§25 history).

## 7. Async Execution

**Inputs.** `push_transcript_delta()` validates input (length and count limits; no caller-controlled priority, routing or deadline) and enqueues it. The session lane drains the queue in one step (§12).

**Session logic.** The lane runs the unchanged Phase 4–7 `StreamingSession` (controller, decomposition, delta planning, fusion, Phase 6 state) on the loop. Its retrievals go to the scheduler through the `RuntimeRetrievalExecutor` (same interface as the Phase 4 executor).

**Answers.** They go through the per-session **AnswerLane**:
- **Snapshot isolation:** the worker gets a snapshot of exactly what the engine reads (evidence store, claims, intents and constraints, conflicts).
- **Buffered output:** its events are buffered.
- **Commit:** events and fallback evidence are committed in one step on the loop, or discarded with an engine rollback (§14).
- **Drafts** (verified, extractive) are coalesced to the latest. **Finals** run in request order.

**Event-loop health.** No blocking call runs on the loop:
- a loop-lag probe is part of the runtime;
- tests assert max < 300 ms while each answer takes the simulated model 600 ms, and a negative control proves a 250 ms block is detected;
- with the real model the runtime's max lag was **16.7–18.8 ms**, against **2,669.6 ms** for the Phase 7 pipeline (§20).

## 8. Concurrency Model

- **Hybrid queries** run as two concurrent subtasks (lexical BM25 and dense embedding + search), plus an assembly task. Independent intents' subtasks run in parallel up to the pool size. The assembled evidence is **identical** to Phase 3's `retrieve()` (test `test_split_retrieval_stages_reproduce_the_phase3_result`), because `retrieve()` now composes the same stage methods (`plan`, `search_lexical`, `search_dense`, `assemble`).
- **Bounded threads.** Threads are used for blocking libraries (ONNX Runtime, numpy, the HTTP model call); processes are not used (no measured need, doc runtime/04). Bounds: per-pool slots, `max_concurrent_sessions`, all queues.
- **Thread-safety fixes:**
  - one Snowball stemmer per thread (the shared instance corrupted results under concurrency);
  - dense embeddings run inline in the worker, not through the service's 2-thread timeout executor, which would serialize concurrent embeddings and time them out.

## 9. Cancellation

**Cooperative.** Every task has a token in a session → task tree; workers stop at checkpoints:
- retrieval: between embedding and vector search, and in interruptible waits;
- generation: per streamed LLM chunk (closing the HTTP stream);
- answering: between answer stages, per verified claim.

**Triggers:**
- query superseded (refinement, constraint, delta plan): `cancel_superseded: cooperative`;
- need superseded or removed;
- correction / entity change / question change detected (barge-in on the previous turn's in-flight answer, at detection time);
- a final superseding a running draft;
- session reset / cancel; shutdown; deadline.

**Superseded turns** are not answered. A turn whose need's query was cancelled because a later utterance refined it gets `TURN_COMPLETED.answer.status = SUPERSEDED`. Before this rule, the end-to-end demo committed a false validated "the documents do not contain an answer" for it (§25).

**Cancel latency.** A running draft was first cancelled about 430 ms after the request (stage-level checkpoints only). Per-claim checkpoints brought it to about 100 ms (development measurement, realtime smoke run).

## 10. Timeout Management

- **Per-type timeouts** (`runtime.timeouts_ms`; configuration, not targets).
- **Turn budget** (`budget.turn_ms` = 30 s) with downstream reserves: retrieval deadline = turn deadline − (generation reserve 4.2 s + validation reserve 0.4 s). The reserves are Phase 7's measured p95 stage times.
- **The model call gets the remaining time** (call-context deadline → HTTP timeout).
- **On expiry:**
  - pending → never runs;
  - running → TIMED_OUT, with the slot held until return;
  - dense timeout → lexical-only (§17);
  - generation timeout → extractive answer.

  All tested (`tests/timeouts`, scenario 07 in §23).

## 11. Retry Strategy

- **Transient only:** timeouts, connection / network errors, temporary service errors, rate limits (exceptions or LLM error text). Deterministic errors are never retried (tested: 1 attempt).
- **Backoff:** bounded exponential, 50 ms × 2, cap 1 s, ±20 % jitter from a seeded RNG, at most 2 retries, and only while the deadline allows.
- **Where:**
  - retrieval subtasks are retried by the scheduler (same task, `TASK_RETRIED`);
  - LLM calls by `RetryingLLM` inside the generation task (only the final response is recorded, so replay stays exact).
- **Idempotency:** a retried retrieval emits its partial result once and adds its evidence once (tested). Identical retrievals in one session are joined or reused; never across sessions or epochs.
- **Measured** (§23): a transient network failure ×2 → 2 retries, then ok; a transient LLM failure → 1 retry, full answer.

## 12. Backpressure

**Strategy, in order:**
1. Adaptive batching: one controller decision per drained batch.
2. Coalescing: an intermediate transcript state replaced by a later pending delta is dropped; never content, never utterance or session ends.
3. Bounded queues everywhere:
   - input (64 deltas; when full, coalesce, else reject with an event; control inputs are never rejected);
   - task pools (shed or reject; rejected generations go extractive);
   - subscribers (a slow consumer loses DRAFT events first, then is disconnected; the producer never blocks).

**Measured:** §22.

## 13. Race Condition Prevention

**Version check at commit.**
- Every task records the session's `state_version` and `epoch`.
- At commit the `StateCoordinator` checks the epoch (reset → `epoch_changed`) and the relevance of the result:
  - retrieval: the query is still the current version of its need;
  - draft: it is still the newest request and the utterance is not finalized;
  - final: it was not cancelled.
- A stale result is discarded with `STALE_RESULT_DISCARDED` (its version vs the current one). A superseded query's completion is still recorded in the ledger (bookkeeping); fusion and claims ignore it.

**Mandatory race (brief §55).** Q1 starts; Q2 starts later and completes first; Q1 completes last.

| Mode | Runs | Race reproduced | Stale discards | Overwrites | Final answer about the correction |
|---|---|---|---|---|---|
| no running-work cancellation | 10 | 10 | 10 | **0** | 10 / 10 |
| cancellation (default) | 10 | 0 (Q1 cancelled in flight, 10 / 10) | 0 | **0** | 10 / 10 |

**Tests:** stale result (virtual, deterministic); simultaneous evidence updates (arrival order changes nothing in the answer); racing answer updates (versions committed in order, no draft after the final); cancellation during retrieval and during generation; session reset during active tasks (`tests/runtime`, `tests/cancellation`).

## 14. State Coordination

- **Fine-grained, not global.** Serialisation is per session (the actor on the loop); there is no global lock. Shared components are read-only or bounded.
- **Atomic commits:**
  - an answer's buffered events and its fallback-retrieval evidence are applied in one step;
  - the session is snapshotted first (Phase 6 `create_snapshot`) and restored on failure (`ERROR {action: rolled_back_to_last_valid_state}`, unit-tested);
  - the answer engine is checkpointed and restored for every cancelled or stale answer.
- **Reset** bumps the epoch; results computed before it are discarded.

## 15. Partial Results

- **`RETRIEVAL_PARTIAL`** when the lexical or dense part of a query arrives (hits, top citations). The aggregator assembles once both are in, or degrades if one failed.
- **Early generation:** drafts (verified extractive answers) are produced after each provisional evidence batch while the user is still speaking.
- **Not done:** per-section LLM finals (brief §31–32). The engine generates one answer version per request. Splitting finals per section gives no gain with a local server that runs one request at a time (§25).

## 16. Streaming Answer Output

- **Event contract** per answer version (Phase 7): ANSWER_STARTED → SECTION_STARTED → CLAIM_READY / CITATION_READY → SECTION_COMPLETED → VALIDATED → COMPLETED.
- **New per version:** `ANSWER_DELTA` (added / removed / modified claims vs the previous version) and `ANSWER_COMMITTED` (the version became current).
- **Ordering:** user-visible events carry a contiguous `output_seq`; commits are serialized per session.
- **`OrderedReleaseBuffer`** (N+1 waits for N, bounded hold of 2 s, gaps reported) sits in the output path as a safety net. It held **0** events out of order in every measured load run (§24) and in the live-stream test, so the structural order held.

**End-to-end demo** (brief §75; real retrieval / NLI / `qwen3:4b`; dense index +1,200 ms injected so supersession happens mid-flight; `research/phase8/results/e2e_demo/README.md`, generated from the event log):

1. `What are the eligibility` → early query Q1 at 30 ms.
2. `requirements … for the permit?` → Q2 supersedes Q1 → **Q1 cancelled in flight**.
3. Utterance end. The user adds `For international` → Q3 supersedes Q2 (cancelled); u1's turn is **SUPERSEDED**, not answered "no evidence".
4. `applicants.` → Q4 supersedes Q3 (cancelled); lexical / dense partials → assembly → evidence revalidated.
5. Final generated off the loop → **validated answer committed** about 6 s after the first word:
   - eligibility facts and the presented proof-of-residence conflict, all cited;
   - "Not established: The retrieved documents do not say how this applies for international applicants."

**Demo trace quality:** 165 events, 0 untraceable. A virtual-clock replay with the recorded LLM output is **behaviour-identical** (not byte-identical: realtime timing).

## 17. Failure Recovery

**Isolation.** Worker failures are task results; result-handler failures are reported; one session's faults leave another's answers intact (tested).

**Degraded modes** (`DEGRADED_MODE_CHANGED`, never silent):
- RETRIEVAL_DEGRADED: lexical-only / dense-only / failed retrieval, which keeps earlier evidence and reports insufficiency;
- GENERATION_DEGRADED: extractive final on the cpu pool, still verified and cited;
- VALIDATION_DEGRADED: rules-only verification;
- last resort: keep the previous validated answer and report the error.

**Deterministic LLM fallback chain:** transient retry → extractive answer from the same plan → previous validated answer. Nothing is fabricated.

## 18. Observability

**Recorded:**
- task latency (queue wait, execution, attempts);
- per-type worker latency;
- `LLM_CALL` (first token / total / tokens);
- cancellations, timeouts, retries, failures, rejections, sheds;
- queue depth per pool (sampled);
- backpressure and coalescing;
- event-loop lag;
- stale discards and rollbacks;
- degraded modes;
- per-turn milestones (first event / evidence / answer, validated answer, utterance end, wait after end);
- the trace tree.

**Access:** `runtime.summary()` and `docs/runtime/10`.

## 19. Replay

**Inputs from the log alone.** A runtime trace records every input with its arrival time:
- chunk offsets;
- coalesced and rejected inputs with payloads;
- utterance-relative ends;
- `SESSION_UPDATED` for the session end;
- the model outputs (`LLM_CALL`) and the fault configuration.

**`replay_runtime`** re-runs them on the virtual clock with a recorded LLM backend.

**Results:**
- **Virtual-mode traces replay exactly** (tested with coalescing, retries and injected faults: every event identical).
- **Realtime traces** replay their orchestration; `behaviour` compares the per-turn transcript, queries and statuses, and the validated answer claims. Exact equality is impossible by design: coalescing batches, cancellation arrival and deadline hits depend on wall timing.
- **LLM nondeterminism** does not matter for replay (outputs come from the log).
- **`reconstruct(events)`** rebuilds queries, evidence, claims and answers from events only (tested against the live state).
- **CLI:** `streamrag replay` detects runtime traces.

## 20. Concurrency Benchmark

### A / B / C (brief §52)

**Setup** (`concurrency.json`, SYNTHETIC: `SimulatedLLM` 800 ms; +40 ms injected per retrieval subtask as a remote index):
- 8 concurrent sessions × 2 multi-intent utterances, 3 words per 150 ms;
- barge-in off (it is measured in §21);
- 3 runs each; medians.

| Config | Slots (retrieval / cpu / llm) | Wall | Throughput | Validated answer p50 / p95 | First evidence p50 | Peak threads | CPU | Loop lag max (p95) | Unanswered |
|---|---|---|---|---|---|---|---|---|---|
| A sequential | 1 / 1 / 1, unsplit | 14.29 s | 1.12 turns/s | 7,293 / 12,527 ms | 831 ms | 5 | 3.57 s | 19.6 (0.8) ms | 0 % |
| B parallel, unbounded | 64 / 64 / 64 | 2.91 s | 5.50 turns/s | 1,692 / 1,823 ms | 526 ms | 54 | 2.14 s | **104.0 (10.0) ms** | 0 % |
| C parallel, bounded (default-like) | 4 / 2 / 4 | 4.20 s | 3.81 turns/s | 2,345 / 3,199 ms | 614 ms | 12 | 2.88 s | 59.3 (6.7) ms | 0 % |

**Reading:**
- **Parallelism is the big win:** A → C is 3.4× throughput and 3.1× lower median latency.
- **Unbounded is faster but costs:** 4.5× the threads, and it starved the event loop (max lag 104 ms). Its LLM parallelism (8 calls at once) exists only for the simulated model: the local Ollama server runs one request at a time.
- **Failure rate:** 0 in all runs.
- **Memory per config: NOT MEASURED.** Process RSS is a lifetime maximum (about 1.2 GB, models included) and does not separate the configs.

### Pipeline comparison with the real model (brief §76)

**Setup** (`comparison.json`): local `qwen3:4b`; 10 Phase 7 end-to-end scenarios; speech-paced (400 ms per chunk, 2.5 s between utterances); real retrieval and verification; one run each; 14 turns per pipeline.

- **BASELINE:** the Phase 7 pipeline (retrieval after the utterance, sequential, generation on the loop).
- **ASYNC:** the runtime without cancellation, drafts or cache.
- **FULL:** all runtime defaults.

| p50 (p95), ms | BASELINE | ASYNC | FULL |
|---|---|---|---|
| time to first evidence | 911 (1,311) | 206 (582) | **7** (608) |
| time to first answer (draft or final) | 2,798 (4,655) | 2,658 (4,018) | **239** (1,254) |
| time to validated answer | 2,798 (4,655) | 2,658 (4,018) | 2,656 (3,990) |
| wait after the utterance ended | 1,900 (3,616) | 1,758 (2,977) | 1,755 (2,949) |
| total latency (last event of the turn) | 2,798 (4,655) | 2,658 (4,018) | 2,656 (3,990) |
| retrieval calls (all turns) | 14 | 23 | 23 |
| LLM calls / validated answers | 11 / 14 | 11 / 14 | 11 / 14 |
| drafts shown before the final | 0 | 0 | 21 |
| queries reused (fast path) | 1 | 0 | 1 |
| cancelled work | 0 | 0 | 0 |
| max event-loop lag | **2,670** | 16.7 | 18.8 |

**Honest reading:**
- **The validated answer is dominated by the local model.** The runtime cuts the median wait after the utterance by about 145 ms (−8 %) and p95 by about 670 ms; that is all it can do for the final.
- **What changes for the user:**
  - evidence appears almost immediately, because retrieval starts mid-sentence;
  - a verified draft appears at a median of 239 ms instead of 2.8 s;
  - the loop is never blocked.
- **The cost:** early retrieval issues **64 % more retrieval calls**.
- **No cancellation occurred in these scenarios.** Fixture retrieval takes about 10 ms, so nothing was running when superseded. Cancellation savings need slow work (§21).

## 21. Cancellation Benchmark

**Setup** (`cancellation.json`, SYNTHETIC: `SimulatedLLM` 1,500 ms; dense index +350 ms): "What are the rules for ladders in the orchard?", then "Sorry, I meant crates instead of ladders." while u1's retrieval and answer run. 5 runs; medians.

| | Wasted work | of which retrieval / generation | Total task execution | Corrected turn: validated answer | LLM calls | Answers committed |
|---|---|---|---|---|---|---|
| no cancellation | 2,244 ms | 722 / 1,672 ms | 4,133 ms | 3,164 ms | 2 | u1 and u2 |
| cancellation | 737 ms | 608 / 130 ms | 2,623 ms | 2,148 ms | 1 | u2 only |

**Result:**
- cancellation saved **1,507 ms of work (−67 % wasted)** and delivered the corrected answer **1,016 ms sooner**;
- the obsolete answer for the ladders question was never shown.

**Definition of wasted work:** execution of tasks that were cancelled, whose query was superseded or stale, or that answered the corrected turn. What remains wasted is work done before the correction arrived.

## 22. Backpressure Benchmark

**Setup** (`backpressure.json`): 2,000 ASR partial hypotheses of one utterance, then the final, into one session (SYNTHETIC `SimulatedLLM` 300 ms). Two input patterns:
- *paced*: the loop runs every 20 deltas, about 1 kHz;
- *instant burst*: the loop cannot run until all deltas are queued.

The unbounded baseline is a finite burst, so no uncontrolled growth was ever allowed to run.

| | Max input queue | Python heap peak | Chunk events processed | Coalesced | Events logged | Drain after last delta | Final transcript intact |
|---|---|---|---|---|---|---|---|
| paced, unbounded / no coalescing | 21 | 20.86 MB | 2,001 | 0 | 5,390 | 447 ms | yes |
| paced, bounded + coalescing | 21 | 13.51 MB | 101 | 1,900 | 1,671 | 426 ms | yes |
| burst, unbounded / no coalescing | **2,001** | 13.45 MB | 2,001 | 0 | 4,118 | 603 ms | yes |
| burst, bounded + coalescing | **65** | **0.95 MB** | 1 | 2,000 | 150 | 453 ms | yes |

**Results:**
- the bounded queue stays at its capacity (64 deltas + the session start);
- coalescing removed 95–100 % of the processing;
- heap peak −35 % (paced) and −93 % (burst);
- 0 deltas rejected; the final transcript was always the last state.

## 23. Failure Injection Results

### Injected faults (brief §56)

**Setup** (`failures.json`, SYNTHETIC `SimulatedLLM` 200 ms): "How high should the wicks be trimmed?"; generation timeout 1 s, dense timeout 0.8 s for the run.

| Fault | Degraded mode | Retrieval | Answer |
|---|---|---|---|
| retrieval timeout (dense never returns) | RETRIEVAL_DEGRADED | degraded (lexical) | VALIDATED_FINAL, correct (4 mm) |
| vector DB failure | RETRIEVAL_DEGRADED | degraded (lexical) | VALIDATED_FINAL, correct |
| lexical failure | RETRIEVAL_DEGRADED | degraded (dense) | VALIDATED_FINAL, correct |
| network failure, transient ×2 | none (2 retries) | ok | VALIDATED_FINAL, correct |
| network failure, persistent | RETRIEVAL_DEGRADED | error | VALIDATED_FINAL: "Not established …" (honest insufficiency) |
| LLM timeout | GENERATION_DEGRADED | ok | VALIDATED_FINAL, extractive, correct |
| LLM error | GENERATION_DEGRADED | ok | VALIDATED_FINAL (engine extractive fallback), correct |
| LLM transient ×1 | none (1 retry) | ok | VALIDATED_FINAL, model-written |
| validation failure (NLI down) | VALIDATION_DEGRADED | ok | VALIDATED_FINAL with rules-only verification |

Every session closed; no fault produced an unverified or fabricated answer.

### Final validation (brief §83)

`final_validation.json`: **13 / 13 scenarios pass** their explicit checks. Each scenario also checks 0 untraceable events, an ordered user stream and clean termination.
- **Real local model** (1–5, 9–10):
  - **1. normal streaming question:** early retrieval, correct, validated 2.68 s after the first word;
  - **2. multi-intent:** two sections, overlapping subtasks;
  - **3. late constraint:** answered again with no LLM call;
  - **4. entity correction:** the final is about crates, with no ladders;
  - **5. concurrent retrieval:** overlapping, bounded subtasks;
  - **9. cancellation:** the generation was cancelled for the correction, and only the corrected turn was answered;
  - **10. stale race:** stale result discarded, no overwrite.
- **Simulated model** (6–8, 11–13):
  - **6. retrieval failure;**
  - **7. retrieval timeout:** lexical-only, no zombie workers;
  - **8. LLM failure;**
  - **11. high-frequency transcript:** 500 deltas, bounded queue, more than 400 coalesced, final state intact;
  - **12. four concurrent sessions:** isolated tasks, events and answers;
  - **13. graceful shutdown with active work:** cancelled within 0.21 s, no zombies, tokens or pending tasks; new sessions refused.

## 24. Load Test Results

**Setup** (`load.json`, SYNTHETIC: `SimulatedLLM` 800 ms; no injected latency):
- N concurrent sessions × 2 multi-intent utterances (staggered 50 ms); after a warm-up run;
- two LLM capacities: 1 slot (the local server's reality) and 4 (a server with parallel capacity).

| Sessions | LLM slots | Throughput | Validated p50 / p95 | First evidence p50 | Unanswered | Extractive (overload) | Error events | Max LLM queue | Loop lag p95 / max | Peak threads |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.76 turns/s | 1,533 / 1,583 ms | 459 ms | 0 | 0 | 0 | 1 | 0.8 / 12.7 ms | 7 |
| 2 | 1 | 0.99 | 2,239 / 2,814 | 458 | 0 | 0 | 0 | 2 | 0.8 / 27.6 | 8 |
| 4 | 1 | 1.08 | 3,862 / 5,912 | 456 | 0 | 0 | 0 | 4 | 0.8 / 22.6 | 8 |
| 8 | 1 | 1.12 | 7,134 / 12,213 | 463 | 0 | 0 | 0 | 8 | 0.8 / 25.8 | 8 |
| 16 | 1 | 2.12 | 2,218 / 12,370 | 466 | 0 | **15 of 32** | 0 | 8 (full) | 2.4 / 55.1 | 8 |
| 1 | 4 | 0.76 | 1,532 / 1,580 | 460 | 0 | 0 | 0 | 1 | 0.7 / 18.0 | 6 |
| 2 | 4 | 1.51 | 1,521 / 1,593 | 457 | 0 | 0 | 0 | 1 | 2.5 / 24.2 | 9 |
| 4 | 4 | 2.91 | 1,528 / 1,604 | 471 | 0 | 0 | 0 | 1 | 6.0 / 32.1 | 11 |
| 8 | 4 | 3.82 | 2,146 / 2,819 | 475 | 0 | 0 | 0 | 5 | 9.4 / 29.3 | 11 |
| 16 | 4 | 5.44 | 2,221 / 4,227 | 471 | 0 | **9 of 32** | 0 | 8 (full) | 12.4 / 76.1 | 11 |

**Reading:**
- **The model slot is the bottleneck.** With one slot, throughput saturates near the model rate (1.25 answers/s at 800 ms), and latency grows linearly with the queue.
- **Overload degrades, it does not fail.** At 16 sessions the bounded LLM queue (8) fills, and the overflow is answered extractively (still verified and cited) instead of dropped. Before that fix, 14 turns went unanswered (§25).
- **First evidence is load-independent** at this scale (about 460 ms, set by the controller's early-retrieval gate).
- **Loop lag grows with concurrent session logic** (p95 12.4 ms at 16 sessions / 4 slots): the single loop thread is the next bottleneck (§25).
- **Out-of-order output events held:** 0 in every row.
- **This is not production scale:** one machine, a fixture corpus, a simulated model.

**Resource leaks** (`leaks.json`, 5 runtime lifecycles, each with 2 sessions, one cancelled mid-run): threads 2 → 2, open fds 18 → 18, pending asyncio tasks 0, live cancellation tokens 0, zombie workers 0, Python heap 0.07 → 0.08 MB. **No growth.**

## 25. Limitations

1. **Synthetic model in most benchmarks; one machine; fixture corpus; at most 16 sessions.** Not production scale; nothing reportable. The real-model comparison is one run per scenario.
2. **The runtime cannot make the local model faster.** The validated answer latency is model-bound (§20). The early answer is an extractive draft, not model-written prose.
3. **Early retrieval costs about 64 % more retrieval calls** (§20).
4. **One event-loop thread runs all session logic.** Loop lag rises with concurrent sessions (p95 12.4 ms, max 76 ms at 16 sessions). Unbounded worker pools starve it (max 104 ms). Multi-process or multi-host scale-out is not implemented.
5. **Barge-in depends on Phase 6's change classification.** Phase 6 classifies some overlapping follow-ups ("…and the application process for the permit" after "…and how long does processing take") as ENTITY_CHANGE, and barge-in then cancels a valid answer. This was observed in the first concurrency run; the benchmarks run with barge-in off.
6. **"User relevance / answer importance" priorities are not modelled** (no measurable basis); query priority uses finalization and evidence presence only.
7. **No per-section LLM finals** (brief §31–32): drafts give section-level early answers; finals are one generation per version.
8. **The event log is in memory**, bounded only by `max_inputs_per_session`. There is no persistence or log compaction.
9. **Realtime traces replay only behaviourally** (by design); virtual traces replay exactly.
10. **Memory per configuration: NOT MEASURED** (only process-lifetime RSS, models included); Python heap peaks are measured instead.
11. **Thread-safety audit is by component inspection plus targeted tests**, not a race detector. The NLI cache relies on GIL atomicity.
12. **Phase 6 behaviours visible through the runtime:**
    - a repeated identical question becomes a second identical answer section;
    - a constraint gap quotes the user's words ("…applies for international applicants").

**Defects found and fixed during Phase 8** (disclosed):
- the shared, non-thread-safe Snowball stemmer, latent since Phase 4 realtime parallel retrieval;
- the hidden 2-thread embedding executor;
- a batch-statistics crash after a never-started cancelled query, which hung a session; callback isolation was added;
- answers dropped when the LLM queue was full (now extractive on the cpu pool);
- a superseded turn answered "no evidence" (now SUPERSEDED);
- draft cancel latency of about 430 ms (now about 100 ms);
- control inputs rejected by a full queue (now never rejected);
- graceful shutdown reporting before cancelled workers returned.

## 26. Phase 9 Requirements

**Ready for Phase 9:**
- one API (`start_session` / `push_transcript_delta` / `get_events` / `cancel` / `reset` / `shutdown`) that a service, UI or voice layer can wrap;
- an ordered, typed, traceable user event stream (drafts, finals, deltas, degraded modes);
- bounded resources with measured behaviour under overload;
- replayable traces for debugging and demos;
- a fault-injection harness for robustness demos.

**Exact prerequisites:**
1. **The official Theme 4 corpus and evaluation cases**, still blocked. Every quality and latency number so far is fixture-domain and NOT REPORTABLE.
2. **A serving decision:** in-process library vs a network service (HTTP / WebSocket over `get_events`). With a service: authentication, per-client quotas (`max_concurrent_sessions`, input limits) and the event wire format.
3. **A model-serving decision for latency:**
   - stay with the local 4B model (validated answer about 2.7 s p50 after the first word on the dev machine);
   - or a faster / parallel-capacity server or hosted model (changes the local-only decision, ADR-017);
   - and a target per-turn latency budget (no official target exists).
4. **A scale-out decision**, if multi-session load beyond one machine is in scope: multiple loop processes (one actor set each) with session affinity, or accept the single-loop bound measured here.
5. **Persistence:** whether the event log / session state must survive restarts (append-only store, retention), which today lives in memory.
6. **The packaging and demo plan** (Phase 2 milestones M8 container, M10 demo): container image with the pinned models and Ollama, and a demo script built on `research/phase8/e2e_demo.py`.
7. **Commit Phase 8.** The working tree is not committed (commit only on request).
