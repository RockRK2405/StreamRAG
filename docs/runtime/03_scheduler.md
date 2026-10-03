# 03: Task Model, Scheduler and Priorities

**Code:** `runtime/tasks.py` (`Task`, `TaskResult`, `TaskStatus`, `TaskType`, `POOL_OF`), `runtime/scheduler.py` (`TaskScheduler`, `VirtualRunner`, `ThreadRunner`, `WorkContext`)

## Task (brief §8)

| Field | Content |
|---|---|
| `task_id` | `T<n>`, runtime-wide (deterministic order) |
| `task_type` | `lexical`, `dense`, `retrieval` (unsplit), `assemble`, `generation`, `answer_extractive`, `draft`, `validation_retrieval`, `analytics` |
| `session_id`, `correlation_id`, `causation_id`, `parent_task_id` | ownership and lineage |
| `priority`, `deadline_ms`, `idempotency_key`, `attempt` | scheduling |
| `state_version`, `epoch` | the session state it was computed from (doc 09) |
| `status` | PENDING → RUNNING → COMPLETED / FAILED / CANCELLED / TIMED_OUT / SUPERSEDED |

A `TaskResult` carries `task_id`, `state_version`, `epoch`, `correlation_id`, `status`, `result`, `error` and `metadata` (queue wait, execution time, attempts). Nothing in it is applied before the state coordinator accepts it.

## Pools

Each resource class has its own slots and its own bounded pending queue. A slow LLM never starves retrieval:

| Pool | Task types | Slots (config) |
|---|---|---|
| retrieval | lexical, dense, retrieval, validation_retrieval | `max_concurrent_retrievals` (4) |
| cpu | assemble, draft, answer_extractive, analytics | `max_concurrent_cpu` (2) |
| llm | generation | `max_concurrent_llm_calls` (1: the local Ollama server serves one request at a time) |

**A final that needs no model call runs on the cpu pool** (extractive backend, or degraded). An overloaded LLM queue therefore never leaves a turn unanswered (measured, report §24).

## Priorities (brief §10–11), configurable (`runtime.priorities`)

| Level | Value | Tasks |
|---|---|---|
| CRITICAL | 0 | the current turn's final answer |
| HIGH | 1 | retrieval for a finalized utterance, or for a need without evidence yet; validation retrieval |
| MEDIUM | 2 | early (provisional) retrieval for a need that already has evidence; drafts |
| LOW | 3 | analytics / optional enrichment |

**Selection:** the next pending task of a pool minimises `priority - waited_ms / aging_ms` (ties: FIFO). With `aging_ms = 500`, a LOW task overtakes a fresh HIGH one after waiting 1 s, so no task starves (tested).

**Query priority** uses what the runtime knows without guessing user intent:
- is the utterance finished (the answer is waiting)?
- does the need already have evidence?

"User relevance" and "answer importance" have no measurable basis yet. They are not used.

## Scheduling rules

- **Slots are held until the worker returns.** A task reported TIMED_OUT whose thread is still finishing keeps its slot (a "zombie"). Timeouts can never create more threads than configured.
- **Expired before it started:** a pending task whose deadline has passed never starts (TIMED_OUT).
- **Running past its deadline:** reported TIMED_OUT at the deadline; its token is cancelled, so the worker stops at its next checkpoint.
- **Retries** (doc 07): transient failures are resubmitted after backoff, while the deadline allows.
- **Idempotency** (brief §38): a task whose key is pending or running is joined, not started twice. A completed result is reused, and reuse never crosses a session or an epoch (keys include both; `forget_session` on reset). A retry reuses the task; nothing is emitted twice (tested: one `RETRIEVAL_PARTIAL` per subtask despite two retries).
- **Full pending queue:** the least important pending task is shed (CANCELLED, `shed_by_backpressure`) if it is less important than the new one. Otherwise the new task is rejected (`TASK_REJECTED`). Nothing grows without bound.
- **A failing result handler is isolated:** reported as an `ERROR` event, and the pool keeps pumping (tested).

## Runners

- **`ThreadRunner` (realtime):** a bounded `ThreadPoolExecutor` per pool; results are delivered on the asyncio loop.
- **`VirtualRunner`:**
  - computes the task when it starts (real computation);
  - delivers it after a modelled latency on the virtual clock;
  - injected delays become virtual latency.

  Replays are deterministic.
