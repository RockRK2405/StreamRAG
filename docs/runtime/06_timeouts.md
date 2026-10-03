# 06: Timeouts, Deadlines and the Latency Budget

**Code:** `runtime/timeouts.py` (`TimeoutManager`, `Deadline`), `runtime/runtime.py` (`new_task`, `turn_deadline`), `generation/llm.py` (`CallContext`)

## Per-task timeouts (brief §24)

`runtime.timeouts_ms`: lexical 2000, dense 3000, assemble 2000, generation 60000, draft 10000, validation retrieval 3000. These are configuration values, not measured targets.

## Deadline propagation (brief §25)

**A turn starts** when its first input arrives. Its budget is `budget.turn_ms` (30 s, configuration).

**Every task's deadline:**

```
deadline = min(now + own_timeout, turn_deadline - reserve_after(task_type))
reserve_after(retrieval tasks) = generation_reserve + validation_reserve
reserve_after(generation, draft) = validation_reserve
```

- Retrieval can never use the time generation and validation will need.
- The reserves are the measured Phase 7 p95 stage times on the dev machine (generation 4.2 s, validation 0.4 s).
- A budget already spent still yields a 1 ms slot, so the task fails fast instead of never running.
- `TASK_*` events record `deadline_limited_by: task_timeout | turn_budget`.

**The LLM call receives the remaining time:** the worker sets a `CallContext` deadline, and the Ollama request timeout is `min(configured, remaining)`.

## On expiry

- **Pending:** never starts (TIMED_OUT).
- **Running:** TIMED_OUT is reported at the deadline and its token cancelled. The slot is held until the worker returns (doc 03).
- **Retrieval timeout:**
  - a dense subtask timeout degrades the query to lexical-only (`RETRIEVAL_DEGRADED`) and the turn continues;
  - a timeout of both subtasks is a failed retrieval, and the session keeps its earlier evidence (doc 09).
- **Generation timeout:** the turn is answered extractively (`GENERATION_DEGRADED`, doc 09).

## Latency budget: measured, not invented (brief §26)

| Stage | Where it is measured |
|---|---|
| input processing | `CHUNK_RECEIVED` → decision (`payload.wall`) |
| intent detection | decompose time (`wall.decompose_ms_total`) |
| retrieval | subtask `TASK_*` queue wait / exec |
| reranking / assembly | assemble task |
| generation | generation task (+ `LLM_CALL` first-token / total) |
| validation | answer stage timings (Phase 7) |
| output | ordered release (event emission) |

The report gives measured values only (§19–§24).
