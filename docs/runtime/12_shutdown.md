# 12: Shutdown, Session Lifecycle and Resource Hygiene

**Code:** `StreamingRuntime.shutdown / force_shutdown / _close`, `RuntimeSession.cancel / reset`, `ThreadRunner.close`

## Graceful shutdown (brief §64)

```
stop accepting new sessions and inputs
  -> let active work finish (<= runtime.shutdown_grace_ms, default 5 s)
  -> cancel what is left (session tokens; SESSION_CANCELLED {reason: shutdown})
  -> wait (bounded) until every running worker reached a checkpoint and returned
  -> close the worker pools, flush and close trace files, stop the loop-lag monitor
```

`force_shutdown()` skips the grace period. Everything is cancelled at once; workers get at most 1 s to reach a checkpoint, then the pools close without waiting. Sessions that had already closed get `RUNTIME_SHUTDOWN`. A runtime that shut down refuses new sessions (`RuntimeClosedError`).

## Session lifecycle

- **`cancel_session`:** all of the session's tasks are cancelled, `SESSION_CANCELLED` is emitted, and the session emits nothing afterwards.
- **`reset_session`:**
  - tasks are cancelled;
  - the epoch is bumped, so results computed before the reset are discarded (`epoch_changed`);
  - cached results are forgotten;
  - the Phase 4–7 session state is rebuilt from scratch, and `SESSION_RESET` is emitted;
  - the event log continues.

## Leaks (brief §58)

Checked by tests (`tests/concurrency`) and `research/phase8` (`leaks.json`):
- after shutdown there are no pending asyncio tasks, no live cancellation tokens, no zombie workers and no queued work;
- thread count and open file descriptors are flat over repeated runtime lifecycles;
- the Python heap is flat across cycles (tracemalloc).

The retrieval service's own small thread pool is shared and outlives runtimes by design (`RetrievalService.close`).
