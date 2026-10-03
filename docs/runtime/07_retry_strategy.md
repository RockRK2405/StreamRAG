# 07: Retries, Backoff and Idempotency

**Code:** `runtime/retry.py` (`RetryManager`, `is_transient`, `TransientError`, `PermanentError`), `runtime/scheduler.py` (task retries), `runtime/runtime.py` (`RetryingLLM`)

## What is retried (brief §36)

| Failure | Transient? |
|---|---|
| `TimeoutError`, `ConnectionError`, `urllib` URL errors, `RetrieverTimeoutError`, `TransientError` | yes |
| error text: timeout / connection refused / reset / unavailable / temporary / rate limit / 429 / 502 / 503 / 504 | yes (LLM responses are values, not exceptions) |
| invalid input, schema violation, validation failure, `PermanentError`, any other exception | **no**: a retry would repeat it |

## Bounded exponential backoff (brief §37)

```
delay(attempt) = min(max_delay_ms, initial_delay_ms * multiplier ** attempt) * (1 ± jitter)
```

Defaults (`runtime.retry`): 50 ms, ×2, capped at 1000 ms, jitter ±20 % from an RNG seeded with `seed` (7). Virtual runs are therefore replayable. At most `max_retries` (2) retries follow the first attempt, and only while the task's deadline allows.

## Where

- **Retrieval subtasks:** the scheduler resubmits the same task (`attempt + 1`, `TASK_RETRIED` with the error and the backoff).
- **LLM calls:** `RetryingLLM` retries transient errors inside one generation task. The wait honours the call's cancellation token and deadline. Only the final response is recorded in `LLM_CALL`, so replay stays exact.

## Idempotency (brief §38)

A retry is the *same* task:
- the same id and idempotency key;
- its partial results are emitted once;
- its evidence is added once;
- it creates no extra answer version (tested).

Identical retrievals in one session (same query, options, index; same epoch) are executed once and joined or reused (doc 03).
