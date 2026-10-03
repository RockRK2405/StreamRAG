# 09: State Coordination, Race Conditions, Failure Isolation and Degraded Modes

**Code:** `runtime/state.py` (`StateCoordinator`), `runtime/aggregator.py`, `runtime/answers.py`, `runtime/faults.py` (`FaultInjector`, `Fault`, `FaultyLLM`), `runtime/sim.py`

## Race-condition prevention (brief §21–23, §44–45)

**Ownership:** one session actor on the loop (doc 02). So the question is not *how to lock* but *which results may be applied*.

**Every task captures** its session's `state_version` and `epoch`. Before a result is applied, `StateCoordinator.commit(result, apply, relevant)` checks two things:
1. the epoch is unchanged (a reset bumps it): otherwise `epoch_changed`;
2. the task is still relevant: otherwise `superseded`:
   - retrieval: the query is still the current version of its need;
   - draft: it is still the newest answer request and the utterance is not finalized;
   - final: it was not cancelled.

A stale result is discarded with `STALE_RESULT_DISCARDED` (task, reason, its version vs the current one). It never overwrites newer state.

- **Retrieval reconciliation:** for a superseded query, the ledger still records the completion (`stale_at_completion`), so the turn's bookkeeping completes; fusion and claims ignore stale queries.
- **Commits are atomic:**
  - an answer's buffered events and its fallback-retrieval evidence are applied in one step;
  - the session is snapshotted first (`SessionMemory.create_snapshot`) and restored if applying raises (`ERROR {action: rolled_back_to_last_valid_state}`);
  - the answer engine itself is checkpointed and restored for every cancelled or stale answer.
- **Mandatory race test** (brief §55): Q1 starts; Q2 starts later and completes first; Q1 completes last. Q1 is discarded as stale, and the final answer is built on Q2. Reproduced in 10 of 10 realtime runs, with 0 overwrites (report §23).

## Failure isolation (brief §39)

- **A worker failure is a task result, not an exception on the loop.** A failing result handler is reported and the pool continues.
- **Faults stay inside their session:** one session's injected failures leave a concurrent session's answers intact (tested).
- **A failed retrieval part does not fail the query:**
  - dense failed → lexical-only;
  - lexical failed → dense-only;
  - both failed → `RETRIEVAL_COMPLETED {status: error}` plus `ERROR {action: keep_previous_evidence}`; the turn is answered from the evidence it has, or as "not established".

## Degraded modes (brief §40–43), never silent

`DEGRADED_MODE_CHANGED` is emitted the first time each mode becomes active in a session.

| Mode | When | Behaviour |
|---|---|---|
| RETRIEVAL_DEGRADED | a dense / lexical subtask failed or timed out, or a whole retrieval failed | lexical-only / dense-only evidence; failed retrieval: keep earlier evidence, report insufficiency |
| GENERATION_DEGRADED | the LLM failed (after transient retries), timed out, its queue was full, or the engine fell back internally | the final is redone **extractively** (verbatim evidence sentences, still verified and cited) on the cpu pool |
| VALIDATION_DEGRADED | the entailment model failed during verification | the final is redone with **rules-only** verification (`verifier: rules`, doc answer/04) |
| (last resort) | the fallback also failed | `ERROR {action: keep_previous_validated_answer}`; the previous validated answer stays current; nothing is fabricated |

**Deterministic fallback chain** for an LLM failure (brief §41): retry (transient only) → extractive answer from the same plan → previous validated answer. "Smaller context / smaller generation" variants are not used: they have no measured benefit here, and the extractive answer is already the smallest grounded one.

## Failure injection (brief §56)

`FaultInjector([Fault(target, kind, times, delay_ms, session_id)])`:
- **targets:** `lexical`, `dense` (vector index), `network` (any retrieval subtask), `llm`, `verification`;
- **kinds:** `error`, `transient`, `timeout`, `delay`.

Injected delays are real interruptible waits in realtime mode, and modelled latency in virtual mode. Used by tests and by `research/phase8` only; TEST / BENCHMARK USE.
