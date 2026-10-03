# 05: Cancellation

**Code:** `runtime/cancellation.py` (`CancellationToken`, `CancellationManager`), `streaming/session.py` (`cancel_obsolete`, `on_cancelled`), `multi_retrieval/coordinator.py` (`_supersede`, `_cancel_queued`), `runtime/answers.py`

## Cooperative, never killed (brief §19)

**Tokens.** Every task has a token, a child of its session's token. Cancelling a session (reset, cancel, shutdown) cancels all of its tasks.

**Checkpoints.** Workers check their token at fixed points:

| Worker | Checkpoint |
|---|---|
| retrieval subtask | before starting; between query embedding and vector search; injected I/O waits are interruptible |
| assembly | before fusion |
| LLM call | before the call and **per streamed chunk**: the HTTP stream is closed, which stops the server-side generation |
| answer engine | between stages, per verified claim, per kept-claim re-verification |

A cancelled worker raises at its next checkpoint and releases what it holds:
- the slot frees when the worker returns;
- the engine is rolled back to its checkpoint;
- the token leaves the tree.

## When (brief §18)

| Trigger | What is cancelled |
|---|---|
| query superseded (refinement, constraint, delta plan) | the old query's pending subtasks are dropped; running ones are signalled (`cancel_superseded: cooperative`) |
| need superseded / removed (correction, entity change) | its queries, pending and running |
| correction / entity change / question change detected | the previous turn's in-flight final answer (`cancel_on_correction`, spec §7.3 barge-in), at detection time - not at the next turn end |
| a final requested | the session's running draft (superseded by the final); pending drafts are dropped |
| session reset / cancel, shutdown | every task of the session (session token) |
| deadline passed | the task (doc 06) |

**Superseded turns.** A turn whose need's query was cancelled because a later utterance refined it is **not answered**:
- `TURN_COMPLETED.answer = {status: SUPERSEDED, superseded_by: …}`;
- the later turn answers;
- before this rule, the cancelled turn committed a validated "the documents do not contain an answer" (found in the end-to-end demo; test `test_turn_superseded_by_a_refining_utterance_is_not_answered`).

## What the session sees

- **A query cancelled before it ran:** `RETRIEVAL_CANCELLED {reason: superseded_before_start}`; ledger status `cancelled`.
- **A query cancelled while running:**
  - `TASK_CANCELLED` per subtask;
  - then `RETRIEVAL_CANCELLED {reason: superseded_in_flight, wall: {wasted_ms}}`;
  - ledger `cancelled` / `cancelled_running`.

  Its result is never applied.
- **A cancelled answer:** `TASK_CANCELLED {error: correction | superseded_by_final | …}`. Nothing else: none of its events were ever emitted.

## Measured

Report §21: wasted work and latency, with and without cancellation.
