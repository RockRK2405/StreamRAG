# 08: Backpressure and Transcript Coalescing

**Code:** `runtime/backpressure.py` (`InputQueue`, `coalesce`), `runtime/streamer.py` (`Subscription`, `OrderedReleaseBuffer`), `runtime/scheduler.py` (pool queues)

## Strategy (brief §13), in order

1. **Adaptive batching.** A session's input queue is drained in one lane step:
   - all pending deltas are applied to the transcript;
   - the expensive work (controller decision, decomposition, retrieval) runs once, for the last delta of the batch.

   With `coalescing_window_ms = 0` (default) this happens only when deltas arrive faster than they are processed, so there is no added latency when idle.
2. **Coalescing (brief §14).** An intermediate transcript state that a later pending delta replaces is dropped (`TRANSCRIPT_COALESCED`, with the dropped inputs for replay). An example is an ASR partial hypothesis ("what" → "what are" → "what are the") revised before it was processed. Never dropped:
   - the latest state of each chunk;
   - appended chunks (content);
   - utterance and session ends.

   The final meaningful transcript is always the one processed (tested).
3. **Bounded queues.**
   - **Input queue:** `queues.input` (64) transcript deltas. When full, it first coalesces in place; if still full, the delta is rejected with `BACKPRESSURE_APPLIED {action: rejected, input}`. That is never silent, and replay re-pushes it to be rejected again. Control inputs (utterance and session ends, at most one per utterance) are never rejected: they are what makes the final state meaningful.
   - **Task queues:** `queues.retrieval / cpu / llm`. A full queue sheds the least important pending task, or rejects the new one (doc 03). A rejected generation is answered extractively on the cpu pool, never dropped.
   - **Subscriber queues:** `queues.output`. A slow consumer first loses intermediate DRAFT answer events, which a newer draft or the final supersedes. If it still overflows, it is disconnected (it resynchronises from the log). The producer never blocks.

## Output ordering (brief §34–35)

- **Structural ordering.** All of a session's events are emitted on the loop, and answer versions are committed by the lane one at a time in request order. User-visible events carry a contiguous `output_seq`.
- **`OrderedReleaseBuffer`** is the general mechanism for out-of-order producers:
  - item N+1 waits for N;
  - an item held longer than `output_max_hold_ms` (2 s) releases with the gap skipped and reported, so buffering is never indefinite;
  - duplicates are never re-released.

  In the measured runs no event ever needed it (the structural order held).

## Measured

Report §22: unbounded versus bounded/coalescing, paced at about 1 kHz and as an instant burst.
