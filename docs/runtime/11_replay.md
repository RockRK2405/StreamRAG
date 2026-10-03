# 11: Event Sourcing and Replay

**Code:** `runtime/replay.py` (`inputs_from_runtime_trace`, `replay_runtime`, `behaviour`, `reconstruct`), `replay/replay.py`, CLI `streamrag replay`

## The log is enough (brief §46)

A runtime trace records every input with its arrival time on the runtime clock:
- **chunks:** `utterance_offset_s + timestamp_s`;
- **coalesced deltas:** `TRANSCRIPT_COALESCED.dropped_inputs`;
- **rejected deltas:** `BACKPRESSURE_APPLIED.input`, `arrival_ms`;
- **utterance ends:** utterance-relative time;
- **session end:** `SESSION_UPDATED.arrival_ms`;
- **model outputs:** `LLM_CALL` (request hash and output);
- **fault configuration:** `SESSION_STARTED.runtime.faults`.

Raw tokens are not stored beyond the chunk texts themselves.

## Replay (brief §47–48)

`replay_runtime(cfg, stack, events)` re-pushes those inputs into a **virtual-clock** runtime, with a `RecordedBackend` for the model and the recorded faults, then compares event by event (wall-clock fields excluded).

| Trace | Result |
|---|---|
| virtual-mode runtime trace | **exact**: every event identical, coalescing, retries and faults included (tested) |
| realtime trace | **orchestration replay**: the same inputs on a deterministic clock. Exact equality is impossible by design, because which deltas were coalesced, when a cancellation reached a worker and whether a deadline fired depend on wall timing. `behaviour` compares what is timing-independent: per utterance, the final transcript, the queries issued and their final status, and the claims of the validated answer. Tested; end-to-end demo: behaviour identical. |

**LLM nondeterminism** does not affect replay: model outputs come from the trace. A live re-run with the model would be equivalent only to the extent the model is deterministic (temperature 0, seed 7; not guaranteed by the server).

**Reconstruction.** `reconstruct(events)` rebuilds queries and their status, evidence per query, claims, and committed answers from events only. Tests compare it with the live session's ledger and answers.

**CLI.** `streamrag replay trace.jsonl` detects runtime traces (`SESSION_STARTED.runtime`) and runs `replay_runtime`. Phase 4–7 traces replay as before.
