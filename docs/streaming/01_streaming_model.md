# 01: Streaming Model

**Code:** `streaming/simulator.py`, `streaming/clock.py`, `streaming/runner.py`, `models/events.py`

## Purpose

Process a conversation as it arrives, as timestamped transcript chunks, not as finished queries. Retrieval can then start before the user stops speaking.

## Input contract (Phase 2 §5; unchanged)

| Event | Key fields |
|---|---|
| `TRANSCRIPT_CHUNK` | `session_id`, `utterance_id`, `chunk_index`, `timestamp_s` (utterance-relative), `text` (delta), `stability` (`final`/`partial`), `replaces_chunk_index`, `utterance_offset_s` |
| `UTTERANCE_END` | `timestamp_s`, `reason` (`endpoint`/`explicit`/`timeout`/`eof`), `last_chunk_index` |
| `SESSION_START` / `SESSION_END` | Optional; the session auto-starts on the first chunk |

**Mapping to the Phase 4 brief's chunk fields:**

| Brief field | Here |
|---|---|
| `received_at` | `ChunkRecord.received_at_ms`, the session stream clock |
| `is_final` | `stability == "final"` (ASR hypothesis) |

End of utterance is a separate timed event, as in the guide. It is not a flag on the last chunk: the guide's end marker arrives 0.5 s after the last chunk and carries no text.

## Two execution modes (same session logic)

| Mode | Clock | Retrieval | Use |
|---|---|---|---|
| `virtual` (default) | Discrete-event: callbacks in (time, priority, seq) order | Executed when it starts (real evidence); completion delivered at `start + sim_retrieval_latency_ms` | Deterministic tests, benchmarks, **exact replay** |
| `realtime` | asyncio loop + wall clock (`speed` factor). Same (time, priority, seq) heap; one loop timer armed at the earliest entry. | `asyncio.to_thread(retrieve)` with a timeout | Real latencies; proves the stream is not blocked; behavior-identical replay |

In both modes, simultaneous events run in this order: retrieval completion, then input, then timer, then in scheduling order.

Controller decisions and their timers use **stream time** (`logical_now_ms`: when the triggering input or timer was due). Event timestamps use actual time. Realtime decisions therefore depend only on the input stream (ADR-014 §1b).

## Simulator

| Function | Behavior |
|---|---|
| `stream(chunks, interval_ms=200)` | Chunk *i* at *i*·interval (the first at 0, as in the guide's timeline); `UTTERANCE_END` after `end_gap_ms` |
| `timed(chunks, words_per_second=2.6, jitter, seed)` | A chunk arrives when its last word has been spoken. Seeded, so deterministic. |
| `from_case(BenchmarkCase)` | Uses the case's chunk timestamps; multi-turn sessions are offset by `utterance_offset_s` |
| `write_inputs` / `read_inputs` | JSONL input files (CLI: `streamrag stream --input`) |

Real audio can be added later by an ASR adapter that emits the same input events.

## Configuration

`streaming.mode`, `speed`, `sim_retrieval_latency_ms` (10; Phase 3 measured hybrid p95 at ~5.6 ms), `retrieval_timeout_ms`, `retrieval_mode`, `top_k`, `rerank`, `words_per_second`, `end_gap_ms`.

## Trade-offs

- Virtual mode models retrieval latency as a constant. The measured wall latency is reported in `payload.wall`, and realtime mode measures the real thing.
- Realtime runs with `speed > 1` scale stream-time latencies by `speed`. Report latencies from speed 1.0 runs or from `wall.measured_ms`.
