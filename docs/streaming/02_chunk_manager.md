# 02: Transcript Chunk Manager

**Code:** `streaming/chunk_manager.py`
**Tests:** `tests/streaming/test_chunk_manager.py`

## Purpose

Validated, idempotent accumulation of transcript chunks per utterance. The transcript is never silently corrupted.

## API

| Method | Behavior |
|---|---|
| `append_chunk(chunk, now_ms)` | Returns `AppendResult{status, changed, missing, transcript}` |
| `get_current_transcript(utterance_id)` | Rebuilt from chunks in `chunk_index` order. Punctuation chunks attach without a space. |
| `get_recent_chunks(n, utterance_id)` | The last *n* chunks by arrival |
| `finalize_utterance(utterance_id, now_ms, reason)` | `False` if already finalized |
| `open_utterance()`, `missing_indices()`, `is_finalized()`, `reset()`, `close()` | State queries and lifecycle |

## Statuses (each is emitted in `CHUNK_RECEIVED.payload.status`)

| Status | Condition | Transcript |
|---|---|---|
| `accepted` | Next index | Updated |
| `duplicate` | Same index, same text | Unchanged; no decision is run |
| `revision` | Same index with different text, or `replaces_chunk_index` | Replaced |
| `out_of_order` | Index below the max seen, filling a gap | Inserted in index order |
| `gap` | Index skips ahead | Accepted; `missing_indices` reported |
| `empty` | Whitespace text | Unchanged (keep-alive) |
| `rejected_finalized` | Utterance already ended | Unchanged, plus an `ERROR` event |
| `rejected_closed` | Session closed | Unchanged |

## Utterance lifecycle (handled by `StreamingSession`)

**Started by** the first chunk. The start time is `utterance_offset_s`, if given.

**Finalized by:**

| Trigger | `reason` |
|---|---|
| `UTTERANCE_END` | The input's reason (`source=input`) |
| No new chunk for `controller.endpoint_timeout_ms` | `timeout` |
| A chunk for a *new* utterance while one is open | `implicit_new_utterance` |
| `SESSION_END` | `session_end` |

A second `UTTERANCE_END` for the same utterance is an `ERROR`, and is ignored.

## Failure modes

- A chunk arriving after a premature timeout is rejected, with an explicit error.
- The timeout is therefore a fallback, set above the longest expected inter-chunk gap. It was raised from 2,000 to 3,000 ms after a dev-suite truncation (Phase 4 report §17).
