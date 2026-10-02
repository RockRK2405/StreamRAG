# 08: Replay Engine

**Code:** `replay/replay.py`
**CLI:** `streamrag replay TRACE.jsonl`
**Tests:** `tests/streaming/test_replay_and_metrics.py`

## Purpose

Deterministic debugging: given a saved event trace, re-run its inputs and verify the same sequence of decisions and events.

## A trace is self-describing

Inputs are recovered from the trace itself:

| Trace event | Input recovered |
|---|---|
| `CHUNK_RECEIVED.payload.input` | Every chunk, including duplicates and rejected ones |
| `UTTERANCE_FINALIZED.payload.input` | The original `UTTERANCE_END`, when `source = input` |
| `ERROR.payload.input_event` | Inputs that only produced an error (e.g. a duplicate `UTTERANCE_END`) |
| `SESSION_STARTED.payload.input`, `SESSION_CLOSED.payload.input` | Session start and end |

System-generated events (timeouts, decisions, queries, retrievals, turn summaries) are *outputs* that the replay must reproduce.

## Comparison

**Exact comparison (virtual traces).** `canonical_for_replay(event)` drops `t_wall_ms` and every `payload.wall` sub-object. Everything else must be equal:
- event order and ids;
- `t_session_ms` / `t_stream_s`;
- decisions, reasons and signals;
- query versions and lineage;
- evidence ids;
- metrics.

## Guarantees

| Trace source | Result |
|---|---|
| Virtual-mode trace + same config, index and code | **Identical** (tested; verified in the Phase 4 benchmark run) |
| Realtime trace | Inputs replayed in virtual mode. Timestamps, measured latencies, `mode` and the config hash necessarily differ, so exact equality is not expected. The check is **`behavior_identical`**: per utterance, the same decisions (decision, reason, query text), query versions and finalizations, in order, plus the same retrieval outcomes (query, status, evidence ids). This holds because controller decisions run on stream time (ADR-014 §1b). It is tested, and checked for every realtime dev-suite trace (`results/realtime_replay_parity.json`). The exception is the in-flight guard (`allow_parallel_retrieval: false`; off by default), which depends on real completion times. `stale` flags are not compared, because they record completion timing. |
| Changed configuration (e.g. a different policy) | Differences detected (tested) |

## Usage

```bash
.venv/bin/streamrag stream --text "I need | information | about the fog signal" --interval-ms 300 --trace t.jsonl
```

```bash
.venv/bin/streamrag replay t.jsonl
```

`replay` prints `identical`, `behavior_identical` and `ok`. It exits with code 0 when `ok` is true (exact match for a virtual trace; behavior match for a realtime trace), and 1 otherwise.
