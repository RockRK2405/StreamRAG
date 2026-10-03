# 04: Concurrency Model

**Code:** `runtime/aggregator.py` (`RuntimeRetrievalExecutor`, `StreamingEvidenceAggregator`), `retrieval/service.py` (`plan`, `search_lexical`, `search_dense`, `assemble`)

## Why each mechanism (brief §61–62)

| Mechanism | Used for | Why |
|---|---|---|
| asyncio event loop (one thread) | input, timers, every session-state mutation, event emission, subscribers | single owner per session without locks; ordered event log |
| thread pools (bounded per pool) | BM25, ONNX embedding + vector search, evidence assembly, answer generation (HTTP to the local model), NLI verification | blocking libraries (numpy / ONNX Runtime release the GIL; the model call is network I/O); `run_in_executor` keeps the loop free |
| processes | not used | no measured need: the hot paths release the GIL, and the fixture-scale CPU work is small. A process pool would add serialization of index and model state for no measured gain. |

**Event-loop health.** No long operation runs on the loop:
- the loop-lag monitor records how late a 5 ms sleep wakes up;
- tests assert max < 300 ms and p95 < 20 ms, while every answer takes the simulated model 600 ms;
- a negative-control test proves a 250 ms blocking call is detected;
- with the real local LLM, the runtime's max lag was 19 ms. The Phase 7 pipeline blocked the loop for 2.7 s (report §20).

## Parallel retrieval (brief §15)

A hybrid query becomes two subtasks on the retrieval pool, which run concurrently:
- **lexical:** BM25;
- **dense:** embedding plus vector search.

Each result is published as `RETRIEVAL_PARTIAL` (hits and top citations) as soon as it arrives. When both are in, one `assemble` task fuses them (RRF, dedup, optional rerank). The assembled `EvidenceSet` is identical to the Phase 3 `retrieve()` result, because `_retrieve` composes the same stage functions.

Independent intents' subtasks run in parallel, up to the pool size. In the dense subtask, the embedding runs inline in the worker thread: the service's own 2-thread timeout executor would serialize concurrent embeddings and time them out. The runtime's deadlines replace it.

## Bounded everywhere (brief §16–17)

- **Slots:** per pool, plus `max_concurrent_sessions`.
- **Queues:** per-session input, per-pool pending, per-subscriber output (doc 08).
- **Memory:**
  - every queue is capped;
  - the session event log grows with the session (event sourcing, doc 11) and is bounded only by `max_inputs_per_session`;
  - the cancellation token tree shrinks as tasks finish.

## Thread safety of shared components

| Component | Concurrent use | Safety |
|---|---|---|
| Snowball stemmer (text analyzer) | BM25 query analysis and claim terms on several workers | **was unsafe**: a shared instance gave 25 exceptions and 123 wrong stems in 300 calls on 8 threads. Now one stemmer per thread (regression test). This also affected Phase 4/5 realtime parallel retrieval; their virtual-mode results were single-threaded and unaffected. |
| ONNX sessions (embedder, NLI) | several workers | ONNX Runtime `run` is thread-safe |
| HF tokenizer | `encode_batch` from several workers | configured once at load; read-only use |
| NLI result cache | dict writes from several workers | atomic under the GIL; a racing duplicate computation is harmless |
| LLM backend | HTTP per call | stateless; the per-call context (cancellation, deadline) is a context variable set in the worker thread |
| Grounded answer engine | per session, only its AnswerLane | one answer task at a time per session |
