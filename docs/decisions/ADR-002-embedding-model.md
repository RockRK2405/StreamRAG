# ADR-002: Embedding Model

- **Status:** Accepted with a pending experiment (the final choice comes from Exp 1 on the corpus)
- **Phase 3 update:** ONNX runtime confirmed by measurement; bge-small and MiniLM ONNX outputs match sentence-transformers at cosine 1.000000; `dense.batch_size` = 8. See ADR-013 and `research/phase3/embedding_benchmark.md`. Model quality is still unmeasured.
- **Date:** 2026-10-02
- **Spec:** §10.2, §25.2

## Context

The embedder sits on the streaming critical path: query encoding for every dispatched intent, plus optional novelty and L2-verification encodings. It also drives index build time.

Measured on the dev machine (`research/phase1`):

| Model | Query encoding | Passage throughput |
|---|---|---|
| `all-MiniLM-L6-v2` | 3–4 ms | 180–210 passages/s |
| `multilingual-e5-base` | 22–26 ms | 30–35 passages/s; similarity range is compressed |

MiniLM truncates at 256 word-pieces and was trained mainly for symmetric similarity. Our workload is asymmetric: short spoken queries against section-level chunks.

The global Python environment cannot import sentence-transformers (transformers 5.9 vs torch 2.2.2), so a pinned environment is required.

## Decision

- **Default:** `BAAI/bge-small-en-v1.5` (33M, 384-d, 512 tokens, MIT).
- **Fallback:** `all-MiniLM-L6-v2` (already cached, measured).
- **Multilingual option:** `multilingual-e5-small`, only if Q7 requires it.
- **Runtime:** ONNX Runtime preferred (smaller image, no torch); torch-CPU ≥ 2.4 as the alternative. Decided in a Phase 3 spike.
- **Final selection:** Exp 1 (Recall@k + latency on the corpus dev set).

## Alternatives

| Model | Status |
|---|---|
| e5-small-v2, arctic-embed-s, gte-small | Close peers. Included in Exp 1 if time allows. |
| bge-base | Possible quality step-up at ~3–4× the cost [E] |
| nomic-embed-v1.5 | Rejected: needs `trust_remote_code`; long context is unnecessary |
| bge-m3, Qwen3-Embedding-0.6B, embeddinggemma | Rejected: too heavy for CPU streaming, or license overhead |

## Why selected

It is retrieval-trained for asymmetric query→passage search, has the same dimension as the measured fallback (so swapping is trivial), has a 512-token window that fits section chunks, and carries a permissive license.

## Trade-offs

- bge-small's latency is an estimate (~6–9 ms) until it is downloaded and measured.
- It needs download approval (~130 MB).

## Consequences

- All similarity thresholds (`τ_dup`, `τ_merge`, `θ_cov_dense`, L2 cosine) are **model-specific** and must be recalibrated whenever the embedder changes.
- The index cache key includes the embedder revision.
