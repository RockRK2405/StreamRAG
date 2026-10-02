# Phase 3 — Retrieval Stack Performance Profile

- **Script:** `research/phase3/profile_retrieval.py`
- **Raw results:** `results/retrieval_profile.json`
- **Date:** 2026-10-02
- **Machine:** Apple M5 Pro dev machine
- **Models:** bge-small-en-v1.5 (ONNX fp32), cross-encoder ms-marco-MiniLM-L6-v2 (ONNX fp32)

> **These are latency and memory measurements on SYNTHETIC corpora** (generated words, TEST_FIXTURE_ONLY). They say nothing about retrieval quality. They are not official Samsung thresholds; the targets referenced are Phase 2 **team targets** `[T]`.

## Setup

| | Small | Large |
|---|---|---|
| Synthetic documents | 135 | 670 |
| Chunks (default chunking) | 2,032 | 9,961 |
| Query set | 200 synthetic queries (4–8 words), after 5 warm-up queries | same |
| Thread settings | default; 2 intra-op threads (container proxy) | same |

## Build (index time)

| Stage | 2,032 chunks | 9,961 chunks |
|---|---|---|
| Corpus scan + hash | 6.2 ms | 31.4 ms |
| Load (txt) | 2.0 ms | 10.2 ms |
| Normalization | 42.1 ms | 211.0 ms |
| Section parsing | 11.1 ms | 57.7 ms |
| Chunking | 42.5 ms | 214.6 ms |
| BM25 indexing | 1.29 s | 6.69 s |
| Embedding indexing (batch 32) | 10.97 s (185 chunks/s) | 54.87 s (182 chunks/s) |
| **Build total** | **12.6 s** | **62.7 s** |
| Index load (verify hashes) | 16.1 ms | 104.4 ms |
| Peak RSS during build (batch 32) | 1093 MB | 1297 MB |
| Peak RSS during build (batch 8, now the default) | 589 MB | — |
| Truncated chunks (> 512 wordpieces) | 0 | 0 |

**Index size on disk (9,961 chunks), 49.1 MB total:**

| Artifact | Size |
|---|---|
| `chunks.jsonl` | 16.6 MB |
| `dense/embeddings.npy` | 15.3 MB (= 9,961 × 384 × 4 bytes) |
| `documents.jsonl` | 13.8 MB (normalized texts kept for traceability) |
| `bm25/weights.npz` | 3.1 MB |

At 2,032 chunks the total is 9.9 MB.

BM25 indexing (6.7 s at 10k chunks) is dominated by Python-side tokenization and Snowball stemming. Embedding dominates everything else.

## Query latency (ms; p50 / p95; 9,961 chunks; default threads)

| Mode | Lexical | Embed | Dense search | Fusion | Dedup | Rerank | **Total** |
|---|---|---|---|---|---|---|---|
| bm25 | 0.17 / 0.26 | — | — | 0.01 | 0.58 / 0.60 | — | **0.83 / 0.94** |
| dense | — | 2.40 / 2.53 | 0.26 / 0.28 | 0.02 | 0.79 / 0.81 | — | **3.52 / 3.66** |
| hybrid (RRF) | 0.23 / 0.29 | 2.44 / 2.58 | 0.26 / 0.28 | 0.10 | 2.21 / 2.37 | — | **5.34 / 5.60** |
| hybrid + rerank (20) | 0.32 / 0.41 | 3.07 / 3.52 | 0.33 / 0.43 | 0.12 | 2.22 / 2.44 | 95.2 / 105.7 | **101.3 / 113.0** |

## Query latency with 2 threads (container proxy; 9,961 chunks)

| Mode | Total p50 / p95 | Rerank p50 / p95 |
|---|---|---|
| bm25 | 0.82 / 0.91 | — |
| dense | 2.98 / 3.18 | — |
| hybrid | 4.36 / 4.71 | — |
| hybrid + rerank (fp32 cross-encoder) | 172.4 / 189.8 | 167.2 / 184.5 |
| hybrid + rerank (int8-arm64 cross-encoder) | 183.8 / 202.0 | 179.1 / 197.7 |

The 2,032-chunk results are within about 0.5 ms of these for every non-rerank mode. Search cost barely grows from 2k to 10k chunks; dense search goes from 0.08 to 0.26 ms.

## Query-process memory and start-up

| Metric | Value |
|---|---|
| Service init (index + bge-small + cross-encoder) | 194–239 ms |
| RSS after loading index + both models (2k chunks) | 560 MB |
| Peak RSS during hybrid + rerank queries | 1.06–1.15 GB |
| Same, with the ONNX Runtime memory arena disabled (2k chunks) | **1.86 GB** (worse), rerank p50 111 ms (slower), so the arena stays enabled |

## Interpretation against the Phase 2 latency model

- **Hybrid retrieval fits the budget with room to spare.** It costs ~4–5 ms end to end, including query embedding, at 10k chunks, on both thread settings, against the `[T]` targets p95 < 50 ms per query (REQ-PERF-002). Retrieval is not the streaming bottleneck, which confirms Phase 1's conclusion with the real stack.
- **The cross-encoder dominates when enabled.** It takes ~95 ms (default threads) to ~170 ms (2 threads) for 20 candidates. This is consistent with the Phase 2 probe (105–124 ms per 20 pairs at 128 tokens; these chunks are slightly shorter). Phase 2's rerank-on-stability and `finalize_wait_ms` mitigations remain necessary on slow hardware. **Whether reranking is worth this cost is a quality question (Exp 3, blocked).**
- **Int8 reranker: rejected.** On this CPU it is slower than fp32 (184 vs 172 ms p50 with 2 threads).
- **Dedup is the largest non-model stage** in hybrid mode (~1.5–2.2 ms). It is a Python pairwise near-duplicate check over about 100 fused candidates. That is acceptable now, and it is an optimization target: vectorize it, or restrict it to the top `top_k + margin` candidates.

## Not measured

- **Judge hardware.** The 2-thread rows are only a proxy. Re-measure inside the container on the target machine (Phase 8).
- **Container overhead (Docker VM).** Packaging is out of scope for Phase 3.
- **Corpora much larger than 10k chunks.** Phase 1 measured exact dense search at 1.6 ms for 100k × 384 on synthetic vectors.
