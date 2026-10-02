# 04: Dense Retrieval

**Code:** `retrieval/embedders.py`, `retrieval/dense.py`
**ADR:** ADR-002 (runtime confirmed in Phase 3)
**Benchmark:** `research/phase3/embedding_benchmark.md`

## Purpose

Semantic matching between spoken phrasing and policy wording. This is the dense half of hybrid retrieval.

## Architecture

**`OnnxEmbedder`** is the production path:
- HF `tokenizers`, with truncation at the model's `max_length` and dynamic padding.
- ONNX Runtime on the **CPU execution provider only** (no GPU or CoreML; deterministic).
- Pooling read from the model's own `1_Pooling/config.json` and cross-checked against `configs/models.yaml`: CLS for bge, mean for MiniLM.
- L2 normalization; per-model query and document prefixes.
- Length-sorted batching, with results restored to the original order.
- Its output matches `sentence-transformers` at **cosine 1.000000** (measured).

**`HashingEmbedder`** is **TEST/OFFLINE ONLY.** It uses signed feature hashing of stemmed tokens and bigrams. It is not semantic, and exists so tests and keyless smoke runs exercise the dense path.

**`DenseIndex`** is an exact inner-product search over a read-only float32 matrix, which equals cosine because vectors are normalized. Top-k uses scores rounded to 1e-6 for ordering, with ties broken by row; this guards against BLAS thread jitter. An optional `dense_min_similarity` cut-off is available.

## Models (pinned in `configs/models.yaml`)

| Name | Repo @ revision | Dim | Max tokens | Role |
|---|---|---|---|---|
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 @ 5c38ec7c | 384 | 512 | **Default (provisional, pending Exp 1)** |
| `all-minilm-l6-v2` | sentence-transformers/all-MiniLM-L6-v2 @ 1110a243 | 384 | 256 | Fallback |
| `hashing` | — | 256 | — | Tests and offline only |

## Vector index decision (evidence-based)

| Property | Value |
|---|---|
| Type | Exact flat inner product, `numpy` matmul |
| Similarity | Cosine |
| Persistence | `dense/embeddings.npy` + `dense/meta.json` (EmbeddingInfo) |
| Search latency (measured) | 0.08 ms @ 2k, 0.26 ms @ 10k chunks (Phase 1: 1.6 ms @ 100k) |
| Memory | N × 384 × 4 bytes (15.3 MB @ 10k) |
| Build time | Embedding-bound: ~185 chunks/s on the dev machine |

**Why no vector database or ANN:** exact search costs well under 2 ms up to 100k chunks, far below the ~2–3 ms query-embedding cost. An ANN index or vector DB would add a dependency, approximation error and build parameters for no measurable gain at the plausible corpus size.

## Inputs, outputs and configuration

- **Input:** query text, then `embed([q], "query")`, then `DenseIndex.search(qvec, k, mask, min_similarity)`.
- **Output:** `[(row, cosine)]`. Evidence gets `dense_score` and `dense_rank`.
- **Configuration:** `dense.embedder`, `dense.batch_size` (8; measured, same throughput as 32 at half the peak RSS), `dense.intra_op_threads` (0 = default), `retrieval.dense_k` (50), `retrieval.dense_min_similarity`, `retrieval.on_dense_failure` (`degrade|error`), `retrieval.dense_timeout_ms`.

## Failure modes

| Failure | Result |
|---|---|
| Model files missing | `ModelNotAvailableError` (hint: `streamrag fetch-models`) |
| Pooling mismatch | `ConfigError` |
| Matrix, query or index dimension mismatch | `EmbeddingDimensionMismatchError` |
| Embed timeout | `RetrieverTimeoutError` |
| Dense unavailable in hybrid mode | Hybrid **degrades to lexical-only**: `status="degraded"`, a warning, and evidence labeled `bm25` (never mislabeled) |
| Dense unavailable in dense-only mode | Raises |

## Trade-offs

- Model quality on the Theme 4 domain is **unmeasured** (blocked).
- Embeddings are bit-identical for the same machine, config and threads. Across machines or thread counts they may differ at around 1e-7; ranking order is stabilized by the rounding.
