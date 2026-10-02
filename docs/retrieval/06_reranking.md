# 06: Reranking (pluggable, optional)

**Code:** `retrieval/rerank.py`, reranker handling in `retrieval/service.py`
**ADR:** ADR-003

## Purpose

Optionally reorder the head of the fused list with a cross-encoder. The **official minimum re-rank is RRF + dedup**, which always runs. The cross-encoder is an ablation-gated enhancement.

## Architecture

- **`Reranker` protocol:** `score(query, passages) -> np.ndarray`.
- **`OnnxCrossEncoder`:** `cross-encoder/ms-marco-MiniLM-L6-v2` @ 233902d2 (Apache-2.0).
  - Pair tokenization with `only_second` truncation, at 512 tokens.
  - ONNX CPU; logits are used as scores.
  - Passages are the chunk `index_text`, so the section header gives context.
- **Service integration:**
  - The top `rerank_k` candidates (default 20) are scored and reordered by score descending, then by fused order.
  - Candidates beyond `rerank_k` keep their fused order.
  - Reranked evidence gets `rerank_score` and `retrieval_method = "*_rerank"`.

**Running with and without the reranker:**

| How | Effect |
|---|---|
| `rerank.enabled: false` (default) | Off |
| `--rerank` / `RetrievalOptions(rerank=True)` | On for this call. Raises `ModelNotAvailableError` if no reranker is loaded, never a silent no-op. |
| `streamrag bench --modes hybrid hybrid_rerank` | Runs the ablation directly |

## Measurements

| Measurement | Value | Status |
|---|---|---|
| Rerank latency, 20 candidates, 10k-chunk synthetic corpus | 95 ms p50 / 106 ms p95 (default threads); 167 / 185 ms (2 threads) | Measured |
| Candidates reranked | `min(rerank_k, fused candidates)` (20 by default; 14 on the 14-chunk fixture) | Measured |
| Int8-arm64 variant | Slower than fp32 with 2 threads (179 vs 167 ms p50). **Rejected.** | Measured |
| Memory | Query-process peak RSS 1.06–1.15 GB with bge-small + cross-encoder loaded (560 MB after load). Disabling the ONNX memory arena made it worse (1.86 GB). | Measured |
| Score distribution | Fixture only, **NOT REPORTABLE**: logits −11.5 … 8.7, median top-1/top-2 gap 18.1 (6 queries, `research/phase3/results/rerank_fixture_scores.json`) | Fixture only |
| Quality gain | **Blocked** (Exp 3 needs official labels) | Not available |

## Configuration

`rerank.enabled`, `rerank.model`, `rerank.rerank_k` (20), `rerank.timeout_ms` (600), `rerank.batch_size`.

## Failure modes

| Case | Behavior |
|---|---|
| Model missing | `ModelNotAvailableError` |
| Timeout | Fused order kept, `status="partial"`, warning `rerank_timeout_fused_order_kept` |

## Trade-offs

The reranker costs 20–35× the rest of the retrieval path, and it is trained on web QA (MS MARCO), which may not transfer to policy text. Phase 2's latency model only allows it with rerank-on-stability during speech. It stays **off by default** until Exp 3 shows a gain.
