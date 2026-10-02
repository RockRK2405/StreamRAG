# Phase 3 — Retrieval Comparison (Experiments A–D)

| Exp | Configuration (identical index, queries, top_k = 10) |
|---|---|
| A | BM25 only (`--modes bm25`) |
| B | Dense only (`--modes dense`, bge-small ONNX) |
| C | Hybrid RRF (`--modes hybrid`, k_rrf = 60). This is the **B0 baseline** (ADR-011). |
| D | Hybrid RRF + cross-encoder (`--modes hybrid_rerank`, rerank_k = 20) |

## Retrieval quality (Recall@1/5/10, MRR@10, nDCG@10)

**Blocked: the official labeled corpus and evaluation data are unavailable.**

No quality numbers are reported for any experiment. The harness is implemented and tested (`streamrag bench`). Runs on test fixtures are automatically stamped `REPORTABLE: false`, and fixture metrics are deliberately not reproduced here.

**To run on the real data,** once `./corpus` holds the official corpus and `eval/` holds gold labels authored from it (spec §22):

```bash
.venv/bin/streamrag build-index
.venv/bin/streamrag bench --eval eval/<dataset> --modes bm25 dense hybrid hybrid_rerank
```

Outputs go to `runs/<run_id>/`:

| File | Contents |
|---|---|
| `metrics.json` | Per-mode Recall@k, Success@k, MRR@10, nDCG@10, per-category and per-intent breakdowns, latency percentiles |
| `per_query.jsonl` / `.csv` | Per-query results |
| `run_manifest.json` | Corpus, index and config hashes, model revisions, environment |

## What can be compared now (measured; synthetic corpus, 9,961 chunks)

| Exp | Mode | Query latency p50 / p95 (default threads) | 2 threads p50 / p95 | Index components needed | On-disk size of those components |
|---|---|---|---|---|---|
| A | bm25 | 0.83 / 0.94 ms | 0.82 / 0.91 ms | BM25 | 3.1 MB |
| B | dense | 3.52 / 3.66 ms | 2.98 / 3.18 ms | dense matrix + bge-small (128 MB model) | 15.3 MB |
| C | hybrid | 5.34 / 5.60 ms | 4.36 / 4.71 ms | both | 18.4 MB |
| D | hybrid + rerank | 101.3 / 113.0 ms | 172.4 / 189.8 ms | both + cross-encoder (88 MB model) | 18.4 MB |

**Memory.** The process loads all models at start-up, so memory is not separable per mode. The measured figures:
- 560 MB RSS after loading index + bge-small + cross-encoder (2k chunks);
- 1.06–1.15 GB peak during D-mode queries;
- a BM25-only process would not need the ONNX models.

Source: `performance_profile.md` / `results/retrieval_profile.json`.

## Interpretation (without quality data)

- Latency rules out none of A–C for streaming use: all run in under 6 ms.
- D costs 20–35× more, and needs Exp 3 on real labels to justify itself. Rerank stays **off by default** (ADR-003) until then.
- B and C need the dense model at query time (~2–3 ms per query embedding). A needs no model at all, which matters for the degraded mode (`on_dense_failure: degrade`).
