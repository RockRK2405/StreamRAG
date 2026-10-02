# 08: Benchmarking and Reproducibility

**Code:** `bench/` (dataset, metrics, harness), `corpus/manifest.py`, `provenance.py`
**CLI:** `streamrag bench`

## Purpose

Measure retrieval quality and latency reproducibly, and make it **impossible to report fixture results by accident**.

## Inputs

Either of:
- **JSONL** of `RetrievalEvalItem` (`item_id`, `query`, `gold[]`, `gold_level`, optional `case_id`, `intent_id`, `category`, `split`, `is_fixture`);
- a **directory of `BenchmarkCase` JSON files** (spec §22.2). Each answerable gold intent becomes one item, giving **per-intent Recall@k**.

Gold refs are citation keys (`gold_level=section`, the default), chunk IDs, or document IDs. Gold must be authored from the official corpus. `BenchmarkCase` rejects an answerable intent without gold evidence.

## Metrics (over the first k returned evidence items, mapped to the gold level)

| Metric | Definition |
|---|---|
| Recall@k (k = 1, 5, 10) | `|gold ∩ keys[:k]| / |gold|` |
| Success@k | Any gold item in the top k |
| MRR@10 | Reciprocal rank of the first gold hit |
| nDCG@10 | Binary relevance; each gold key credited once |

Breakdowns: per category, `all_intents_covered@5` per case (multi-intent), and per-stage latency p50/p95/mean/min/max. Index load latency is also recorded.

## Modes (experiments A–D)

`bm25`, `dense`, `hybrid` (RRF, the B0 baseline per ADR-011), and `hybrid_rerank`. Every mode uses the same index, queries and `top_k = 10`, after a warm-up query.

## Outputs (`runs/<run_id>/`)

| File | Contents |
|---|---|
| `metrics.json` | `REPORTABLE`, banner, per-mode metrics and latency |
| `per_query.jsonl` / `per_query.csv` | Per-query results |
| `run_manifest.json` | Run ID, command, config hash, index path, version and content hash, corpus hash, `corpus_is_test_fixture`, embedding model (repo, revision, file sha256s), reranker, eval-data path, sha256 and item counts, environment (Python, platform, package versions, git SHA, dirty flag) |

## Not-reportable rule

`REPORTABLE` is `false` whenever the index was built from a `TEST_FIXTURE_ONLY` corpus, or any evaluation item has `is_fixture: true`. The CLI then prints:

```
*** TEST FIXTURE / NON-OFFICIAL DATA - NOT A BENCHMARK RESULT - DO NOT REPORT ***
```

## Reproducibility

Same corpus + configuration + model + code gives identical artifacts. This is tested for the hashing embedder and for bge-small ONNX on the same machine.

| Record | Where |
|---|---|
| `corpus_hash` | sha256 over every file's path and bytes, plus the fixture flag |
| `index_config_hash` | Index-affecting settings only; changing `top_k` does not force a rebuild |
| `config_hash` | Full configuration |
| `INDEX_VERSION` (3.0) | Manifest |
| `content_hash` | Manifest, over everything except timestamps and environment |
| Per-artifact sha256 | Verified on every load |
| Model revisions and file sha256 | `MODEL_INFO.json` |

## Status

The harness is implemented and tested. **Quality results are blocked** until the official corpus and labels exist; see `research/phase3/retrieval_comparison.md`.
