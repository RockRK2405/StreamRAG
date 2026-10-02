# 05: Hybrid Retrieval, RRF and Deduplication

**Code:** `retrieval/service.py`, `retrieval/fusion.py`, `retrieval/dedup.py`
**ADRs:** ADR-001, ADR-009

## Purpose

Combine lexical and dense evidence into one deterministic ranked list. This is the official "dense/sparse hybrid scoring + re-rank & deduplicate" stage.

## Pipeline (per query)

```
query ─► BM25 (top lexical_k) ─┐
      └► embed ► dense (top dense_k) ─► candidate union (by chunk) ─► RRF ─► dedup ─► [rerank] ─► top_k ─► EvidenceSet
```

## RRF

`RRF(d) = Σ_lists 1 / (k_rrf + rank)`, with 1-based ranks and `k_rrf = 60` (configurable).

**Why RRF:** it is rank-based, so BM25 scores and cosines never need calibrating against each other. That makes it robust on a held-out corpus, which is why the guide's roadmap names it. Weighted score fusion needs an α tuned per corpus; it remains an Exp 2 ablation arm.

**Ordering:** RRF descending, then best single-list rank, then corpus order (deterministic).

## Deduplication (conservative; the best-ranked member is kept and others become `alternates`)

1. **Exact duplicate text** (same text hash): merged.
2. **Overlapping chunks of one section:** merged only if the overlap is ≥ `overlap_merge_ratio` (80%) of the shorter chunk. Otherwise both are kept and annotated in `overlaps_with`.
3. **Near-duplicates:** merged only if all three hold:
   - cosine ≥ `near_dup_cosine` (0.97);
   - word-set Jaccard ≥ `near_dup_jaccard` (0.9);
   - the negation and number tokens are identical.

   The guard means "X is permitted" and "X is **not** permitted", or "40 crates" and "50 crates", are **never** merged. Both cases are tested.

The thresholds are Phase 2 values. They are validated **only on fixtures**, and are configurable pending real-corpus calibration.

## API

```python
svc = RetrievalService.from_config(cfg)                       # loads index + query embedder (+ reranker if enabled)
es  = svc.retrieve("query", RetrievalOptions(mode="hybrid", top_k=10, rerank=False,
                                             filters=RetrievalFilters(document_ids=["Doc_07"])))
ess = svc.retrieve_batch(["q1", "q2"])                        # one batched query embedding
```

The service knows nothing about streaming, voice, generation or UI. It has no method for adding documents, and that is tested.

## Configuration

`retrieval.mode` / `top_k` / `lexical_k` / `dense_k` / `rrf_k`, plus `dedup.*`.

## Failure modes

| Case | Behavior |
|---|---|
| Empty query | `InvalidQueryError` |
| Filters match nothing | `status="empty"`, warning `filters_matched_no_chunks` |
| Dense failure | Lexical-only, `status="degraded"` (or an error, per config) |
| No results | `status="empty"` (never padded with unrelated evidence) |

## Performance (measured, 10k synthetic chunks)

| Setting | Hybrid total p50 / p95 |
|---|---|
| Default threads | 5.3 / 5.6 ms |
| 2 threads | 4.4 / 4.7 ms |

Dedup (~2 ms) is the largest non-model stage, and is an optimization target.

## Trade-offs

RRF discards score magnitudes. A very confident single-retriever hit can be outranked by an item that is moderately ranked in both lists. Exp 2 measures that against weighted fusion on real labels.
