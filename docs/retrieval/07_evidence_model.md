# 07: Evidence Model

**Code:** `models/evidence.py`, `models/retrieval.py`
**Schemas:** `docs/schemas/Evidence.schema.json`, `EvidenceSet.schema.json`

## Purpose

One canonical, serializable, traceable object for "a corpus chunk retrieved for a query".

## `Evidence` fields

| Field | Meaning |
|---|---|
| `evidence_id` | Equals `chunk_id` (one chunk is one evidence object) |
| `document_id`, `section_id`, `chunk_id` | Corpus identity |
| `citation` | `"Doc_ID §Section"` |
| `section_title`, `section_path` | Context |
| `text` | Exact chunk text |
| `source_path`, `char_start`, `char_end`, `page_start`, `page_end` | **Traceability** into `documents.jsonl` and the source file |
| `rank`, `score`, `retrieval_method` | Final position. `score` is the value that produced the rank: rerank > rrf > single-list score. |
| `bm25_score/rank`, `dense_score/rank`, `rrf_score`, `rerank_score` | Score provenance; `null` when a stage did not produce one |
| `alternates`, `overlaps_with` | Dedup outcomes, kept traceable |
| `metadata` | Document title, token count, part |

## `EvidenceSet` fields

| Field | Meaning |
|---|---|
| `evidence_set_id` | Deterministic: hash of query + resolved options + index content hash |
| `query`, `items` | Ranked evidence |
| `token_count` | Sum of chunk tokens |
| `per_intent` | Reserved for multi-intent fusion (Phase 5) |
| `trace: RetrievalTrace` | `request_id`, mode, `status` (`ok`, `partial`, `degraded`, `empty`), `rerank_applied`, candidate counts, `dedup_removed`, per-stage `timings_ms`, `warnings`, `index_version`, `corpus_hash`, `index_config_hash` |

`RetrievalService.to_retrieval_result(es)` produces the `RetrievalResult` telemetry payload (spec §6.3, `RETRIEVAL_COMPLETED`).

## Flow

```
CorpusChunk (index, read-only)
  ─► Candidate (row + per-list ranks/scores)
  ─► RRF / dedup / rerank
  ─► Evidence (+ trace)
  ─► EvidenceSet
```

Later phases (session evidence store, grounding) consume `Evidence` by `evidence_id` and cite with `citation`.

## Guarantees (tested)

- Serialization round-trips exactly. `canonical_json()` is deterministic (sorted keys).
- Every returned evidence slice equals the persisted normalized document text at `[char_start:char_end]`.
- Every `evidence_id` exists in the loaded index.
- There is no free-form evidence: evidence can only come from `RetrievalService`, which only reads the index.

## Trade-offs

Session-level provenance (`hits[]` across many retrievals, `cited_in_versions`) from spec §11.3 is intentionally **not** in this per-query object. It belongs to the session evidence store (Phase 6), keeping the retrieval layer stateless.
