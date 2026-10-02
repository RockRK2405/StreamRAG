# ADR-001: Retrieval Strategy

- **Status:** Accepted (Phase 2)
- **Date:** 2026-10-02
- **Spec:** §10, §12

## Context

- The guide mandates dense/sparse hybrid scoring plus re-rank and dedup [G§2 p2], and names RRF in its roadmap [G§7 p5].
- Spoken queries carry exact entities and numbers (which favor lexical search) *and* paraphrase (which favors dense search).
- Corpus size is unknown. We designed for 10 to 100k chunks.
- Measured on the dev machine: BM25 0.04–0.4 ms; exact dense search 0.01–1.6 ms; 100k × 384 float32 = 154 MB (`research/phase1`).

## Decision

1. Each retrieval runs **BM25 (own scipy-sparse implementation) and dense search (exact numpy inner product)**, each producing a top-50 list.
2. The two lists are fused with **RRF (k=60)**.
3. Retrieval is **in-process and in-memory**, with no vector database. Index arrays are read-only and cached by corpus hash.
4. Chunking is **section-bounded**, ~200–400 tokens with 15% overlap within a section. Embeddings use a contextual header (`title > heading: text`).

## Alternatives

| Alternative | Why not chosen |
|---|---|
| BM25 only | Misses paraphrase |
| Dense only | Misses entities and numbers (kept as an ablation) |
| Weighted score fusion | Needs α tuning per corpus (kept as an ablation) |
| SPLADE | Heavy expansion on CPU |
| ColBERT / multi-vector | Index ~100× larger; heavy |
| FAISS, Chroma, Qdrant, LanceDB | Unnecessary at this scale; extra dependencies or services |

## Why selected

It satisfies the official hybrid requirement with the fewest moving parts. It needs no calibration between score scales, and RRF is robust on a held-out corpus. Latency is negligible relative to LLM calls.

## Trade-offs

- Exact search is O(N·d). It would need ANN far beyond ~1M chunks, which is out of the expected range.
- Our own BM25 means owning the tokenizer and its tests.

## Consequences

- Retrieval can run speculatively during speech at almost no cost.
- Exp 1 and Exp 2 compare hybrid against BM25-only and dense-only (REQ-RET-008).
- Index build time is dominated by embedding throughput (ADR-002).
