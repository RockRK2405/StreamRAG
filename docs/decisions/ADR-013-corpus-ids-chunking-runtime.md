# ADR-013: Corpus IDs, Chunking Default and Embedding Runtime (Phase 3 implementation decisions)

- **Status:** Accepted (Phase 3). Chunk size remains provisional, pending Exp 10.
- **Date:** 2026-10-02
- **Refines:** Phase 2 spec §10.1, ADR-001, ADR-002

## Context

Phase 3 implemented the corpus and retrieval foundation without the official corpus. Several Phase 2 defaults had to be made concrete, and some were changed on the basis of measurements (`research/phase3/`).

## Decisions

1. **Document IDs: `native_or_stem` by default.** Phase 2 §10.1 specified an ordinal fallback (`Doc_<n>`); this is a recorded deviation.
   - Native IDs (front matter, or a `Doc_<digits>` file name) still always win.
   - The fallback is now the sanitized relative path, because ordinal IDs shift when a file is added. A stable citation key is worth more than uniform naming.
   - `native_or_ordinal` remains available via config.
2. **Chunk ID and citation formats stay as in Phase 2:** `{document_id}§{section_id}#{part}` and `{document_id} §{section_id}`. Both are configurable templates, so an official convention can be adopted without code changes.
3. **Section IDs:** native numbers win. Derived ordinal paths get a `u` prefix in natively numbered documents, `0` marks the preamble, and heading-less PDFs get `p<N>`. Sequential-numbering list detection prevents lists from swallowing headings (a regression-tested bug).
4. **Chunking default: `paragraph`, 180/300 tokens, sentence overlap 40.** This is provisional. Measured on the real-layout PDF smoke set:
   - 0 chunks end mid-sentence (fixed token windows end 57 of 67 mid-sentence);
   - 0 chunks cross a section;
   - 0 truncations at 512 wordpieces.

   Exp 10 decides between this and `section(max300)` or `paragraph(120/200)` on the real corpus.
5. **Embedding runtime: ONNX Runtime CPU** (torch is not a runtime dependency). Measured against PyTorch for bge-small:
   - 4× lower query latency (2.1 vs 8.6 ms);
   - about 25% higher throughput;
   - load 10× faster;
   - embeddings identical (cosine 1.000000).
6. **`dense.batch_size` = 8.** Same build throughput as 32, at about half the peak RSS (589 vs 1093 MB). The batch size is part of `index_config_hash` and is recorded in the manifest.
7. **Rejected after measurement:** the int8-arm64 MiniLM embedder (no latency gain at default threads; unmeasured quality risk), the int8-arm64 cross-encoder (slower than fp32 with 2 threads), and disabling the ONNX memory arena (1.86 GB vs 1.08 GB peak, and slower).

## Consequences

- Indexes built before decision 6 hash differently, so they are rebuilt automatically.
- All quality-dependent choices (embedder model, chunk size, reranker adoption, dedup thresholds) remain open until official labels exist.
