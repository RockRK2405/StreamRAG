# Samsung PRISM Theme 4
# Phase 3 — Corpus & Retrieval Foundation

| | |
|---|---|
| Date | 2026-10-02 |
| Status | Phase 3 complete. **OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE** |
| Code | `src/streamrag/` (~3.7k lines), `tests/` (140 tests, all passing) |
| Inputs | `PHASE_1_RESEARCH_DOSSIER.md`, `PHASE_2_SYSTEM_SPECIFICATION.md`, `docs/architecture/`, `docs/decisions/` |

**Labeling conventions.**

| Label | Meaning |
|---|---|
| **MEASURED** | Numbers produced by scripts in `research/phase3/` on the dev machine (Apple M5 Pro, macOS 26.5.1 arm64, Python 3.12.0). Raw JSON is in `research/phase3/results/`. |
| **BLOCKED** | Cannot be measured without the official corpus and gold labels |
| `[T]` | A Phase 2 *team* target. Never an official Samsung threshold. |

---

## 1. Implementation Summary

**What exists now.**

| Area | Implemented |
|---|---|
| **M0 bootstrap** | Git repo (no commits made); `pyproject.toml` (`streamrag` package, `src/` layout); isolated `.venv` (the global Python env is broken, Phase 1); `requirements.lock` (31 pinned runtime+dev packages); `requirements-research.lock`; `.env.example`; `.gitignore`; `README.md`; `configs/default.yaml` + `configs/models.yaml`; structured JSON logging; pytest |
| **M1 contracts** | Pydantic v2 models for every Phase 2 contract (below), exported to `docs/schemas/*.schema.json` (18 schemas) with a drift test |
| **Corpus engineering** | `CorpusSource` → loaders (txt, md, pdf) → normalizer → section parser → stable IDs → chunker → manifest |
| **Retrieval** | BM25 (scipy sparse) · dense (bge-small, ONNX Runtime CPU, exact search) · RRF hybrid · guarded deduplication · pluggable cross-encoder reranker · `Evidence` / `EvidenceSet` · `RetrievalService.retrieve()` / `retrieve_batch()` |
| **Index store** | Versioned, hash-verified artifacts; reuse by `(corpus_hash, index_config_hash)` |
| **Benchmark harness** | Recall@1/5/10, Success@k, MRR@10, nDCG@10, per-category and per-intent breakdowns, per-stage latency p50/p95; JSON + JSONL + CSV + run manifest; automatic **NOT REPORTABLE** stamping for fixtures |
| **CLI** | `streamrag corpus-status`, `build-index`, `search`, `bench`, `export-schemas`, `fetch-models` |
| **Research** | Chunking experiment, embedding runtime benchmark (ONNX vs PyTorch), retrieval-stack profile, retrieval comparison A–D (quality blocked) |
| **Docs** | `docs/retrieval/01–08`, `docs/architecture/07_retrieval_foundation.md` (Mermaid, parse-verified), ADR-013 plus Phase 3 notes on ADR-002/003 |

**Contracts implemented** (M1): TranscriptChunk, UtteranceEnd, SessionStart, SessionEnd, TelemetryEvent (16 event types, deterministic `session:seq` IDs), Intent, IntentSet, Evidence, EvidenceSet (+ RetrievalTrace), Claim, Citation, AnswerVersion, RetrievalRequest, RetrievalResult, BenchmarkCase, RetrievalEvalItem, CorpusChunk, CorpusManifest.

**Not implemented, by design** (Phase 4+): streaming controller, transcript simulation, multi-intent orchestration, session refinement, answer synthesis, final grounding pipeline, UI, Docker.

### Quality gate

| Gate | Status | Evidence |
|---|---|---|
| Repository structure is clean | ✅ | `README.md` layout; `src/` packages per brief §34 |
| Contracts are implemented | ✅ | `models/`, 25 contract tests, schema drift test |
| Corpus ingestion works | ✅ | Fixture, error and PDF corpora; real-layout PDF smoke set |
| Stable IDs work | ✅ | `test_ids.py`, `test_reproducibility.py` |
| Chunking is deterministic | ✅ | `test_chunking.py` (all 3 strategies) |
| Corpus manifest is generated | ✅ | `test_manifest_and_store.py` |
| BM25 works | ✅ | `test_bm25.py` |
| Dense retrieval works | ✅ | Real bge-small ONNX tests ran (not skipped). Quality is BLOCKED. |
| Hybrid retrieval works | ✅ | `test_service.py` |
| RRF works | ✅ | Exact-value tests |
| Evidence objects are traceable | ✅ | Slice-equality test against persisted documents |
| Reranker is pluggable | ✅ | Off / on / fake / timeout / real-model tests |
| Benchmark harness exists | ✅ | `test_bench.py`; CLI `bench` |
| No fabricated benchmark metrics exist | ✅ | Quality sections say BLOCKED; fixture runs stamped NOT REPORTABLE |
| No corpus facts were invented | ✅ | Fixtures are fictional and domain-unrelated; guard test forbids guide-example vocabulary in `src/` and `configs/` |
| No external knowledge enters retrieval | ✅ | `test_isolation.py`: static import scan, network-blocked run, no injection API |
| Tests pass | ✅ | **140 passed, 0 skipped** |
| Configuration is externalized | ✅ | `configs/default.yaml` (strict schema), env + CLI overrides |
| Reproducibility metadata is recorded | ✅ | Manifest, run manifest, lockfiles, pinned model revisions |
| Performance profiling exists | ✅ | `research/phase3/performance_profile.md` |

---

## 2. Corpus Status

```
$ streamrag corpus-status
OFFICIAL_CORPUS_STATUS = NOT_AVAILABLE
corpus_path = /Users/rudrakhale/Desktop/SamsungHackathon/corpus
detected = NOT_AVAILABLE
```

- I re-checked the repository, `~/Downloads` and the hackathon folder on 2026-10-02. No Theme 4 corpus is present. The only new file is a team submission `.pptx`.
- **What was done instead,** per the brief:
  - the full ingestion architecture;
  - **TEST FIXTURES** in `tests/fixtures/{corpus,corpus_pdf,corpus_errors}`. These are fictional documents (a lighthouse manual, an orchard handbook, observatory notes) deliberately unrelated to the hackathon domain. Each fixture directory has a `TEST_FIXTURE_ONLY` marker.
- **How fixtures cannot be mistaken for the corpus.** The marker:
  - sets `is_test_fixture: true` in the manifest;
  - makes `corpus-status` report `TEST_FIXTURE`, never `AVAILABLE`;
  - forces benchmark outputs to `REPORTABLE: false` with a banner.
- **Switching to the official corpus needs no code changes.** Drop the files in `./corpus` (or set `STREAMRAG_CORPUS` / `--corpus`) and run `streamrag build-index`. Native `Doc_NN` IDs and `§N` section numbers are preserved automatically.

---

## 3. Corpus Pipeline

```
CorpusSource → Loader (txt|md|pdf) → Normalizer → SectionParser → IDs → Chunker → Manifest → BM25Index / DenseIndex
```

Details: `docs/retrieval/01_corpus_pipeline.md`.

- **Discovery.** Sorted POSIX relative paths; hidden files ignored; extension allow-list; `corpus_hash` = sha256 over every file's path and bytes (plus the fixture flag).
- **Loaders.**
  - UTF-8 with a logged cp1252 fallback.
  - Markdown front matter (`id`, `title`).
  - PDF via pypdf, per page. Image-only, encrypted or unreadable PDFs raise `DocumentLoadError`; these are skipped and recorded, or raised, per `on_invalid_document`.
  - Demonstrated on real files: the **Theme 4 guide PDF itself and the Theme 2 guide are image-only** and were skipped with that reason.
- **Normalization** (every removal is logged in the document's `normalization_log`):
  - NFKC;
  - invisible characters;
  - whitespace;
  - consecutive duplicate lines;
  - repeated headers/footers and page numbers (paged sources only; top and bottom tracked separately; one edge line per side on short pages);
  - de-hyphenation;
  - joining of wrapped lines (headings, list items, tables and ALL-CAPS titles protected);
  - page joining with offsets and cross-page paragraph continuation.
- **Traceability invariant** (tested for every chunk): `document.text[char_start:char_end] == chunk.text`. Normalized texts are persisted in `documents.jsonl`, and every chunk keeps `source_path`, char span and page span.

---

## 4. Chunking Strategy

**Default (provisional): `paragraph`, target 180 / max 300 tokens (regex words), sentence overlap ≤ 40, min 25.**

- Whole paragraphs are packed into chunks.
- Oversized paragraphs are split only at sentence boundaries, with an abbreviation guard.
- A paragraph ending in `:` is glued to the list that follows it.
- A chunk never crosses a section.

**Experiment** (`research/phase3/chunking_report.md`, MEASURED, three non-official datasets):

| Finding | Result |
|---|---|
| Statement integrity | Fixed token windows end **57 of 67** (128-token) and **37 of 46** (256-token) chunks mid-sentence on real-layout PDFs. Every paragraph- or section-aware configuration ends **0**. |
| Section boundaries | 0 chunks cross sections, in every configuration |
| Truncation | 0 chunks exceed bge-small's 512-wordpiece window, including the title/section header |
| Overlap | Nearly free with paragraph packing; it only applies inside split paragraphs |

**Open:** whether 180/300, 120/200 or `section(max300)` retrieves best. That is Exp 10, **BLOCKED**.

---

## 5. Stable ID Strategy

| Object | Rule | Example |
|---|---|---|
| `document_id` | Native ID wins (front-matter `id`, or file name `Doc_<digits>…`). Otherwise `native_or_stem` (sanitized relative path), or `native_or_ordinal`. Duplicate native IDs fail fast. | `Doc_07`, `fixture_lighthouse_manual` |
| `section_id` | Native number from the source (`2.1`, `§3`). Otherwise an ordinal path, `u`-prefixed in natively numbered docs. `0` = preamble, `p<N>` = heading-less PDF page, `-2` suffix on collision. | `2.1`, `0`, `u2`, `p3` |
| `chunk_id` | Template `{document_id}§{section_id}#{part}` | `Doc_07§2.1#1` |
| Citation | Template `{document_id} §{section_id}` (the guide's `[Doc_ID §Section]`) | `Doc_07 §2.1` |

- **Determinism:** same corpus + same configuration gives identical IDs, tested across builds.
- **Deviation from Phase 2** (recorded in ADR-013): the fallback document-ID strategy is `native_or_stem` instead of an ordinal, because ordinal IDs shift when files are added. Templates are configurable, so an official convention can be adopted without code changes.

---

## 6. BM25 Implementation

`docs/retrieval/03_bm25.md`

- **Analyzer** (identical at index and query time): NFKC + lowercase → letter/number tokens → spoken number words to digits → generic English stopwords (negations kept; spoken fillers removed) → Snowball stemming.
- **Index:** precomputed BM25 weights (k1 = 1.5, b = 0.75, Lucene-style IDF) in a scipy CSC matrix. A query sums the columns of its unique query terms. Only positive scores qualify. Order is deterministic (score descending, then row).
- **API:** `build()`, `load()`, `save()`, `search()`. Evidence carries `bm25_score`, `bm25_rank`, `retrieval_method`.
- **Tests:** exact keyword, partial match, terminology variation (stemming), number words, empty and stopword-only queries, no-result queries (returns `[]`, never padding), tie determinism, mask filters, save/load round trip.

---

## 7. Dense Retrieval

`docs/retrieval/04_dense_retrieval.md`, `research/phase3/embedding_benchmark.md`

**Runtime: ONNX Runtime CPU** (`OnnxEmbedder`).
- Pooling is read from each model's own config and cross-checked against the pinned registry.
- L2 normalization; per-model prefixes; length-sorted batching.

**Vector index:** exact float32 inner product (= cosine) in numpy. Read-only matrix; deterministic top-k (scores rounded to 1e-6 for ordering).

**Embedding experiment (MEASURED, synthetic text, default threads):**

| Model / runtime | Load (s) | Query p50 (ms) | Passages/s | Peak RSS (MB) |
|---|---|---|---|---|
| **bge-small-en-v1.5 / ONNX** | 0.32 | **2.13** | **108.5** | 1120 (batch 32) |
| all-MiniLM-L6-v2 / ONNX | 0.20 | 0.97 | 214.3 | 1053 |
| all-MiniLM-L6-v2 / ONNX int8-arm64 | 0.20 | 0.84 | 189.7 | 758 |
| bge-small-en-v1.5 / PyTorch | 3.09 | 8.55 | 87.3 | 1002 |
| e5-small-v2 / PyTorch | 3.75 | 8.09 | 86.2 | 1016 |

**Correctness:** our ONNX output equals `sentence-transformers` at **cosine 1.000000** for bge-small and MiniLM (MEASURED).

**Decisions:**
- ONNX runtime (confirmed).
- **bge-small kept as default, provisionally.** Retrieval quality on the domain is **BLOCKED** (Exp 1).
- `dense.batch_size` = **8**: same build throughput as 32, at 589 vs 1093 MB peak RSS (MEASURED).

**Not used and why:**

| Option | Reason |
|---|---|
| Vector DB / ANN | Exact search costs 0.26 ms at 10k chunks (MEASURED) and 1.6 ms at 100k (Phase 1) |
| Int8 MiniLM | No latency gain; unmeasured quality risk |

---

## 8. Hybrid Retrieval

`docs/retrieval/05_hybrid_retrieval.md`, `docs/architecture/07_retrieval_foundation.md`

```
query → BM25 (top 50) ┐
      → embed → dense (top 50) → candidate union → RRF → dedup → [rerank] → top_k → EvidenceSet
```

- **Modes:** `bm25`, `dense`, `hybrid` (default; the B0 baseline per ADR-011).
- **API:** `RetrievalService.retrieve(query, RetrievalOptions)` → `EvidenceSet`; `retrieve_batch()` batches query embeddings. Filters: document IDs, section IDs or citation keys.
- **Separation:** the service knows nothing about streaming, voice, generation or UI. It has no document-injection API (tested). The LLM plays no part in retrieval.
- **Degradation is explicit.** If the dense path fails or times out, hybrid falls back to lexical-only with `status="degraded"`, a warning, and evidence labeled `bm25`. That labeling was a bug the tests caught and that is now fixed. `on_dense_failure: error` makes it raise instead.

---

## 9. RRF Fusion

- `RRF(d) = Σ 1/(k_rrf + rank)`, with `k_rrf = 60` configurable (`retrieval.rrf_k`, or per call).
- Deterministic order: RRF descending, then best single-list rank, then corpus order.
- **Why RRF** (consistent with Phase 2 and the guide's roadmap): it is rank-based, so BM25 and cosine scales never need calibration, which makes it robust on a held-out corpus. Weighted fusion is the Exp 2 ablation arm; the harness supports adding it.
- **Tests:** exact values, union, tie-break, sensitivity to `k`.
- **Comparison of BM25 vs dense vs RRF on quality: BLOCKED** (no gold labels). Latency comparison is in §14.

---

## 10. Reranking

`docs/retrieval/06_reranking.md`

- **Pluggable.** The `Reranker` protocol; the ONNX cross-encoder `cross-encoder/ms-marco-MiniLM-L6-v2` @ 233902d2 (pair tokenization, `only_second` truncation, logits).
- **Off by default.** Per ADR-003 the official minimum re-rank is RRF + dedup, which always runs.
- **Run without and with it:** `rerank: false|true` per call; `--rerank`; `bench --modes hybrid hybrid_rerank`. Requesting rerank with no model loaded raises `ModelNotAvailableError`; it is never a silent no-op. A timeout keeps the fused order with `status="partial"`.

**MEASURED:**
- **Latency**, 20 candidates at 10k synthetic chunks: 95 ms p50 / 106 ms p95 (default threads); 167 / 185 ms (2 threads).
- **Candidates reranked:** `min(rerank_k = 20, fused)`.
- **Memory:** query-process peak 1.06–1.15 GB with both models loaded (560 MB after load). Disabling the ONNX memory arena made it worse (1.86 GB) and slower, so the arena stays on.
- **Int8 cross-encoder:** slower (179 vs 167 ms p50 at 2 threads). Rejected.

**Score distribution:** fixture only and **NOT REPORTABLE**. Logits range −11.5 … 8.7, and the median top-1/top-2 gap is 18.1 over 6 queries.

**Quality gain: BLOCKED** (Exp 3).

---

## 11. Evidence Model

`docs/retrieval/07_evidence_model.md`, `docs/schemas/Evidence.schema.json`

**`Evidence`:**
- Identity: `evidence_id` (= chunk_id), `document_id`, `section_id`, `chunk_id`, `citation`.
- Context: `section_title`, `section_path`.
- `text`.
- **Traceability:** `source_path`, `char_start/end`, `page_start/end`.
- Ranking: `rank`, `score` (the value that produced the rank), `retrieval_method`.
- Score provenance: `bm25_score/rank`, `dense_score/rank`, `rrf_score`, `rerank_score`.
- Dedup outcomes: `alternates`, `overlaps_with`.
- Minimal `metadata`.

**`EvidenceSet`:** deterministic `evidence_set_id`, `items`, `token_count`, `per_intent` (reserved for Phase 5), and `trace` (status, candidate counts, `dedup_removed`, per-stage timings, warnings, index version, corpus and config hashes).

`to_retrieval_result()` produces the Phase 2 `RETRIEVAL_COMPLETED` payload.

**Tests:** serialization round trip; canonical JSON; slice equality against the persisted documents; IDs ⊆ index.

---

## 12. Benchmark Harness

`docs/retrieval/08_benchmarking.md`

| Aspect | Detail |
|---|---|
| Input | JSONL `RetrievalEvalItem`, or a directory of `BenchmarkCase` files (flattened per answerable gold intent → per-intent Recall@k) |
| Gold levels | chunk, section (default, citation keys), document |
| Metrics | Recall@1/5/10, Success@1/5/10, MRR@10, nDCG@10; per category; `all_intents_covered@5` per case; per-stage latency p50/p95/mean/min/max; index load latency |
| Outputs | `metrics.json`, `per_query.jsonl`, `per_query.csv`, `run_manifest.json` |
| Modes | `bm25` (A), `dense` (B), `hybrid` (C, B0 baseline), `hybrid_rerank` (D). Same index, queries and `top_k`; warm-up excluded. |
| Safety | `REPORTABLE: false` and a banner whenever the corpus or the dataset is a fixture |

**Tests:** hand-computed metric math; a harness run on the fixture (asserts NOT REPORTABLE and all outputs present); BenchmarkCase flattening; duplicate-ID rejection.

---

## 13. Experimental Results

| Experiment | Status | Result |
|---|---|---|
| Chunking comparison (8 configurations × 3 datasets) | **MEASURED** (structural) | §4. Paragraph/section-aware chunking preserves statements (0 mid-sentence ends); token windows do not (57/67). Default kept, provisionally. |
| Embedding runtime, ONNX vs PyTorch (5 model/runtime combinations × 2 thread settings) | **MEASURED** (latency, memory, agreement) | §7. ONNX is 4× faster per query, with identical vectors. |
| Exp A — BM25 | Quality **BLOCKED**; latency MEASURED | 0.83 ms p50 (10k chunks) |
| Exp B — Dense | Quality **BLOCKED**; latency MEASURED | 3.52 ms p50 |
| Exp C — Hybrid RRF | Quality **BLOCKED**; latency MEASURED | 5.34 ms p50 |
| Exp D — Hybrid + reranker | Quality **BLOCKED**; latency MEASURED | 101.3 ms p50 |
| Rerank score distribution | Fixture only, **NOT REPORTABLE** | §10 |

Full write-up: `research/phase3/retrieval_comparison.md`. For every quality metric it states: **"Blocked — official labeled corpus/evaluation data unavailable."**

---

## 14. Performance Results

All MEASURED on synthetic corpora (`research/phase3/performance_profile.md`). These are not official thresholds.

**Build:**

| Stage | 2,032 chunks | 9,961 chunks |
|---|---|---|
| Scan + hash / load | 6 / 2 ms | 31 / 10 ms |
| Normalize / sections / chunking | 42 / 11 / 43 ms | 211 / 58 / 215 ms |
| BM25 indexing | 1.29 s | 6.69 s |
| Embedding indexing (bge-small ONNX, batch 32) | 10.97 s (185 chunks/s) | 54.87 s (182 chunks/s) |
| Index load (with checksum verification) | 16 ms | 104 ms |
| Service init (index + both models) | 194 ms | 239 ms |
| **Index size on disk** | **9.9 MB** | **49.1 MB** (chunks 16.6, dense 15.3, documents 13.8, BM25 3.1) |

**Query, ms, p50 / p95, 9,961 chunks:**

| Stage / mode | Default threads | 2 threads (container proxy) |
|---|---|---|
| BM25 search | 0.17 / 0.26 | 0.16 / 0.22 |
| Query embedding | 2.40 / 2.53 | 2.11 / 2.28 |
| Dense search | 0.26 / 0.28 | 0.21 / 0.26 |
| RRF fusion | 0.10 / 0.11 | 0.08 / 0.09 |
| Dedup (hybrid) | 2.21 / 2.37 | 1.65 / 1.77 |
| Rerank (20 candidates) | 95.2 / 105.7 | 167.2 / 184.5 |
| **Total: bm25** | **0.83 / 0.94** | **0.82 / 0.91** |
| **Total: dense** | **3.52 / 3.66** | **2.98 / 3.18** |
| **Total: hybrid** | **5.34 / 5.60** | **4.36 / 4.71** |
| **Total: hybrid + rerank** | **101.3 / 113.0** | **172.4 / 189.8** |

**Against the Phase 2 team targets `[T]`:**
- Hybrid retrieval p95 < 50 ms (REQ-PERF-002): met (4.7–5.6 ms).
- Rerank ≤ 150 ms p95 at 20 pairs on the dev machine (REQ-PERF-004): met with default threads (106 ms); exceeded with 2 threads (185 ms).
- Container RSS ≤ 2 GB (REQ-PERF-008): met (≤ 1.3 GB peak, including builds at batch 32).

---

## 15. Failure Cases

**Bugs found by tests and experiments during Phase 3, and fixed:**

1. **A numbered list swallowed the next heading.** "1. Red… / 2. Gold… / 3. Blue…" followed by heading "2. Pruning" was treated as one list run. That dropped §2 and gave §2.1 and §2.2 the wrong parent. Fix: list runs must be numbered *sequentially*. Regression test added.
2. **Header/footer stripping removed body text on short pages.** With digits normalized, a body line repeated across short pages looked like a running header. Fix: track top and bottom separately, and use one edge line per side when a page has fewer than 6 lines.
3. **Degraded hybrid mislabeled its evidence.** When dense failed, lexical-only results were still labeled `hybrid_rrf`. Fix: label by the *effective* method.
4. **The anti-hardcoding guard fired on a docstring.** An example ID in `ids.py` used `Doc_12`, which appears in the guide. Replaced with a neutral example. The guard works as intended.
5. **First-run PyTorch "load time" included the weight download** (~50 s). It was re-measured with a warm cache and annotated, so download time is not reported as load time.

**Explicit error behavior (tested):**

| Condition | Behavior |
|---|---|
| Missing corpus | `CorpusNotFoundError` |
| Empty corpus | `EmptyCorpusError` |
| Empty, whitespace-only, image-only or unreadable document | Skipped with a recorded reason, or `DocumentLoadError` |
| Duplicate native ID | `CorpusIntegrityError` |
| Empty query | `InvalidQueryError` |
| Missing model | `ModelNotAvailableError` |
| Corrupt or missing artifact | `IndexCorruptError` |
| Index version mismatch | `IndexVersionMismatchError` |
| Dimension mismatch | `EmbeddingDimensionMismatchError` |
| Dense timeout | Degraded, lexical-only, flagged |
| Reranker timeout | Partial, fused order kept, flagged |
| Filters match nothing | Empty, flagged |
| No-match query | Empty, never padded |
| Invalid config | `ConfigError` |
| CLI errors | Exit code 2, with the error class |

**Remaining known weak spots:**
- Heading heuristics are unverified on the official layout.
- Scanned PDFs need OCR, which is not implemented.
- Near-duplicate thresholds are validated on fixtures only.

---

## 16. Reproducibility

- **Pinned:** `requirements.lock` (runtime + dev closure), `requirements-research.lock` (full freeze incl. torch); model revisions plus per-file sha256 in `models/<name>/MODEL_INFO.json`; `configs/models.yaml`.
- **Manifest** (`indexes/<corpus_hash>-<index_config_hash>/manifest.json`):
  - `index_version` 3.0;
  - `corpus_version` (corpus hash);
  - `is_test_fixture`;
  - `generated_at`;
  - source files with sha256, status and reason;
  - document, section and chunk counts;
  - chunking, normalization, section, ID, index-text and BM25 configurations;
  - embedding model (repo, revision, file hashes, dimension, pooling, max length, truncation count, build batch size);
  - `index_config_hash`;
  - per-artifact sha256;
  - environment (Python, platform, package versions, git SHA and dirty flag);
  - `content_hash` (everything except timestamps and environment).
- **Run manifest** (`runs/<id>/run_manifest.json`): command, config hash, index hashes, model info, eval-data sha256, environment, `REPORTABLE` flag.
- **Tests:** two independent builds give identical `content_hash` and identical artifact hashes, for both the hashing embedder and **bge-small ONNX**. Loading verifies every artifact hash.
- **Caveats:**
  - Embedding bits are guaranteed identical for the same machine, configuration and thread count. Across CPUs or thread counts, ONNX results can differ at about 1e-7. Ranking order is stabilized by rounding scores to 1e-6.
  - The lockfile was produced on macOS arm64. The container build will need a Linux lock.
  - There are no commits yet, so `git_sha = "uncommitted"`.

---

## 17. Known Limitations

1. **No official corpus, so no retrieval-quality evidence.** Every quality-dependent choice is provisional: embedder model, chunk size, reranker adoption, dedup thresholds, BM25 parameters.
2. **Format coverage** is txt, md and pdf only. No DOCX, HTML or OCR yet; image-only PDFs are skipped, as the Theme 4 guide itself would be.
3. **Heading and section heuristics** are generic and lexical. Unusual layouts (tables, multi-column PDFs) are untested.
4. **Dedup costs ~2 ms** (Python, pairwise). It is the largest non-model stage in hybrid mode, and an optimization target.
5. **BM25 build is Python-bound** (6.7 s per 10k chunks). Fine at build time.
6. **Memory:** ~560 MB after loading both models, and about 1.1 GB peak during reranked queries.
7. **Latency is measured only on the M5 Pro.** The 2-thread rows are a proxy; judge hardware and container overhead are unmeasured.
8. **`HashingEmbedder` lives in `src/`** for tests and offline runs. It is labeled TEST/OFFLINE ONLY, and indexes record `runtime: hashing-test-only`, but it must never be configured for real runs.
9. **The sentence splitter is regex-based** (with an abbreviation guard). It is a proxy for statement boundaries.
10. **Session-level evidence provenance** (spec §11.3 `hits[]` across retrievals) is intentionally deferred to the session store (Phase 6).

---

## 18. Phase 4 Prerequisites

**Ready for Phase 4:**
- A stable retrieval API (`retrieve`, `retrieve_batch`, filters, deterministic IDs and evidence);
- the event/telemetry contracts;
- the JSONL sink;
- config and manifest machinery;
- the benchmark harness;
- 140 green tests.

**Needed (in parallel; they block *quality* claims, not Phase 4 plumbing):**
1. **The official Theme 4 corpus** (Q1). Without it, nothing in Phases 4–8 can produce reportable quality numbers.
2. **Gold labels authored from that corpus** (spec §22) for Exp 1/2/3/10, and later G2–G5.
3. **The deadline** (Q6), which sets the scope cut.

**Small Phase 3 → Phase 4 hand-offs:**
- Expose a BM25 IDF lookup for the controller's anchor-strength signal. The vocabulary already exists in `BM25Index.vocab`; `df` can be derived from the matrix.
- Make the retrieval executor awaitable from asyncio, so `RetrievalService` calls run via `asyncio.to_thread`.
- Optionally vectorize dedup.

**Recommended Phase 4 order** (streaming simulation → controller → early retrieval):
1. **Async event bus + session actor skeleton.** One task and mailbox per session, emitting `TelemetryEvent`s through the existing sink. Commits serialized (ADR-010).
2. **Transcript Streamer.** JSONL replay with real and virtual clocks, a speed factor, cumulative-to-delta conversion and end-of-file handling.
3. **Chunk Manager.** Idempotency, reordering window, revisions, segmentation (boundary, dangling, stability timers, force-close).
4. **Intent Analyzer (rules).** Dialog acts, slots, anchor strength from corpus IDF, and the turn-type hypothesis.
5. **Retrieval Controller (rules)** with the reason-code vocabulary, plus the **Query Ledger** (dedup and reuse). Dispatch through `RetrievalService.retrieve_batch` in a thread.
6. **Turn Orchestrator (LISTENING → FINALIZING only).** No synthesis yet: emit `TURN_COMPLETED` with `retrieval_events` and the evidence set.
7. **Streaming metrics in the harness:** early-retrieval rate (G2), false-trigger rate, lead time, TTFR, retrieval calls per turn. Add streaming benchmark-case structure (utterances and chunkings; gold evidence once the corpus exists).
8. **Experiments 5, 7 and 8:** static vs early retrieval; retrieve-always vs controller; rules vs a prototype-classifier controller.
