# StreamRAG — Samsung PRISM GenAI Hackathon 2026–27, Theme 4: Streaming Live RAG

**Current phase: Phase 5, multi-intent decomposition, parallel per-intent retrieval and evidence fusion**, built on the Phase 4 streaming engine and the Phase 3 retrieval foundation.

Implemented:
- corpus ingestion with stable citation IDs;
- BM25 + dense hybrid retrieval with RRF fusion, deduplication and an optional reranker;
- **incremental transcript streaming** with a WAIT / RETRIEVE / SKIP **retrieval controller**;
- a versioned **query ledger**, with stale-query handling;
- async retrieval;
- structured telemetry and deterministic **replay**;
- **multi-intent decomposition** (rule-first, validated), versioned intent sets with **delta retrieval** while the user speaks, parallel per-intent retrieval and **intent-aware evidence fusion** into a `UnifiedEvidenceSet`.

Not implemented yet: session refinement / late-detail updates (Phase 6), answer generation and grounding (Phase 7). See `PHASE_2_SYSTEM_SPECIFICATION.md` for the full design.

> **Official corpus status: NOT_AVAILABLE.** No Theme 4 corpus has been supplied yet. Everything under `tests/fixtures/` is a **TEST FIXTURE** (synthetic, fictional) used only to test software. No retrieval-quality results exist yet. Check the status with `streamrag corpus-status`.

## Setup (Python ≥ 3.11, CPU only)

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.lock && .venv/bin/pip install -e . --no-deps
```

```bash
.venv/bin/streamrag fetch-models bge-small-en-v1.5 ms-marco-minilm-l6-v2
```

`fetch-models` downloads the pinned ONNX models into `./models/` at build time; this is the only step that uses the network. The revisions are in `configs/models.yaml`.

## Using the official corpus (no code changes)

Put the corpus documents (`.txt`, `.md`, `.pdf`) in `./corpus/`, or point `STREAMRAG_CORPUS` or `--corpus` at their directory. Then:

```bash
.venv/bin/streamrag corpus-status
```

```bash
.venv/bin/streamrag build-index
```

```bash
.venv/bin/streamrag search "your question" --mode hybrid
```

```bash
.venv/bin/streamrag bench --eval eval/<labels> --modes bm25 dense hybrid hybrid_rerank
```

- `corpus-status` prints `OFFICIAL_CORPUS_STATUS = AVAILABLE`.
- `build-index` writes `indexes/<corpus_hash>-<config_hash>/` with its manifest.
- `search` takes `--rerank` and `--json`.
- `bench` writes `runs/<run_id>/` (metrics.json, per_query.jsonl/csv, run_manifest.json).

Native document IDs (e.g. files named `Doc_12_….pdf`, or a front-matter `id:`) and native section numbers (`§4`, `2.1`) are preserved automatically. Citations render as `Doc_ID §Section`.

## Streaming (Phase 4)

```bash
.venv/bin/streamrag stream --text "I need | information | about the fog signal | during a storm" --interval-ms 300 --trace trace.jsonl
```

```bash
.venv/bin/streamrag replay trace.jsonl
```

- `stream` prints controller decisions, query versions, retrievals and lead time. Add `--mode realtime` for the asyncio wall-clock mode, or `--policy end_only|every_chunk` for the ablation baselines.
- `--multi-intent` (Phase 5) decomposes each utterance into intents, retrieves per intent (only new or changed intents while the user speaks) and prints the fused evidence.
- `replay` re-runs the trace. A virtual trace must match exactly. A realtime trace must match in behavior (decisions, queries, retrieval outcomes), because its timestamps carry real jitter.

Dev-suite benchmarks (fixture domain; NOT official results):

```bash
.venv/bin/python research/phase4/run_streaming_benchmarks.py --index-root /tmp/idx
```

```bash
.venv/bin/python research/phase5/run_multi_intent_benchmarks.py --index-root /tmp/idx5
```

## Trying it on the test fixture

```bash
.venv/bin/streamrag build-index --corpus tests/fixtures/corpus
```

```bash
.venv/bin/streamrag search "how high are wicks trimmed" --corpus tests/fixtures/corpus
```

Any fixture-based index or benchmark is labeled `TEST FIXTURE … NOT A BENCHMARK RESULT`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests that need the downloaded models skip automatically if `./models` is absent.

## Layout

```
configs/           default.yaml (all tunables), models.yaml (pinned model registry)
src/streamrag/
  config/          typed config loading + hashes
  models/          data contracts (pydantic) -> docs/schemas/*.schema.json
  corpus/          source, loaders, normalization, sections, IDs, chunker, manifest
  retrieval/       analyzer, BM25, embedders (ONNX), dense index, RRF, dedup, rerank, index store, service API
  bench/           evaluation datasets, metrics, retrieval + streaming harnesses
  streaming/       chunk manager, simulator, schedulers, async executor, session, metrics, CLI printer
  controller/      act classifier, signals, query builder, WAIT/RETRIEVE/SKIP policies (+ ablation baselines)
  ledger/          query versions, lineage, stale evidence
  replay/          deterministic trace replay
  intents/         intent decomposition, tracker (versions/deltas), per-intent queries, validation, optional LLM check
  multi_retrieval/ per-intent retrieval (sequential/parallel/batched) + streaming multi-intent coordinator
  fusion/          cross-intent dedup, fusion strategies, intent-aware rerank, conflicts -> UnifiedEvidenceSet
  telemetry/       structured logging, JSONL event sink, timing
  tools/           build-time model download (the only network code)
tests/             unit/integration tests; tests/fixtures = TEST FIXTURES only
research/          phase1–5 measurements and reports
docs/              architecture diagrams, ADRs, retrieval docs, JSON schemas
```

## Key documents

| Document | Contents |
|---|---|
| `PHASE_5_MULTI_INTENT_REPORT.md` | Phase 5: multi-intent decomposition, delta retrieval, fusion; dev-suite results |
| `PHASE_4_STREAMING_REPORT.md` | Phase 4: streaming engine, controller, ledger; dev-suite results |
| `PHASE_3_RETRIEVAL_REPORT.md` | Phase 3: what was built, what was measured, what is blocked |
| `PHASE_2_SYSTEM_SPECIFICATION.md` | Full system specification |
| `PHASE_1_RESEARCH_DOSSIER.md` | Research dossier |
| `docs/retrieval/`, `docs/streaming/`, `docs/multi_intent/` | Component docs (Phases 3, 4, 5) |
| `docs/decisions/` | ADRs |
| `research/phase3/` | Chunking, embedding, profiling and comparison reports |
