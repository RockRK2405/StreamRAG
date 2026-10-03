# StreamRAG — Samsung PRISM GenAI Hackathon 2026–27, Theme 4: Streaming Live RAG

**Current phase: Phase 8, the streaming runtime** (asynchronous orchestration: concurrent workers, cancellation, timeouts, retries, backpressure, stale-result protection, degraded modes, replay), running the Phase 7 grounded answers, the Phase 6 adaptive session, Phase 5 multi-intent retrieval, the Phase 4 streaming engine and the Phase 3 retrieval foundation.

Implemented:
- corpus ingestion with stable citation IDs;
- BM25 + dense hybrid retrieval with RRF fusion, deduplication and an optional reranker;
- **incremental transcript streaming** with a WAIT / RETRIEVE / SKIP **retrieval controller**;
- a versioned **query ledger**, with stale-query handling;
- async retrieval;
- structured telemetry and deterministic **replay**;
- **multi-intent decomposition** (rule-first, validated), versioned intent sets with **delta retrieval** while the user speaks, parallel per-intent retrieval and **intent-aware evidence fusion** into a `UnifiedEvidenceSet`;
- **adaptive sessions** (`session.enabled: true`): late details, retractions and corrections across turns become typed context changes; a delta planner re-queries only the affected needs (or reuses earlier evidence); evidence and claims carry lifecycles; the answer is a versioned, claim-level state with diffs and minimal-regeneration markers (`docs/session/`, ADR-016, `PHASE_6_ADAPTIVE_RAG_REPORT.md`).

- **grounded answers** (`generation.enabled: true`): evidence-derived claim plans, generation by a local LLM (Ollama, structured JSON, extractive fallback), entailment verification of every claim, citations to the supporting sentence of the indexed chunk, repair / bounded retrieval fallback for unsupported claims, drafts while the user speaks and validated finals per turn (`docs/answer/`, ADR-017, `PHASE_7_GROUNDED_GENERATION_REPORT.md`). See `PHASE_2_SYSTEM_SPECIFICATION.md` for the full design.

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

## Streaming runtime (Phase 8)

```bash
.venv/bin/streamrag --set generation.enabled=true stream --session --runtime --interval-ms 300 --text "Tell me the eligibility requirements | and the application process | for the permit." --corpus tests/fixtures/corpus_grounding
```

- `--runtime` runs the turn through `streamrag.runtime.StreamingRuntime`: one actor per session on the event loop, lexical / dense retrieval subtasks in parallel on bounded worker pools, answers off the loop, cooperative cancellation of superseded work, bounded coalescing input queues (`runtime:` in `configs/default.yaml`, `docs/runtime/`, ADR-018).
- In code: `rt = await StreamingRuntime(cfg, stack).start()`, `rt.start_session()`, `rt.push_transcript_delta(...)`, `async for ev in rt.get_events(sid)`, `await rt.shutdown()`.
- Runtime traces replay exactly from the event log (`streamrag replay trace.jsonl`).

```bash
.venv/bin/python research/phase8/run_runtime_benchmarks.py --index-root /tmp/idx8
```

```bash
.venv/bin/python research/phase8/compare_pipelines.py --index-root /tmp/idx8
```

```bash
.venv/bin/python research/phase8/e2e_demo.py --index-root /tmp/idx8
```

```bash
.venv/bin/python research/phase8/final_validation.py --index-root /tmp/idx8
```

The last three need `ollama serve` (real local model); `run_runtime_benchmarks.py` uses a simulated LLM (synthetic, labelled).

## Grounded answers (Phase 7)

```bash
ollama pull qwen3:4b
```

```bash
.venv/bin/streamrag fetch-models nli-deberta-v3-xsmall
```

```bash
.venv/bin/streamrag build-index --corpus tests/fixtures/corpus_grounding
```

```bash
.venv/bin/streamrag --set generation.enabled=true stream --session --interval-ms 300 --text "Tell me the eligibility requirements | and the application process | for the permit." --corpus tests/fixtures/corpus_grounding
```

- With the Ollama server running and the model pulled, answers are written by the local LLM; otherwise by the extractive generator (same verification).
- The LLM URL must be a loopback address. Every claim is verified against the evidence before it is released, cited from the verification.

## Adaptive sessions (Phase 6)

```bash
.venv/bin/streamrag stream --session --interval-ms 300 --text "What are the rules | for ladders | in the orchard?" --text "Specifically | overnight." --text "Okay." --text "Actually, ignore | the overnight restriction." --trace trace.jsonl
```

- `--session` turns on multi-intent mode plus the adaptive session (`session.enabled`).
- It prints context changes, delta plans, query reuse, evidence and claim transitions, answer versions with diffs, and a per-turn summary. `replay` detects session traces and replays them exactly.
- The synchronous drivers for experiments are `streamrag.session.AdaptivePipeline` and `FullRestartPipeline`.

Dev-suite benchmarks (fixture domain; NOT official results):

```bash
.venv/bin/python research/phase4/run_streaming_benchmarks.py --index-root /tmp/idx
```

```bash
.venv/bin/python research/phase5/run_multi_intent_benchmarks.py --index-root /tmp/idx5
```

```bash
.venv/bin/python research/phase6/run_adaptive_benchmarks.py --index-root /tmp/idx6 --reps 7
```

```bash
.venv/bin/python research/phase7/run_grounded_benchmarks.py --index-root /tmp/idx7
```

```bash
.venv/bin/python research/phase7/e2e_streaming.py --index-root /tmp/idx7
```

```bash
.venv/bin/python research/phase7/label_agreement.py
```

```bash
.venv/bin/python research/phase7/report_tables.py
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
  session/         Phase 6: session memory (4 layers, versions, snapshot/restore/reset/archive), adaptive engine,
                   synchronous incremental + full-restart pipelines, PII redaction
  context/         change detection + taxonomy, topic frames, late-detail gate, relevant-context selection/compression
  delta/           delta planner, delta queries, semantic cache, evidence store + lifecycle rules
  claims/          extractive claims, claim-evidence graph, targeted revalidation (Phase 6); claim planning,
                   decomposition, entailment alignment and verification (Phase 7)
  answers/         versioned, sectioned answer state, answer diffs, minimal-regeneration markers
  generation/      Phase 7: LLM gateway (local Ollama, scripted, recorded), prompts, answer planner, generators,
                   claim extraction from generated output
  citations/       citation model, mapping from verification to the supporting span, validation, orphans
  validation/      unsupported-claim policy, repair, coverage, consistency, grounding metrics
  answer_state/    grounded answer engine (plan -> generate -> verify -> repair -> cite -> validate), renderer
  runtime/         Phase 8: StreamingRuntime - event bus, task scheduler, worker pools, cancellation, timeouts,
                   retries, backpressure, state coordinator, answer lane, streamer, replay, fault injection
  telemetry/       structured logging, JSONL event sink, timing
  tools/           build-time model download (the only network code)
tests/             unit/integration tests; tests/fixtures = TEST FIXTURES only
research/          phase1–7 measurements and reports
docs/              architecture diagrams, ADRs, retrieval docs, JSON schemas
```

## Key documents

| Document | Contents |
|---|---|
| `PHASE_7_GROUNDED_GENERATION_REPORT.md` | Phase 7: grounded generation, claim verification, citations; benchmark, ablations, hallucination tests |
| `PHASE_8_STREAMING_RUNTIME_REPORT.md` | Phase 8: streaming runtime - concurrency, cancellation, backpressure, race protection, degraded modes, replay; concurrency / cancellation / backpressure / failure / load benchmarks |
| `PHASE_6_ADAPTIVE_RAG_REPORT.md` | Phase 6: adaptive session RAG; dev-suite, stress-set, full-restart, ablation and scaling results |
| `PHASE_5_MULTI_INTENT_REPORT.md` | Phase 5: multi-intent decomposition, delta retrieval, fusion; dev-suite results |
| `PHASE_4_STREAMING_REPORT.md` | Phase 4: streaming engine, controller, ledger; dev-suite results |
| `PHASE_3_RETRIEVAL_REPORT.md` | Phase 3: what was built, what was measured, what is blocked |
| `PHASE_2_SYSTEM_SPECIFICATION.md` | Full system specification |
| `PHASE_1_RESEARCH_DOSSIER.md` | Research dossier |
| `docs/retrieval/`, `docs/streaming/`, `docs/multi_intent/`, `docs/session/`, `docs/answer/` | Component docs (Phases 3–7) |
| `docs/decisions/` | ADRs |
| `research/phase3/` | Chunking, embedding, profiling and comparison reports |
