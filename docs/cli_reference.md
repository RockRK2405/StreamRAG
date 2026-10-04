# CLI reference (component-level commands, Phases 3–9)

These are the per-component commands from the development phases, moved here unchanged from the Phase 9 README when
the README was rewritten for the final submission. The final system is started with `streamrag serve` (see
`../README.md`). Every result these commands print on `tests/fixtures/` is a TEST FIXTURE result, not a benchmark.

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

## Adaptive retrieval (Phase 9)

```bash
.venv/bin/streamrag --set adaptive_retrieval.enabled=true --set generation.enabled=true --set generation.backend=extractive stream --session --runtime --interval-ms 300 --text "What documents does | an applicant from | Zemland need?" --corpus tests/fixtures/corpus_adaptive
```

- `adaptive_retrieval.enabled` replaces the fixed per-need retrieval with `streamrag.adaptive.AdaptiveRetrievalController`: explainable complexity analysis, claim slots, a logged strategy choice (`strategy_reason`), a bounded retrieve-assess loop with explicit stop reasons, hops, contradiction search and validity-checked caches (`adaptive_retrieval:` in `configs/default.yaml`, `configs/retrieval_lexicon.yaml`, `docs/retrieval/01_query_analysis.md` … `11_budget_management.md`, `docs/architecture/13_adaptive_retrieval.md`, ADR-019). Off by default: fixture results only.

```bash
.venv/bin/python research/phase9/calibrate_routing.py --index-root /tmp/idx9
```

```bash
.venv/bin/python research/phase9/run_benchmarks.py --index-root /tmp/idx9
```

```bash
.venv/bin/python research/phase9/run_experiments.py --index-root /tmp/idx9 --reps 5
```

```bash
.venv/bin/python research/phase9/runtime_h5.py --index-root /tmp/idx9
```

```bash
.venv/bin/python research/phase9/llm_quality.py --index-root /tmp/idx9
```

```bash
.venv/bin/python research/phase9/e2e_final.py --index-root /tmp/idx9
```

```bash
.venv/bin/python research/phase9/make_tables.py
```

`llm_quality.py` and `e2e_final.py` need `ollama serve`; `make_tables.py` renders `research/phase9/results/tables.md`.

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
