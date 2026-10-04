# Evaluation design

## Datasets

| set | domain | turns | role |
|---|---|---|---|
| `eval/dev_*`, `streamrag_eval_v1/dev` | several fixture domains | 129 (v1 dev) | development throughout |
| `streamrag_eval_v1/test` | Northvale transit | 87 | Phase 10 held-out; after its error analysis, **development data** |
| `streamrag_eval_v2/test` | Lakeside utility services | 81 | **Phase 11 held-out**, frozen before any Phase 11 change (`FROZEN.sha256`) |

* **Coverage.** Both test sets cover all 14 query types:
  * SIMPLE, EXACT_TERM, SEMANTIC;
  * MULTI_INTENT, MULTI_CONSTRAINT, MULTI_HOP;
  * TEMPORAL, CONTRADICTORY, INSUFFICIENT_EVIDENCE, AMBIGUOUS;
  * CONTEXTUAL_FOLLOWUP, ENTITY_CORRECTION, STREAMING_CORRECTION, REPEATED_QUERY.
* **Turn structure.** Single turns, multi-turn sessions and streamed corrections with scripted chunk timing.
* **Labels per turn.** Gold sections, expected claims with key strings, forbidden (stale) values, conflict values and
  the expected state.
* **Leakage tests.** `tests/evaluation/test_dataset_and_leakage.py` checks schemas, that the corpora are separate, and
  that no v2 string occurs in source code, configs or demo data.

## Systems

`experiments/configs/systems.yaml` defines every variant on the same index and models:
* **Baselines:**
  * A: naive dense top-5;
  * B: hybrid BM25 + dense with RRF;
  * C: hybrid + cross-encoder;
  * D: streaming with fixed retrieval.
* **The final system:** streaming runtime, session, delta retrieval, cancellation, adaptive retrieval, verified
  generation.
* **Its batch mode.**
* **Ablations:**
  * answerability flag;
  * fixed top-5 vs adaptive (extractive);
  * no cache;
  * full re-retrieval;
  * cancellation on / off.

## Metrics

Definitions are in `../docs/evaluation/metrics.md`; the code is in `../src/streamrag/evaluation/metrics/`.
* **Retrieval:** Recall@k, MRR, nDCG against gold sections.
* **Evidence:** precision, recall, coverage and redundancy of what reaches the answer stage.
* **Answer:**
  * correctness: all key strings stated, no forbidden value;
  * stale or forbidden values;
  * insufficiency handled;
  * conflict reported.
* **Grounding:**
  * model-free hallucinated-value rate;
  * verifier-judged claim support and citation precision (biased: it is the system's own NLI model).
* **Latency:** TTFE / TTFA / TTVA from the first chunk, and after the utterance end.
* **Efficiency:** retrieval calls, embeddings, LLM calls, tokens, worker time.
* **Theme 4 gates G2–G6:** computed from the runtime traces (`../experiments/runners/final_tables.py`).

## Statistics

* Paired by turn: exact McNemar for binary metrics, Wilcoxon signed-rank otherwise, with a bootstrap 95% CI of the
  mean difference.
* No test below 20 pairs or 6 non-zero differences.
* No multiple-comparison correction. The p-values are descriptive.

## Protocol rules

* v2 was run **once** on the final system. No system change was made after seeing v2 outputs.
* A re-run is allowed only for an instrument or environment fault, and both runs are reported. Phase 11 has one: the
  v1 regression run during an LLM-server slowdown (`../FINAL_BENCHMARK_RESULTS/README.md` §5).
* Missing measurements are written as NOT MEASURED.
