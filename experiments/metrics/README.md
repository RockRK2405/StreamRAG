# Metrics: where they are defined and computed

The metric code is a library in `src/streamrag/evaluation/metrics/`. The experiment runner calls it, and
`tests/evaluation/` validates it against hand-computed cases. No system module imports it; a test checks this. The
full definitions, formulas and caveats are in `docs/evaluation/metrics.md`. This file is the index.

| group | module | metrics | needs |
|---|---|---|---|
| retrieval | `retrieval.py` | Recall@1/3/5/10, Precision@1/3/5, MRR, nDCG@10 (binary relevance on gold sections; `any` semantics for ambiguous questions) | gold sections |
| evidence | `evidence.py` | evidence recall / precision / coverage, unsupported evidence, stale / contradictory evidence, redundancy (embedding cosine), source diversity | gold sections, expected claims, forbidden patterns |
| claims | `claims.py` | claims per answer, claim support / unsupported / contradicted rate (verifier-judged), claim coverage | the Phase 7 ClaimVerifier as an instrument |
| generation | `generation.py` | answer correctness, completeness, forbidden value asserted, abstention, insufficiency handling, conflict reporting, faithfulness, groundedness | expected claims (key strings), forbidden patterns, expected state |
| citation | `citation.py` | citation precision / recall / completeness / entailment, source validity, position correctness | verifier-judged claims with their cited sections |
| hallucination | `hallucination.py` | unsupported claim rate, hallucinated claim rate (numbers / identifiers in neither the evidence nor the question), citation-less fact rate, grounding-failure turns | judged claims, evidence text |
| latency | `latency.py` | TTFT, TTFE, TTFA, TTVA, total (from the first transcript chunk) and the same after the utterance ends; p50 / p90 / p95 / p99 with minimum n (10 / 20 / 100) | runtime events or measured stage times |
| streaming | `streaming.py` | answer updates, revisions, stale-update rate, time to useful / final answer, ordering errors, update gaps | runtime events |
| efficiency | `efficiency.py` | retrieval calls, embeddings, lexical searches, reranker calls, chunks retrieved, avg k, iterations, expansions, documents, cache hit, evidence reused, LLM calls, prompt / output tokens | counts of operations actually executed |
| robustness | `robustness.py` | recovery, degraded success, incorrect-answer and failure-propagation rates | `runners/robustness.py` rows |

Statistics (`src/streamrag/evaluation/stats.py`): bootstrap 95% CIs (10,000 resamples, seed 20261004), paired
Wilcoxon signed-rank (continuous metrics), and exact McNemar (binary metrics). Effect sizes are Cohen's d_z and the
matched-pairs rank-biserial. No test is run below 20 pairs or below 6 non-zero differences. See
`docs/evaluation/statistics.md`.

Error categories (`src/streamrag/evaluation/errors.py`) are assigned by rule from the scored row. See
`docs/evaluation/error_analysis.md`.

Not measured, and why:

* answer relevance and human preference: the human study was not run, and no LLM judge is used;
* GPU memory: there is no GPU, and inference runs on CPU and Ollama's Metal backend;
* currency cost: no priced API is used, so only operation and token counts are reported.
