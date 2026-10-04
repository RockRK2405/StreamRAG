# Results

Authoritative tables:
* `../FINAL_BENCHMARK_RESULTS/README.md`: the Phase 11 held-out v2 set, regression, robustness, demo and profile;
* `../PHASE_10_EVALUATION_REPORT.md`: v1 and experiments 1–10.

Fixture data: **NOT REPORTABLE** as official results.

## Held-out v2 (81 turns, new domain, run once)

| | naive RAG | hybrid + rerank | full system | full system, batch |
|---|---|---|---|---|
| answer correct | 0.761 | **0.803** | 0.718 | 0.732 |
| stale values (of 13) | 3 | 2 | **0** | **0** |
| hallucinated-value claim rate | 0.070 | 0.062 | **0** | **0** |
| evidence precision | 0.247 | 0.242 | **0.552** | 0.551 |
| Recall@5 | **0.994** | 0.961 | 0.883 | 0.935 |
| first evidence / first answer content, from the first word (p50) | after end | after end | **264 / 271 ms** | 767 / 2,477 ms |
| verified answer after the user stops (p50) | **1,411 ms** | 1,489 ms | 1,643 ms | 1,659 ms |
| retrieval / LLM calls per turn | **1.00 / 1.00** | **1.00 / 1.00** | 1.86 / 1.27 | 1.35 / 1.28 |

**Significant on v2:**
* fewer hallucinated values vs every RAG baseline (p ≤ 0.008);
* higher evidence precision (p < 0.001);
* earlier first answer content than batch mode (p < 0.001);
* lower Recall@5 than naive and hybrid RAG (p ≤ 0.03);
* more retrieval and LLM calls than naive RAG (p < 0.001);
* fewer retrieval calls with caches (p = 0.031) and with delta retrieval (p = 0.016).

**Not significant:** every answer-correctness difference between the full system and the baselines.

**Theme 4 gates (v2 traces):**
* G2 early retrieval: 98.7%;
* G3 multi-intent: 4 / 4;
* G4: claim support 1.000, 0 of 173 citations unresolvable;
* G6 telemetry: 100%;
* G5 session refinement: weak, 0.61 streaming vs 0.72 batch.

**Cancellation on corrections:** worker time −27%, LLM calls 15 → 9, final answers unchanged.

**Robustness:** all eight fault types recovered 10 / 10, including the LLM timeout, which was 0 / 10 in Phase 10.

**Demo check:** 10 / 10 scenarios with and without the LLM.

## Development data (v1): Phase 10 → Phase 11

* Answer correctness: 0.730 → 0.797.
* Stale values: 3 / 13 → 0 / 13.
* Streaming vs batch: turns lost 7 → 2.
* TTVA after the end, p50: 1,690 → 1,261 ms.

These are optimistic, because v1 was used to develop the fixes.
