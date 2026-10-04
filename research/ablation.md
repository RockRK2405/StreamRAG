# Ablation summary (brief §23)

Each row removes or changes one component and states what that did, with the data set and the source table.
* **v2** = held-out (`../FINAL_BENCHMARK_RESULTS/README.md`).
* **v1** = Phase 10 test split (`../PHASE_10_EVALUATION_REPORT.md` §22, `../experiments/results/ablation_table.json`).

Answer correctness = all key strings stated and no stale value. Fixture data, not reportable.

| component | ablation | effect | set | verdict |
|---|---|---|---|---|
| streaming (early retrieval, drafts) | batch mode of the same pipeline | First evidence p50 264 → 767 ms and first answer content 271 → 2,477 ms. Verified answer after end: 1,643 vs 1,659 ms. Correctness 0.718 vs 0.732 (net 1 turn). | v2 | keeps: big gain in time to first content; the verified answer is not faster |
| adaptive retrieval | fixed top-5 (extractive answers) | Evidence precision 0.55 → 0.24. Stale values 0 → 7 / 13. Embeddings 0.83 → 1.06 per turn. Recall@5 0.94 → 0.98. Correctness 0.718 → 0.662 (n.s.). | v2 | keeps: precision, cost and staleness; recall is the price |
| adaptive retrieval | streaming with fixed retrieval (baseline D) | Correctness 0.718 → 0.620 (n.s., p = 0.14). Stale values 0 → 7 / 13. TTVA after end p50 1,643 → 2,275 ms. | v2 | keeps |
| claim-driven requirements | query-driven sufficiency | Correctness −2 turns (runtime), n.s. | v1 | weak evidence |
| session memory and caches | no cache | Retrieval calls 1.35 → 1.42 per turn (p = 0.031). Cache answers 7.4% of turns. Quality unchanged. | v2 | small saving |
| session memory | memoryless runtime | Correctness 0.730 → 0.716. Five more unanswered turns. | v1 | small gain |
| delta retrieval | full re-retrieval | Retrieval calls 1.35 → 1.49 per turn (p = 0.016). Quality unchanged. | v2 | keeps: less work |
| cancellation | off | Worker time 16.0 → 21.8 s (+36%). LLM calls 9 → 15. Final-turn correctness unchanged. Corrected answer verified 147 ms later (p50). | v2, 14 turns | keeps |
| claim verification | none (S_batch − verification) | Correctness 0.797 → 0.865 (more true claims kept). Claim support (verifier) 1.000 → 0.847. Unsupported claims reach the user. | v1 | keeps: grounding over recall of facts; the cost is reported |
| citation validation | none | Correctness 0.797 → 0.770. Citations not guaranteed to support their claim. | v1 | keeps |
| cross-encoder reranking | on (S_batch + rerank) | No correctness change on v1. +29 ms wall / +262 ms CPU per search. On v2, the reranked *baseline* had the best correctness (0.803). | v1, v2 | off; a re-tune is open |
| LLM answerability flag | on | Insufficiency handled 1 → 2 of 4. Correctness 0.718 → 0.662 (p = 0.22). TTVA +103 ms (mean). | v2; dev: same direction | off |
| Phase 11 early-commitment fixes | Phase 10 system | Correctness 0.730 → 0.797. Stale values 3 / 13 → 0 / 13. Streaming vs batch gap 7 → 2 turns. | v1 (development data, optimistic) | keeps; v2 confirms 0 stale values |

**Overall.**
* The components that pay for themselves on held-out data:
  * streaming, for time to first content;
  * adaptive retrieval, for precision, cost and staleness;
  * delta retrieval and cancellation, for work;
  * verification, for grounding.
* No single component raised answer correctness significantly on v2.
* Two switches stay off because they cost correctness or time without a held-out gain: the reranker and the
  answerability flag.
