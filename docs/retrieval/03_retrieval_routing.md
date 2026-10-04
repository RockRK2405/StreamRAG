# 03 · Retrieval routing (Phase 9)

Code: `src/streamrag/adaptive/policy.py` (`RetrievalPolicySelector` = the router), `controller.py`.

## Strategies (each serves a project requirement; measured in research/phase9)
| Strategy | Retrievers | Used when | Requirement served |
|---|---|---|---|
| CACHE_REUSE | none | a valid cached result exists (08) | no redundant retrieval |
| SESSION_REUSE | none | session evidence meets every requirement (08) | cross-query evidence reuse |
| MULTI_HOP | hybrid + hops | complexity MULTI_HOP (07) | chained answers |
| LEXICAL | BM25 | exact identifiers; the calibrated simple-query fast path; tight latency | exact terms, cheap path |
| FILTERED | hybrid + metadata / validity filter | a user-stated metadata constraint or an explicit date / period | exact constraints, temporal questions |
| SEMANTIC | dense | ≥ `semantic_oov_ratio` of the content terms unknown to BM25 | vocabulary mismatch |
| ITERATIVE | hybrid, more rounds | complexity COMPLEX | multi-constraint / comparison needs |
| HYBRID | BM25 + dense + RRF | everything else | default |
| FAST_VECTOR | dense, small k | only if `simple_strategy: FAST_VECTOR` | kept for the calibration; never selected by default |

Rules are applied in the order of the table (after cache / session reuse); a stack without a dense index falls back
to LEXICAL (recorded). Every plan stores `strategy_reason`; RETRIEVAL_POLICY_SELECTED carries it with the
complexity reasons, k, retrievers, filters, reranking and budget.

## Calibration (not tuned on the evaluation set)
`research/phase9/calibrate_routing.py` decides `simple_strategy` and `rerank` with rules fixed before running, on the
Phase 7 grounded dev questions + Phase 3 fixture retrieval items (a different question set from the Phase 9 eval):
the cheapest strategy whose Recall@5 and MRR on SIMPLE needs are within 0.02 of HYBRID's, and reranking only if it
raises COMPLEX MRR by ≥ 0.02 at < 50 ms. Results: `research/phase9/results/calibration.json`.

## Reranking policy
`never` (default after calibration) | `always` | `policy`: rerank COMPLEX / MULTI_HOP / ITERATIVE needs when a
cross-encoder is loaded and the measured per-candidate cost × `rerank_max_candidates` fits half the latency budget;
in the loop a `rerank` action is proposed when requirements are unmet and more candidates than k exist. Skipped when
the budget is tight. The reranker evaluation (baseline C vs D, ADAPTIVE vs ADAPTIVE_rerank_policy) is in the report §28-29.

## Security
Corpus text never selects a strategy, k, budget, iteration count, filter *field* or prompt. Corpus *metadata* values
can only narrow a search, and only when the user stated the value as a constraint (inside a Phase 5/6 constraint, or
after a constraint marker — `constraint_markers` in `configs/retrieval_lexicon.yaml`): a document declaring
`applicant_type: fee` cannot filter "what is the fee?" (tests/adaptive_retrieval/test_security.py).
