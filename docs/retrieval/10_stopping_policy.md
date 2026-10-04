# 10 · Stopping policy and marginal value (Phase 9)

Code: `stopping.py` (`RetrievalStoppingPolicy`, `actual_gain`, `expected_gain`).

Stop reasons, checked after every round in this order: `USER_CANCELLED`, `ERROR`, `SUFFICIENT_EVIDENCE`,
`CONTRADICTION` (unresolved after the targeted search), `NO_RESULTS`, `ITERATION_LIMIT`, `QUERY_LIMIT` (queries or
results), `LATENCY_LIMIT` (spent + measured mean search cost > budget), `NO_EXPECTED_GAIN` (a follow-up round gained
nothing; every unmet slot is unattainable; or the best next action's expected gain < `min_gain`). Every run ends with
exactly one reason (`RetrievalState.stop_reason`, RETRIEVAL_STOPPED).

**Marginal value.** Measured after a round: `actual_gain = Δcoverage + 0.5·Δquality` (coverage = met slots / slots;
quality = mean best term coverage). Expected gain of a candidate action = share of unmet / conflicting slots ×
prior(action): hop 0.7, requirement query 0.6, contradiction search 0.5, filter relaxation 0.5, k expansion 0.4 (0 if
the last search had no candidates beyond k), rerank 0.4. The priors are documented heuristics, not learned, and no
optimality is claimed; their effect is only what the ablations measure.

**No infinite loops:** every loop iteration consumes a round of `max_iterations`, every search a unit of
`max_queries` and k of `max_results`; repeated searches are impossible (signatures); tests run every eval question and
check the bounds.
