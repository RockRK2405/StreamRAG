"""RetrievalStoppingPolicy and marginal value (docs/retrieval/10_stopping_policy.md).

Checked after every round, in this order (the first that applies is the stop reason):

  USER_CANCELLED       the caller cancelled (superseded query, session reset)
  ERROR                the round failed and no fallback is left
  SUFFICIENT_EVIDENCE  every requirement MET
  CONTRADICTION        a value conflict is still unresolved after the targeted contradiction search (or that search
                       is disabled / impossible): retrieval cannot decide - the answer must report both values
  NO_RESULTS           the round returned nothing and nothing can be relaxed
  ITERATION_LIMIT / QUERY_LIMIT / LATENCY_LIMIT   budget (the latency check uses the measured mean search cost:
                       stop if spent + expected next cost > max_latency)
  NO_EXPECTED_GAIN     the best next action's expected gain < ``min_gain``, a follow-up round gained nothing
                       (marginal value <= 0), or every unmet requirement is unattainable (terms unknown to the index)

Marginal value of a round (measured after it ran):
  actual_gain = (coverage_after - coverage_before) + 0.5 * (quality_after - quality_before)
Expected gain of a candidate action (a documented heuristic prior - not learned, no optimality claim):
  expected_gain = share of unmet requirements x prior(action), prior: hop 0.7, requirement query 0.6,
  contradiction search 0.5, filter relaxation 0.5, k expansion 0.4 (0 when the last search had no candidates beyond
  its k), rerank 0.4 (policy mode only, when more candidates than k exist).
"""

from __future__ import annotations

from streamrag.adaptive.models import RetrievalBudgetState, StopReason, SufficiencyAssessment

PRIOR = {"hop": 0.7, "requirement_query": 0.6, "contradiction_search": 0.5, "relax_filter": 0.5, "expand_k": 0.4,
         "rerank": 0.4}


def actual_gain(before: SufficiencyAssessment | None, after: SufficiencyAssessment) -> float:
    cb = before.coverage if before else 0.0
    qb = before.quality if before else 0.0
    return round((after.coverage - cb) + 0.5 * (after.quality - qb), 4)


def expected_gain(action: str, assessment: SufficiencyAssessment, n_reqs: int, more_candidates: bool = True) -> float:
    if n_reqs == 0:
        return 0.0
    share = (len(assessment.unmet) + len(assessment.conflicts)) / n_reqs
    prior = PRIOR.get(action, 0.3)
    if action == "expand_k" and not more_candidates:
        prior = 0.0
    return round(share * prior, 4)


class RetrievalStoppingPolicy:
    def __init__(self, min_gain: float) -> None:
        self.min_gain = min_gain

    def check(self, a: SufficiencyAssessment, budget: RetrievalBudgetState, *, cancelled: bool = False,
              error: bool = False, no_results: bool = False, contradiction_exhausted: bool = False,
              mean_search_ms: float = 0.0, stalled_rounds: int = 0) -> StopReason | None:
        if cancelled:
            return StopReason.USER_CANCELLED
        if error:
            return StopReason.ERROR
        if a.status == "SUFFICIENT":
            return StopReason.SUFFICIENT_EVIDENCE
        if a.status == "CONTRADICTORY" and contradiction_exhausted:
            return StopReason.CONTRADICTION
        if no_results:
            return StopReason.NO_RESULTS
        if budget.iterations_used >= budget.max_iterations:
            return StopReason.ITERATION_LIMIT
        if budget.queries_used >= budget.max_queries or budget.results_used >= budget.max_results:
            return StopReason.QUERY_LIMIT
        if budget.latency_used_ms + mean_search_ms > budget.max_latency_ms:
            return StopReason.LATENCY_LIMIT
        if stalled_rounds >= 1:
            return StopReason.NO_EXPECTED_GAIN
        return None

    def worth(self, gain: float) -> bool:
        return gain >= self.min_gain
