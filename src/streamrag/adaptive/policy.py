"""RetrievalPolicySelector / RetrievalRouter (docs/retrieval/03_retrieval_routing.md, 04_adaptive_top_k.md).

Inputs are the query analysis (query + interpreted need), the session state (cache / reusable evidence), the
runtime budget and what the stack can do (dense index loaded, reranker loaded). Output: a ``RetrievalPlan`` with the
strategy and the reason (``strategy_reason``), logged as RETRIEVAL_POLICY_SELECTED. Corpus *text* is never an input.

Routing rules, first match wins (each one is a project requirement, see docs/retrieval/03 for the measured effect):

  1 CACHE_REUSE    a cached result with a valid signature exists (same query terms, filters, validity date, corpus,
                   unchanged source versions)                                          -> no search if SUFFICIENT
  2 SESSION_REUSE  usable session evidence exists for the need                         -> assess before searching
  3 MULTI_HOP      complexity MULTI_HOP (entity_aspect_gap)
  4 LEXICAL        exact identifiers known to the index, no filters (``exact_id_strategy``)
  5 FILTERED       metadata constraint or explicit date / period
  6 SEMANTIC       vocabulary mismatch: share of query terms unknown to BM25 >= ``semantic_oov_ratio``
  7 ITERATIVE      complexity COMPLEX
  8 simple path    complexity SIMPLE -> ``simple_strategy`` (calibrated, research/phase9/calibrate_routing.py)
  9 HYBRID         everything else
  capability fallback: no dense index / embedder -> LEXICAL for any strategy that needs dense (recorded)
  latency-aware: remaining budget < ``tight_latency_ms`` -> LEXICAL fast path (no embedding), one round

Initial k by complexity (``initial_k``); expansion follows ``k_schedule`` (5 -> 10 -> 20) only while requirements
are unmet and the previous search had more candidates than it returned (docs/retrieval/04).
Reranking: ``never`` | ``always`` | ``policy`` (COMPLEX / MULTI_HOP / ITERATIVE needs, reranker loaded, and the
measured per-candidate cost fits the remaining latency budget).
"""

from __future__ import annotations

from streamrag.adaptive.models import QueryAnalysis, QueryComplexity, RetrievalPlan, RetrievalStrategy as S
from streamrag.models.retrieval import RetrievalFilters

_RETRIEVERS = {S.FAST_VECTOR: ["dense"], S.LEXICAL: ["bm25"], S.SEMANTIC: ["dense"], S.HYBRID: ["bm25", "dense"],
               S.FILTERED: ["bm25", "dense"], S.MULTI_HOP: ["bm25", "dense"], S.ITERATIVE: ["bm25", "dense"],
               S.CACHE_REUSE: [], S.SESSION_REUSE: []}


def retrievers_for(strategy: S) -> list[str]:
    return list(_RETRIEVERS[strategy])


class RetrievalPolicySelector:
    def __init__(self, cfg, dense_available: bool, reranker_available: bool) -> None:
        self.cfg = cfg.adaptive_retrieval
        self.dense, self.rerank_ok = dense_available, reranker_available

    def base_strategy(self, a: QueryAnalysis, latency_budget_ms: float | None = None) -> tuple[S, str]:
        c = self.cfg
        oov = a.signals.get("oov_ratio", 0.0)
        if not c.routing:
            s, why = S.HYBRID, "routing disabled (ablation): fixed hybrid"
        elif latency_budget_ms is not None and latency_budget_ms < c.tight_latency_ms:
            s, why = S.LEXICAL, (f"latency-aware: {latency_budget_ms:.0f} ms left < {c.tight_latency_ms:.0f} ms - "
                                 "fast lexical path (no embedding), one round")
        elif a.complexity == QueryComplexity.MULTI_HOP and not c.multi_hop:
            s, why = S.ITERATIVE, "complexity MULTI_HOP but multi_hop disabled (ablation): iterative"
        elif a.complexity == QueryComplexity.MULTI_HOP:
            s, why = S.MULTI_HOP, "complexity MULTI_HOP: " + ";".join(a.reasons)
        elif a.exact_ids and not a.metadata_filters and not a.temporal.explicit \
                and a.complexity in (QueryComplexity.SIMPLE, QueryComplexity.MODERATE):
            s, why = S[c.exact_id_strategy], f"exact identifiers {a.exact_ids} known to the lexical index"
        elif a.metadata_filters or a.temporal.explicit:
            s, why = S.FILTERED, (f"metadata constraint {a.metadata_filters}" if a.metadata_filters else "") + (
                f" explicit date '{a.temporal.cue}'" if a.temporal.explicit else "")
        elif a.terms and oov >= c.semantic_oov_ratio:
            s, why = S.SEMANTIC, f"vocabulary mismatch: {oov:.0%} of query terms unknown to BM25 {a.oov_terms}"
        elif a.complexity == QueryComplexity.COMPLEX:
            s, why = S.ITERATIVE, "complexity COMPLEX: " + ";".join(a.reasons)
        elif a.complexity == QueryComplexity.SIMPLE:
            s, why = S[c.simple_strategy], "simple-query fast path (simple_strategy, calibrated)"
        else:
            s, why = S.HYBRID, "complexity MODERATE: " + ";".join(a.reasons)
        if not self.dense and "dense" in _RETRIEVERS[s]:
            s, why = S.LEXICAL, why.strip() + " | fallback: dense index / embedder unavailable"
        return s, why.strip()

    def filters_for(self, a: QueryAnalysis, period_end: str | None) -> RetrievalFilters | None:
        if not self.cfg.routing:
            return None
        dated = a.temporal.explicit and self.cfg.temporal
        if not a.metadata_filters and not dated:
            return None
        return RetrievalFilters(metadata=a.metadata_filters or None, valid_at=a.temporal.valid_at if dated else None,
                                valid_to=period_end if dated else None)

    def rerank_for(self, a: QueryAnalysis, strategy: S, latency_budget_ms: float,
                   rerank_ms_per_candidate: float | None) -> tuple[bool, str]:
        mode = self.cfg.rerank
        if not self.rerank_ok:
            return False, "no reranker loaded"
        if mode == "never":
            return False, "rerank=never (reranker evaluation, docs/retrieval/03)"
        if mode == "always":
            return True, "rerank=always"
        if a.complexity not in (QueryComplexity.COMPLEX, QueryComplexity.MULTI_HOP) and strategy != S.ITERATIVE:
            return False, f"policy: {a.complexity.value} need"
        est = (rerank_ms_per_candidate or 0.0) * self.cfg.rerank_max_candidates
        if est > 0.5 * latency_budget_ms:
            return False, f"policy: estimated rerank {est:.0f} ms > half the latency budget"
        return True, f"policy: {a.complexity.value} need, estimated rerank {est:.0f} ms fits the budget"

    def plan(self, plan_id: str, a: QueryAnalysis, intent_id: str | None, strategy: S, reason: str,
             filters: RetrievalFilters | None, rerank: tuple[bool, str], requirements: list[str],
             latency_budget_ms: float, max_iterations: int) -> RetrievalPlan:
        c = self.cfg
        k0 = c.initial_k.get(a.complexity.value, c.k_schedule[0]) if c.adaptive_k else c.k_schedule[0]
        return RetrievalPlan(
            plan_id=plan_id, query_id=a.query_id, intent_id=intent_id, strategy=strategy, strategy_reason=reason,
            retrievers=retrievers_for(strategy), top_k=k0,
            max_top_k=max(c.k_schedule[-1], k0) if c.adaptive_k else k0, filters=filters,
            reranking=rerank[0], rerank_reason=rerank[1], max_iterations=max_iterations,
            latency_budget_ms=latency_budget_ms, evidence_threshold=1.0,
            stop_condition="all requirements MET (or a value conflict that retrieval cannot resolve) | budget "
                           "(queries / results / iterations / latency) | no expected gain | cancelled",
            requirements=requirements)


def next_k(current: int, schedule: list[int]) -> int | None:
    """Adaptive top-k: the next size in the schedule above ``current`` (None when exhausted)."""
    bigger = [k for k in schedule if k > current]
    return bigger[0] if bigger else None
