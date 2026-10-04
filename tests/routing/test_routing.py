"""Retrieval policy tests (brief §59): strategy, reason, retrievers, top-k, reranking and budget per query."""

import pytest
from adaptive_helpers import run

CASES = [
    # query, strategy, reason substring, retrievers, top_k
    ("How often must a residence permit be renewed?", "LEXICAL", "simple-query fast path", ["bm25"], 3),
    ("What is Form PX-204 used for?", "LEXICAL", "exact identifiers ['PX-204']", ["bm25"], 3),
    ("How long does processing take for domestic applicants?", "FILTERED", "metadata constraint", ["bm25", "dense"], 5),
    ("What was the application fee in December 2025?", "FILTERED", "explicit date 'December 2025'",
     ["bm25", "dense"], 5),
    ("What documents does an applicant from Zemland need?", "MULTI_HOP", "entity_aspect_gap:zemland",
     ["bm25", "dense"], 5),
    ("Is there parking space at the office?", "SEMANTIC", "vocabulary mismatch", ["dense"], 5),
    ("Compare the business licence fee and the vehicle registration fee", "ITERATIVE", "comparison",
     ["bm25", "dense"], 5),
]


@pytest.mark.parametrize("q,strategy,reason,retrievers,k", CASES)
def test_policy_per_query(adaptive_env, q, strategy, reason, retrievers, k):
    r = run(adaptive_env, q)
    p = r.plan
    sel = next(d for d in r.decisions if d.action == "select")
    assert sel.strategy.value == strategy and reason in sel.reason
    assert reason in next(payload for t, payload in r.events if t == "RETRIEVAL_POLICY_SELECTED")["strategy_reason"]
    assert p.retrievers == retrievers and p.top_k == k and p.max_top_k == 20
    assert p.reranking is False and p.rerank_reason == "no reranker loaded"       # test stack: no cross-encoder
    assert p.max_iterations == 3 and p.latency_budget_ms == 1500 and p.evidence_threshold == 1.0
    assert p.requirements and p.stop_condition


def test_filters_in_plan(adaptive_env):
    r = run(adaptive_env, "What was the application fee in December 2025?")
    assert r.plan.filters.valid_at == "2025-12-01" and r.plan.filters.valid_to == "2025-12-31"
    r2 = run(adaptive_env, "How long does processing take for domestic applicants?")
    assert r2.plan.filters.metadata == {"applicant_type": ["domestic"]} and r2.plan.filters.valid_at is None


def test_routing_disabled_is_fixed_hybrid(adaptive_env):
    r = run(adaptive_env, "What is Form PX-204 used for?", **{"adaptive_retrieval.routing": False})
    assert r.plan.strategy.value == "HYBRID" and r.plan.filters is None and "routing disabled" in r.plan.strategy_reason


def test_tight_latency_budget_selects_fast_path(adaptive_env):
    from adaptive_helpers import controller
    from streamrag.adaptive.controller import AdaptiveRequest
    ctl = controller(adaptive_env)
    try:
        r = ctl.run(AdaptiveRequest("q", "How long does processing take for domestic applicants?",
                                    latency_budget_ms=50))
    finally:
        ctl.close()
    assert r.plan.strategy.value == "LEXICAL" and "latency-aware" in r.plan.strategy_reason
    assert r.plan.max_iterations == 1 and r.plan.reranking is False and r.state.iteration == 1


def test_dense_unavailable_falls_back_to_lexical(adaptive_env):
    from adaptive_helpers import controller
    ctl = controller(adaptive_env)
    ctl.selector.dense = False
    try:
        from streamrag.adaptive.controller import AdaptiveRequest
        r = ctl.run(AdaptiveRequest("q", "Is there parking space at the office?"))
    finally:
        ctl.close()
    assert r.plan.strategy.value == "LEXICAL" and "dense index / embedder unavailable" in r.plan.strategy_reason


def test_rerank_policy_mode(adaptive_env):
    from adaptive_helpers import controller
    ctl = controller(adaptive_env, **{"adaptive_retrieval.rerank": "policy"})
    try:
        a = ctl.analyzer.analyze("q", "Compare the business licence fee and the vehicle registration fee")
        assert ctl.selector.rerank_for(a, ctl.selector.base_strategy(a)[0], 1500, None)[0] is False   # no reranker
        ctl.selector.rerank_ok = True
        ok, why = ctl.selector.rerank_for(a, ctl.selector.base_strategy(a)[0], 1500, 1.0)
        assert ok and "COMPLEX" in why
        assert ctl.selector.rerank_for(a, ctl.selector.base_strategy(a)[0], 20, 1.0)[0] is False      # too slow
        s = ctl.analyzer.analyze("q", "How often must a residence permit be renewed?")
        assert ctl.selector.rerank_for(s, ctl.selector.base_strategy(s)[0], 1500, 1.0)[0] is False      # simple
    finally:
        ctl.close()
