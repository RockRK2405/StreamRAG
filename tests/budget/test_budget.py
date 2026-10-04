"""Retrieval budgets (docs/retrieval/11): queries, results, iterations, latency, parallelism, cancellation, failures."""

import time

import pytest
from adaptive_helpers import controller, run

from streamrag.adaptive.controller import AdaptiveRequest
from streamrag.adaptive.models import StopReason
from streamrag.config.settings import AdaptiveBudget


def budget(**kw):
    base = dict(max_queries=5, max_results=40, max_iterations=3, max_latency_ms=1500, max_parallel_tasks=2, max_hops=2)
    return AdaptiveBudget(**{**base, **kw})


def test_query_budget_exhaustion(adaptive_env):
    r = run(adaptive_env, "Is there parking near the hearing room for appeals?",
            **{"adaptive_retrieval.budget": budget(max_queries=1), "adaptive_retrieval.min_gain": 0.0})
    assert r.ops.searches == 1 and r.state.stop_reason in (StopReason.QUERY_LIMIT, StopReason.NO_EXPECTED_GAIN,
                                                            StopReason.SUFFICIENT_EVIDENCE)
    assert r.state.budget_remaining["queries"] == 0


def test_latency_limit_with_slow_retriever(adaptive_env):
    ctl = controller(adaptive_env, **{"adaptive_retrieval.budget": budget(max_latency_ms=60),
                                      "adaptive_retrieval.min_gain": 0.0})
    ctl.search_hook = lambda spec: time.sleep(0.05)   # a slow (remote) retriever: ~50 ms per search
    try:
        r = ctl.run(AdaptiveRequest("q", "What are the eligibility requirements for international applicants "
                                         "from Zemland?"))
    finally:
        ctl.close()
    assert r.state.stop_reason in (StopReason.LATENCY_LIMIT, StopReason.SUFFICIENT_EVIDENCE)
    if r.state.stop_reason == StopReason.LATENCY_LIMIT:
        assert r.ops.searches <= 2 and r.evidence.items       # partial evidence is still handed on


def test_retrieval_timeout_degrades_to_lexical(adaptive_env):
    from streamrag.errors import RetrieverTimeoutError
    ctl = controller(adaptive_env)
    orig = ctl.svc.search_dense

    def boom(*a, **k):
        raise RetrieverTimeoutError("dense exceeded 3000 ms")
    ctl.svc.search_dense = boom
    try:
        r = ctl.run(AdaptiveRequest("q", "How long does processing take for domestic applicants?"))
    finally:
        ctl.svc.search_dense = orig
        ctl.close()
    assert r.evidence.items and r.state.stop_reason != StopReason.ERROR
    assert any(s["status"] == "degraded" for s in r.searches)


def test_dense_only_failure_falls_back_to_lexical(adaptive_env):
    from streamrag.errors import ModelNotAvailableError
    ctl = controller(adaptive_env)
    orig = ctl.svc.search_dense

    def boom(*a, **k):
        raise ModelNotAvailableError("embedder gone")
    ctl.svc.search_dense = boom
    try:
        r = ctl.run(AdaptiveRequest("q", "Is there parking space at the office?"))     # SEMANTIC = dense only
    finally:
        ctl.svc.search_dense = orig
        ctl.close()
    assert any(d.action == "fallback_lexical" for d in r.decisions)


def test_user_cancelled_and_cooperative_checkpoint(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        r = ctl.run(AdaptiveRequest("q", "What is the application fee?", cancelled=lambda: True))
        assert r.state.stop_reason == StopReason.USER_CANCELLED and r.ops.searches == 0

        class TaskCancelled(Exception):
            pass

        def checkpoint():
            raise TaskCancelled("superseded")
        with pytest.raises(TaskCancelled):
            ctl.run(AdaptiveRequest("q2", "What is the application fee?", checkpoint=checkpoint))
    finally:
        ctl.close()


def test_parallel_tasks_bounded(adaptive_env):
    r = run(adaptive_env, "What documents does an applicant from Zemland need?")
    rounds: dict[int, int] = {}
    started = [p for t, p in r.events if t == "RETRIEVAL_STARTED"]
    assessed = [p["round"] for t, p in r.events if t == "EVIDENCE_ASSESSED" and p["round"] > 0]
    assert len(started) == r.ops.searches and len(assessed) == r.state.iteration
    assert r.ops.searches <= r.state.iteration * 2


def test_resource_exhaustion_input_is_bounded(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        t0 = time.perf_counter()
        r = ctl.run(AdaptiveRequest("q", "fee " * 50000, constraints=[]))
        assert (time.perf_counter() - t0) < 5 and len(r.analysis.text) <= 2000
        assert r.ops.searches <= ctl.ac.budget.max_queries
    finally:
        ctl.close()
