"""Adaptive stopping (brief §60), bounded iteration, adaptive top-k, dedup / diversity."""

import json

from adaptive_helpers import controller, run
from conftest import REPO

from streamrag.adaptive.controller import AdaptiveRequest
from streamrag.adaptive.models import RetrievalBudgetState, StopReason, SufficiencyAssessment
from streamrag.adaptive.stopping import RetrievalStoppingPolicy, actual_gain, expected_gain


def test_sufficient_first_round_means_no_second_retrieval(adaptive_env):
    r = run(adaptive_env, "What is Form PX-204 used for?")
    assert r.assessment.status == "SUFFICIENT" and r.ops.searches == 1 and r.state.iteration == 1
    assert r.state.stop_reason == StopReason.SUFFICIENT_EVIDENCE


def test_insufficient_first_round_triggers_more_retrieval(adaptive_env):
    r = run(adaptive_env, "What are the eligibility requirements for international applicants?",
            **{"adaptive_retrieval.initial_k": {"SIMPLE": 1, "MODERATE": 1, "COMPLEX": 1, "MULTI_HOP": 1}})
    first = [e for t, e in r.events if t == "EVIDENCE_ASSESSED" and e["round"] == 1][0]
    if first["status"] != "SUFFICIENT":
        assert r.ops.searches >= 2 and any(t == "RETRIEVAL_EXPANDED" for t, _ in r.events)


def test_iteration_limit_stops_safely(adaptive_env):
    ctl = controller(adaptive_env, **{"adaptive_retrieval.min_gain": 0.0})
    pol = RetrievalStoppingPolicy(0.0)
    b = RetrievalBudgetState(max_queries=9, max_results=99, max_iterations=2, max_latency_ms=1e9,
                             max_parallel_tasks=2, max_hops=2, iterations_used=2)
    a = SufficiencyAssessment(status="PARTIAL", coverage=0.5, quality=0.5, met=["R1"], unmet=["R2"])
    assert pol.check(a, b) == StopReason.ITERATION_LIMIT
    ctl.close()


def test_every_eval_question_stops_within_bounds(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        for p in sorted((REPO / "eval" / "dev_adaptive_retrieval").glob("A*.json")):
            q = json.loads(p.read_text())["turns"][0]["utterance_text"]
            r = ctl.run(AdaptiveRequest(p.stem, q))
            b = ctl.ac.budget
            assert r.state.stop_reason is not None
            assert r.state.iteration <= b.max_iterations and r.ops.searches <= b.max_queries
            assert all(s["k"] <= 20 for s in r.searches)
            assert len(r.evidence.items) <= ctl.ac.final_k
            docs = [e.document_id for e in r.evidence.items]
            assert all(docs.count(d) <= ctl.ac.max_per_document for d in docs)
            ids = [e.evidence_id for e in r.evidence.items]
            assert len(ids) == len(set(ids))                       # dedup across searches
            assert [d.decision_id for d in r.decisions] == sorted({d.decision_id for d in r.decisions},
                                                                  key=lambda x: int(x.split("D")[-1]))
            assert all(d.reason for d in r.decisions)
    finally:
        ctl.close()


def test_adaptive_k_schedule(adaptive_env):
    r = run(adaptive_env, "Is there parking space at the office near the hearing room?",
            **{"adaptive_retrieval.min_gain": 0.0})
    ks = [s["k"] for s in r.searches]
    assert ks == sorted(ks) and set(ks) <= {3, 5, 10, 20}
    off = run(adaptive_env, "What is Form PX-204 used for?", **{"adaptive_retrieval.adaptive_k": False})
    assert {s["k"] for s in off.searches} == {5}


def test_marginal_value_formulas():
    a0 = SufficiencyAssessment(status="PARTIAL", coverage=0.5, quality=0.6, met=["R1"], unmet=["R2"])
    a1 = SufficiencyAssessment(status="SUFFICIENT", coverage=1.0, quality=0.8, met=["R1", "R2"])
    assert actual_gain(a0, a1) == 0.6
    assert expected_gain("hop", a0, 2) == 0.35 and expected_gain("expand_k", a0, 2, more_candidates=False) == 0.0


def test_retrieval_state_is_complete(adaptive_env):
    r = run(adaptive_env, "How long does processing take for domestic applicants?")
    s = r.state
    assert s.current_strategy.value == "FILTERED" and s.iteration >= 1 and s.queries_executed == r.ops.searches
    assert s.evidence_count == len(r.evidence.items) and 0 <= s.coverage <= 1 and s.latency_spent_ms > 0
    assert set(s.budget_remaining) == {"queries", "results", "iterations", "latency_ms"}
