"""Brief §58: the 18 adaptive retrieval test cases (fixture corpus tests/fixtures/corpus_adaptive, hashing embedder).

Several cases have deeper suites of their own (routing, cache, contradiction, budget, multi_hop,
streaming_integration); here each case is checked end to end on the plan, the stop reason and the evidence."""

import time

from adaptive_helpers import citations, controller, run

from streamrag.adaptive.controller import AdaptiveRequest
from streamrag.adaptive.models import StopReason as R
from streamrag.config.settings import AdaptiveBudget


def test_01_simple_query(adaptive_env):
    r = run(adaptive_env, "How often must a residence permit be renewed?")
    assert (r.analysis.complexity.value, r.plan.strategy.value, r.ops.searches, r.ops.dense_searches) == \
        ("SIMPLE", "LEXICAL", 1, 0)
    assert r.state.stop_reason == R.SUFFICIENT_EVIDENCE and citations(r)[0] == "RENEW §1"


def test_02_exact_keyword_query(adaptive_env):
    r = run(adaptive_env, "Who has to present Form VR-12?")
    assert r.plan.strategy.value == "LEXICAL" and r.analysis.exact_ids == ["VR-12"]
    assert citations(r)[0] == "VEH-2026 §2"


def test_03_semantic_query(adaptive_env):
    r = run(adaptive_env, "Is there parking space for cars near the office?")
    assert r.plan.strategy.value == "SEMANTIC" and r.plan.retrievers == ["dense"]


def test_04_multi_intent_query(tmp_path_factory):
    from streaming_integration.test_streaming_integration import ADAPTIVE
    from conftest import make_cfg
    from streamrag.retrieval import build_index
    from streamrag.session import AdaptivePipeline
    from streaming_helpers import hashing_stack
    cfg = make_cfg(tmp_path_factory.mktemp("mi"), corpus="tests/fixtures/corpus_adaptive",
                   **{"multi_intent.enabled": True, "session.enabled": True, **ADAPTIVE})
    p = AdaptivePipeline(hashing_stack(cfg, build_index(cfg)))
    try:
        res = p.process("u1", "How much is the vehicle registration fee and what documents are needed for "
                              "registration?", 1000.0)
    finally:
        p.close()
    assert len(res.adaptive) == 2 and len({a.plan.intent_id for a in res.adaptive}) == 2   # one plan per need
    assert all(a.analysis.n_intents == 2 for a in res.adaptive)


def test_05_multi_hop_query(adaptive_env):
    r = run(adaptive_env, "Does an applicant from Estria have to attend an interview?")
    assert any(h.bridge == "estria -> Group B" for h in r.hops) and "GROUP-RULES §2" in citations(r)


def test_06_ambiguous_query(adaptive_env):
    r = run(adaptive_env, "What about the fee?")
    assert r.analysis.ambiguous and r.assessment.status != "CONTRADICTORY"     # one word: values not compared
    assert r.state.stop_reason is not None and r.evidence.items


def test_07_exact_constraints(adaptive_env):
    r = run(adaptive_env, "How long does processing take for international applicants?")
    assert r.plan.filters.metadata == {"applicant_type": ["international"]}
    assert "INTL-2026 §2" in citations(r) and "DOM-2026 §2" not in citations(r)


def test_08_temporal_query(adaptive_env):
    r = run(adaptive_env, "What was the application fee in December 2025?")
    assert r.plan.strategy.value == "FILTERED" and r.plan.filters.valid_to == "2025-12-31"
    assert citations(r)[0] == "ELIG-2024 §2" and "ELIG-2026 §2" not in citations(r)   # published 2025-11-15,
    # effective 2026-01-01: publication is not validity


def test_09_cached_query(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        ctl.run(AdaptiveRequest("q1", "How soon must a vehicle be registered after purchase?"))
        r = ctl.run(AdaptiveRequest("q2", "After purchase, how soon must a vehicle be registered?"))
        assert r.cache == "hit" and r.ops.searches == 0
    finally:
        ctl.close()


def test_10_stale_cache(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        ctl.run(AdaptiveRequest("q1", "How soon must a vehicle be registered after purchase?"))
        r = ctl.run(AdaptiveRequest("q2", "How soon must a vehicle be registered after purchase?",
                                    session=__import__("streamrag.adaptive.controller", fromlist=["SessionView"])
                                    .SessionView(usable=lambda i: False)))
        assert r.cache == "invalidated" and r.ops.searches >= 1
        assert any(t == "RETRIEVAL_INVALIDATED" and p["reason"] == "stale_evidence" for t, p in r.events)
    finally:
        ctl.close()


def test_11_insufficient_evidence(adaptive_env):
    r = run(adaptive_env, "Can applicants pay the fee by credit card?")
    assert r.assessment.status == "INSUFFICIENT" and r.state.stop_reason == R.NO_EXPECTED_GAIN
    assert r.ops.searches <= 3


def test_12_contradictory_evidence(adaptive_env):
    r = run(adaptive_env, "What time does the PPO open on weekdays?")
    assert r.assessment.status == "CONTRADICTORY" and r.state.stop_reason == R.CONTRADICTION


def test_13_retrieval_timeout(adaptive_env):
    from streamrag.errors import RetrieverTimeoutError
    ctl = controller(adaptive_env)
    real = ctl.svc.search_dense
    ctl.svc.search_dense = lambda *a, **k: (_ for _ in ()).throw(RetrieverTimeoutError("dense timeout"))
    try:
        r = ctl.run(AdaptiveRequest("q", "What are the eligibility requirements for international applicants?"))
    finally:
        ctl.svc.search_dense = real
        ctl.close()
    assert r.evidence.items and all(s["status"] in ("degraded", "ok", "empty") for s in r.searches)


def test_14_budget_exhaustion(adaptive_env):
    b = AdaptiveBudget(max_queries=2, max_results=40, max_iterations=3, max_latency_ms=1500, max_parallel_tasks=2,
                       max_hops=2)
    r = run(adaptive_env, "What documents does an applicant from Zemland need?", **{"adaptive_retrieval.budget": b})
    assert r.ops.searches <= 2 and r.state.stop_reason in (R.QUERY_LIMIT, R.SUFFICIENT_EVIDENCE, R.NO_EXPECTED_GAIN)


def test_15_late_transcript_correction(tmp_path_factory):
    """An ASR revision replaces an earlier chunk: the query built on the wrong words is superseded."""
    from grounding_helpers import grounding_stack
    from runtime_helpers import of
    from streaming_integration.test_streaming_integration import ADAPTIVE
    from streamrag.runtime import StreamingRuntime
    st = grounding_stack(tmp_path_factory, "corpus_adaptive", overrides=ADAPTIVE, verifier="rules")
    rt = StreamingRuntime(st.cfg, st, mode="virtual")
    sid = rt.start_session("s1")
    rt.push_transcript_delta(sid, "u1", "How long does processing take for domestic", at_ms=0)
    rt.push_transcript_delta(sid, "u1", "applicants?", at_ms=400)
    rt.push_transcript_delta(sid, "u1", "How long does processing take for international", stability="final",
                             replaces=0, at_ms=800)
    rt.end_utterance(sid, "u1", at_ms=1400)
    rt.end_session(sid, at_ms=6000)
    rt.run()
    evs = rt.events(sid)
    queries = [e.payload["query_text"] for e in of(evs, "QUERY_GENERATED")]
    assert any("international" in q for q in queries)
    last = [e for e in of(evs, "RETRIEVAL_POLICY_SELECTED")][-1]
    assert last.payload["filters"]["metadata"] == {"applicant_type": ["international"]}


def test_16_entity_correction(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        ctl.run(AdaptiveRequest("q1", "Do applicants from Zemland need an interview?"))
        inv = ctl.cache.invalidate_entities({"zemland"})
        r = ctl.run(AdaptiveRequest("q2", "Do applicants from Norvia need an interview?",
                                    dropped_entities=["Zemland"]))
        assert inv["reason"] == "entity_changed" and r.cache == "miss"
        assert all("zemland" not in s["query"].lower() for s in r.searches)
    finally:
        ctl.close()


def test_17_repeated_question(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        a = ctl.run(AdaptiveRequest("q1", "Who hears the appeals?"))
        b = ctl.run(AdaptiveRequest("q2", "Who hears the appeals?"))
        assert a.ops.searches >= 1 and b.ops.searches == 0 and b.plan.strategy.value == "CACHE_REUSE"
    finally:
        ctl.close()


def test_18_high_frequency_streaming(adaptive_env):
    """Every prefix of a question arriving word by word: each run is bounded, nothing grows unboundedly."""
    ctl = controller(adaptive_env)
    words = "What are the eligibility requirements for international applicants from Zemland".split()
    try:
        t0 = time.perf_counter()
        runs = [ctl.run(AdaptiveRequest(f"q{i}", " ".join(words[:i]))) for i in range(2, len(words) + 1)]
        assert (time.perf_counter() - t0) < 10
        assert all(r.ops.searches <= ctl.ac.budget.max_queries for r in runs)
        assert len(ctl.cache.entries) <= ctl.ac.cache_max_entries
    finally:
        ctl.close()
