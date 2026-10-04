"""Phase 9 integration with the Phase 6 session pipeline and the Phase 8 runtime (brief §52-§55)."""

import pytest
from runtime_helpers import of, virtual

from conftest import make_cfg
from grounding_helpers import grounding_stack
from streamrag.retrieval import build_index
from streamrag.session import AdaptivePipeline
from streaming_helpers import hashing_stack

ADAPTIVE = {"adaptive_retrieval.enabled": True, "adaptive_retrieval.reference_date": "2026-10-03"}


@pytest.fixture(scope="module")
def ad_stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p9s")
    cfg = make_cfg(tmp, corpus="tests/fixtures/corpus_adaptive", **{"multi_intent.enabled": True,
                                                                     "session.enabled": True, **ADAPTIVE})
    return hashing_stack(cfg, build_index(cfg))


def turns(stack, texts):
    p = AdaptivePipeline(stack)
    try:
        return p, [p.process(f"u{n}", t, n * 3000.0) for n, t in enumerate(texts, start=1)]
    finally:
        p.close()


def test_late_constraint_retrieves_only_the_delta(ad_stack):
    p, rs = turns(ad_stack, ["How long does processing take for a residence permit?", "For international applicants."])
    assert [c.change_type for c in rs[1].changes] == ["CONSTRAINT_ADDITION"]
    (a,) = rs[1].adaptive                                  # one need re-planned, nothing restarted
    assert a.plan.strategy.value == "FILTERED" and a.plan.filters.metadata == {"applicant_type": ["international"]}
    assert "INTL-2026 §2" in [e.citation for e in a.evidence.items]


def test_entity_correction_invalidates_and_uses_the_corrected_entity(ad_stack):
    p, rs = turns(ad_stack, ["Do applicants from Zemland need an interview?", "Sorry, I meant Norvia, not Zemland."])
    assert [c.change_type for c in rs[1].changes] == ["CORRECTION"]
    inv = [e.payload for e in p.events if e.type.value == "RETRIEVAL_INVALIDATED"]
    assert inv and inv[0]["reason"] == "entity_changed" and "zemland" in inv[0]["detail"]
    (a,) = rs[1].adaptive
    assert all("zemland" not in s["query"].lower() for s in a.searches)
    assert any(h.bridge == "norvia -> Group A" for h in a.hops)


def test_repeated_question_is_not_retrieved_again(ad_stack):
    p, rs = turns(ad_stack, ["What is the application fee for a residence permit?",
                             "How long does processing take for domestic applicants?",
                             "What is the application fee for a residence permit?"])
    assert rs[2].retrievals == 0 and rs[2].cache_hits == 1


def test_cross_query_evidence_reuse(ad_stack):
    p, rs = turns(ad_stack, ["What is the application fee for a residence permit?",
                             "Is there a reduced application fee for applicants under 25?"])
    (a,) = rs[1].adaptive
    assert a.plan.strategy.value == "SESSION_REUSE" and a.ops.searches == 0
    assert [e.citation for e in a.evidence.items] == ["ELIG-2026 §2"]
    assert rs[1].retrievals == 0


def test_adaptive_events_reach_the_session_bus(ad_stack):
    p, rs = turns(ad_stack, ["What documents does an applicant from Zemland need?"])
    types = {e.type.value for e in p.events if e.component == "adaptive_retrieval"}
    assert {"RETRIEVAL_POLICY_SELECTED", "QUERY_REWRITTEN", "CACHE_MISS", "RETRIEVAL_STARTED", "RETRIEVAL_COMPLETED",
            "EVIDENCE_ASSESSED", "RETRIEVAL_EXPANDED", "RETRIEVAL_STOPPED", "HOP_CREATED"} <= types
    assert all(e.query_id for e in p.events if e.component == "adaptive_retrieval")


# ------------------------------------------------------------------ Phase 8 runtime
@pytest.fixture(scope="module")
def rt_ad_stack(tmp_path_factory):
    return grounding_stack(tmp_path_factory, "corpus_adaptive", overrides=ADAPTIVE, verifier="rules")


def test_runtime_virtual_adaptive_turn(rt_ad_stack):
    rt, evs = virtual(rt_ad_stack, [["What documents does", "an applicant from", "Zemland need?"]])
    pol = of(evs, "RETRIEVAL_POLICY_SELECTED")
    assert pol and pol[-1].payload["strategy"] == "MULTI_HOP"
    assert of(evs, "HOP_CREATED") and of(evs, "TURN_COMPLETED")
    final = of(evs, "ANSWER_COMMITTED")[-1].payload["text"]
    assert "Group B" in final
    tasks = [e for e in of(evs, "TASK_SCHEDULED") if e.payload.get("kind") == "adaptive"]
    assert tasks and all(e.payload["task_type"] == "retrieval" for e in tasks)


def test_runtime_high_frequency_streaming_is_bounded(rt_ad_stack):
    words = "What are the eligibility requirements for international applicants from Zemland please".split()
    rt, evs = virtual(rt_ad_stack, [words])                # one word per chunk
    started = [e for e in of(evs, "RETRIEVAL_STARTED") if e.payload.get("adaptive_search")]
    stops = of(evs, "RETRIEVAL_STOPPED")
    assert stops and all(s.payload["searches"] <= 5 for s in stops)
    assert len(started) <= 5 * len(stops)
    assert of(evs, "TURN_COMPLETED")


def test_runtime_superseded_adaptive_work_is_cancelled(rt_ad_stack):
    """A late detail supersedes the running adaptive retrieval (1.2 s remote dense latency, injected): its task is
    cancelled cooperatively and only the new need's retrieval is applied."""
    from streamrag.runtime import Fault, FaultInjector
    faults = FaultInjector([Fault("dense", "delay", times=-1, delay_ms=1200)])
    rt, evs = virtual(rt_ad_stack, [["How long does", "processing take", "for international", "applicants?"]],
                      faults=faults)
    cancelled = [e for e in of(evs, "TASK_CANCELLED") if e.payload.get("kind") == "adaptive"]
    assert [e.query_id for e in cancelled] == ["Q1"]
    assert [e.payload["reason"] for e in of(evs, "RETRIEVAL_CANCELLED")] == ["superseded_in_flight"]
    assert [e.query_id for e in of(evs, "RETRIEVAL_STOPPED")] == ["Q2"]      # Q1's buffered events never emitted
    assert of(evs, "TURN_COMPLETED")
