"""Contradiction-aware retrieval (brief §64) and temporal conflict resolution (§40)."""

from adaptive_helpers import citations, controller, evidence, run


def test_conflict_detected_then_targeted_search(adaptive_env):
    r = run(adaptive_env, "When does the Permit Processing Office open?")
    assert r.assessment.status == "CONTRADICTORY" and r.state.stop_reason.value == "CONTRADICTION"
    req = next(q for q in r.requirements if q.status == "CONFLICT")
    vals = {v for vs in req.values.values() for v in vs}
    assert vals == {"09:00", "08:30"}
    cs = [s for s in r.searches if s["action"] == "contradiction_search"]
    assert len(cs) == 1                                  # one targeted search, not more random documents
    allowed = set(cs[0]["filters"]["document_ids"])
    assert "NOTICE-A" not in allowed and "NOTICE-B" not in allowed   # it looks for a *third* source
    assert "current version effective" in cs[0]["query"]
    assert {"NOTICE-A §1", "NOTICE-B §1"} <= set(citations(r))      # both sides are handed on (reported)


def test_contradiction_search_can_be_disabled(adaptive_env):
    r = run(adaptive_env, "When does the Permit Processing Office open?",
            **{"adaptive_retrieval.contradiction_retrieval": False})
    assert r.state.stop_reason.value == "CONTRADICTION" and r.ops.searches == 1


def test_temporal_conflict_resolved_by_validity_with_reason(adaptive_env):
    r = run(adaptive_env, "What is the current application fee for a residence permit?")
    assert r.assessment.status == "SUFFICIENT"
    assert "ELIG-2024 §2" not in citations(r) and "ELIG-2026 §2" in citations(r)


def test_past_question_keeps_the_old_version(adaptive_env):
    ctl = controller(adaptive_env)
    _, svc = adaptive_env
    try:
        text = "What was the previous application fee for a residence permit?"
        a = ctl.analyzer.analyze("q", text)
        assert a.temporal.kind == "past"
        reqs = ctl.builder.build(a, text, None)
        pool = [evidence(svc, c) for c in ("ELIG-2024 §2", "ELIG-2026 §2", "ELIG-2026 §3")]
        st, exc = ctl.evaluator.assess(reqs, pool, "past")
        assert st.status == "SUFFICIENT" and reqs[0].resolution == "past_version"
        assert exc == {"ELIG-2026§2#1": "conflict_resolution:past_version:R1"}
        assert "ELIG-2026§3#1" in reqs[0].evidence_ids   # also states the old value: not discarded
    finally:
        ctl.close()


def test_supersession_needs_both_documents_to_agree(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        assert ctl.catalog.superseded == {"ELIG-2024": "ELIG-2026"}
        assert ctl.catalog.authority("ELIG-2024") < ctl.catalog.authority("ELIG-2026")
    finally:
        ctl.close()


def test_values_for_different_populations_are_not_a_conflict(adaptive_env):
    ctl = controller(adaptive_env)
    _, svc = adaptive_env
    try:
        text = "How long does processing take?"
        a = ctl.analyzer.analyze("q", text)
        reqs = ctl.builder.build(a, text, ctl.ref_date)
        st, _ = ctl.evaluator.assess(reqs, [evidence(svc, "DOM-2026 §2"), evidence(svc, "INTL-2026 §2")], "none")
        assert st.status != "CONTRADICTORY"              # 10 vs 30 days: domestic vs international applicants
    finally:
        ctl.close()
