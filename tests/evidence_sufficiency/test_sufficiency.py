"""Claim-driven requirements and the EvidenceSufficiencyEvaluator (docs/retrieval/05, 09)."""

from adaptive_helpers import controller, evidence

from streamrag.adaptive.models import EvidenceRequirement


def build(env, text):
    ctl = controller(env)
    a = ctl.analyzer.analyze("q", text, cue_text=text)
    reqs = ctl.builder.build(a, text, ctl.ref_date, ctl.analyzer.period_end(a.temporal))
    return ctl, a, reqs


def test_requirements_per_claim_slot(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "How long does processing take for domestic applicants?")
    try:
        assert [(r.kind, r.value_kind) for r in reqs] == [("value", "duration"), ("condition", "duration")]
        assert reqs[1].metadata == {"applicant_type": ["domestic"]}
        assert "take" not in reqs[0].terms and "long" not in reqs[0].terms       # question words are not content
    finally:
        ctl.close()


def test_value_slot_needs_a_value_not_just_the_topic(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "How much does it cost to renew a residence permit?")
    _, svc = adaptive_env
    try:
        st, _ = ctl.evaluator.assess(reqs, [evidence(svc, "RENEW §1"), evidence(svc, "ELIG-2026 §2")], "none")
        assert st.status == "INSUFFICIENT"            # renewal text has no amount; the fee text is not about renewal
    finally:
        ctl.close()


def test_sufficient_with_value_and_condition(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "How long does processing take for domestic applicants?")
    _, svc = adaptive_env
    try:
        st, exc = ctl.evaluator.assess(reqs, [evidence(svc, "DOM-2026 §2"), evidence(svc, "INTL-2026 §2")], "none")
        assert st.status == "SUFFICIENT" and st.coverage == 1.0
        assert exc == {"INTL-2026§2#1": "not_applicable:applicant_type"}   # another population: not usable
    finally:
        ctl.close()


def test_general_rule_meets_a_condition(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "What is the application fee for international applicants?")
    _, svc = adaptive_env
    try:
        st, _ = ctl.evaluator.assess(reqs, [evidence(svc, "ELIG-2026 §2")], "none")
        assert st.status == "SUFFICIENT"              # applicant_type: all -> the general fee applies
    finally:
        ctl.close()


def test_partial_and_unattainable(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "What are the eligibility requirements for international applicants?")
    _, svc = adaptive_env
    try:
        st, _ = ctl.evaluator.assess(reqs, [evidence(svc, "ELIG-2026 §1")], "none")
        assert st.status == "PARTIAL" and st.met == ["R1"] and st.unmet == ["R2"]
    finally:
        ctl.close()
    ctl, a, reqs = build(adaptive_env, "Is there parking space at the office?")
    try:
        st, _ = ctl.evaluator.assess(reqs, [evidence(svc, "FORMS §2")], "none")
        assert st.status == "INSUFFICIENT" and st.unattainable == ["R1"]
    finally:
        ctl.close()


def test_temporal_validity_excludes_expired_source(adaptive_env):
    ctl, a, reqs = build(adaptive_env, "What is the application fee for a residence permit?")
    _, svc = adaptive_env
    try:
        st, exc = ctl.evaluator.assess(reqs, [evidence(svc, "ELIG-2026 §2"), evidence(svc, "ELIG-2024 §2")], "none")
        assert st.status == "SUFFICIENT" and exc == {"ELIG-2024§2#1": "temporal_validity:R1"}
    finally:
        ctl.close()


def test_acronym_and_morphology_matching(adaptive_env):
    ctl, _, _ = build(adaptive_env, "x")
    _, svc = adaptive_env
    try:
        r = EvidenceRequirement(requirement_id="R1", claim_slot="t", kind="fact", terms=["ppo", "open"])
        assert ctl.evaluator.supports(r, evidence(svc, "NOTICE-A §1"))[0]       # "Permit Processing Office" = PPO
        r2 = EvidenceRequirement(requirement_id="R1", claim_slot="t", kind="fact", terms=["pay", "appli"])
        assert ctl.evaluator.supports(r2, evidence(svc, "ELIG-2026 §2"))[0]     # appli ~ applic (stem variant)
    finally:
        ctl.close()


def test_single_requirement_when_claim_driven_off(adaptive_env):
    ctl = controller(adaptive_env, **{"adaptive_retrieval.claim_driven": False})
    try:
        text = "How long does processing take for domestic applicants?"
        a = ctl.analyzer.analyze("q", text)
        reqs = ctl.builder.build(a, text, None)
        assert len(reqs) == 1 and reqs[0].kind == "fact" and "domest" in reqs[0].terms
    finally:
        ctl.close()
