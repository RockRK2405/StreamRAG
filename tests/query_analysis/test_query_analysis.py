"""QueryComplexityAnalyzer (docs/retrieval/01): explainable classes, signals from the need and corpus statistics."""

from adaptive_helpers import controller

from streamrag.adaptive.models import QueryComplexity as C


def analyze(env, text, **kw):
    ctl = controller(env)
    try:
        return ctl.analyzer.analyze("q", text, **kw)
    finally:
        ctl.close()


def test_simple_query(adaptive_env):
    a = analyze(adaptive_env, "How often must a residence permit be renewed?")
    assert a.complexity == C.SIMPLE and a.reasons == ["single_need_no_constraints"]
    assert a.value_kind == "frequency" and "often" in a.question_terms


def test_exact_identifier_detected(adaptive_env):
    a = analyze(adaptive_env, "What is Form PX-204 used for?")
    assert a.exact_ids == ["PX-204"] and a.signals["n_exact_ids"] == 1


def test_constraint_becomes_metadata_filter(adaptive_env):
    a = analyze(adaptive_env, "How long does processing take for domestic applicants?")
    assert a.metadata_filters == {"applicant_type": ["domestic"]}
    assert a.complexity == C.MODERATE and "metadata_filter" in a.reasons


def test_temporal_period_and_current_and_past(adaptive_env):
    ctl = controller(adaptive_env)
    try:
        t = ctl.analyzer.temporal("What was the fee in December 2025?")
        assert (t.kind, t.valid_at, t.explicit) == ("as_of", "2025-12-01", True)
        assert ctl.analyzer.period_end(t) == "2025-12-31"
        y = ctl.analyzer.temporal("Which holidays are there in 2026?")
        assert (y.valid_at, ctl.analyzer.period_end(y)) == ("2026-01-01", "2026-12-31")
        cur = ctl.analyzer.temporal("What is the current fee?")
        assert cur.kind == "current" and cur.valid_at == "2026-10-03" and not cur.explicit
        assert ctl.analyzer.temporal("What was the previous fee?").kind == "past"
        assert ctl.analyzer.temporal("What is the fee?").kind == "none"
    finally:
        ctl.close()


def test_multi_hop_from_entity_aspect_gap(adaptive_env):
    a = analyze(adaptive_env, "What documents does an applicant from Zemland need?")
    assert a.complexity == C.MULTI_HOP and a.bridge_terms == ["zemland"]
    assert a.reasons == ["entity_aspect_gap:zemland"]


def test_lowercase_rare_word_is_not_a_multi_hop_entity(adaptive_env):
    a = analyze(adaptive_env, "Is there any way to avoid paying the permit charge?")
    assert a.complexity != C.MULTI_HOP and not a.bridge_terms


def test_question_words_do_not_count_as_content(adaptive_env):
    a = analyze(adaptive_env, "How high is the fee?")
    assert "high" in a.question_terms and a.complexity != C.MULTI_HOP


def test_multi_intent_and_comparison_are_complex(adaptive_env):
    assert analyze(adaptive_env, "What is the fee", n_intents=2).complexity == C.COMPLEX
    a = analyze(adaptive_env, "Compare the business licence fee and the vehicle registration fee")
    assert a.comparison and a.complexity == C.COMPLEX and "comparison" in a.reasons


def test_ambiguous_and_vocabulary_mismatch(adaptive_env):
    a = analyze(adaptive_env, "What about the fee?")
    assert a.ambiguous and "ambiguous" in a.reasons
    b = analyze(adaptive_env, "Is there parking space at the office?")
    assert "vocabulary_mismatch" in b.reasons and set(b.oov_terms) >= {"park", "space"}


def test_every_class_is_explained(adaptive_env):
    for q in ("How often must a residence permit be renewed?", "What documents does an applicant from Zemland need?",
              "How long does processing take for domestic applicants?", "Compare fee and processing time"):
        a = analyze(adaptive_env, q)
        assert a.reasons and all(isinstance(r, str) and r for r in a.reasons)
