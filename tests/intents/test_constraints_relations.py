"""Global vs local constraints, intent relationships and context carry-over (brief §6, §7, §14; TESTS 4, 10)."""

from intent_helpers import dec, qb, tracker  # noqa: F401  (pytest fixtures)

from streamrag.intents.decomposer import DecompositionContext


def scopes(d):
    return [(k.text, k.scope, k.applies_to, k.scope_reason) for k in d.constraints]


def test_trailing_restriction_after_coordination_is_global(dec):
    d = dec.decompose("Tell me about the dome and the telescope for visitors from abroad.", "u1")
    assert [i.text for i in d.active] == ["the dome", "the telescope"]
    assert scopes(d) == [("for visitors from abroad", "global", ["d1", "d2"], "trailing_pp_after_coordination")]


def test_shared_topic_is_distributed_not_a_constraint(dec):
    d = dec.decompose("what are the rules and the schedule for the telescope", "u1")
    assert d.constraints == []
    first = d.active[0]
    assert [s.text for s in first.inherited] == ["for the telescope"] and first.inherited[0].reason == "distributed_pp"


def test_fronted_restriction_applies_to_every_need(dec):
    d = dec.decompose("For visitors, what are the rules and when is the dome opened?", "u1")
    assert len(d.active) == 2
    assert scopes(d) == [("For visitors", "global", ["d1", "d2"], "fronted_restriction")]
    assert d.constraints[0].scope_confidence == 1.0


def test_focus_sharing_a_term_is_local(dec):
    d = dec.decompose("Tell me the requirements for ladders, especially the safety requirement.", "u1")
    assert scopes(d) == [("the safety requirement", "local", ["d1"], "shares_term_with_one_intent")]


def test_focus_with_question_refines_previous_need(dec):
    d = dec.decompose("I was wondering about the harvest limits specifically how high the crates can be stacked", "u1")
    assert [i.text for i in d.active] == ["the harvest limits"]
    assert scopes(d)[0][1:] == ("local", ["d1"], "focus_question_refines_previous")


def test_ambiguous_trailing_restriction_is_global_with_lower_scope_confidence(dec):
    d = dec.decompose("I need information about the fog signal and also the lens especially during a storm", "u1")
    k = d.constraints[0]
    assert (k.scope, k.applies_to, k.scope_confidence) == ("global", ["d1", "d2"], 0.6)
    assert d.confidence < min(i.confidence for i in d.active)      # ambiguity lowers decomposition confidence


def test_explicit_all_scope(dec):
    d = dec.decompose("Tell me about the wicks and the lens, for both of them during a storm.", "u1")
    assert d.constraints and d.constraints[0].scope_reason == "explicit_all"
    assert d.constraints[0].applies_to == ["d1", "d2"]


# ---------------------------------------------------------------------------------------- relationships
def test_10_pronoun_resolved_within_utterance(dec):
    d = dec.decompose("Tell me about the lens. And how does it apply there?", "u1")
    i2 = d.active[1]
    assert i2.resolved_text == "how does lens apply"
    assert i2.unresolved == ["there"]
    assert [(r.type, r.source, r.target) for r in d.relations] == [("DEPENDENT", "d2", "d1")]


def test_pronoun_with_antecedent_inside_the_need_is_not_replaced(dec):
    d = dec.decompose("what do workers do before they leave the tool shed", "u1")
    assert d.active[0].resolved_text == "what do workers do before they leave the tool shed"


def test_aspect_only_need_inherits_topic_within_utterance(dec):
    d = dec.decompose("What are the requirements for ladders? And the deadlines?", "u1")
    i2 = d.active[1]
    assert [s.text for s in i2.inherited] == ["ladders"] and i2.inherited[0].reason == "follow_up_ellipsis"


def test_follow_up_inherits_only_relevant_context(tracker):
    tracker.update("u1", "What are the eligibility requirements for the observatory visits?", 0)
    iset, _, _ = tracker.update("u2", "And what about the application process?", 1000)
    i = iset.intents[0]
    assert [c.text for c in i.inherited_context] == ["observatory visits"]
    assert i.inherited_context[0].source_span.utterance_id == "u1"
    assert [(r.type, r.target) for r in iset.relationships] == [("FOLLOW_UP", "I1")]
    # Phase 6: a follow-up naming its own topic is a *parallel* question: it inherits the previous need's aspect,
    # never its topic (no blind copying of X-specific context; the corpus has no chunk with both lens and visits)
    iset3, _, d3 = tracker.update("u3", "And what about the lens?", 2000)
    assert [c.text for c in iset3.intents[0].inherited_context] == ["application process"]
    assert "observatory" not in iset3.intents[0].resolved_text
    assert d3.follow_ups[0]["decision"] == "parallel"


def test_cross_utterance_pronoun(tracker):
    tracker.update("u1", "Tell me about the lens.", 0)
    iset, _, _ = tracker.update("u2", "And how often is it cleaned?", 1000)
    assert iset.intents[0].resolved_text == "how often is lens cleaned"


def test_carryover_can_be_disabled(fixture_bundle):
    from streamrag.intents import make_decomposer
    cfg, b = fixture_bundle
    off = make_decomposer(cfg.model_copy(update={"multi_intent": cfg.multi_intent.model_copy(
        update={"carryover": False})}), b)
    d = off.decompose("Tell me about the lens. And how does it apply there?", "u1")
    assert d.active[1].resolved_text == "how does it apply there"


def test_context_object_defaults():
    assert DecompositionContext().prior == [] and DecompositionContext().recent == []
