"""ContextChangeDetector, semantic diff, change taxonomy, topic frames, late-detail gate (Phase 6, brief §6-9, §28-33).
TEST FIXTURE corpus (fictional)."""

from types import SimpleNamespace

from streamrag.context import CHANGE_TYPES, ContextChangeDetector, semantic_diff
from streamrag.context.cues import late_detail_cue, late_detail_gate

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)

LADDERS = "What are the rules for ladders in the orchard?"


def kinds(r):
    return [c.change_type for c in r.changes]


def test_taxonomy_is_closed_and_documented():
    assert set(CHANGE_TYPES) == {"NO_CHANGE", "REFINEMENT", "CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL", "NEW_INTENT",
                                 "INTENT_REMOVAL", "CORRECTION", "ENTITY_CHANGE", "QUESTION_CHANGE"}


def test_change_carries_semantic_diff_not_raw_strings(p6_stack):
    p, (_, r2) = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    [ch] = r2.changes
    [d] = ch.diffs
    assert d.intent_id == "I1" and d.from_version == 1 and d.to_version == 2
    assert d.constraints_added == ["K1"] and d.terms_added == d.terms_removed == []   # need words unchanged
    assert d.topic_before == d.topic_after
    assert ch.session_version_from >= 1 and ch.affected_queries == ["Q1"]


def test_semantic_diff_ignores_surface_form(p6_stack):
    p, _ = run_turns(p6_stack, [LADDERS])
    tr = engine_of(p).tracker
    it = tr.intents["I1"]
    louder = it.model_copy(update={"resolved_text": it.resolved_text.upper() + " !!", "version": 2})
    d = semantic_diff(it, louder, tr)
    assert d.terms_added == [] and d.terms_removed == []


def test_confidence_is_minimum_of_computed_signals(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    for r in rs:
        for c in r.changes:
            assert c.confidence == min(c.confidence_signals.values(), default=1.0)
            assert all(0.0 <= v <= 1.0 for v in c.confidence_signals.values())


def test_text_change_classification():
    det = ContextChangeDetector(SimpleNamespace(dec=SimpleNamespace(terms_fn=lambda t: t.lower().split())))

    def it(text, topic, aspect="rules", typ="REQUIREMENT"):
        return SimpleNamespace(topic=topic, aspect=aspect, intent_type=typ, resolved_text=text)

    diff = SimpleNamespace(terms_removed=[])
    assert det._text_change(it("rules ladders", "ladders"), it("rules ladders orchard", "ladders"), diff) == "REFINEMENT"
    diff = SimpleNamespace(terms_removed=["ladders"])
    assert det._text_change(it("rules ladders", "ladders"), it("rules crates", "crates"), diff) == "ENTITY_CHANGE"
    assert det._text_change(it("rules ladders", "ladders"), it("cost ladders", "ladders", aspect="cost"),
                            diff) == "QUESTION_CHANGE"


def test_follow_up_vs_new_question(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "What about overnight?", "What about crates?",
                                 "Now explain how the telescope is recalibrated."])
    assert kinds(rs[1]) == ["CONSTRAINT_ADDITION"]                    # the corpus discusses ladders overnight
    assert kinds(rs[2]) == ["NEW_INTENT"] and rs[2].changes[0].relation == "follow_up"
    eng = engine_of(p)
    assert eng.tracker.intents["I2"].aspect == "rules"                # parallel need inherits the facet only
    assert eng.tracker.intents["I2"].constraint_ids == []             # ...not the other need's constraint
    assert kinds(rs[3]) == ["NEW_INTENT"] and rs[3].changes[0].relation == "independent"
    assert rs[3].changes[0].frame_action == "new_frame"


def test_constraint_update_replaces_previous_value(p6_stack):
    p, rs = run_turns(p6_stack, ["What are the rules for crates in the orchard?", "Only for the night shift.",
                                 "Sorry, for the day shift."])
    eng = engine_of(p)
    ks = list(eng.tracker.constraints.values())
    assert [k.text for k in ks] == ["for the night shift", "for the day shift"]
    assert ks[0].status == "retracted" and ks[1].replaces == ks[0].constraint_id and ks[1].op == "update"
    assert sorted(kinds(rs[2])) == ["CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL"]


def test_frames_isolate_and_reactivate(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Now explain how the telescope is recalibrated.",
                                 "Back to the ladders, what about overnight?"])
    fm = engine_of(p).frames
    assert [(f.frame_id, f.status) for f in fm.frames] == [("T1", "active"), ("T2", "dormant")]
    assert rs[2].changes[0].frame_action == "reactivated" and rs[2].changes[0].affected_intents == ["I1"]
    assert "named_topic" in rs[2].changes[0].cue


def test_late_detail_cue_requires_content(p6_stack):
    lx = p6_stack.intent_stack.decomposer.lx
    terms = p6_stack.intent_stack.decomposer.terms_fn
    assert late_detail_cue("Specifically overnight.", lx, terms)
    assert late_detail_cue("Actually, ignore the overnight restriction.", lx, terms)
    assert late_detail_cue("Only for students.", lx, terms)
    assert not late_detail_cue("Specifically", lx, terms)              # marker without content (mid-stream)
    assert not late_detail_cue("Actually, ignore", lx, terms)
    assert not late_detail_cue("Okay.", lx, terms)
    no = SimpleNamespace(reason="backchannel")
    yes = SimpleNamespace(reason="not_retrieval_worthy")
    assert not late_detail_gate(no, True, "Specifically overnight.", lx, terms)       # suppressed acts never open
    assert not late_detail_gate(yes, False, "Specifically overnight.", lx, terms)     # nothing to refine
    assert late_detail_gate(yes, True, "Specifically overnight.", lx, terms)


def test_net_change_types_summarise_provisional_streaming_changes():
    from streamrag.context import net_change_types
    from streamrag.context.models import ContextChange, SemanticDiff

    def ch(t, aff=(), new=(), add=(), rem=()):
        return ContextChange(change_id="CH", change_type=t, utterance_id="u1", affected_intents=list(aff),
                             new_intents=list(new), confidence=1.0,
                             diffs=[SemanticDiff(intent_id="I1", constraints_added=list(add),
                                                 constraints_removed=list(rem))])
    # need created this turn, then refined and constrained while the utterance streamed -> just NEW_INTENT
    assert net_change_types([ch("NEW_INTENT", new=["I1"]), ch("REFINEMENT", aff=["I1"]),
                             ch("CONSTRAINT_ADDITION", aff=["I1"], add=["K1"])]) == ["NEW_INTENT"]
    # late detail revised mid-utterance (K1 "for fruit" -> K2 "for fruit picked after sunset") -> one addition
    assert net_change_types([ch("CONSTRAINT_ADDITION", aff=["I9"], add=["K1"]),
                             ch("CONSTRAINT_ADDITION", aff=["I9"], add=["K2"]),
                             ch("CONSTRAINT_REMOVAL", aff=["I9"], rem=["K1"])]) == ["CONSTRAINT_ADDITION"]
    # a fragment need created and dropped again leaves nothing
    assert net_change_types([ch("NEW_INTENT", new=["I2"]), ch("INTENT_REMOVAL", aff=["I2"])]) == ["NO_CHANGE"]
    assert net_change_types([]) == []
