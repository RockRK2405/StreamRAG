"""Phase 6 CRITICAL TEST CASES 1-10 (brief §46) on the TEST FIXTURE corpus (fictional; mechanics only).

The brief's abstract X / Y / "international applicants" are instantiated with entities of the fixture corpus
(ladders / crates / overnight / night shift); what is tested is the change type and the targeted update, never
retrieval quality.
"""

from streamrag.context.models import ContextChange, SemanticDiff
from streamrag.delta.evidence import EvidenceStore, EvidenceValidityManager

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)
from streaming_helpers import make_es

LADDERS = "What are the rules for ladders in the orchard?"


def types(res):
    return [c.change_type for c in res.changes]


def test_case_1_constraint_addition_not_new_intent(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    assert types(r2) == ["CONSTRAINT_ADDITION"]
    ch = r2.changes[0]
    assert ch.affected_intents == ["I1"] and ch.new_intents == [] and ch.added_constraints == ["overnight"]
    eng = engine_of(p)
    assert [i for i, it in eng.tracker.intents.items() if it.status == "ACTIVE"] == ["I1"]   # no new need
    [a] = r2.plan.queries_to_create                              # delta query = previous query + the delta
    assert a.intent_id == "I1" and a.query.text == "What are the rules for ladders in the orchard overnight"
    assert a.parent_query_id == "Q1" and a.supersedes_query_id == "Q1" and a.change_id == ch.change_id
    assert eng.ledger.get("Q2").parent_query_id == "Q1" and eng.ledger.get("Q2").derived_from_change_id == ch.change_id


def test_case_2_entity_replacement(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, ["What are the requirements for ladders?", "Actually, I meant crates."])
    [ch] = r2.changes
    assert ch.change_type == "CORRECTION" and "entity_replacement" in ch.cue
    assert ch.superseded_intents == ["I1"] and ch.new_intents == ["I2"]
    eng = engine_of(p)
    assert eng.tracker.intents["I1"].status == "SUPERSEDED" and eng.tracker.intents["I2"].topic == "crates"
    assert r2.plan.queries_to_supersede == ["Q1"]
    [a] = r2.plan.queries_to_create                                    # lineage continues across the correction
    assert a.intent_id == "I2" and a.parent_query_id == "Q1" and a.reason == "full_query_for_corrected_need"
    q1, q2 = eng.ledger.get("Q1"), eng.ledger.get("Q2")
    assert q1.stale and q1.stale_reason == "intent_superseded" and q2.parent_query_id == "Q1"
    assert all(a.status == "SUPERSEDED" for a in eng.store.for_intent("I1"))          # old evidence superseded
    assert {c.status for c in (eng.graph.claims[x] for x in eng.graph.claims_of("I1"))} == {"SUPERSEDED"}
    assert r2.answer is not None and [s.intent_id for s in r2.answer.sections] == ["I2"]
    assert set(r2.answer.diff.retracted) == set(eng.graph.claims_of("I1"))
    assert eng.frames.frame_of("I2").frame_id == eng.frames.frame_of("I1").frame_id == "T1"   # same topic frame


def test_case_3_new_intent(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, [LADDERS, "And how is the lens cleaned?"])
    assert types(r2) == ["NEW_INTENT"] and r2.changes[0].new_intents == ["I2"]
    assert [a.intent_id for a in r2.plan.queries_to_create] == ["I2"]                     # I1 is not re-retrieved
    assert r2.plan.claims_to_revalidate == []


def test_case_4_backchannel_changes_nothing(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, [LADDERS, "Okay."])
    assert r2.gate.startswith("closed") and types(r2) == ["NO_CHANGE"]
    assert r2.plan is None and r2.retrievals == 0 and r2.answer is None
    eng = engine_of(p)
    assert len(eng.ledger.all()) == 1 and len(eng.answers.versions) == 1


def test_case_5_constraint_removal(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, ["What are the rules for ladders in the orchard, especially overnight?",
                                       "Ignore the overnight restriction."])
    [ch] = r2.changes
    assert ch.change_type == "CONSTRAINT_REMOVAL" and ch.removed_constraints == ["overnight"]
    eng = engine_of(p)
    k = next(iter(eng.tracker.constraints.values()))
    assert k.status == "retracted" and k.retracted_in == "u2"
    assert eng.tracker.intents["I1"].constraint_ids == []
    assert r2.plan.queries_to_create[0].query.text == "What are the rules for ladders in the orchard"
    # evidence specific to the removed constraint was revalidated, general evidence retained
    assert any(t.endswith("@I1") for t in r2.plan.evidence_to_revalidate)
    assert r2.plan.evidence_to_retain


def test_case_6_constraint_affects_only_some_evidence(fixture_bundle):
    """Q1 retrieved E1, E2, E3; the new constraint touches only E2 -> E2 REVALIDATION_REQUIRED, E1/E3 reusable."""
    _, b = fixture_bundle
    terms = lambda t: list(dict.fromkeys(b.analyzer.tokens(t)))  # noqa: E731
    store = EvidenceStore()
    es = make_es("q", ["E1", "E2", "E3"])
    texts = {"E1": "Ladders must be inspected before use.", "E2": "Ladders used on the day shift are logged.",
             "E3": "A second worker holds the ladder base."}
    es = es.model_copy(update={"items": [e.model_copy(update={"text": texts[e.evidence_id]}) for e in es.items]})
    store.add_results("I1", 1, "Q1", es, 0.0)
    vm = EvidenceValidityManager(store, terms)
    ch = ContextChange(change_id="CH2", change_type="CONSTRAINT_ADDITION", utterance_id="u2", affected_intents=["I1"],
                       diffs=[SemanticDiff(intent_id="I1", constraints_added=["K1"])], confidence=1.0)
    acts = vm.apply(ch, lambda k: "for the night shift", lambda i: None, 1.0)
    got = {a.evidence_id: (a.decision, a.to_status, a.rule) for a in acts}
    assert got["E2"] == ("REVALIDATE", "REVALIDATION_REQUIRED", "constraint_dimension_other_value")
    assert got["E1"][:2] == got["E3"][:2] == ("RETAIN", "RETAINED")
    # delta retrieval that does not return E2 again -> STALE; E1/E3 stay usable
    vm.confirm_after_retrieval("I1", {"E1"}, "Q2", 2.0)
    assert store.assign[("E2", "I1")].status == "STALE"
    assert {a.evidence_id for a in store.usable("I1")} == {"E1", "E3"}


def test_case_7_entity_change_makes_old_evidence_stale(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, [LADDERS, "Sorry, I meant crates instead of ladders."])
    eng = engine_of(p)
    old = eng.store.for_intent("I1")
    assert old and all(a.status == "SUPERSEDED" and a.history[-1].rule == "intent_superseded" for a in old)
    carried = [a for a in eng.store.for_intent("I2") if a.history[0].rule == "carried_to_correction"]
    # evidence that names the new entity is carried over but must be revalidated (never blindly reused)
    assert all("crate" in eng.store.text(a.evidence_id).lower() for a in carried)
    assert all(t.rule in ("carried_to_correction", "confirmed_by_delta_retrieval", "not_confirmed_by_delta_retrieval")
               for a in carried for t in a.history)


def test_case_8_unrelated_detail_triggers_no_unnecessary_retrieval(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Now explain how the telescope is recalibrated.", "Okay, thanks."])
    eng = engine_of(p)
    assert [a.intent_id for a in rs[1].plan.queries_to_create] == ["I2"]
    assert set(rs[1].plan.claims_unaffected) >= set(eng.graph.claims_of("I1"))           # I1 claims untouched
    assert rs[2].retrievals == 0 and rs[2].plan is None
    assert [r.intent_id for r in eng.ledger.all()] == ["I1", "I2"]


def test_case_9_context_isolation_across_unrelated_questions(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Now explain how the telescope is recalibrated.",
                                 "How is the lens cleaned?", "Back to the ladders, what about crates?"])
    eng = engine_of(p)
    f_ladders, f_tel = eng.frames.frame_of("I1"), eng.frames.frame_of("I2")
    assert f_ladders.frame_id != f_tel.frame_id
    k = [c for c in eng.tracker.constraints.values() if c.text == "overnight"][0]
    assert k.applies_to == ["I1"]                                    # the late detail never leaked to other needs
    for iid in ("I2", "I3"):
        q = eng.ledger.active_for_intent(iid)
        assert "overnight" not in q.query_text and "ladder" not in q.query_text.lower()
    # context selection for the telescope frame excludes the ladders frame
    from streamrag.context import RelevantContextSelector
    pkg = RelevantContextSelector(eng.memory).select("structured", intent_id="I2")
    assert {i.intent_id for i in pkg.items if i.intent_id} == {"I2"}
    assert all("ladder" not in i.text.lower() for i in pkg.items if i.kind in ("need", "constraint"))


def test_case_10_same_question_rephrased_reuses_evidence(p6_stack):
    p, (r1, r2) = run_turns(p6_stack, [LADDERS, "Tell me the orchard ladder rules."])
    assert r2.retrievals == 0 and r2.cache_hits == 1
    [a] = r2.plan.queries_to_reuse
    assert a.action == "cache_hit" and a.reused_query_id == "Q1"
    eng = engine_of(p)
    reused = eng.ledger.all()[-1]
    assert reused.status == "reused" and reused.reused_from == "Q1" and reused.evidence_ids == eng.ledger.get("Q1").evidence_ids


def test_case_10_reuse_is_unsafe_after_index_change(p6_stack):
    """The semantic cache key includes the index content hash: a different corpus snapshot never reuses evidence."""
    p, _ = run_turns(p6_stack, [LADDERS])
    eng = engine_of(p)
    terms = eng.ledger.get("Q1").terms
    from streamrag.delta.planner import SemanticCache
    other = SemanticCache("another-index", eng.cache.options_hash, True)
    assert other.key(terms) != eng.cache.key(terms)
