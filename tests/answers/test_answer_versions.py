"""AnswerVersion, answer diff, minimal regeneration, answer-state API (Phase 6, brief §24-27). TEST FIXTURE corpus."""

from streamrag.models.answers import AnswerVersion

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)

LADDERS = "What are the rules for ladders in the orchard?"


def test_versions_link_and_carry_required_fields(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Okay.", "Actually, ignore the overnight restriction."])
    vs = engine_of(p).answers.versions
    assert [v.answer_id for v in vs] == ["A1", "A2", "A3"]                     # "Okay." committed nothing
    assert [v.supersedes_answer_id for v in vs] == [None, "A1", "A2"]
    assert [v.kind for v in vs] == ["initial", "refinement", "refinement"]
    for v in vs:
        AnswerVersion.model_validate(v.model_dump())
        assert v.claim_ids and v.evidence_ids and v.session_version >= 1 and v.change_summary
        assert v.text == ""                                                    # rendering is Phase 7
    assert vs[1].frame_slots and list(vs[1].frame_slots.values()) == ["overnight"] and vs[2].frame_slots == {}
    assert vs[1].delta_queries == ["Q2"] and not vs[1].full_rerun


def test_diff_and_minimal_regeneration(p6_stack):
    """Two needs in one frame; a late detail aimed at one of them re-renders only its section."""
    p, rs = run_turns(p6_stack, [LADDERS, "What about crates?", "Back to the ladders, what about overnight?"])
    last = engine_of(p).answers.versions[-1]
    assert last.topic_id == "T1" and [s.intent_id for s in last.sections] == ["I1", "I2"]
    status = {s.intent_id: (s.status, s.needs_regeneration) for s in last.sections}
    assert status == {"I1": ("changed", True), "I2": ("unchanged", False)}
    d = last.diff
    assert d.sections_changed == ["S-I1"] and d.sections_unchanged == ["S-I2"]
    crates = next(s for s in last.sections if s.intent_id == "I2")
    assert set(crates.claim_ids) <= set(d.kept)                              # untouched claims reused as they are
    assert set(d.kept).isdisjoint({m.claim_id for m in d.modified}) and set(d.kept).isdisjoint(d.added)


def test_compare_answer_versions_api(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight."])
    am = engine_of(p).answers
    a1, a2 = am.versions
    d = am.compare_answer_versions(a1, a2)
    assert d == a2.diff
    assert [(m.claim_id, m.from_status, m.to_status) for m in d.modified] == [
        (c, "SUPPORTED", "PARTIALLY_SUPPORTED") for c in a2.claim_ids if c in {m.claim_id for m in d.modified}]
    assert len(d.kept) + len(d.modified) == len(a1.claim_ids) and d.added == d.retracted == []
    assert am.get_current_answer_state().answer_id == "A2"
    assert am.update_answer_state("u9", 99, 1.0) is None                     # nothing changed -> no new version


def test_correction_retracts_old_claims(p6_stack):
    p, (_, r2) = run_turns(p6_stack, [LADDERS, "Sorry, I meant crates instead of ladders."])
    d = r2.answer.diff
    assert d.retracted and d.added and not d.kept
    assert d.sections_changed == ["S-I1"]                                    # same section lineage, re-rendered
    assert any("Doc_07 §3" == c for c in d.citations_added)


def test_uncertainty_when_constraint_not_covered(p6_stack):
    p, (_, r2) = run_turns(p6_stack, ["What are the rules for crates in the orchard?", "Only for the night shift."])
    [s] = r2.answer.sections
    assert [u.kind for u in s.uncertainty] == ["constraint_not_covered"]
    assert "I1:constraint_not_covered" in r2.answer.diff.uncertainty_introduced
