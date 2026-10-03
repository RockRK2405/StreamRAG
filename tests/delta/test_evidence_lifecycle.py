"""EvidenceStore + EvidenceValidityManager rules (Phase 6, brief §11-13; docs/session/05).

Evidence texts here are TEST_FIXTURE_ONLY strings written for the rules, not corpus documents."""

from streamrag.context.models import ContextChange, SemanticDiff
from streamrag.delta.evidence import EvidenceStore, EvidenceValidityManager

from streaming_helpers import make_es

TEXTS = {"E1": "Ladders must be inspected before use.", "E2": "Ladders stay in the shed overnight.",
         "E3": "Crates are stacked five high.", "E4": "Ladders used on the day shift are logged."}


def setup(fixture_bundle, ids=("E1", "E2", "E3"), intent="I1"):
    _, b = fixture_bundle
    terms = lambda t: list(dict.fromkeys(b.analyzer.tokens(t)))  # noqa: E731
    store = EvidenceStore()
    es = make_es("q", list(ids))
    es = es.model_copy(update={"items": [e.model_copy(update={"text": TEXTS[e.evidence_id]}) for e in es.items]})
    store.add_results(intent, 1, "Q1", es, 0.0)
    return store, EvidenceValidityManager(store, terms), es


def change(kind, **kw):
    diff = SemanticDiff(intent_id=kw.pop("intent", "I1"), **kw.pop("diff", {}))
    return ContextChange(change_id="CH9", change_type=kind, utterance_id="u2", diffs=[diff], confidence=1.0, **kw)


def statuses(store, intent="I1"):
    return {a.evidence_id: a.status for a in store.for_intent(intent)}


def test_constraint_addition(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    acts = vm.apply(change("CONSTRAINT_ADDITION", diff={"constraints_added": ["K1"]}), lambda k: "overnight",
                    lambda i: None, 1.0)
    assert statuses(store) == {"E1": "RETAINED", "E2": "ACTIVE", "E3": "RETAINED"}
    rules = {a.evidence_id: a.rule for a in acts}
    assert rules["E1"] == "general_evidence_still_applicable"
    assert rules["E2"].startswith("constraint_covered")
    assert store.assign[("E1", "I1")].history[-1].change_id == "CH9"          # every transition names its change


def test_constraint_removal_revalidates_specific_evidence(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    vm.apply(change("CONSTRAINT_REMOVAL", diff={"constraints_removed": ["K1"]}), lambda k: "overnight",
             lambda i: None, 1.0)
    assert statuses(store) == {"E1": "RETAINED", "E2": "REVALIDATION_REQUIRED", "E3": "RETAINED"}
    vm.confirm_after_retrieval("I1", {"E1", "E2"}, "Q2", 2.0)
    assert statuses(store)["E2"] == "ACTIVE"
    assert store.assign[("E2", "I1")].history[-1].rule == "confirmed_by_delta_retrieval"


def test_entity_change(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    vm.apply(change("ENTITY_CHANGE", diff={"topic_after": "crates"}), lambda k: "", lambda i: None, 1.0)
    assert statuses(store) == {"E1": "SUPERSEDED", "E2": "SUPERSEDED", "E3": "REVALIDATION_REQUIRED"}


def test_refinement_that_names_the_topic(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    vm.apply(change("REFINEMENT", diff={"terms_added": ["crate"], "topic_after": "crates"}), lambda k: "",
             lambda i: None, 1.0)
    assert statuses(store) == {"E1": "REVALIDATION_REQUIRED", "E2": "REVALIDATION_REQUIRED", "E3": "ACTIVE"}


def test_question_change_and_intent_removal(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    vm.apply(change("QUESTION_CHANGE"), lambda k: "", lambda i: None, 1.0)
    assert set(statuses(store).values()) == {"REVALIDATION_REQUIRED"}
    vm.apply(change("INTENT_REMOVAL"), lambda k: "", lambda i: None, 2.0)
    assert set(statuses(store).values()) == {"STALE"}
    assert store.usable("I1") == []


def test_correction_supersedes_and_carries_matching_evidence(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    ch = ContextChange(change_id="CH3", change_type="CORRECTION", utterance_id="u2", affected_intents=["I1"],
                       superseded_intents=["I1"], new_intents=["I2"], confidence=1.0)
    acts = vm.apply(ch, lambda k: "", lambda i: "crates" if i == "I2" else None, 1.0)
    assert set(statuses(store, "I1").values()) == {"SUPERSEDED"}
    assert statuses(store, "I2") == {"E3": "REVALIDATION_REQUIRED"}
    assert [a.rule for a in acts if a.intent_id == "I2"] == ["carried_to_correction"]


def test_lifecycle_history_and_never_deleted(fixture_bundle):
    store, vm, es = setup(fixture_bundle)
    vm.apply(change("INTENT_REMOVAL"), lambda k: "", lambda i: None, 1.0)
    assert len(store.records) == 3 and all(len(a.history) == 2 for a in store.assign.values())
    react = store.add_results("I1", 2, "Q5", es, 3.0)                 # retrieved again for the need
    assert {e for e, _ in react} == {"E1", "E2", "E3"} and set(statuses(store).values()) == {"ACTIVE"}


def test_source_unavailable_invalidates(fixture_bundle):
    store, vm, _ = setup(fixture_bundle)
    acts = vm.check_sources({"E1", "E3"}, 5.0)
    assert [(a.evidence_id, a.to_status, a.decision, a.rule) for a in acts] == [
        ("E2", "INVALID", "DISCARD", "source_unavailable")]
    assert statuses(store)["E2"] == "INVALID"
