"""Unsupported-claim policy, repair templates, coverage and consistency (brief §18-19, §23-24, §42)."""

from types import SimpleNamespace

from conftest import REPO
from grounding_helpers import requires_nli
from streamrag.claims.models import ClaimVerification, IntentGap
from streamrag.claims.nli import NliModel
from streamrag.validation.consistency import ConsistencyChecker
from streamrag.validation.coverage import AnswerCoverageValidator
from streamrag.validation.policy import decide, material, needs_revision
from streamrag.validation.repair import ClaimRepairer, gap_sentence


def V(status, supporting=(), contradicting=()):
    return ClaimVerification(claim_id="c", status=status, supported=status == "SUPPORTED",
                             supporting_evidence=list(supporting), contradicting_evidence=list(contradicting))


def test_policy_table_is_deterministic():
    assert decide(V("SUPPORTED", ["E1"]), "x", [], 1) == "keep"
    assert decide(V("CONTRADICTED", ["E1"], ["E2"]), "x", [], 1) == "present_conflict"
    assert decide(V("PARTIALLY_SUPPORTED"), "x", [], 1) == "keep_atoms"
    assert decide(V("UNSUPPORTED"), "x", ["F1"], 1) == "restore_facts"
    assert decide(V("CONTRADICTED", [], ["E2"]), "x", ["F1"], 1) == "restore_facts"
    assert decide(V("UNSUPPORTED"), "It takes 30 days.", [], 1) == "retrieve"
    assert decide(V("UNSUPPORTED"), "It takes 30 days.", [], 0) == "remove"      # budget exhausted
    assert decide(V("UNSUPPORTED"), "It is quick.", [], 1) == "remove"           # not material
    assert material("costs 40 euros") and not material("is reviewed by an officer")
    assert needs_revision("strict", ["supplementary"], False) and not needs_revision("relaxed", ["supplementary"], False)
    assert needs_revision("relaxed", [], True)


def test_uncertainty_sentences_never_state_facts():
    s = gap_sentence(IntentGap(intent_id="I1", kind="value_not_stated", aspect="How long does processing take?"))
    assert s == "The retrieved documents do not state the value asked for in “How long does processing take”."
    assert gap_sentence(IntentGap(intent_id="I1", kind="no_evidence", aspect="parking rules")).startswith(
        "The retrieved documents do not contain an answer")
    assert "to night shifts" in gap_sentence(IntentGap(intent_id="I1", kind="constraint_not_covered",
                                                       aspect="night shifts"))


def test_keep_atoms_returns_only_supported_atoms():
    a1, a2 = V("SUPPORTED", ["E1"]), V("UNSUPPORTED")
    v = V("PARTIALLY_SUPPORTED").model_copy(update={"atoms": [a1, a2], "atom_texts": ["A.", "B."]})
    assert [t for t, _ in ClaimRepairer.keep_atoms(v)] == ["A."]


def test_coverage_requires_facts_or_explicit_uncertainty():
    claims = {"f": SimpleNamespace(claim_id="f", kind="fact", evidence_ids=["E1"], facts=["F1"]),
              "u": SimpleNamespace(claim_id="u", kind="uncertainty", evidence_ids=[], facts=[])}
    secs = [SimpleNamespace(section_id="S1", intent_id="I1", claim_ids=["f"]),
            SimpleNamespace(section_id="S2", intent_id="I2", claim_ids=["u"]),
            SimpleNamespace(section_id="S3", intent_id="I3", claim_ids=[])]
    rep = AnswerCoverageValidator().validate(secs, claims, {"S1": ["F1", "F2"]}, {"S1": {"F1"}})
    assert rep.covered == ["I1"] and rep.uncertain_only == ["I2"] and rep.failures == ["I3"]
    assert rep.intent_coverage == round(2 / 3, 4) and rep.critical_facts_missing == {"S1": ["F2"]}
    assert rep.claim_to_evidence == {"f": ["E1"]} and rep.section_to_intent["S2"] == "I2"


@requires_nli
def test_consistency_compares_only_the_same_proposition():
    nli = NliModel.load(REPO / "models", "nli-deberta-v3-xsmall")
    cc = ConsistencyChecker(lambda t: [w.strip(".").lower() for w in t.split() if len(w) > 3], nli)
    assert cc.conflicts([("n", "Applicants do not need proof of residence.")],
                        [("k", "Applicants need proof of residence.")]) == [("n", "k")]
    assert cc.conflicts([("n", "Ladders must be returned to the tool shed.")],
                        [("k", "Ladders are permitted only when a second worker holds the base.")]) == []
