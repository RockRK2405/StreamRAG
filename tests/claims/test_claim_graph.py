"""Claim model, extraction, claim-evidence graph, invalidation and targeted revalidation (Phase 6, brief §19-23).

Unit-level evidence texts are TEST_FIXTURE_ONLY strings; pipeline tests use the fictional fixture corpus."""

from types import SimpleNamespace

from streamrag.claims.graph import ClaimExtractor, ClaimGraph, ClaimRevalidator, sentences
from streamrag.delta.evidence import EvidenceStore
from streamrag.models.answers import Claim

from session_helpers import engine_of, p6_stack, run_turns  # noqa: F401  (fixture)
from streaming_helpers import make_es

LADDERS = "What are the rules for ladders in the orchard?"


def store_with(texts: dict[str, str], docs: dict[str, str] | None = None, intent="I1") -> EvidenceStore:
    es = make_es("q", list(texts))
    es = es.model_copy(update={"items": [e.model_copy(update={"text": texts[e.evidence_id],
                                                              "document_id": (docs or {}).get(e.evidence_id, "D")})
                                         for e in es.items]})
    s = EvidenceStore()
    s.add_results(intent, 1, "Q1", es, 0.0)
    return s


def test_sentence_spans_strip_list_markers():
    text = "- First rule here.\n2) Second rule!\nThird"
    assert [text[s:e] for s, e in sentences(text)] == ["First rule here.", "Second rule!", "Third"]


def test_extracted_claims_are_verbatim_and_relevant():
    terms = lambda t: [w.strip(".,").lower().rstrip("s") for w in t.split()]  # noqa: E731
    st = store_with({"E1": "Ladders need a second worker. The orchard opens at dawn. Ladders stay dry.",
                     "E2": "Crates are stacked five high."})
    ex = ClaimExtractor(terms, per_intent=4, min_relevance=0.2)
    cands = ex.select(["ladder", "rule", "orchard"], st, "I1", topic_terms=["ladder"])
    texts = [c.text for c in cands]
    assert texts and all(st.text(c.evidence_id)[c.start:c.end] == c.text for c in cands)
    assert "Crates are stacked five high." not in texts
    assert "The orchard opens at dawn." not in texts            # shares only a non-topic word with the need


def test_register_is_stable_per_lineage_and_span():
    g = ClaimGraph()
    it = SimpleNamespace(intent_id="I1", lineage_root="I1", version=1)
    cand = SimpleNamespace(evidence_id="E1", start=0, end=10, text="0123456789", relevance=0.5, rank=1)
    c1, new1 = g.register(it, "T1", cand, 0.0)
    it2 = SimpleNamespace(intent_id="I1", lineage_root="I1", version=2)
    c2, new2 = g.register(it2, "T1", cand, 1.0)
    assert new1 and not new2 and c1.claim_id == c2.claim_id == "C1" and c2.intent_version == 2
    assert g.links[("C1", "E1")].relation == "SUPPORTS" and g.linked_to("E1") == ["C1"]
    Claim.model_validate(c2.model_dump())                     # brief §19 fields present on the contract
    for f in ("claim_id", "text", "intent_id", "evidence_ids", "status", "confidence", "created_at_ms", "updated_at_ms"):
        assert f in Claim.model_fields


def test_claim_lifecycle_through_constraint_changes(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Specifically overnight.", "Actually, ignore the overnight restriction."])
    g = engine_of(p).graph
    by_text = {c.text: c for c in g.claims.values()}
    general = by_text["Ladders are permitted in the orchard only when a second worker holds the base of the ladder."]
    night = by_text["Ladders are not permitted in the orchard overnight and must be returned to the tool shed."]
    hist = lambda cid: [(t.from_status, t.to_status, t.reason.split(":")[0]) for t in g.transitions  # noqa: E731
                        if t.claim_id == cid]
    assert hist(general.claim_id) == [
        (None, "PENDING_VALIDATION", "registered"),
        ("PENDING_VALIDATION", "SUPPORTED", "verbatim_in_valid_evidence"),
        ("SUPPORTED", "PENDING_VALIDATION", "affected_by_change"),
        ("PENDING_VALIDATION", "PARTIALLY_SUPPORTED", "does_not_address_constraint"),
        ("PARTIALLY_SUPPORTED", "PENDING_VALIDATION", "affected_by_change"),
        ("PENDING_VALIDATION", "SUPPORTED", "verbatim_in_valid_evidence")]
    assert g.claims[night.claim_id].status == "SUPPORTED"
    assert rs[1].plan.claims_to_revalidate == sorted(g.claims_of("I1"), key=lambda c: int(c[1:]))


def test_unrelated_claims_are_not_revalidated(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "Now explain how the telescope is recalibrated.", "Specifically monthly."])
    g = engine_of(p).graph
    before = [t for t in g.transitions if t.claim_id in g.claims_of("I1")]
    assert all(t.at_ms <= 3000.0 for t in before)                    # nothing after the ladders turn touched them


def test_revalidator_statuses_and_links():
    terms = lambda t: [w.strip(".,").lower() for w in t.split()]  # noqa: E731
    st = store_with({"E1": "Ladders stay dry.", "E2": "Ladders are logged overnight."})
    g = ClaimGraph()
    it = SimpleNamespace(intent_id="I1", lineage_root="I1", version=1, status="ACTIVE")
    for eid, (s, e) in {"E1": (0, 17), "E2": (0, 29)}.items():
        g.register(it, "T1", SimpleNamespace(evidence_id=eid, start=s, end=e, text=st.text(eid)[s:e], relevance=1.0,
                                             rank=1), 0.0)
    g.selected["I1"] = ["C1", "C2"]
    k = SimpleNamespace(constraint_id="K1", text="overnight")
    tracker = SimpleNamespace(intents={"I1": it}, constraints_for=lambda i: [k])
    rv = ClaimRevalidator(terms)
    rv.validate(["C1", "C2"], g, st, tracker, 1.0)
    assert g.claims["C1"].status == "PARTIALLY_SUPPORTED" and g.links[("C1", "E1")].relation == "PARTIALLY_SUPPORTS"
    assert g.claims["C2"].status == "SUPPORTED" and g.links[("C2", "E2")].relation == "SUPPORTS"
    st.transition("E2", "I1", "REVALIDATION_REQUIRED", "test", None, 2.0)
    rv.validate(["C2"], g, st, tracker, 2.0)
    assert g.claims["C2"].status == "PENDING_VALIDATION"            # never left SUPPORTED on unconfirmed evidence
    st.transition("E2", "I1", "STALE", "test", None, 3.0)
    rv.validate(["C2"], g, st, tracker, 3.0)
    assert g.claims["C2"].status == "UNSUPPORTED" and g.links[("C2", "E2")].relation == "UNSUPPORTED"
    g.selected["I1"] = ["C2"]
    rv.validate(["C1"], g, st, tracker, 4.0)
    assert g.claims["C1"].status == "STALE" and g.claims["C1"].status_reason == "not_selected_for_version"
    tracker.intents["I1"] = SimpleNamespace(**{**vars(it), "status": "SUPERSEDED"})
    rv.validate(["C1", "C2"], g, st, tracker, 5.0)
    assert {g.claims[c].status for c in ("C1", "C2")} == {"SUPERSEDED"}


def test_numeric_contradiction_is_flagged_not_resolved():
    terms = lambda t: [w.strip(".,").lower() for w in t.split()]  # noqa: E731
    st = store_with({"E1": "Visitors may stay 2 hours in the dome.", "E2": "Visitors may stay 3 hours in the dome.",
                     "E3": "Keepers rest 3 hours after a storm."}, docs={"E1": "A", "E2": "B", "E3": "B"})
    g = ClaimGraph()
    it = SimpleNamespace(intent_id="I1", lineage_root="I1", version=1)
    for eid in ("E1", "E2", "E3"):
        t = st.text(eid)
        g.register(it, "T1", SimpleNamespace(evidence_id=eid, start=0, end=len(t), text=t, relevance=1.0, rank=1), 0.0)
    out = ClaimRevalidator(terms).conflicts(["C1", "C2", "C3"], g, st)
    assert [(a, b, u) for a, b, u in out] == [("C1", "C2", "hour")]
    assert g.links[("C1", "E2")].relation == "CONTRADICTS" and g.links[("C2", "E1")].relation == "CONTRADICTS"
    assert g.claims["C1"].status == g.claims["C2"].status == "PENDING_VALIDATION"    # status untouched


def test_new_need_sharing_evidence_does_not_revalidate_other_needs_claims(p6_stack):
    p, rs = run_turns(p6_stack, [LADDERS, "What about crates?"])
    eng = engine_of(p)
    shared = {a.evidence_id for a in eng.store.for_intent("I1")} & {a.evidence_id for a in eng.store.for_intent("I2")}
    assert shared                                                    # the two needs retrieved common chunks
    i1 = set(eng.graph.claims_of("I1"))
    assert not [t for t in eng.graph.transitions if t.claim_id in i1 and t.at_ms > 3000.0]
    assert rs[1].plan.claims_to_revalidate == []
