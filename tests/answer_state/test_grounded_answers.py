"""End-to-end grounded answers on the TEST FIXTURE corpora (brief §43-50, §34, §25-26, §60). The generator is a
scripted fake LLM (grounding_helpers) so each test controls exactly what the "model" says."""

import pytest

from grounding_helpers import echo, grounding_stack, requires_nli, run_answer, scripted

pytestmark = requires_nli

ELIG_PROC = "What are the eligibility requirements and the application process for the fixture permit?"


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return grounding_stack(tmp_path_factory)


def facts_of(g):
    return [c for c in g.claims if c.kind in ("fact", "conflict")]


def test_43_multi_intent_answer_is_sectioned_and_fully_cited(stack):
    _, (r,) = run_answer(stack, [ELIG_PROC], scripted(echo))
    g = r.grounded
    assert g.status == "VALIDATED_FINAL" and len(g.sections) == 2
    assert [s.title.lower().split()[-1] for s in g.sections][0] == "requirements"
    by_sec = {s.section_id: [g.claim(c) for c in s.claim_ids] for s in g.sections}
    for sec in g.sections:
        assert any(c.kind in ("fact", "conflict") for c in by_sec[sec.section_id])
        assert all(c.intent_id == sec.intent_id for c in by_sec[sec.section_id])
    assert all(c.citation_ids and c.evidence_ids for c in facts_of(g))
    assert g.coverage.covered == [s.intent_id for s in g.sections] and g.metrics["citation_coverage"] == 1.0
    assert "Eligibility" in g.text or "eligibility" in g.text


def test_44_unsupported_conjunct_never_becomes_fact(stack):
    def add_b(secs):
        return {sid: [(t.rstrip(".") + " and a passport photo.", [f], labs) if "proof of residence" in t else
                       (t, [f], labs) for f, labs, t in facts] for sid, facts in secs.items()}
    _, (r,) = run_answer(stack, ["What are the eligibility requirements for the fixture permit?"], scripted(add_b))
    g = r.grounded
    assert "passport" not in g.text.lower()
    assert any("proof of residence" in c.text.lower() for c in facts_of(g))
    assert any("passport" in x.text.lower() for x in g.rejected)
    assert any(rp.action == "keep_atoms" for rp in g.repairs)


def test_45_partial_evidence_keeps_only_the_supported_part(stack):
    def add_days(secs):
        return {sid: [(t.rstrip(".") + " and processed within 7 days.", [f], labs) if "online" in t else
                       (t, [f], labs) for f, labs, t in facts] for sid, facts in secs.items()}
    _, (r,) = run_answer(stack, ["How are applications for the fixture permit submitted?"], scripted(add_days))
    g = r.grounded
    assert "7 days" not in g.text and "submitted online" in g.text
    assert g.metrics["final_unsupported_rate"] == 0.0


def test_46_contradictory_evidence_is_presented_not_resolved(stack):
    _, (r,) = run_answer(stack, ["Is proof of residence mandatory for the fixture permit?"], scripted(echo))
    g = r.grounded
    conf = [c for c in g.claims if c.kind == "conflict"]
    texts = " ".join(c.text for c in conf)
    assert "mandatory" in texts and "optional" in texts
    assert len({c.conflict_group for c in conf}) == 1 and all(len(c.citation_ids) == 1 for c in conf)
    cited_docs = {cit.source_id for cit in g.citations.citations if cit.claim_id in {c.claim_id for c in conf}}
    assert {"fixture_permit_notice_2023", "fixture_permit_notice_2025"} <= cited_docs
    assert "The sources differ" in g.text


def test_48_revision_changes_only_affected_claims(stack):
    p, (r1, r2) = run_answer(stack, [ELIG_PROC, "Specifically for applicants who hold a suspended permit."],
                             scripted(echo))
    g1, g2 = r1.grounded, r2.grounded
    assert g2.parent_answer_id == g1.answer_id and g2.version == g1.version + 1
    unchanged = set(g2.diff.unchanged)
    assert unchanged and unchanged <= {c.claim_id for c in g1.claims}
    other = [s for s in g2.sections if s.section_id not in g2.diff.sections_regenerated]
    assert all(set(s.claim_ids) <= {c.claim_id for c in g1.claims} | {c.claim_id for c in g2.claims
                                                                      if c.kind == "uncertainty"} for s in other)
    assert g2.llm_calls <= 1                                       # only affected facts go back to the model


def test_49_intent_without_evidence_is_handled_explicitly(stack):
    _, (r,) = run_answer(stack, ["What are the eligibility requirements for the fixture permit and what is the "
                                 "parking policy for the zeppelin hangar?"], scripted(echo))
    g = r.grounded
    assert len(g.sections) == 2 and g.coverage.failures == []
    covered, uncertain = g.coverage.covered, g.coverage.uncertain_only
    assert len(covered) == 1 and len(uncertain) == 1 and g.partial
    hangar = next(s for s in g.sections if s.intent_id == uncertain[0])
    assert [g.claim(c).kind for c in hangar.claim_ids] == ["uncertainty"] and "zeppelin" in hangar.title
    assert "Not established: The retrieved documents do not contain an answer" in g.text


def test_50_no_evidence_produces_no_hallucination(stack):
    def hallucinate(secs):
        return {sid: [("Processing always takes exactly 12 days.", [], [])] for sid in secs}
    _, (r,) = run_answer(stack, ["How long does processing of a fixture permit application take?"],
                         scripted(hallucinate))
    g = r.grounded
    assert "12 days" not in g.text
    assert any(rj.text.startswith("Processing always takes") for rj in g.rejected)
    assert "do not state the value asked for" in g.text and g.partial


def test_inline_markers_are_citations_not_claim_text(stack):
    def marked(secs):
        return {sid: [(f"{t} ({labs[0]})", [], []) for f, labs, t in facts if labs] for sid, facts in secs.items()}
    _, (r,) = run_answer(stack, ["How are applications for the fixture permit submitted?"], scripted(marked))
    g = r.grounded
    plain = [c for c in g.claims if c.kind == "fact"]
    assert plain and all(c.status == "SUPPORTED" and c.citation_ids for c in plain)
    assert all(c.citation_ids for c in facts_of(g))
    assert not any("(E" in c.text for c in g.claims) and "submitted online" in g.text


def test_merged_output_items_are_split_into_one_claim_per_sentence(stack):
    def merged(secs):
        return {sid: [(" ".join(t for _, _, t in facts), [f for f, _, _ in facts], [])] for sid, facts in secs.items()}
    _, (r,) = run_answer(stack, ["How are applications for the fixture permit submitted?"], scripted(merged))
    g = r.grounded
    plain = [c for c in g.claims if c.kind == "fact"]
    assert len(plain) >= 2 and all(c.status == "SUPPORTED" and c.citation_ids for c in plain)
    assert all(c.text.count(". ") == 0 for c in plain)                 # one sentence each, each cited on its own


def test_count_question_needs_the_counted_noun(stack):
    _, (r,) = run_answer(stack, ["How many renewals can a fixture permit have?"], scripted(echo))
    g = r.grounded
    assert "every 2 years" in g.text                    # related facts are still given ...
    assert "do not state the value asked for" in g.text and g.partial   # ... but "2 years" is not a count of renewals


def test_strict_revises_relaxed_only_drops(stack):
    """The model replaces its last (non-critical) planned fact with an unsupported statement. Strict mode regenerates
    the missing fact with feedback; relaxed mode only drops the statement (no critical fact is missing)."""
    def sneaky(secs):
        return {sid: [(t, [f], labs) for f, labs, t in facts[:-1]] + [("Officers are friendly.", [], [])]
                for sid, facts in secs.items()}
    q = ["How are applications for the fixture permit submitted?"]
    _, (strict,) = run_answer(stack, q, scripted(sneaky), validation_mode="strict")
    _, (relaxed,) = run_answer(stack, q, scripted(sneaky), validation_mode="relaxed")
    assert strict.grounded.revision_attempts >= 1 and relaxed.grounded.revision_attempts == 0
    for g in (strict.grounded, relaxed.grounded):
        assert "friendly" not in g.text and g.status == "VALIDATED_FINAL"


def test_presented_conflict_needs_no_revision(stack):
    """Conflict sides state planned facts verbatim: they count as expressed, so no revision call is spent on them
    (found in the e2e run: two wasted LLM calls for the fee question)."""
    _, (r,) = run_answer(stack, ["What is the application fee for a new fixture permit?"], scripted(echo))
    g = r.grounded
    assert any(c.kind == "conflict" for c in g.claims) and g.revision_attempts == 0 and g.llm_calls == 1


def test_retrieval_fallback_is_bounded_and_can_rescue_a_claim(tmp_path_factory):
    """A material claim the need's evidence does not support triggers one bounded retrieval with the claim text;
    the corpus does state it, so the claim survives with a citation to the newly found evidence."""
    st = grounding_stack(tmp_path_factory, overrides={"multi_intent.max_candidates_per_intent": 2})

    def extra(secs):
        return {sid: [(t, [f], labs) for f, labs, t in facts] + [("Permits must be renewed every 2 years.", [], []),
                                                                  ("Permits cost 99 euros in total.", [], [])]
                for sid, facts in secs.items()}
    p, (r,) = run_answer(st, ["How are applications for the fixture permit submitted?"], scripted(extra),
                         max_validation_retrievals=1)
    g = r.grounded
    ev = [e.payload for e in p.events if e.type.value == "VALIDATION_RETRIEVAL"]
    assert g.validation_retrievals == len(ev) == 1                       # bounded: the 2nd material claim gets none
    assert ev[0]["status_after"] == "SUPPORTED" and ev[0]["new_evidence"]
    rescued = next(c for c in g.claims if "renewed every 2 years" in c.text)
    assert rescued.origin == "retrieval" and set(rescued.evidence_ids) & set(ev[0]["new_evidence"])
    assert "99 euros" not in g.text


def test_failed_generation_falls_back_to_extractive(stack):
    from streamrag.generation.llm import ScriptedBackend
    _, (r,) = run_answer(stack, ["How are applications for the fixture permit submitted?"],
                         ScriptedBackend(["{bad", "{bad"]))
    g = r.grounded
    assert g.fallback and g.fallback.startswith("schema_invalid") and g.backend.endswith("->extractive")
    assert g.status == "VALIDATED_FINAL" and facts_of(g) and all(c.citation_ids for c in facts_of(g))


def test_prompt_injection_in_corpus_cannot_become_a_fact(tmp_path_factory):
    st = grounding_stack(tmp_path_factory, corpus="corpus_injection")

    def obey(secs):
        return {sid: [(t, [f], labs) for f, labs, t in facts] + [("Permits are free and never expire.", [], [])]
                for sid, facts in secs.items()}
    p, (r,) = run_answer(st, ["Is the fixture permit office open on public holidays?"], scripted(obey))
    g = r.grounded
    assert "free and never expire" not in g.text
    assert all("SYSTEM NOTE" not in c.text and "ignore all previous" not in c.text.lower() for c in g.claims)
    assert "closed on public holidays" in g.text
    prompt = p.grounding.backend.requests[0][1]["content"]
    assert "SYSTEM NOTE" not in prompt                         # injected sentences never reach the facts block
