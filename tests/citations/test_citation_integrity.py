"""Citation model, mapping, validation and orphans (brief §12-15, §31-32, §47). TEST FIXTURE corpus_grounding."""

import pytest

from grounding_helpers import echo, grounding_stack, requires_nli, run_answer, scripted
from streamrag.citations.mapper import ChunkCatalog, CitationMapper
from streamrag.citations.validator import CitationValidator
from streamrag.claims.models import ClaimVerification, EvidenceAlignment


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    return grounding_stack(tmp_path_factory)


def _verif(cid, eid, strength="STRONG", span=(0, 10)):
    return ClaimVerification(claim_id=cid, status="SUPPORTED", supported=True, supporting_evidence=[eid],
                             alignments=[EvidenceAlignment(claim_id=cid, evidence_id=eid, strength=strength,
                                                           basis="t", premise="sentence", premise_span=span)])


def test_mapper_uses_index_metadata_and_smallest_unit(stack):
    cat = ChunkCatalog.from_bundle(stack.bundle)
    ch = next(c for c in stack.bundle.chunks if c.section_title == "Eligibility")
    cmap = CitationMapper(cat).map("GA1", {"AC-1": _verif("AC-1", ch.chunk_id, span=(5, 20))}, {ch.chunk_id: "E3"})
    [c] = cmap.citations
    assert c.chunk_id == c.evidence_id == ch.chunk_id and c.source_id == ch.document_id
    assert (c.location.char_start, c.location.char_end) == (ch.char_start + 5, ch.char_start + 20)
    assert c.location.chunk_char_start == ch.char_start and c.location.page_start is None   # md: no pages
    assert c.display_metadata == {"key": ch.citation, "document_title": ch.title, "section_title": ch.section_title,
                                  "label": "E3"}


def test_mapper_never_cites_unknown_chunks(stack):
    cat = ChunkCatalog.from_bundle(stack.bundle)
    assert CitationMapper(cat).map("GA1", {"AC-1": _verif("AC-1", "made_up§9#1")}, {}).citations == []


def test_validator_detects_fabrication_and_orphans(stack):
    cat = ChunkCatalog.from_bundle(stack.bundle)
    ch = stack.bundle.chunks[1]
    v = {"AC-1": _verif("AC-1", ch.chunk_id)}
    cmap = CitationMapper(cat).map("GA1", v, {})
    good = cmap.citations[0]
    fakes = [good.model_copy(update={"citation_id": "X1", "chunk_id": "nope", "evidence_id": "nope"}),
             good.model_copy(update={"citation_id": "X2", "location": good.location.model_copy(update={"page_start": 3})}),
             good.model_copy(update={"citation_id": "X3", "claim_id": "AC-404"}),
             good.model_copy(update={"citation_id": "X4", "source_id": "Doc_99"}),
             good.model_copy(update={"citation_id": "X5", "location": good.location.model_copy(
                 update={"char_end": ch.char_end + 50})})]                      # span reaching past the chunk
    cmap = cmap.model_copy(update={"citations": cmap.citations + fakes})
    ev = {ch.chunk_id: ch.text, "other": "x"}
    fixed, rep = CitationValidator(cat).validate(cmap, v, ["AC-1", "AC-2"], ev, [ch.chunk_id, "other"])
    kinds = {i.citation_id: {x.kind for x in rep.issues if x.citation_id == i.citation_id} for i in fakes}
    assert "unknown_chunk" in kinds["X1"] and "missing_evidence" in kinds["X1"]
    assert kinds["X2"] == {"fabricated_page"} and "wrong_claim" in kinds["X3"] and "unknown_document" in kinds["X4"]
    assert kinds["X5"] == {"bad_span"}
    assert [c.citation_id for c in fixed.citations if c.status == "valid"] == [good.citation_id]
    assert rep.orphan_claims == ["AC-2"] and rep.orphan_citations == ["X3"] and rep.unused_evidence == ["other"]


def test_validator_rejects_text_mismatch_and_non_supporting(stack):
    cat = ChunkCatalog.from_bundle(stack.bundle)
    ch = stack.bundle.chunks[1]
    v = {"AC-1": _verif("AC-1", ch.chunk_id)}
    cmap = CitationMapper(cat).map("GA1", v, {})
    weak = {"AC-1": _verif("AC-1", ch.chunk_id, strength="WEAK")}
    _, rep = CitationValidator(cat).validate(cmap, weak, ["AC-1"], {ch.chunk_id: ch.text + " tampered"}, [])
    assert {i.kind for i in rep.issues} == {"text_mismatch", "not_supporting"} and rep.valid == 0


@requires_nli
def test_pipeline_citations_resolve_to_index(stack):
    """Brief §47: no citation without evidence, no claim without validation, no unrelated chunk, no fabricated ids /
    pages - even when the generator cites a label that does not exist and lists an unknown fact."""
    def bad_labels(secs):
        return {sid: [(t, [f, "F99"], ["E99"]) for f, _, t in facts] for sid, facts in secs.items()}
    _, (r,) = run_answer(stack, ["What are the eligibility requirements for the fixture permit?"], scripted(bad_labels))
    g = r.grounded
    chunks = {c.chunk_id: c for c in stack.bundle.chunks}
    docs = {c.document_id for c in stack.bundle.chunks}
    facts = [c for c in g.claims if c.kind in ("fact", "conflict")]
    assert facts and all(c.citation_ids for c in facts)
    for cit in g.citations.citations:
        assert cit.status == "valid" and cit.chunk_id in chunks and cit.source_id in docs
        assert cit.location.page_start is None and cit.display_metadata["key"] == chunks[cit.chunk_id].citation
        v = g.verifications[cit.claim_id]
        assert any(a.evidence_id == cit.evidence_id and a.strength in ("STRONG", "MODERATE")
                   for a in v.alignments + [x for at in v.atoms for x in at.alignments])
    assert g.citation_report.orphan_claims == [] and g.citation_report.orphan_citations == []
    assert "[E99]" not in g.text and "E99" not in str(g.citations.keys())


@requires_nli
def test_lineage_answer_claim_citation_evidence_chunk_document(stack):
    _, (r,) = run_answer(stack, ["How are applications for the fixture permit submitted?"], scripted(echo))
    g = r.grounded
    claim = next(c for c in g.claims if c.kind == "fact")
    cit = next(c for c in g.citations.citations if c.citation_id == claim.citation_ids[0])
    chunk = next(c for c in stack.bundle.chunks if c.chunk_id == cit.chunk_id)
    assert cit.claim_id == claim.claim_id and cit.evidence_id in claim.evidence_ids
    assert cit.source_id == chunk.document_id
    full = chunk.text[cit.location.char_start - chunk.char_start:cit.location.char_end - chunk.char_start]
    assert full.strip() and g.answer_id.startswith("GA")
