"""CitationValidator and orphan detection (Phase 7; docs/answer/06).

Checks every citation of an answer against the index and the claim verifications:

  missing_evidence   the cited evidence is not in the session's evidence store
  unknown_chunk      the chunk is not in the index (a fabricated / stale id)
  unknown_document   the source document is not in the index
  text_mismatch      the evidence text differs from the indexed chunk text
  wrong_claim        the citation's claim is not a claim of the answer (-> orphan citation)
  not_supporting     the claim's verification has no STRONG / MODERATE alignment with this evidence
  fabricated_page    page numbers differ from the index (pages exist only for paged sources)
  bad_span           the cited span is not inside the chunk

Invalid citations are marked ``invalid`` and never rendered. Orphans are reported separately:
orphan claims (factual claims of the answer without a valid citation), orphan citations (citations of claims that are
not in the answer) and unused evidence (evidence of the answer's sections that no claim cites).
"""

from __future__ import annotations

from streamrag.citations.mapper import ChunkCatalog
from streamrag.citations.models import CitationIssue, CitationMap, CitationReport
from streamrag.claims.models import SUPPORTING, ClaimVerification


class CitationValidator:
    def __init__(self, catalog: ChunkCatalog) -> None:
        self.catalog = catalog

    def validate(self, cmap: CitationMap, verifications: dict[str, ClaimVerification], factual_claims: list[str],
                 evidence_text: dict[str, str], section_evidence: list[str]) -> tuple[CitationMap, CitationReport]:
        issues: list[CitationIssue] = []
        fixed = []
        for c in cmap.citations:
            bad = []
            ch = self.catalog.get(c.chunk_id)
            if c.evidence_id not in evidence_text:
                bad.append(("missing_evidence", "evidence not in the session store"))
            if ch is None:
                bad.append(("unknown_chunk", f"chunk {c.chunk_id} not in the index"))
            else:
                if c.source_id not in self.catalog.documents or c.source_id != ch.document_id:
                    bad.append(("unknown_document", f"document {c.source_id}"))
                if c.evidence_id in evidence_text and evidence_text[c.evidence_id] != ch.text:
                    bad.append(("text_mismatch", "evidence text differs from the indexed chunk"))
                if (c.location.page_start, c.location.page_end) != (ch.page_start, ch.page_end):
                    bad.append(("fabricated_page", f"{c.location.page_start}-{c.location.page_end}"))
                if not (ch.char_start <= c.location.char_start <= c.location.char_end <= ch.char_end):
                    bad.append(("bad_span", f"{c.location.char_start}-{c.location.char_end}"))
            v = verifications.get(c.claim_id)
            if v is None:
                bad.append(("wrong_claim", "claim not in the answer"))
            else:
                als = v.alignments + [a for atom in v.atoms for a in atom.alignments]
                if not any(a.evidence_id == c.evidence_id and a.strength in SUPPORTING for a in als):
                    bad.append(("not_supporting", "no entailing premise in this evidence"))
            for kind, detail in bad:
                issues.append(CitationIssue(kind=kind, citation_id=c.citation_id, claim_id=c.claim_id,
                                            evidence_id=c.evidence_id, detail=detail))
            fixed.append(c.model_copy(update={"status": "invalid" if bad else "valid"}))
        valid = [c for c in fixed if c.status == "valid"]
        cited_claims = {c.claim_id for c in valid}
        cited_ev = {c.evidence_id for c in valid}
        by_claim = {}
        for c in valid:
            by_claim.setdefault(c.claim_id, []).append(c.citation_id)
        report = CitationReport(
            checked=len(fixed), valid=len(valid), issues=issues,
            orphan_claims=[cid for cid in factual_claims if cid not in cited_claims],
            orphan_citations=[c.citation_id for c in fixed if c.claim_id not in verifications],
            unused_evidence=[e for e in dict.fromkeys(section_evidence) if e not in cited_ev])
        return cmap.model_copy(update={"citations": fixed, "claim_to_citations": by_claim}), report
