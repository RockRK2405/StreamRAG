"""CitationMapper: verified claims -> citations to the smallest supporting unit (Phase 7; docs/answer/05).

Citations are built from the *verification* (the evidence whose alignment is STRONG or MODERATE), never from the
labels the generator wrote: a wrongly cited label is replaced by the evidence that actually supports the claim, and
evidence the generator cited but that does not support the claim is not cited. Only supporting evidence is cited
(at most ``max_per_claim`` per claim, STRONG first). The location is the supporting sentence (or window) inside the
chunk, converted to document offsets with the chunk's own span from the index.

``ChunkCatalog`` is the read-only view of the index used for all metadata: chunk -> document, section, titles,
source path, character span, pages. A chunk that is not in the index cannot be cited.
"""

from __future__ import annotations

from streamrag.citations.models import Citation, CitationLocation, CitationMap
from streamrag.claims.models import SUPPORTING, ClaimVerification


class ChunkCatalog:
    def __init__(self, chunks, documents=None, boilerplate_min_docs: int | None = None) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.documents = ({d.document_id: d for d in documents} if documents is not None
                          else {c.document_id: None for c in chunks})
        # sentences repeated verbatim in several documents (disclaimers, headers, footers: >= 2 documents in a corpus
        # of up to 4, else >= 3) are template text, not document-specific facts (claims/planner.py excludes them)
        from streamrag.claims.graph import sentences
        seen: dict[str, set[str]] = {}
        for c in chunks:
            for s, e in sentences(c.text):
                seen.setdefault(" ".join(c.text[s:e].lower().split()), set()).add(c.document_id)
        n_docs = len({c.document_id for c in chunks})
        k_min = boilerplate_min_docs or max(2, min(3, -(-n_docs // 2)))     # 2 for small corpora, else 3
        self.boilerplate = {k for k, docs in seen.items() if len(docs) >= k_min} if n_docs >= 2 else set()

    def is_boilerplate(self, sentence: str) -> bool:
        return " ".join(sentence.lower().split()) in self.boilerplate

    @classmethod
    def from_bundle(cls, bundle) -> "ChunkCatalog":
        return cls(bundle.chunks, bundle.documents())

    def get(self, chunk_id: str):
        return self.chunks.get(chunk_id)


class CitationMapper:
    def __init__(self, catalog: ChunkCatalog, max_per_claim: int = 2) -> None:
        self.catalog, self.max_per_claim = catalog, max_per_claim

    def map(self, answer_id: str, verifications: dict[str, ClaimVerification], labels: dict[str, str]) -> CitationMap:
        """verifications: claim id -> verification (only claims that will be in the answer); labels: evidence id ->
        label shown to the generator (display only)."""
        cits, by_claim = [], {}
        n = 0
        for cid, v in verifications.items():
            als = [a for a in v.alignments if a.strength in SUPPORTING]
            if v.atoms:                                   # support composed from atoms
                als = [a for atom in v.atoms for a in atom.alignments if a.strength in SUPPORTING]
            als.sort(key=lambda a: (a.strength != "STRONG", a.evidence_id not in v.cited_evidence, a.evidence_id))
            seen = set()
            for a in als:
                if a.evidence_id in seen or len(seen) >= self.max_per_claim:
                    continue
                ch = self.catalog.get(a.evidence_id)
                if ch is None:
                    continue                              # never cite what the index does not contain
                seen.add(a.evidence_id)
                s, e = a.premise_span or (0, len(ch.text))
                n += 1
                cit = Citation(
                    citation_id=f"{answer_id}-CIT{n}", claim_id=cid, evidence_id=a.evidence_id,
                    source_id=ch.document_id, chunk_id=ch.chunk_id,
                    location=CitationLocation(document_id=ch.document_id, section_id=ch.section_id,
                                              chunk_id=ch.chunk_id, source_path=ch.source_path,
                                              char_start=ch.char_start + s, char_end=ch.char_start + e,
                                              chunk_char_start=ch.char_start, chunk_char_end=ch.char_end,
                                              page_start=ch.page_start, page_end=ch.page_end),
                    display_metadata={"key": ch.citation, "document_title": ch.title,
                                      "section_title": ch.section_title, "label": labels.get(a.evidence_id, "")},
                    strength=a.strength)
                cits.append(cit)
                by_claim.setdefault(cid, []).append(cit.citation_id)
        return CitationMap(answer_id=answer_id, claim_to_citations=by_claim, citations=cits)
