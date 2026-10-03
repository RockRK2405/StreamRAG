"""Citation contracts (Phase 7; docs/answer/05-06).

A ``Citation`` links one verified claim to one supporting evidence chunk, down to the supporting sentence span.
Every location field is copied from the index (``CorpusChunk``), never from the generator: lineage is
answer -> claim -> citation -> evidence -> chunk -> source document, and each hop can be resolved from the index alone.
Pages are present only when the source format had pages (PDF); they are never guessed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.claims.models import SupportStrength
from streamrag.models.base import Contract


class CitationLocation(Contract):
    document_id: str
    section_id: str
    chunk_id: str
    source_path: str
    char_start: int = Field(ge=0)                    # supporting span in the normalized document text
    char_end: int = Field(ge=0)
    chunk_char_start: int = Field(ge=0)              # the chunk's span (the smallest indexed unit)
    chunk_char_end: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None


class Citation(Contract):
    citation_id: str                                 # "CIT<n>" (answer-scoped)
    claim_id: str
    evidence_id: str
    source_id: str                                   # the source document id
    chunk_id: str
    location: CitationLocation
    display_metadata: dict[str, str] = {}            # key ("Doc_ID §Section"), document / section title, label
    strength: SupportStrength
    status: Literal["valid", "invalid"] = "valid"


class CitationMap(Contract):
    answer_id: str
    claim_to_citations: dict[str, list[str]] = {}
    citations: list[Citation] = []

    def keys(self) -> list[str]:
        return list(dict.fromkeys(c.display_metadata.get("key", "") for c in self.citations if c.status == "valid"))


class CitationIssue(Contract):
    kind: Literal["missing_evidence", "unknown_chunk", "unknown_document", "text_mismatch", "not_supporting",
                  "wrong_claim", "fabricated_page", "bad_span", "orphan_citation"]
    citation_id: str | None = None
    claim_id: str | None = None
    evidence_id: str | None = None
    detail: str = ""


class CitationReport(Contract):
    checked: int = Field(0, ge=0)
    valid: int = Field(0, ge=0)
    issues: list[CitationIssue] = []
    orphan_claims: list[str] = []                    # factual claims without a valid citation
    orphan_citations: list[str] = []                 # citations whose claim is not in the answer
    unused_evidence: list[str] = []                  # evidence of the answer's sections that no claim cites
