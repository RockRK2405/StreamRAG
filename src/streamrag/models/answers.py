"""Claim and answer contracts (spec §16-§17; evolved in Phase 6, docs/session/06-08).

Phase 6 establishes claim-level and answer-level *state*; rendering answer text is Phase 7. A claim is a factual
statement that may appear in the answer. In Phase 6 claims are **extractive**: verbatim sentences of retrieved
evidence (``source`` gives the exact span), so a SUPPORTS link is justified by construction; no generated text.

Claim IDs are stable across answer versions. Nothing is deleted: superseded/stale claims stay for lineage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from streamrag.models.base import Contract

ClaimStatus = Literal["PENDING_VALIDATION", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "SUPERSEDED", "STALE"]
LinkRelation = Literal["SUPPORTS", "PARTIALLY_SUPPORTS", "CONTRADICTS", "UNSUPPORTED"]


class Citation(Contract):
    key: str                      # "Doc_ID §Section"
    evidence_ids: list[str] = Field(min_length=1)
    document_id: str
    section_id: str

    @field_validator("key")
    @classmethod
    def _format(cls, v: str) -> str:
        if " §" not in v:
            raise ValueError("citation key must look like 'Doc_ID §Section'")
        return v


class ClaimSource(Contract):
    """Exact origin of an extractive claim: characters [char_start, char_end) of the evidence text."""

    evidence_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class Claim(Contract):
    claim_id: str
    text: str = Field(min_length=1)
    type: Literal["FACTUAL", "UNCERTAINTY", "CONNECTIVE", "CLARIFICATION"] = "FACTUAL"
    origin: Literal["extractive", "generated"] = "extractive"
    intent_id: str | None = None
    intent_version: int | None = None
    frame_id: str | None = None
    evidence_ids: list[str] = []                     # evidence this claim cites
    source: ClaimSource | None = None
    status: ClaimStatus = "PENDING_VALIDATION"
    status_reason: str = ""
    confidence: float = Field(0.0, ge=0, le=1)       # computed relevance to the intent (docs/session/06)
    confidence_signals: dict[str, float] = {}
    introduced_in: int | None = None                 # answer version (within the frame) that first used it
    modified_in: list[int] = []
    introduced_by_constraint: str | None = None
    created_at_ms: float = Field(0.0, ge=0)
    updated_at_ms: float = Field(0.0, ge=0)


class ClaimEvidenceLink(Contract):
    claim_id: str
    evidence_id: str
    relation: LinkRelation
    basis: str                                       # rule that justified it
    active: bool = True


class ClaimTransition(Contract):
    claim_id: str
    from_status: ClaimStatus | None
    to_status: ClaimStatus
    reason: str
    change_id: str | None = None
    at_ms: float = Field(0.0, ge=0)


class ClaimChange(Contract):
    claim_id: str
    from_text: str
    to_text: str
    from_citations: list[str] = []
    to_citations: list[str] = []
    from_status: ClaimStatus | None = None
    to_status: ClaimStatus | None = None


class UncertaintyItem(Contract):
    intent_id: str | None = None
    kind: Literal["no_evidence", "conflict", "unverified_claim", "ambiguous", "constraint_not_covered"]
    aspect: str


class AnswerSection(Contract):
    """One section per active intent of the frame; the unit of (later) partial regeneration."""

    section_id: str                                  # stable per intent lineage ("S-I1")
    intent_id: str
    intent_version: int = Field(ge=1)
    title: str
    order: int = Field(0, ge=0)
    claim_ids: list[str] = []
    evidence_ids: list[str] = []
    constraints: list[str] = []
    status: Literal["new", "unchanged", "changed"] = "new"
    needs_regeneration: bool = True
    uncertainty: list[UncertaintyItem] = []


class AnswerDiff(Contract):
    kept: list[str] = []                             # unchanged claims (carried verbatim)
    modified: list[ClaimChange] = []
    added: list[str] = []
    retracted: list[str] = []                        # removed from the answer (claims stay in the store)
    citations_added: list[str] = []
    citations_removed: list[str] = []
    evidence_added: list[str] = []
    evidence_removed: list[str] = []
    sections_added: list[str] = []
    sections_removed: list[str] = []
    sections_changed: list[str] = []
    sections_unchanged: list[str] = []
    uncertainty_resolved: list[str] = []
    uncertainty_introduced: list[str] = []


class AnswerVersion(Contract):
    answer_id: str = ""                              # "A<n>" (session-scoped)
    version: int = Field(ge=1)                       # within the frame (topic)
    parent_version: int | None = None
    supersedes_answer_id: str | None = None
    topic_id: str                                    # frame id
    utterance_id: str
    session_version: int = Field(0, ge=0)
    kind: Literal["initial", "refinement", "presentation", "meta", "social", "clarification"]
    created_t_stream_s: float | None = None
    created_at_ms: float = Field(0.0, ge=0)
    text: str = ""                                   # rendered by Phase 7; empty in Phase 6 (structured state only)
    sections: list[AnswerSection] = []
    claim_ids: list[str] = []
    evidence_ids: list[str] = []
    citations: list[str] = []                        # citation keys, order of first appearance
    evidence_set_id: str | None = None
    uncertainty: list[UncertaintyItem] = []
    frame_slots: dict[str, str] = {}                 # active constraints of the frame (id -> text)
    diff: AnswerDiff = AnswerDiff()
    change_summary: str = ""
    delta_queries: list[str] = []
    full_rerun: bool = False
    unverified: bool = False

    @field_validator("parent_version")
    @classmethod
    def _parent_before(cls, v: int | None, info) -> int | None:
        version = info.data.get("version")
        if v is not None and version is not None and v >= version:
            raise ValueError("parent_version must be smaller than version")
        return v
