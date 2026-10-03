"""Grounded answer contracts (Phase 7; docs/answer/08, 10).

``GroundedAnswer`` is the final answer state of a topic frame: rendered text, the claims it consists of, their
verification, citations, coverage, what was rejected or repaired, the version diff, and the measured cost.
Claim ids are stable across versions (hash of section lineage + normalized text), so an unchanged sentence keeps its
id and its citations.

Status (brief §40): DRAFT (streamed before the turn ends; verified, but not final), VALIDATED_FINAL, BLOCKED
(validation could not establish the answer: never presented as validated). ``partial`` marks a validated answer
in which some need is only answered with an uncertainty statement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.citations.models import CitationMap, CitationReport
from streamrag.claims.models import ClaimVerification, Importance, IntentGap
from streamrag.models.base import Contract
from streamrag.validation.coverage import CoverageReport

AnswerStatus = Literal["DRAFT", "VALIDATED_FINAL", "BLOCKED"]
ClaimKind = Literal["fact", "uncertainty", "conflict", "connective"]


class AnswerClaim(Contract):
    claim_id: str
    section_id: str
    intent_id: str
    text: str = Field(min_length=1)
    kind: ClaimKind = "fact"
    origin: Literal["llm", "extractive", "repair", "template", "kept", "retrieval"] = "llm"
    importance: Importance = "important"
    facts: list[str] = []                          # planned-fact ids of the plan this version used
    fact_keys: list[str] = []                      # stable fact identities (phase-6 claim id | text)
    status: Literal["DRAFT", "SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNCERTAINTY"] = "DRAFT"
    evidence_ids: list[str] = []                   # supporting evidence (from the verification)
    citation_ids: list[str] = []
    repaired_from: str | None = None
    conflict_group: str | None = None
    introduced_in: int = Field(1, ge=1)


class RejectedClaim(Contract):
    claim_id: str
    section_id: str
    text: str
    status: str
    action: str
    reasons: list[str] = []
    importance: Importance = "important"
    origin: str = "llm"


class RepairRecord(Contract):
    claim_id: str
    action: str
    from_text: str
    to_texts: list[str] = []
    ok: bool = True
    detail: str = ""


class GroundedSection(Contract):
    section_id: str
    intent_id: str
    title: str
    order: int = Field(0, ge=0)
    claim_ids: list[str] = []
    text: str = ""
    status: Literal["new", "changed", "unchanged"] = "new"
    regenerated: bool = True
    uncertainty: list[IntentGap] = []


class AnswerClaimDiff(Contract):
    added: list[str] = []
    modified: list[tuple[str, str]] = []           # (old claim id, new claim id) expressing the same facts
    removed: list[str] = []
    unchanged: list[str] = []
    sections_regenerated: list[str] = []
    sections_reused: list[str] = []


class GroundedAnswer(Contract):
    answer_id: str                                 # "GA<n>" (session-scoped)
    version: int = Field(ge=1)                     # within the frame
    parent_answer_id: str | None = None
    frame_id: str
    phase6_answer_id: str
    utterance_id: str
    status: AnswerStatus
    partial: bool = False
    mode: Literal["strict", "relaxed"] = "strict"
    sections: list[GroundedSection] = []
    claims: list[AnswerClaim] = []
    verifications: dict[str, ClaimVerification] = {}
    rejected: list[RejectedClaim] = []
    repairs: list[RepairRecord] = []
    citations: CitationMap
    citation_report: CitationReport = CitationReport()
    coverage: CoverageReport = CoverageReport()
    consistency_conflicts: list[tuple[str, str]] = []
    blocked_reasons: list[str] = []
    diff: AnswerClaimDiff = AnswerClaimDiff()
    text: str = ""
    backend: str = "extractive"
    model: str | None = None
    fallback: str | None = None
    llm_calls: int = Field(0, ge=0)
    prompt_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    ttft_raw_ms: float | None = None
    validation_retrievals: int = Field(0, ge=0)
    revision_attempts: int = Field(0, ge=0)
    verifier: str = ""
    timings_ms: dict[str, float] = {}
    metrics: dict[str, float | None] = {}
    created_at_ms: float = Field(0.0, ge=0)

    def claim(self, claim_id: str) -> AnswerClaim | None:
        return next((c for c in self.claims if c.claim_id == claim_id), None)
