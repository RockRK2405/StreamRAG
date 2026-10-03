"""Phase 7 claim contracts: claim plans, evidence alignments, verification results (docs/answer/01, 03-04).

Phase 6 ``Claim`` (models/answers.py) stays the stored claim record; these contracts describe *how* a claim was
planned, aligned and verified. Support strengths are categorical on purpose: the entailment model's probabilities
are not calibrated, so they are kept only as raw signals, never as a reported "confidence".
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.answers import ClaimSource
from streamrag.models.base import Contract

SupportStrength = Literal["STRONG", "MODERATE", "WEAK", "NONE", "CONTRADICTORY"]
Importance = Literal["critical", "important", "supplementary"]
VerificationStatus = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
SUPPORTING: tuple[str, ...] = ("STRONG", "MODERATE")


class PlannedClaim(Contract):
    """An atomic, evidence-derived statement the answer may express (never invented: verbatim evidence text or an
    atom of it that the verifier accepted as entailed by its source sentence)."""

    plan_claim_id: str                              # "F<n>" within the plan ("fact" label shown to the generator)
    intent_id: str
    section_id: str
    text: str = Field(min_length=1)
    source: ClaimSource                             # evidence id + span of the source sentence
    evidence_ids: list[str] = Field(min_length=1)   # candidates (the source evidence first)
    phase6_claim_id: str | None = None              # the extractive claim it comes from
    atom_of: str | None = None                      # sentence group: atoms of one sentence share it
    importance: Importance = "important"
    importance_basis: str = ""
    order: int = Field(0, ge=0)
    depends_on: list[str] = []                      # earlier atoms of the same sentence (shared subject)
    entailed_by_source: bool = True                 # verified at planning time (atoms only)


class IntentGap(Contract):
    intent_id: str
    kind: Literal["no_evidence", "constraint_not_covered", "conflict", "unsupported_removed", "value_not_stated"]
    aspect: str
    evidence_ids: list[str] = []


class IntentPlan(Contract):
    intent_id: str
    section_id: str
    title: str
    order: int = Field(0, ge=0)
    constraints: list[str] = []
    claims: list[PlannedClaim] = []
    gaps: list[IntentGap] = []
    needs_regeneration: bool = True


class ClaimPlan(Contract):
    plan_id: str                                    # "CP<n>"
    frame_id: str
    answer_id: str                                  # Phase 6 answer version it plans for
    intents: list[IntentPlan] = []
    conflicts: list[tuple[str, str]] = []           # planned-claim pairs whose evidence disagrees


class EvidenceAlignment(Contract):
    claim_id: str
    evidence_id: str
    strength: SupportStrength
    basis: str                                      # rule that set the strength
    premise: Literal["sentence", "window", "chunk", "none"] = "none"
    premise_span: tuple[int, int] | None = None     # characters of the evidence text that support it
    signals: dict[str, float] = {}                  # raw (uncalibrated) entailment probabilities, overlap


class ClaimVerification(Contract):
    claim_id: str
    status: VerificationStatus
    supported: bool                                 # Q1
    supporting_evidence: list[str] = []             # Q2
    entailed: bool = False                          # Q3: an entailment premise exists (not similarity)
    sufficient: bool = False                        # Q4: entailed and every number / date is in the premise
    contradicting_evidence: list[str] = []          # Q5
    cited_evidence: list[str] = []
    invalid_labels: list[str] = []                  # cited labels that are not in the evidence set (L0)
    support_outside_citations: list[str] = []       # supports it but was not cited (relabel candidate)
    alignments: list[EvidenceAlignment] = []
    atoms: list["ClaimVerification"] = []           # set when a compound statement was decomposed
    atom_texts: list[str] = []
    reasons: list[str] = []
    verifier: str = "nli+rules"
