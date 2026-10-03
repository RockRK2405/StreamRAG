"""Generation contracts (Phase 7; docs/answer/02): the answer plan the generator follows and the candidate answer
it returns (before verification)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.claims.models import IntentGap, PlannedClaim
from streamrag.models.base import Contract


class AnswerPlanSection(Contract):
    section_id: str
    intent_id: str
    title: str
    order: int = Field(0, ge=0)
    constraints: list[str] = []
    facts: list[PlannedClaim] = []                  # planned claims to express, in order
    omitted_facts: list[str] = []                   # planned but left out (concise mode / cap), by id
    uncertainties: list[IntentGap] = []             # mentioned deterministically by the renderer, not by the LLM
    regenerate: bool = True                         # False: reuse the previous version's sentences verbatim
    keep_sentences: list[str] = []                  # sentence ids reused verbatim inside a regenerated section


class AnswerPlan(Contract):
    answer_plan_id: str
    claim_plan_id: str
    frame_id: str
    detail: Literal["concise", "detailed"] = "detailed"
    citation_placement: Literal["per_claim"] = "per_claim"
    sections: list[AnswerPlanSection] = []
    labels: dict[str, str] = {}                     # evidence label (E1..) -> evidence id
    conflicts: list[tuple[str, str]] = []


class CandidateSentence(Contract):
    sentence_id: str
    section_id: str
    intent_id: str
    text: str = Field(min_length=1)
    facts: list[str] = []                           # planned-claim ids the sentence says it expresses
    labels: list[str] = []                          # evidence labels the generator cited
    kind: Literal["fact", "uncertainty", "conflict", "connective"] = "fact"
    origin: Literal["llm", "extractive", "repair", "template", "kept"] = "llm"


class CandidateAnswer(Contract):
    sentences: list[CandidateSentence] = []
    backend: str
    model: str | None = None
    structured_ok: bool = True
    fallback: str | None = None                     # why the extractive generator was used, if it was
    llm_calls: int = Field(0, ge=0)
    prompt_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    ttft_ms: float | None = None
    generation_ms: float = 0.0
