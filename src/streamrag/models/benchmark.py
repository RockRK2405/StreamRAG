"""Benchmark contracts (spec §22.2) and the flattened retrieval-evaluation item derived from them.

Gold evidence is authored only from the real corpus. Fixture datasets must set ``is_fixture=True``;
the harness then marks every output NOT REPORTABLE.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from streamrag.models.base import Contract

Category = Literal["single_intent", "multi_intent", "incremental_intent", "early_retrieval", "suppression",
                   "late_detail", "multiple_late_details", "ambiguous", "irrelevant_continuation",
                   "long_streaming", "session_isolation"]


class CaseChunk(Contract):
    chunk_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)
    text: str


class GoldIntent(Contract):
    gold_intent_id: str
    description: str
    paraphrases: list[str] = []
    shared_slots: list[str] = []
    constraints: list[dict[str, str]] = []
    answerable: bool = True
    gold_evidence: list[str] = []      # citation keys "Doc_ID §Section" (or chunk ids)
    min_evidence: int = Field(1, ge=0)

    @model_validator(mode="after")
    def _answerable_needs_evidence(self) -> "GoldIntent":
        if self.answerable and not self.gold_evidence:
            raise ValueError("answerable gold intent needs gold_evidence (authored from the real corpus)")
        return self


class RefinementExpectation(Contract):
    is_refinement: bool = True
    refines_utterance: str
    delta: list[dict[str, str]] = []
    affected_gold_intents: list[str] = []
    must_retain_citations: list[str] = []
    must_not_reissue_prior_queries: bool = True
    expected_new_evidence: list[str] = []


class CitationExpectation(Contract):
    must_include_any_of: list[list[str]] = []
    must_not_include: list[str] = []
    subset_of_previous: bool = False


class TurnExpectation(Contract):
    turn_type: Literal["query", "refinement", "presentation", "social", "backchannel", "meta"]
    retrieval_required: bool
    suppression_reason: str | None = None
    earliest_retrieval_chunk_index: int | None = None
    latest_acceptable_first_retrieval_s: float | None = None
    intents: list[GoldIntent] = []
    refinement: RefinementExpectation | None = None
    citations: CitationExpectation = CitationExpectation()
    uncertainty_expected_for: list[str] = []
    clarification_expected: bool = False
    reference_answer: str | None = None   # only if fully writable from cited corpus text


class CaseTurn(Contract):
    utterance_id: str
    utterance_text: str
    word_timing: list[dict[str, float | str]] = []
    chunks: list[CaseChunk] = []
    utterance_end_s: float = Field(ge=0)
    chunking: dict[str, str | int | float] = {}
    expected: TurnExpectation


class CaseSession(Contract):
    session_id: str
    turns: list[CaseTurn] = Field(min_length=1)


class BenchmarkCase(Contract):
    case_id: str
    schema_version: str = "1.0"
    category: Category
    split: Literal["tune", "test"]
    description: str = ""
    corpus_ref: dict[str, str] = {}
    session: CaseSession
    annotation: dict[str, object] = {}
    provenance: dict[str, object] = {}
    is_fixture: bool = False

    def retrieval_items(self) -> list["RetrievalEvalItem"]:
        """Flatten answerable gold intents into single-query retrieval items (per-intent Recall@k)."""
        items = []
        for turn in self.session.turns:
            for gi in turn.expected.intents:
                if not gi.answerable:
                    continue
                items.append(RetrievalEvalItem(
                    item_id=f"{self.case_id}:{turn.utterance_id}:{gi.gold_intent_id}",
                    query=gi.description, gold=gi.gold_evidence, gold_level="section",
                    case_id=self.case_id, intent_id=gi.gold_intent_id, category=self.category,
                    split=self.split, is_fixture=self.is_fixture))
        return items


class RetrievalEvalItem(Contract):
    item_id: str
    query: str = Field(min_length=1)
    gold: list[str] = Field(min_length=1)      # chunk ids, citation keys ("Doc §Sec") or document ids
    gold_level: Literal["chunk", "section", "document"] = "section"
    case_id: str | None = None
    intent_id: str | None = None
    category: str | None = None
    split: Literal["tune", "test"] | None = None
    is_fixture: bool = False


# ---------------------------------------------------------------------------------------------- Phase 5
MICategory = Literal["A_two_independent", "B_three_independent", "C_incremental_second", "D_intent_constraint",
                     "E_refinement", "F_follow_up", "G_pronoun", "H_shared_context", "I_over_decomposition_trap",
                     "J_under_decomposition_trap", "K_duplicate", "L_changed_intent", "S_single"]


class MIGoldIntent(Contract):
    gold_intent_id: str                                # unique within the case ("u1.g1")
    description: str                                   # the need in annotator words (not the utterance text)
    paraphrases: list[str] = []
    answerable: bool = True
    gold_evidence: list[str] = []                      # citation keys of the (fixture) corpus
    required_query_terms: list[str] = []               # context that must reach this intent's query (carry-over)
    superseded: bool = False                           # replaced later by a correction: must not stay active

    @model_validator(mode="after")
    def _answerable_needs_evidence(self) -> "MIGoldIntent":
        if self.answerable and not self.gold_evidence:
            raise ValueError("answerable gold intent needs gold_evidence")
        return self


class MIGoldConstraint(Contract):
    text: str
    scope: Literal["global", "local"]
    applies_to: list[str]                              # gold intent ids


class MIGoldRelation(Contract):
    type: Literal["DEPENDENT", "FOLLOW_UP", "REFINEMENT", "COMPARISON_WITH"]
    source: str
    target: str


class MIUtterance(Contract):
    utterance_id: str
    utterance_text: str
    chunks: list[CaseChunk] = Field(min_length=1)
    utterance_end_s: float = Field(ge=0)
    expected_intents: list[MIGoldIntent] = []
    expected_constraints: list[MIGoldConstraint] = []
    expected_relationships: list[MIGoldRelation] = []
    expected_query_count: int = Field(ge=0)            # distinct final per-intent queries for this utterance


class MultiIntentBenchmarkCase(Contract):
    """Phase 5 multi-intent case (brief §33). Behaviour + gold evidence only: no answers, no retrieval results."""

    case_id: str
    schema_version: str = "1.0"
    category: MICategory
    split: Literal["tune", "test"] = "tune"
    description: str = ""
    utterances: list[MIUtterance] = Field(min_length=1)
    is_fixture: bool = False
    provenance: dict[str, object] = {}
