"""Intent contracts (spec §9.2, refined in Phase 5; docs/multi_intent/01).

An *intent* is one retrieval-relevant information need inside an utterance. Every intent, constraint and piece of
inherited context carries a ``SourceSpan`` into the transcript it came from, so the query built from it is fully
traceable to what was said. Confidences are computed by documented formulas (never authored); see
``streamrag.intents.decomposer``.

IDs are session-scoped and monotonic (``I1, I2, ...`` for intents, ``K1, K2, ...`` for constraints), because
follow-up utterances can depend on earlier intents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from streamrag.models.base import Contract

IntentType = Literal["FACTUAL", "PROCEDURAL", "REQUIREMENT", "TIMELINE", "EXCEPTION", "DEFINITION", "COMPARISON",
                     "OTHER"]
INTENT_TYPES: tuple[str, ...] = ("FACTUAL", "PROCEDURAL", "REQUIREMENT", "TIMELINE", "EXCEPTION", "DEFINITION",
                                 "COMPARISON", "OTHER")
IntentStatus = Literal["ACTIVE", "SUPERSEDED", "MERGED", "DROPPED"]
ConstraintKind = Literal["focus", "condition", "restriction"]
RelationType = Literal["DEPENDENT", "FOLLOW_UP", "REFINEMENT", "COMPARISON_WITH", "CONSTRAINT_OF"]
InheritReason = Literal["distributed_pp", "fronted_context", "anaphora", "follow_up_ellipsis", "correction_aspect"]
DecompositionSource = Literal["rule", "rule_fallback", "llm", "reconciled"]


class SourceSpan(Contract):
    """Characters ``[start, end)`` of one utterance's transcript; ``text`` must equal that substring."""

    utterance_id: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _ordered(self) -> "SourceSpan":
        if self.end < self.start:
            raise ValueError("span end < start")
        return self


class QueryComponent(Contract):
    text: str
    source: Literal["intent", "inherited", "constraint"]
    ref: str | None = None                          # intent / constraint id
    span: SourceSpan


class Constraint(Contract):
    constraint_id: str
    kind: ConstraintKind
    marker: str | None = None                       # "especially", "if", "for", ...
    text: str = Field(min_length=1)                 # the constraint content as spoken ("category z")
    scope: Literal["global", "local"]
    applies_to: list[str] = []                      # intent ids
    scope_reason: str = ""                          # why this scope (rule name)
    scope_confidence: float = Field(1.0, ge=0, le=1)
    source_span: SourceSpan
    op: Literal["set", "update", "retract"] = "set"
    status: Literal["active", "retracted"] = "active"   # Phase 6: session-level lifecycle
    retracted_in: str | None = None                     # utterance that retracted / replaced it
    replaces: str | None = None                         # constraint id this one updated ("op: update")


class InheritedContext(Contract):
    """Words an intent's query borrows from elsewhere (another clause, intent or utterance) - never invented."""

    text: str = Field(min_length=1)
    reason: InheritReason
    from_intent: str | None = None
    replaces: str | None = None                     # the anaphor it resolves ("it"), if any
    source_span: SourceSpan


class IntentProvenance(Contract):
    source: DecompositionSource
    split: Literal["single", "clause", "coordination", "correction", "fallback"]
    clause_index: int | None = None
    span: SourceSpan


class Intent(Contract):
    intent_id: str
    session_id: str
    utterance_id: str
    version: int = Field(1, ge=1)                   # revision of this intent (modified -> +1)
    text: str = Field(min_length=1)                 # the need as spoken (fillers / request heads removed)
    resolved_text: str = Field(min_length=1)        # text with anaphora resolved (what the query is built from)
    components: list[QueryComponent] = []           # resolved_text as verbatim, span-traceable pieces
    intent_type: IntentType = "OTHER"
    type_cues: list[str] = []
    entities: list[str] = []                        # content phrases, verbatim
    topic: str | None = None                        # what the need is about ("x" in "requirements for x")
    topic_span: SourceSpan | None = None            # where the topic was said (follow-ups inherit it from here)
    aspect: str | None = None                       # facet asked for ("requirements")
    constraint_ids: list[str] = []                  # constraints that apply (local + global in scope)
    inherited_context: list[InheritedContext] = []
    unresolved_references: list[str] = []
    order: int = Field(0, ge=0)                     # order of mention (answer organisation)
    priority: int = Field(1, ge=1)                  # retrieval priority rank (1 = first)
    priority_score: float = Field(0.0, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    confidence_signals: dict[str, float] = {}
    status: IntentStatus = "ACTIVE"
    provenance: IntentProvenance
    supersedes: str | None = None
    superseded_by: str | None = None
    lineage_root: str = ""
    created_at_ms: float = Field(0.0, ge=0)
    updated_at_ms: float = Field(0.0, ge=0)
    query_ids: list[str] = []                       # ledger queries serving this intent


class IntentRelationship(Contract):
    type: RelationType
    source: str                                     # intent id (or constraint id for CONSTRAINT_OF)
    target: str                                     # intent id
    cue: str = ""                                   # what established it ("pronoun 'it'", "terms contained", ...)


class MergeRecord(Contract):
    text: str
    into: str                                       # intent id kept
    reason: Literal["duplicate", "near_duplicate", "fixed_phrase"]


class DropRecord(Contract):
    text: str
    intent_id: str | None = None
    reason: str


class IntentSet(Contract):
    session_id: str
    utterance_id: str
    version: int = Field(ge=0)
    source: DecompositionSource
    original_text: str = ""
    intents: list[Intent] = []                      # ACTIVE intents, in order of mention
    global_constraints: list[Constraint] = []
    local_constraints: list[Constraint] = []
    relationships: list[IntentRelationship] = []
    decomposition_confidence: float = Field(0.0, ge=0, le=1)
    superseded: list[str] = []                      # intent ids superseded in this utterance (provenance retained)
    merged: list[MergeRecord] = []
    dropped: list[DropRecord] = []
    created_at_ms: float = Field(0.0, ge=0)

    def intent(self, intent_id: str) -> Intent | None:
        return next((i for i in self.intents if i.intent_id == intent_id), None)

    def constraints_for(self, intent_id: str) -> list[Constraint]:
        return [c for c in self.global_constraints + self.local_constraints if intent_id in c.applies_to]


class IntentChange(Contract):
    intent_id: str
    from_version: int
    to_version: int
    changed: list[Literal["text", "constraints", "context"]]


class Supersession(Contract):
    old: str
    new: str
    cue: str


class IntentSetDelta(Contract):
    """What changed between two IntentSet versions; drives delta retrieval (docs/multi_intent/09)."""

    utterance_id: str
    version: int = Field(ge=1)
    previous_version: int = Field(ge=0)
    added: list[str] = []
    modified: list[IntentChange] = []
    removed: list[str] = []
    superseded: list[Supersession] = []
    constraints_added: list[str] = []
    constraints_removed: list[str] = []
    affected_intents: list[str] = []                # intents whose query must be (re)built: added + modified
    cross_turn: list[str] = []                      # Phase 6: intents of earlier utterances modified by this one

    @property
    def empty(self) -> bool:
        return not (self.added or self.modified or self.removed or self.superseded or self.constraints_added
                    or self.constraints_removed)


class IntentQuery(Contract):
    """One retrieval query generated for one intent version (docs/multi_intent/04)."""

    intent_id: str
    intent_version: int = Field(ge=1)
    utterance_id: str
    text: str = Field(min_length=1)
    components: list[QueryComponent] = Field(min_length=1)
    terms: list[str] = []                           # analyzed content terms (dedup / lineage)
