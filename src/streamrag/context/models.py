"""Context-change contracts (Phase 6; docs/session/03).

Change taxonomy (9 types; only those the pipeline acts on differently):

  NO_CHANGE            nothing retrieval-relevant changed (backchannel, restatement, narrative aside)
  REFINEMENT           the same need, stated more specifically (terms added, none removed)
  CONSTRAINT_ADDITION  a constraint now applies to a need (also the "set" half of a constraint update)
  CONSTRAINT_REMOVAL   a constraint no longer applies (retracted, or replaced by an update)
  NEW_INTENT           a need was added (follow-up in the current frame, or a new topic -> new frame)
  INTENT_REMOVAL       a need disappeared from the interpretation (e.g. ASR revision)
  CORRECTION           an explicit correction superseded a need ("I meant Y, not X")
  ENTITY_CHANGE        a need's topic words were replaced without a correction marker
  QUESTION_CHANGE      a need's facet / question changed (e.g. requirements -> timeline) without a correction marker

The brief's other labels map onto these: MODIFIED_INTENT = REFINEMENT | ENTITY_CHANGE | QUESTION_CHANGE;
NEW_CONSTRAINT / REMOVED_CONSTRAINT = CONSTRAINT_ADDITION / _REMOVAL; ENTITY_REFINEMENT / QUERY_REFINEMENT =
REFINEMENT; FOLLOW_UP = NEW_INTENT(relation=follow_up) or CONSTRAINT_ADDITION from an elliptical follow-up;
EXPANSION = NEW_INTENT(frame=same); NO_MEANINGFUL_CHANGE = NO_CHANGE.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract

ChangeType = Literal["NO_CHANGE", "REFINEMENT", "CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL", "NEW_INTENT",
                     "INTENT_REMOVAL", "CORRECTION", "ENTITY_CHANGE", "QUESTION_CHANGE"]
CHANGE_TYPES: tuple[str, ...] = ("NO_CHANGE", "REFINEMENT", "CONSTRAINT_ADDITION", "CONSTRAINT_REMOVAL", "NEW_INTENT",
                                 "INTENT_REMOVAL", "CORRECTION", "ENTITY_CHANGE", "QUESTION_CHANGE")


class SemanticDiff(Contract):
    """Interpretation-level difference for one need: terms, topic, facet and constraints (not raw strings)."""

    intent_id: str
    from_version: int | None = None
    to_version: int | None = None
    terms_added: list[str] = []
    terms_removed: list[str] = []
    topic_before: str | None = None
    topic_after: str | None = None
    aspect_before: str | None = None
    aspect_after: str | None = None
    constraints_added: list[str] = []          # constraint ids
    constraints_removed: list[str] = []


class ContextChange(Contract):
    change_id: str                              # "CH<n>" (session-scoped)
    change_type: ChangeType
    utterance_id: str
    affected_intents: list[str] = []            # needs whose interpretation changed (existing ones)
    new_intents: list[str] = []
    superseded_intents: list[str] = []
    added_constraints: list[str] = []           # constraint texts (ids in diffs)
    removed_constraints: list[str] = []
    affected_queries: list[str] = []            # active queries of the affected needs
    relation: Literal["follow_up", "independent", "correction", "same_need", "none"] = "none"
    frame_action: Literal["same", "new_frame", "reactivated", "none"] = "none"
    diffs: list[SemanticDiff] = []
    cue: str = ""                               # what established it (rule / marker)
    confidence: float = Field(ge=0, le=1)       # computed: see detector docstring
    confidence_signals: dict[str, float] = {}
    session_version_from: int = Field(0, ge=0)
    at_ms: float = Field(0.0, ge=0)


class TopicFrame(Contract):
    """A topic thread of the session (spec §13.4): the needs one answer is about."""

    frame_id: str                               # "T<n>"
    intent_ids: list[str] = []
    status: Literal["active", "dormant"] = "active"
    opened_in: str
    topic_terms: list[str] = []
    last_answer_version: int | None = None
