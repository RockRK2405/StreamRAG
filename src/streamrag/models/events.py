"""Input events (spec §5) and the telemetry envelope (spec §6).

Inputs carry only what a real streaming front-end could know. There are deliberately no fields for
intents, turn types or "is_refinement" — those are inferred by the system (spec §5.1).
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import Field, model_validator

from streamrag.models.base import SCHEMA_VERSION, Contract

SessionId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
UtteranceId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$")]


class SessionStartPayload(Contract):
    locale: str = "en"
    client: Literal["replay", "interactive", "asr"] = "replay"
    session_started_wall: str | None = None  # informational only; never used in logic


class TranscriptChunkPayload(Contract):
    chunk_index: int = Field(ge=0)
    timestamp_s: float = Field(ge=0)  # utterance-relative arrival time (guide semantics)
    text: str = ""                    # delta text, not cumulative
    stability: Literal["final", "partial"] = "final"
    replaces_chunk_index: int | None = Field(default=None, ge=0)
    utterance_offset_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _revision_points_backwards(self) -> "TranscriptChunkPayload":
        if self.replaces_chunk_index is not None and self.replaces_chunk_index > self.chunk_index:
            raise ValueError("replaces_chunk_index must refer to an earlier or equal chunk_index")
        return self


class UtteranceEndPayload(Contract):
    timestamp_s: float = Field(ge=0)
    reason: Literal["endpoint", "explicit", "timeout", "eof"] = "endpoint"
    last_chunk_index: int | None = Field(default=None, ge=0)


class SessionEndPayload(Contract):
    reason: str = "client_closed"


class SessionStart(Contract):
    schema_version: str = SCHEMA_VERSION
    type: Literal["SESSION_START"] = "SESSION_START"
    session_id: SessionId
    payload: SessionStartPayload = SessionStartPayload()


class TranscriptChunk(Contract):
    schema_version: str = SCHEMA_VERSION
    type: Literal["TRANSCRIPT_CHUNK"] = "TRANSCRIPT_CHUNK"
    session_id: SessionId
    utterance_id: UtteranceId
    payload: TranscriptChunkPayload


class UtteranceEnd(Contract):
    schema_version: str = SCHEMA_VERSION
    type: Literal["UTTERANCE_END"] = "UTTERANCE_END"
    session_id: SessionId
    utterance_id: UtteranceId
    payload: UtteranceEndPayload


class SessionEnd(Contract):
    schema_version: str = SCHEMA_VERSION
    type: Literal["SESSION_END"] = "SESSION_END"
    session_id: SessionId
    payload: SessionEndPayload = SessionEndPayload()


InputEvent = Annotated[Union[SessionStart, TranscriptChunk, UtteranceEnd, SessionEnd], Field(discriminator="type")]


class EventType(str, Enum):
    """Output event vocabulary: spec §6.2 (16 types) amended in Phase 4 (ADR-012 amendment):
    CONTROLLER_DECISION renamed RETRIEVAL_DECISION; added SESSION_STARTED, TRANSCRIPT_UPDATED, QUERY_UPDATED,
    RETRIEVAL_CANCELLED, UTTERANCE_FINALIZED. Phase 5 (ADR-015) added INTENT_DETECTED, INTENT_UPDATED,
    INTENT_SUPERSEDED, QUERY_GENERATED, MULTI_QUERY_STARTED, MULTI_QUERY_COMPLETED, EVIDENCE_DEDUPLICATED,
    RERANK_STARTED, RERANK_COMPLETED. Phase 6 (ADR-016) added SESSION_VERSION_CREATED, CONTEXT_CHANGE_DETECTED,
    DELTA_PLAN_CREATED, QUERY_REUSED, QUERY_SUPERSEDED, EVIDENCE_RETAINED, EVIDENCE_INVALIDATED, EVIDENCE_REVALIDATED,
    CLAIM_CREATED, CLAIM_INVALIDATED, CLAIM_REVALIDATED, ANSWER_VERSION_CREATED, ANSWER_VERSION_UPDATED.
    Every event is telemetry."""

    SESSION_STARTED = "SESSION_STARTED"
    CHUNK_RECEIVED = "CHUNK_RECEIVED"
    TRANSCRIPT_UPDATED = "TRANSCRIPT_UPDATED"
    RETRIEVAL_DECISION = "RETRIEVAL_DECISION"
    QUERY_UPDATED = "QUERY_UPDATED"
    RETRIEVAL_CANCELLED = "RETRIEVAL_CANCELLED"
    UTTERANCE_FINALIZED = "UTTERANCE_FINALIZED"
    INTENTS_UPDATED = "INTENTS_UPDATED"
    INTENT_DETECTED = "INTENT_DETECTED"
    INTENT_UPDATED = "INTENT_UPDATED"
    INTENT_SUPERSEDED = "INTENT_SUPERSEDED"
    QUERY_GENERATED = "QUERY_GENERATED"
    MULTI_QUERY_STARTED = "MULTI_QUERY_STARTED"
    MULTI_QUERY_COMPLETED = "MULTI_QUERY_COMPLETED"
    EVIDENCE_DEDUPLICATED = "EVIDENCE_DEDUPLICATED"
    RERANK_STARTED = "RERANK_STARTED"
    RERANK_COMPLETED = "RERANK_COMPLETED"
    SESSION_VERSION_CREATED = "SESSION_VERSION_CREATED"
    CONTEXT_CHANGE_DETECTED = "CONTEXT_CHANGE_DETECTED"
    DELTA_PLAN_CREATED = "DELTA_PLAN_CREATED"
    QUERY_REUSED = "QUERY_REUSED"
    QUERY_SUPERSEDED = "QUERY_SUPERSEDED"
    EVIDENCE_RETAINED = "EVIDENCE_RETAINED"
    EVIDENCE_INVALIDATED = "EVIDENCE_INVALIDATED"
    EVIDENCE_REVALIDATED = "EVIDENCE_REVALIDATED"
    CLAIM_CREATED = "CLAIM_CREATED"
    CLAIM_INVALIDATED = "CLAIM_INVALIDATED"
    CLAIM_REVALIDATED = "CLAIM_REVALIDATED"
    ANSWER_VERSION_CREATED = "ANSWER_VERSION_CREATED"
    ANSWER_VERSION_UPDATED = "ANSWER_VERSION_UPDATED"
    RETRIEVAL_STARTED = "RETRIEVAL_STARTED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    RETRIEVAL_SKIPPED = "RETRIEVAL_SKIPPED"
    EVIDENCE_RERANKED = "EVIDENCE_RERANKED"
    EVIDENCE_FUSED = "EVIDENCE_FUSED"
    SYNTHESIS_STARTED = "SYNTHESIS_STARTED"
    ANSWER_DELTA = "ANSWER_DELTA"
    LLM_CALL = "LLM_CALL"
    GROUNDING_CHECKED = "GROUNDING_CHECKED"
    ANSWER_COMMITTED = "ANSWER_COMMITTED"
    TURN_COMPLETED = "TURN_COMPLETED"
    ERROR = "ERROR"
    SESSION_CLOSED = "SESSION_CLOSED"


class TelemetryEvent(Contract):
    """Common envelope (spec §6.1). ``event_id`` is deterministic: ``<session_id>:<seq:06d>``."""

    schema_version: str = SCHEMA_VERSION
    event_id: str
    type: EventType
    session_id: str
    utterance_id: str | None = None
    intent_id: str | None = None                             # Phase 5: set on intent-scoped events
    query_id: str | None = None                              # Phase 5: set on query-scoped events
    seq: int = Field(ge=0)
    t_session_ms: float | None = Field(default=None, ge=0)   # session stream clock (deterministic in virtual mode)
    t_stream_s: float | None = None                          # utterance-relative stream time (spec §6.1)
    t_wall_ms: float = Field(ge=0)                           # monotonic wall clock since run start (non-deterministic)
    component: str
    payload: dict[str, Any] = {}

    @model_validator(mode="after")
    def _deterministic_id(self) -> "TelemetryEvent":
        expected = f"{self.session_id}:{self.seq:06d}"
        if self.event_id != expected:
            raise ValueError(f"event_id must be '{expected}' (deterministic session:seq)")
        return self
