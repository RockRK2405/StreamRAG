"""Session state contracts (Phase 6; docs/session/01-02).

The session's useful state is kept in four separate layers (never mixed):

  A transcript memory   what the user said: the last ``transcript_window`` utterances verbatim (PII-redacted);
                        older utterances survive only as hashes + the structured state derived from them
  B semantic memory     topic frames, needs (intents + versions), constraints (with lifecycle), entities
  C retrieval memory    query lineage (ledger), semantic retrieval cache, evidence store with lifecycle
  D answer memory       claims, claim-evidence links, claim transitions, answer versions

``SessionStateVersion`` records every state change with its parent, the context changes that caused it and a
compact, deterministic ``SessionSnapshot`` (ids -> statuses) for replay comparison.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.context.models import TopicFrame
from streamrag.models.answers import AnswerVersion
from streamrag.models.base import Contract
from streamrag.models.intents import Constraint, SourceSpan


class TranscriptEntry(Contract):
    utterance_id: str
    text: str = ""                              # redacted; "" once compressed out of the window
    text_sha1: str
    chars: int = Field(ge=0)
    final: bool = False
    compressed: bool = False
    redactions: int = Field(0, ge=0)


class EntityRecord(Contract):
    text: str
    terms: list[str]
    first_seen: str                             # utterance id
    last_seen: str
    intent_ids: list[str] = []
    spans: list[SourceSpan] = []


class SessionSnapshot(Contract):
    """Compact, deterministic view (ids -> status) used by versions and replay."""

    frame_id: str | None = None
    frames: dict[str, str] = {}                 # frame id -> status
    intents: dict[str, str] = {}                # intent id -> "v<n>:<status>"
    constraints: dict[str, str] = {}
    queries: dict[str, str] = {}                # query id -> status(+":stale")
    evidence: dict[str, str] = {}               # "evidence_id@intent_id" -> status
    claims: dict[str, str] = {}
    answer_id: str | None = None


class SessionStateVersion(Contract):
    version_id: int = Field(ge=1)
    parent_version: int | None = None
    created_at_ms: float = Field(0.0, ge=0)
    utterance_id: str | None = None
    trigger: Literal["interpretation", "evidence", "answer", "reset", "restore"]
    changes: list[str] = []                     # context change ids
    summary: str = ""
    state_snapshot: SessionSnapshot


class SessionState(Contract):
    """Canonical view of the current session (SessionMemory.get_current_state)."""

    session_id: str
    current_utterance_id: str | None = None
    session_version: int = Field(0, ge=0)
    transcript_state: list[TranscriptEntry] = []
    frames: list[TopicFrame] = []
    active_frame_id: str | None = None
    intent_set: list[dict] = []                 # active needs of the active frame (id, version, text, constraints)
    constraints: list[Constraint] = []          # active constraints
    entities: list[EntityRecord] = []
    query_ledger: list[dict] = []
    active_queries: list[str] = []
    superseded_queries: list[str] = []
    evidence_store: dict[str, int] = {}         # status -> count
    claims: dict[str, int] = {}                 # status -> count
    answer_state: AnswerVersion | None = None
    telemetry_state: dict[str, float] = {}
    configuration: dict[str, str | int | float | bool] = {}


class SessionArchive(Contract):
    """What survives archive_session(): debugging metadata, no raw utterance or evidence text."""

    session_id: str
    archived_at_ms: float = Field(0.0, ge=0)
    config_hash: str
    index_content_hash: str
    utterances: int = Field(0, ge=0)
    transcript_sha1: list[str] = []
    session_versions: int = Field(0, ge=0)
    change_types: dict[str, int] = {}
    queries: dict[str, int] = {}
    evidence_status: dict[str, int] = {}
    claim_status: dict[str, int] = {}
    answer_versions: int = Field(0, ge=0)
    counters: dict[str, float] = {}
