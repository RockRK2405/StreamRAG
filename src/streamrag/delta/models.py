"""Delta-planning and evidence-lifecycle contracts (Phase 6; docs/session/04-05)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract
from streamrag.models.intents import IntentQuery

EvidenceStatus = Literal["ACTIVE", "RETAINED", "REVALIDATION_REQUIRED", "STALE", "SUPERSEDED", "INVALID"]
USABLE_EVIDENCE: tuple[str, ...] = ("ACTIVE", "RETAINED")   # evidence a need's claims/answer may rely on


class EvidenceTransition(Contract):
    from_status: EvidenceStatus | None
    to_status: EvidenceStatus
    rule: str                                   # the validity rule that decided it
    change_id: str | None = None
    query_id: str | None = None
    at_ms: float = Field(0.0, ge=0)


class EvidenceRecord(Contract):
    """A corpus chunk as seen in this session (one per evidence id)."""

    evidence_id: str
    document_id: str
    section_id: str
    citation: str
    text: str
    first_seen_query: str
    first_seen_ms: float = Field(0.0, ge=0)


class EvidenceAssignment(Contract):
    """Applicability of one evidence item to one need (the unit of the evidence lifecycle)."""

    evidence_id: str
    intent_id: str
    status: EvidenceStatus
    intent_version: int = Field(ge=1)          # need version it was last validated for
    query_ids: list[str] = []
    best_rank: int = Field(ge=1)
    history: list[EvidenceTransition] = []


class QueryAction(Contract):
    intent_id: str
    intent_version: int = Field(ge=1)
    action: Literal["retrieve", "reuse_active", "cache_hit"]
    reason: str
    query: IntentQuery
    semantic_key: str
    parent_query_id: str | None = None          # query the delta was derived from
    supersedes_query_id: str | None = None
    reused_query_id: str | None = None          # reuse_active / cache_hit: evidence of this query is used
    change_id: str | None = None
    scope: Literal["corpus", "session_docs_first"] = "corpus"


class EvidenceAction(Contract):
    evidence_id: str
    intent_id: str
    decision: Literal["RETAIN", "REVALIDATE", "SUPERSEDE", "DISCARD"]
    from_status: EvidenceStatus | None
    to_status: EvidenceStatus
    rule: str


class DeltaPlan(Contract):
    plan_id: str                                # "P<n>"
    utterance_id: str
    change_ids: list[str] = []
    affected_intents: list[str] = []
    new_intents: list[str] = []
    queries_to_create: list[QueryAction] = []   # action == retrieve
    queries_to_reuse: list[QueryAction] = []    # reuse_active | cache_hit
    queries_to_supersede: list[str] = []        # previous active queries of changed / superseded needs
    evidence_to_retain: list[str] = []          # "evidence_id@intent_id"
    evidence_to_revalidate: list[str] = []
    evidence_to_discard: list[str] = []         # superseded / stale (never deleted: status only)
    claims_to_revalidate: list[str] = []
    claims_unaffected: list[str] = []
    full_restart: bool = False
