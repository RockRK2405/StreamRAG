"""Query ledger records (Phase 4; intent lineage added in Phase 5). One record per query version; lineage via
supersedes/superseded_by. In multi-intent mode every record belongs to one intent (``intent_id``) and lineage is
per intent: Intent -> Query versions -> Retrieval -> Evidence."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract

QueryStatus = Literal["pending", "queued", "in_flight", "completed", "failed", "cancelled", "reused"]


class QueryRecord(Contract):
    query_id: str                                   # Q1, Q2, ... (session-scoped, monotonic)
    session_id: str
    utterance_id: str
    version: int = Field(ge=1)                      # version within the utterance
    query_text: str
    transcript_snapshot: str                        # transcript the query was built from (traceability)
    source_spans: list[tuple[int, int]] = []        # char spans of kept tokens in transcript_snapshot
    terms: list[str] = []                           # analyzed content terms (novelty / lineage)
    created_at_ms: float = Field(ge=0)              # session stream clock
    trigger: Literal["provisional", "final"]
    trigger_chunk: int | None = None                # chunk_index whose arrival triggered it (None: timer/end)
    tick: Literal["chunk", "stability_timer", "utterance_end"]
    decision_reason: str
    status: QueryStatus = "pending"
    stale: bool = False                             # superseded by a newer version (of the utterance's / intent's query)
    stale_reason: Literal["superseded_query", "intent_superseded", "intent_removed"] | None = None
    intent_id: str | None = None                    # Phase 5: the intent this query serves (None in single-query mode)
    intent_version: int | None = None
    batch_id: str | None = None                     # Phase 5: queries dispatched together (MULTI_QUERY_STARTED)
    parent_query_id: str | None = None              # Phase 6: query this one was derived from (delta query)
    derived_from_change_id: str | None = None       # Phase 6: ContextChange that caused it
    semantic_key: str | None = None                 # Phase 6: cache key (analyzed terms + options + index)
    reused_from: str | None = None                  # Phase 6: status "reused" -> evidence of this earlier query
    supersedes: str | None = None
    superseded_by: str | None = None
    relation: Literal["initial", "refines", "replaces"] = "initial"
    lineage_root: str
    retrieval_queued_at_ms: float | None = None
    retrieval_started_at_ms: float | None = None
    retrieval_completed_at_ms: float | None = None
    retrieval_status: str | None = None             # ok | partial | degraded | empty | error | timeout | cancelled
    stale_at_completion: bool | None = None
    evidence_set_id: str | None = None
    evidence_ids: list[str] = []
    citations: list[str] = []
    error: str | None = None
