"""Fusion contracts (Phase 5; docs/multi_intent/06). The output of Phase 5 is a ``UnifiedEvidenceSet``.

One ``FusedEvidence`` per corpus chunk: evidence retrieved by several intents/queries is a single object whose
``hits`` keep the full provenance (which intent, which query, which retrieval method, at which rank/score).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract
from streamrag.models.evidence import RetrievalMethod


class EvidenceHit(Contract):
    intent_id: str
    query_id: str | None = None
    retrieval_method: RetrievalMethod
    rank: int = Field(ge=1)                          # rank inside that query's result list
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    stale: bool = False                              # from a superseded query version of the intent


class FusedEvidence(Contract):
    label: str                                       # E1..En within this unified set
    evidence_id: str                                 # == chunk_id
    chunk_id: str
    document_id: str
    section_id: str
    citation: str
    section_title: str | None = None
    text: str
    source_path: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    supporting_intents: list[str]                    # every intent that retrieved it (cross-intent dedup)
    supporting_queries: list[str]
    selected_for: list[str]                          # intents whose budget/quota selected it
    hits: list[EvidenceHit] = Field(min_length=1)
    best_rank_by_intent: dict[str, int] = {}
    intent_relevance: dict[str, float] = {}          # cross-intent rerank scores (if a reranker ran)
    fused_score: float = 0.0
    alternates: list[str] = []                       # near-duplicate chunk ids collapsed into this item
    conflict_ids: list[str] = []
    token_count: int = Field(0, ge=0)


class IntentCoverage(Contract):
    intent_id: str
    query_id: str | None = None                      # query whose evidence was used
    status: Literal["ok", "no_candidates", "retrieval_failed", "pending"]
    candidates: int = Field(0, ge=0)
    evidence_ids: list[str] = []                     # items in the unified set supporting this intent
    labels: list[str] = []
    covered: bool = False                            # >= 1 unified item supports this intent
    best_rank: int | None = None


class EvidenceRelation(Contract):
    source: str                                      # evidence id
    target: str
    type: Literal["DUPLICATES", "RELATED", "CONTRADICTS"]
    basis: str                                       # how it was established (rule name)


class EvidenceConflict(Contract):
    """Potential conflict found by a conservative typed-value heuristic. Not verified: a later grounding stage
    must decide; the unified set only refuses to merge such items silently."""

    conflict_id: str
    kind: Literal["numeric_value"]
    status: Literal["potential"] = "potential"
    intent_ids: list[str]
    evidence_ids: list[str]
    detail: dict[str, str | list[str]] = {}


class UnifiedEvidenceSet(Contract):
    unified_set_id: str
    session_id: str
    utterance_id: str
    intent_set_version: int = Field(ge=0)
    strategy: Literal["concat", "global_score", "rrf", "intent_aware"]
    rerank: Literal["none", "intent_ce", "cross_intent_dense", "cross_intent_ce"] = "none"
    top_k: int = Field(ge=1)
    items: list[FusedEvidence] = []
    per_intent: list[IntentCoverage] = []
    relations: list[EvidenceRelation] = []
    conflicts: list[EvidenceConflict] = []
    dedup: dict[str, int] = {}                       # input_hits, unique_chunks, cross_intent_duplicates, ...
    token_count: int = Field(0, ge=0)
    timings_ms: dict[str, float] = {}
    warnings: list[str] = []

    def evidence_ids(self) -> list[str]:
        return [e.evidence_id for e in self.items]
