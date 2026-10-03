"""Evidence contracts (spec §11, refined for the Phase 3 retrieval layer).

``Evidence`` is a corpus chunk *as retrieved for one query*, with full score provenance. It is always
traceable: chunk_id -> (document_id, section_id) -> source_path + char span (+ page span if any).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from streamrag.models.base import Contract

RetrievalMethod = Literal["bm25", "dense", "hybrid_rrf", "hybrid_rrf_rerank", "dense_rerank", "bm25_rerank"]
MetaValue = str | int | float | bool | None


class Evidence(Contract):
    evidence_id: str                    # == chunk_id (one chunk is one evidence object)
    document_id: str
    section_id: str
    chunk_id: str
    citation: str                       # rendered key, e.g. "Doc_07 §3"
    section_title: str | None = None
    section_path: list[str] = []
    text: str
    source_path: str                    # relative to the corpus root
    char_start: int = Field(ge=0)       # span in the normalized document text
    char_end: int = Field(ge=0)
    page_start: int | None = None
    page_end: int | None = None
    rank: int = Field(ge=1)             # final 1-based rank
    score: float                        # the score that produced ``rank`` (see retrieval_method)
    retrieval_method: RetrievalMethod
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    alternates: list[str] = []          # near-duplicate chunk_ids collapsed into this one
    overlaps_with: list[str] = []       # overlapping (not merged) chunk_ids in the same result
    metadata: dict[str, MetaValue] = {}


class RetrievalTrace(Contract):
    """How an EvidenceSet was produced: request echo, status, timings, warnings."""

    request_id: str
    mode: Literal["bm25", "dense", "hybrid"]
    status: Literal["ok", "partial", "degraded", "empty"]
    rerank_applied: bool = False
    candidates_lexical: int = 0
    candidates_dense: int = 0
    candidates_fused: int = 0
    dedup_removed: int = 0
    timings_ms: dict[str, float] = {}
    warnings: list[str] = []
    index_version: str
    corpus_hash: str
    index_config_hash: str


class EvidenceSet(Contract):
    evidence_set_id: str
    query: str
    items: list[Evidence] = []
    per_intent: dict[str, list[str]] | None = None   # filled by later phases (multi-intent fusion)
    token_count: int = Field(0, ge=0)
    trace: RetrievalTrace

    def chunk_ids(self) -> list[str]:
        return [e.chunk_id for e in self.items]
