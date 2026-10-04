"""Retrieval request/result contracts. The retrieval layer knows nothing about streaming, voice,
answer generation or UI: these models only describe a corpus search."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator

from streamrag.models.base import Contract


class RetrievalFilters(Contract):
    document_ids: list[str] | None = None
    section_ids: list[str] | None = None   # matched against "<document_id>§<section_id>" or bare section ids
    # Phase 9: document metadata. field -> allowed values; a document without the field (or with value "all")
    # applies to everyone and passes. ``valid_at`` (ISO date): effective_date <= valid_at <= valid_until; with
    # ``valid_to`` the question is about a period: the document's validity window must overlap [valid_at, valid_to].
    # A publication date is never a validity start.
    metadata: dict[str, list[str]] | None = None
    valid_at: str | None = None
    valid_to: str | None = None


class RetrievalOptions(Contract):
    """Per-call overrides; ``None`` means "use configs/default.yaml"."""

    mode: Literal["bm25", "dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1)
    lexical_k: int | None = Field(default=None, ge=1)
    dense_k: int | None = Field(default=None, ge=1)
    rrf_k: int | None = Field(default=None, ge=1)
    rerank: bool | None = None
    rerank_k: int | None = Field(default=None, ge=1)
    dedup: bool | None = None
    filters: RetrievalFilters | None = None


class RetrievalRequest(Contract):
    request_id: str | None = None
    query: str
    options: RetrievalOptions = RetrievalOptions()

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class RetrievalHit(Contract):
    chunk_id: str
    lexical_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalResult(Contract):
    """Telemetry-facing summary of one search (payload of RETRIEVAL_COMPLETED, spec §6.3 #5)."""

    retrieval_id: str
    status: Literal["ok", "partial", "degraded", "empty", "error"]
    latency_ms: dict[str, float]
    n_candidates: int = Field(ge=0)
    results: list[RetrievalHit] = []
