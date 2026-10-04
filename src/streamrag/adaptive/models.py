"""Adaptive retrieval contracts (Phase 9; docs/retrieval/01-11 Phase 9 series, docs/architecture/13).

Every decision the adaptive layer takes is a record with its inputs and reason (``RetrievalDecision``), so a run can
be explained and replayed. Numbers in these records are measurements (latency, counts) or documented formulas
(coverage, gain); none is authored.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field

from streamrag.models.base import Contract
from streamrag.models.retrieval import RetrievalFilters


class QueryComplexity(str, Enum):
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"
    MULTI_HOP = "MULTI_HOP"


class RetrievalStrategy(str, Enum):
    FAST_VECTOR = "FAST_VECTOR"        # dense only, small k (no lexical search)
    LEXICAL = "LEXICAL"                # BM25 only (exact identifiers; no embedding)
    HYBRID = "HYBRID"                  # BM25 + dense + RRF (the Phase 3 default)
    FILTERED = "FILTERED"              # hybrid restricted by metadata / validity filters
    SEMANTIC = "SEMANTIC"              # dense led: vocabulary mismatch (query words unknown to the index)
    MULTI_HOP = "MULTI_HOP"            # bridge entity -> second search
    ITERATIVE = "ITERATIVE"            # hybrid, requirement-targeted follow-up searches, adaptive k
    CACHE_REUSE = "CACHE_REUSE"        # a validated cached result for the same query / validity signature
    SESSION_REUSE = "SESSION_REUSE"    # evidence already in the session satisfies the requirements


class StopReason(str, Enum):
    SUFFICIENT_EVIDENCE = "SUFFICIENT_EVIDENCE"
    LATENCY_LIMIT = "LATENCY_LIMIT"
    QUERY_LIMIT = "QUERY_LIMIT"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    NO_EXPECTED_GAIN = "NO_EXPECTED_GAIN"
    CONTRADICTION = "CONTRADICTION"
    NO_RESULTS = "NO_RESULTS"
    ERROR = "ERROR"
    USER_CANCELLED = "USER_CANCELLED"


Sufficiency = Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT", "CONTRADICTORY"]


class TemporalSpec(Contract):
    """What point in time a question is about. ``valid_at`` is compared with document validity windows
    (effective_date <= valid_at <= valid_until); a publication date is never used as the validity start."""

    kind: Literal["none", "current", "past", "as_of"] = "none"
    valid_at: str | None = None              # ISO date
    cue: str = ""
    explicit: bool = False                   # a date / year was said (vs. defaulted to the reference date)


class QueryAnalysis(Contract):
    query_id: str
    text: str
    complexity: QueryComplexity
    reasons: list[str] = []                  # rule names that decided the complexity (explainable)
    signals: dict[str, Any] = {}
    terms: list[str] = []                    # analyzed terms of the query
    question_terms: list[str] = []           # question / value-phrase words ("how high" -> high): not content
    exact_ids: list[str] = []                # identifier-like tokens ("AB-123", "XYZ")
    rare_terms: list[str] = []
    oov_terms: list[str] = []                # analyzed terms unknown to the lexical index
    entities: list[str] = []
    constraint_texts: list[str] = []
    metadata_filters: dict[str, list[str]] = {}
    temporal: TemporalSpec = TemporalSpec()
    value_kind: str | None = None            # amount / duration / count / time / age / ... (claim lexicon)
    n_intents: int = Field(1, ge=1)
    context_dependent: bool = False
    ambiguous: bool = False
    comparison: bool = False
    bridge_terms: list[str] = []             # rare terms that never co-occur with the asked aspect (multi-hop)


class Expansion(Contract):
    term: str
    source: Literal["synonym", "acronym", "alias"]
    for_term: str
    reason: str


class RewrittenQuery(Contract):
    """Rewrites never change the need: every content term of ``contextual`` is kept in the retrieval queries
    (a test checks the invariant); expansions are additive and bounded."""

    intent_id: str | None = None
    original: str
    normalized: str
    contextual: str
    expanded: str
    lexical_query: str
    dense_query: str
    expansions: list[Expansion] = []
    dropped_entities: list[str] = []         # entity terms removed by a correction (stale)


class EvidenceRequirement(Contract):
    """What evidence a claim of the answer needs (claim-driven retrieval, docs/retrieval/05)."""

    requirement_id: str
    claim_slot: str                          # human-readable: "amount for: licence fee"
    query_text: str = ""                     # surface text a targeted search for this requirement uses
    kind: Literal["value", "fact", "condition", "bridge", "comparison_item", "definition"]
    terms: list[str] = []                    # analyzed terms the supporting text must contain (>= coverage share)
    key_terms: list[str] = []                # must all occur (identifiers, the multi-hop entity / bridge target)
    alternatives: dict[str, list[str]] = {}  # term -> accepted equivalents (validated acronym / synonym expansion)
    value_kind: str | None = None
    constraint_terms: list[str] = []         # a constraint the evidence must satisfy (terms or metadata)
    metadata: dict[str, list[str]] = {}
    valid_at: str | None = None
    valid_to: str | None = None
    min_sources: int = Field(1, ge=1)
    status: Literal["UNMET", "MET", "CONFLICT"] = "UNMET"
    evidence_ids: list[str] = []
    values: dict[str, list[str]] = {}        # evidence id -> values of the asked kind (conflict detection)
    resolution: str | None = None            # how a conflict was resolved (temporal / version / authority)


class SufficiencyAssessment(Contract):
    status: Sufficiency
    coverage: float = Field(ge=0, le=1)      # met requirements / requirements
    quality: float = Field(ge=0, le=1)       # mean best term coverage over requirements
    met: list[str] = []
    unmet: list[str] = []
    conflicts: list[str] = []
    resolved_conflicts: list[str] = []
    unattainable: list[str] = []             # unmet requirements whose terms the index cannot match (no gain possible)
    reasons: list[str] = []


class RetrievalBudgetState(Contract):
    max_queries: int
    max_results: int
    max_iterations: int
    max_latency_ms: float
    max_parallel_tasks: int
    max_hops: int
    queries_used: int = 0
    results_used: int = 0
    iterations_used: int = 0
    latency_used_ms: float = 0.0

    def remaining(self) -> dict[str, float]:
        return {"queries": self.max_queries - self.queries_used, "results": self.max_results - self.results_used,
                "iterations": self.max_iterations - self.iterations_used,
                "latency_ms": round(self.max_latency_ms - self.latency_used_ms, 3)}


class RetrievalPlan(Contract):
    plan_id: str
    query_id: str
    intent_id: str | None = None
    strategy: RetrievalStrategy
    strategy_reason: str
    retrievers: list[Literal["bm25", "dense"]] = []
    top_k: int = Field(ge=1)
    max_top_k: int = Field(ge=1)
    filters: RetrievalFilters | None = None
    reranking: bool = False
    rerank_reason: str = ""
    max_iterations: int = Field(ge=1)
    latency_budget_ms: float = Field(gt=0)
    evidence_threshold: float = Field(ge=0, le=1)   # share of requirements that must be MET
    stop_condition: str
    requirements: list[str] = []


class RetrievalHop(Contract):
    hop_id: str
    parent_hop_id: str | None = None
    query: str
    purpose: str
    evidence_ids: list[str] = []
    status: Literal["PLANNED", "COMPLETED", "FAILED", "SKIPPED"] = "PLANNED"
    bridge: str | None = None                # the bridging entity / document that produced this hop


class RetrievalDecision(Contract):
    decision_id: str
    query_id: str
    strategy: RetrievalStrategy
    action: str                              # select / search / expand_k / requirement_query / hop / ...
    reason: str
    inputs: dict[str, Any] = {}
    expected_gain: float | None = None
    actual_gain: float | None = None
    latency_cost_ms: float = Field(0.0, ge=0)
    result: dict[str, Any] = {}
    timestamp_ms: float = Field(0.0, ge=0)   # since the start of the adaptive run


class RetrievalState(Contract):
    current_strategy: RetrievalStrategy
    iteration: int = 0
    queries_executed: int = 0
    evidence_count: int = 0
    evidence_quality: float = 0.0
    coverage: float = 0.0
    latency_spent_ms: float = 0.0
    budget_remaining: dict[str, float] = {}
    stop_reason: StopReason | None = None


class OperationCounts(Contract):
    """Measured work of one adaptive run (cost / operation counts, brief §67)."""

    searches: int = 0                        # retrieval calls (one lexical and / or dense search = one call)
    lexical_searches: int = 0
    dense_searches: int = 0                  # = query embeddings computed
    reranker_calls: int = 0
    reranked_candidates: int = 0
    chunks_returned: int = 0                 # evidence items returned by all searches (sum of k actually used)
    llm_calls: int = 0                       # the adaptive layer never calls an LLM (kept for the comparison table)
    cache_hits: int = 0
    cache_misses: int = 0
    session_reuse: int = 0
    cancelled: int = 0
