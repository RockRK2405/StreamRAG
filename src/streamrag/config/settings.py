"""Typed, externalized configuration.

All runtime behavior is driven by ``configs/default.yaml`` (plus optional overrides). Unknown keys are
rejected so that a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from streamrag.errors import ConfigError


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Cfg):
    corpus: Path = Path("./corpus")
    index_root: Path = Path("./indexes")
    models_dir: Path = Path("./models")
    model_registry: Path = Path("./configs/models.yaml")
    runs_dir: Path = Path("./runs")


class CorpusConfig(_Cfg):
    include_extensions: list[str] = [".txt", ".md", ".pdf"]
    fixture_marker: str = "TEST_FIXTURE_ONLY"
    on_invalid_document: Literal["skip", "error"] = "skip"
    doc_id_strategy: Literal["native_or_stem", "native_or_ordinal"] = "native_or_stem"
    native_doc_id_pattern: str = r"^(?i:doc)[ _-]?(\d+)"
    # Phase 9: front-matter fields kept as document metadata (whitelist; values sanitised, see corpus/metadata.py).
    # Metadata is corpus content - untrusted: it can narrow retrieval (filters, validity dates) but never steer
    # routing, budgets or prompts.
    metadata_fields: list[str] = ["version", "effective_date", "published_date", "valid_until", "status",
                                  "supersedes", "doc_type", "applicant_type", "country", "region", "year", "product",
                                  "language"]


class NormalizationConfig(_Cfg):
    unicode_form: Literal["NFC", "NFKC"] = "NFKC"
    dehyphenate: bool = True
    join_wrapped_lines: bool = True
    strip_page_numbers: bool = True
    strip_repeated_headers_footers: bool = True
    header_footer_min_pages: int = Field(3, ge=2)
    header_footer_min_fraction: float = Field(0.6, gt=0, le=1)


class SectionsConfig(_Cfg):
    markdown_headings: bool = True
    numbered_headings: bool = True
    caps_headings: bool = True
    max_heading_words: int = Field(12, ge=1)
    pdf_page_sections_when_no_headings: bool = True


class ChunkingConfig(_Cfg):
    strategy: Literal["section", "paragraph", "token"] = "paragraph"
    target_tokens: int = Field(180, ge=10)
    max_tokens: int = Field(300, ge=10)
    overlap_tokens: int = Field(40, ge=0)
    min_tokens: int = Field(25, ge=0)
    keep_lead_in_with_list: bool = True

    @model_validator(mode="after")
    def _check(self) -> "ChunkingConfig":
        if self.target_tokens > self.max_tokens:
            raise ValueError("chunking.target_tokens must be <= max_tokens")
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("chunking.overlap_tokens must be < target_tokens")
        return self


class IdsConfig(_Cfg):
    chunk_id_template: str = "{document_id}§{section_id}#{part}"
    citation_template: str = "{document_id} §{section_id}"


class IndexTextConfig(_Cfg):
    template: str = "{header}: {text}"


class LexicalConfig(_Cfg):
    k1: float = Field(1.5, gt=0)
    b: float = Field(0.75, ge=0, le=1)
    stemming: Literal["snowball", "none"] = "snowball"
    stopwords: str = "stopwords_en.txt"
    normalize_number_words: bool = True


class DenseConfig(_Cfg):
    embedder: str = "bge-small-en-v1.5"
    batch_size: int = Field(8, ge=1)
    intra_op_threads: int = Field(0, ge=0)


class RetrievalConfig(_Cfg):
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(10, ge=1)
    lexical_k: int = Field(50, ge=1)
    dense_k: int = Field(50, ge=1)
    rrf_k: int = Field(60, ge=1)
    dense_min_similarity: float | None = None
    on_dense_failure: Literal["degrade", "error"] = "degrade"
    dense_timeout_ms: int = Field(1000, ge=1)


class DedupConfig(_Cfg):
    enabled: bool = True
    near_duplicates: bool = True
    near_dup_cosine: float = Field(0.97, gt=0, le=1)
    near_dup_jaccard: float = Field(0.9, gt=0, le=1)
    overlap_merge_ratio: float = Field(0.8, gt=0, le=1)


class RerankConfig(_Cfg):
    enabled: bool = False
    model: str = "ms-marco-minilm-l6-v2"
    rerank_k: int = Field(20, ge=1)
    timeout_ms: int = Field(600, ge=1)
    batch_size: int = Field(32, ge=1)


class ControllerConfig(_Cfg):
    strategy: Literal["rules", "end_only", "every_chunk"] = "rules"
    act_classifier: Literal["rules", "prototype"] = "rules"
    lexicon: Path = Path("./configs/controller_lexicon.yaml")
    min_content_tokens: int = Field(2, ge=1)
    anchor_idf_floor: float = Field(1.0, ge=0)
    strong_anchor_strength: float = Field(0.8, ge=0, le=1)
    min_stability: float = Field(0.65, ge=0, le=1)
    min_worthiness: float = Field(0.5, ge=0, le=1)
    act_suppress_confidence: float = Field(0.8, ge=0, le=1)
    stability_quiet_ms: int = Field(600, ge=0)
    novelty_threshold: float = Field(0.25, ge=0, le=1)
    retrieval_cooldown_ms: int = Field(400, ge=0)
    max_retrievals_per_utterance: int = Field(4, ge=1)
    reserve_final_retrieval: bool = True
    allow_parallel_retrieval: bool = True
    max_concurrent_retrievals: int = Field(2, ge=1)
    # queued_only: a superseded query is dropped while queued, a running one completes and is marked stale (Phase 4);
    # cooperative: a running one is also signalled to stop at its next checkpoint (Phase 8 runtime, docs/runtime/05)
    cancel_superseded: Literal["never", "queued_only", "cooperative"] = "queued_only"
    endpoint_timeout_ms: int = Field(3000, ge=1)


class StreamingConfig(_Cfg):
    mode: Literal["virtual", "realtime"] = "virtual"
    speed: float = Field(1.0, gt=0)
    sim_retrieval_latency_ms: float = Field(10.0, ge=0)
    retrieval_timeout_ms: int = Field(2000, ge=1)
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "hybrid"
    top_k: int = Field(10, ge=1)
    rerank: bool = False
    words_per_second: float = Field(2.6, gt=0)
    end_gap_ms: int = Field(500, ge=0)


class MultiIntentConfig(_Cfg):
    """Phase 5: intent decomposition + per-intent retrieval (docs/multi_intent/). ``enabled: false`` keeps the
    Phase 4 single-active-query behaviour of the streaming session."""

    enabled: bool = False
    lexicon: Path = Path("./configs/intent_lexicon.yaml")
    max_intents: int = Field(4, ge=1)                    # active intents per utterance (excess ranked, dropped w/ reason)
    max_queries_per_utterance: int = Field(10, ge=1)
    max_queries_per_intent: int = Field(3, ge=1)         # provisional + refined + final
    max_candidates_per_intent: int = Field(5, ge=1)      # top_k of each intent's retrieval
    max_concurrent_retrievals: int = Field(3, ge=1)
    dispatch: Literal["parallel", "sequential", "batched"] = "parallel"
    intent_cooldown_ms: int = Field(400, ge=0)           # min stream-time gap between two queries of the same intent
    duplicate_jaccard: float = Field(0.8, gt=0, le=1)    # intra-set duplicate intents (content-term Jaccard)
    match_jaccard: float = Field(0.5, gt=0, le=1)        # version tracking: same intent across updates
    refine_containment: float = Field(0.6, gt=0, le=1)   # query relation 'refines' (as Phase 4)
    carryover: bool = True                               # anaphora / follow-up context inheritance
    llm_check: Literal["off", "gated"] = "off"           # optional structured LLM check (no backend configured)
    llm_check_wait_ms: int = Field(300, ge=0)


class FusionConfig(_Cfg):
    """Phase 5: cross-intent evidence fusion and reranking (docs/multi_intent/06-07)."""

    strategy: Literal["concat", "global_score", "rrf", "intent_aware"] = "intent_aware"
    top_k: int = Field(8, ge=1)                          # unified evidence budget (items)
    min_per_intent: int = Field(2, ge=0)                 # intent-aware coverage floor
    section_cap_per_intent: int = Field(2, ge=1)
    rrf_k: int = Field(60, ge=1)
    rerank: Literal["none", "intent_ce", "cross_intent_dense", "cross_intent_ce"] = "none"
    rerank_k: int = Field(20, ge=1)                      # candidates per intent sent to a cross-encoder
    conflict_check: bool = True


class SessionConfig(_Cfg):
    """Phase 6: session memory, late-arriving details, delta retrieval, claim/answer state (docs/session/).
    ``enabled`` requires ``multi_intent.enabled``."""

    enabled: bool = False
    transcript_window: int = Field(6, ge=1)              # utterances kept verbatim in transcript memory
    redact_pii: bool = True                              # e-mails, phone/card-like numbers, tokens never stored
    delta_scope: Literal["corpus", "session_docs_first"] = "corpus"
    cache: bool = True                                   # semantic retrieval cache (term-set key)
    frame_overlap_min: int = Field(1, ge=0)              # shared topic terms that keep a new need in the frame
    claims_per_intent: int = Field(4, ge=1)              # extractive claims registered per intent version
    claim_min_relevance: float = Field(0.2, ge=0, le=1)  # min share of the intent's terms in a claim sentence


class GenerationConfig(_Cfg):
    """Phase 7: grounded answer generation, claim verification, citations (docs/answer/, ADR-017).
    ``enabled`` requires ``session.enabled``. Backends follow ADR-007: an LLM backend when reachable, the extractive
    generator otherwise (and as the fallback after any LLM failure)."""

    enabled: bool = False
    backend: Literal["auto", "ollama", "extractive"] = "auto"   # auto: ollama if reachable, else extractive
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    temperature: float = Field(0.0, ge=0)
    seed: int = 7
    num_ctx: int = Field(8192, ge=512)
    max_output_tokens: int = Field(1024, ge=64)
    timeout_s: float = Field(120.0, gt=0)
    max_structured_retries: int = Field(1, ge=0)          # re-ask once when the JSON fails the schema
    detail: Literal["concise", "detailed"] = "detailed"   # concise: critical + important claims only
    max_claims_per_section: int = Field(6, ge=1)
    # verification (docs/answer/04)
    verifier: Literal["nli", "rules"] = "nli"             # nli: entailment cross-encoder (L3) + rules (L0-L2)
    nli_model: str = "nli-deberta-v3-xsmall"
    # unsupported-claim policy (docs/answer/07)
    validation_mode: Literal["strict", "relaxed"] = "strict"
    repair: bool = True
    llm_repair: bool = False                              # +1 LLM call per unsupported claim (ablation)
    max_validation_retrievals: int = Field(1, ge=0)       # retrieval fallback budget per answer version
    max_answer_revision_attempts: int = Field(2, ge=0)
    draft_mode: Literal["off", "extractive"] = "extractive"   # streamed drafts before the turn ends


class RuntimeTimeouts(_Cfg):
    """Per-task deadlines (ms). A child task gets min(its own timeout, the remaining turn budget minus the time
    reserved for the stages after it) - deadline propagation, docs/runtime/06."""
    lexical: int = Field(2000, ge=1)
    dense: int = Field(3000, ge=1)
    assemble: int = Field(2000, ge=1)
    generation: int = Field(60000, ge=1)
    draft: int = Field(10000, ge=1)
    validation_retrieval: int = Field(3000, ge=1)


class RuntimeQueues(_Cfg):
    """Capacities of the bounded queues (number of entries)."""
    input: int = Field(64, ge=1)          # per session: transcript deltas / control inputs awaiting the session lane
    retrieval: int = Field(64, ge=1)      # pending retrieval subtasks (all sessions)
    llm: int = Field(8, ge=1)             # pending generation tasks
    cpu: int = Field(32, ge=1)            # pending assembly / draft tasks
    output: int = Field(4096, ge=16)      # per subscriber: user-visible events not yet consumed


class RuntimeRetry(_Cfg):
    max_retries: int = Field(2, ge=0)                 # attempts after the first, transient failures only
    initial_delay_ms: float = Field(50.0, ge=0)
    max_delay_ms: float = Field(1000.0, ge=0)
    multiplier: float = Field(2.0, ge=1.0)
    jitter: float = Field(0.2, ge=0, le=1)            # +- share of the delay, drawn from a seeded RNG (replayable)
    seed: int = 7


class RuntimePriorities(_Cfg):
    """Lower value = served first. Aging: a pending task gains one level per ``aging_ms`` waited (no starvation)."""
    final_answer: int = 0                 # CRITICAL: the current turn's validated answer
    retrieval_final: int = 1              # HIGH: retrieval for a finalized utterance / a need without evidence yet
    validation_retrieval: int = 1
    retrieval_provisional: int = 2        # MEDIUM: early retrieval while the user speaks
    draft: int = 2
    analytics: int = 3                    # LOW: optional enrichment
    aging_ms: float = Field(500.0, gt=0)


class RuntimeBudget(_Cfg):
    """Turn latency budget used for deadline propagation - a configuration value, not a measured or claimed target.
    The reserves are the dev-machine Phase 7 p95 stage times (PHASE_7 report §19), so retrieval cannot consume the
    time generation and validation still need."""
    turn_ms: float = Field(30000.0, gt=0)
    generation_reserve_ms: float = Field(4200.0, ge=0)
    validation_reserve_ms: float = Field(400.0, ge=0)


class RuntimeSimLatency(_Cfg):
    """Virtual-clock execution model (deterministic replay / orchestration tests only; never reported as measured
    latency). Values are ms of virtual time per task."""
    lexical: float = 20.0
    dense: float = 40.0
    assemble: float = 2.0
    generation: float = 2000.0
    draft: float = 50.0
    validation_retrieval: float = 30.0


class RuntimeConfig(_Cfg):
    """Phase 8 streaming runtime (docs/runtime/, ADR-018). Team engineering defaults, not official thresholds."""
    max_concurrent_sessions: int = Field(8, ge=1)
    max_concurrent_retrievals: int = Field(4, ge=1)   # retrieval worker threads (lexical / dense subtasks)
    max_concurrent_llm_calls: int = Field(1, ge=1)    # the local Ollama server runs one request at a time
    max_concurrent_cpu: int = Field(2, ge=1)          # assembly, drafts (NLI verification)
    queues: RuntimeQueues = RuntimeQueues()
    timeouts_ms: RuntimeTimeouts = RuntimeTimeouts()
    retry: RuntimeRetry = RuntimeRetry()
    priorities: RuntimePriorities = RuntimePriorities()
    budget: RuntimeBudget = RuntimeBudget()
    sim_latency_ms: RuntimeSimLatency = RuntimeSimLatency()
    split_retrieval: bool = True                      # lexical and dense as concurrent subtasks (partial results)
    coalescing_window_ms: float = Field(0.0, ge=0)    # 0: coalesce only what is already queued (adaptive batching)
    cancel_running: bool = True                       # superseded running work is signalled (cooperative)
    cancel_on_correction: bool = True                 # a correction cancels the previous turn's in-flight answer
    output_max_hold_ms: float = Field(2000.0, ge=0)   # ordered output buffer: longest wait for a missing sequence
    shutdown_grace_ms: float = Field(5000.0, ge=0)
    max_chunk_chars: int = Field(4000, ge=1)          # input validation (resource exhaustion)
    max_inputs_per_session: int = Field(20000, ge=1)


class AdaptiveBudget(_Cfg):
    """Per-need retrieval budget (docs/retrieval/11). Trusted configuration only: corpus text never changes it."""
    max_queries: int = Field(5, ge=1)                 # searches (lexical and / or dense) per need
    max_results: int = Field(40, ge=1)                # evidence items examined per need (sum of k over searches)
    max_iterations: int = Field(3, ge=1)              # retrieve -> assess rounds
    max_latency_ms: float = Field(1500.0, gt=0)       # wall time of the adaptive loop (search + assessment)
    max_parallel_tasks: int = Field(2, ge=1)          # concurrent searches of one need (lexical || dense)
    max_hops: int = Field(2, ge=1)                    # multi-hop depth (hop 0 = the question itself)


class AdaptiveRetrievalConfig(_Cfg):
    """Phase 9: adaptive retrieval intelligence (docs/retrieval/01-11 Phase 9 series, docs/architecture/13).
    Off by default: the Phase 3-8 fixed policy stays the default until a real corpus validates the adaptive one.
    Defaults are team engineering values calibrated on the dev fixture suites (research/phase9) - NOT REPORTABLE."""
    enabled: bool = False
    lexicon: Path = Path("configs/retrieval_lexicon.yaml")
    initial_k: dict[str, int] = {"SIMPLE": 3, "MODERATE": 5, "COMPLEX": 5, "MULTI_HOP": 5}
    k_schedule: list[int] = [5, 10, 20]               # adaptive top-k: next k after an insufficient round
    final_k: int = Field(8, ge=1)                     # evidence items handed to the claim / answer stages
    max_per_document: int = Field(3, ge=1)            # source diversity cap in the final evidence
    simple_strategy: Literal["HYBRID", "LEXICAL", "FAST_VECTOR"] = "LEXICAL"  # research/phase9/calibrate_routing.py
    exact_id_strategy: Literal["HYBRID", "LEXICAL"] = "LEXICAL"
    semantic_oov_ratio: float = Field(0.5, ge=0, le=1)   # >= this share of query terms unknown to BM25 -> SEMANTIC
    rare_idf: float = Field(1.5, ge=0)                # analyzed term with IDF >= this is "rare" (entity / id signal)
    requirement_coverage: float = Field(0.6, ge=0, le=1)   # share of a requirement's terms evidence must contain
    min_gain: float = Field(0.05, ge=0)               # marginal value below which another round is not worth it
    rerank: Literal["never", "policy", "always"] = "never"   # research/phase9 reranker evaluation decides
    rerank_max_candidates: int = Field(20, ge=1)
    expansion: bool = True                            # bounded alias / acronym / synonym expansion (lexical query)
    max_expansions: int = Field(3, ge=0)
    contradiction_retrieval: bool = True              # one targeted search to resolve a value conflict
    claim_driven: bool = True                         # requirements per claim slot (else: one requirement per need)
    cache: bool = True                                # adaptive query cache (validity-signature checked)
    cache_max_entries: int = Field(256, ge=1)
    session_reuse: bool = True                        # assess session evidence before searching
    filter_fields: list[str] = ["applicant_type", "country", "region", "product", "language"]
    reference_date: str | None = None                 # ISO date for "current" validity; None = today
    tight_latency_ms: float = Field(150.0, ge=0)      # remaining budget below this -> fast path, one round, no rerank
    # ablation switches (research/phase9/ablation): every component can be turned off independently
    routing: bool = True                              # off: always HYBRID (no strategy selection, no filters)
    adaptive_k: bool = True                           # off: k = k_schedule[0] for every need, no k expansion
    iterative: bool = True                            # off: one retrieval round
    multi_hop: bool = True                            # off: no hops (MULTI_HOP needs are routed to ITERATIVE)
    temporal: bool = True                             # off: no validity filtering / temporal conflict resolution
    authority: dict[str, dict[str, float]] = {
        "status": {"current": 1.0, "active": 1.0, "draft": 0.4, "superseded": 0.2, "archived": 0.2, "expired": 0.2},
        "doc_type": {"policy": 1.0, "annex": 0.9, "catalogue": 0.8, "notice": 0.7, "bulletin": 0.6}}
    budget: AdaptiveBudget = AdaptiveBudget()


class TelemetryConfig(_Cfg):
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"


class StreamRagConfig(_Cfg):
    paths: PathsConfig = PathsConfig()
    corpus: CorpusConfig = CorpusConfig()
    normalization: NormalizationConfig = NormalizationConfig()
    sections: SectionsConfig = SectionsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    ids: IdsConfig = IdsConfig()
    index_text: IndexTextConfig = IndexTextConfig()
    lexical: LexicalConfig = LexicalConfig()
    dense: DenseConfig = DenseConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    dedup: DedupConfig = DedupConfig()
    rerank: RerankConfig = RerankConfig()
    controller: ControllerConfig = ControllerConfig()
    streaming: StreamingConfig = StreamingConfig()
    multi_intent: MultiIntentConfig = MultiIntentConfig()
    fusion: FusionConfig = FusionConfig()
    session: SessionConfig = SessionConfig()
    generation: GenerationConfig = GenerationConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    adaptive_retrieval: AdaptiveRetrievalConfig = AdaptiveRetrievalConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

    def resolve_paths(self, base: Path) -> "StreamRagConfig":
        """Return a copy whose relative paths are resolved against ``base`` (the repo root)."""
        p = self.paths
        resolved = PathsConfig(**{k: (v if Path(v).is_absolute() else (base / v)).resolve()
                                  for k, v in p.model_dump().items()})
        lex = self.controller.lexicon
        controller = self.controller.model_copy(update={"lexicon": (lex if lex.is_absolute() else base / lex).resolve()})
        ilex = self.multi_intent.lexicon
        multi = self.multi_intent.model_copy(update={"lexicon": (ilex if ilex.is_absolute() else base / ilex).resolve()})
        alex = self.adaptive_retrieval.lexicon
        adaptive = self.adaptive_retrieval.model_copy(
            update={"lexicon": (alex if alex.is_absolute() else base / alex).resolve()})
        return self.model_copy(update={"paths": resolved, "controller": controller, "multi_intent": multi,
                                       "adaptive_retrieval": adaptive})


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(cfg: StreamRagConfig) -> str:
    """Hash of every behavior-relevant setting (paths excluded: content hashes cover the data)."""
    data = cfg.model_dump(mode="json", exclude={"paths", "telemetry"})
    return hashlib.sha256(_canonical(data).encode()).hexdigest()


# Settings that change the *index artifacts*. Query-time settings (retrieval/dedup/rerank) are excluded,
# so changing top_k does not force a rebuild.
_INDEX_SECTIONS = ("corpus", "normalization", "sections", "chunking", "ids", "index_text", "lexical")


def index_config_hash(cfg: StreamRagConfig) -> str:
    data = {k: cfg.model_dump(mode="json")[k] for k in _INDEX_SECTIONS}
    data["dense.embedder"] = cfg.dense.embedder
    data["dense.batch_size"] = cfg.dense.batch_size   # batch shape can change ONNX outputs at ~1e-7
    return hashlib.sha256(_canonical(data).encode()).hexdigest()


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
        if not isinstance(cur, dict):
            raise ConfigError(f"override path '{dotted}' collides with a non-mapping value")
    cur[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None,
                base_dir: str | Path | None = None) -> StreamRagConfig:
    """Load YAML config, apply dotted-key overrides and STREAMRAG_CORPUS, validate, resolve paths.

    ``base_dir`` (default: the YAML file's parent's parent, i.e. the repo root for configs/default.yaml)
    anchors relative paths so results do not depend on the current working directory.
    """
    raw: dict[str, Any] = {}
    cfg_path = Path(path) if path else None
    if cfg_path is not None:
        if not cfg_path.exists():
            raise ConfigError(f"config file not found: {cfg_path}")
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config root must be a mapping: {cfg_path}")
        raw = loaded
    env_corpus = os.environ.get("STREAMRAG_CORPUS")
    if env_corpus:
        _set_dotted(raw, "paths.corpus", env_corpus)
    for key, value in (overrides or {}).items():
        _set_dotted(raw, key, value)
    try:
        cfg = StreamRagConfig.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError -> explicit config error
        raise ConfigError(f"invalid configuration: {exc}") from exc
    if base_dir is None:
        base_dir = cfg_path.resolve().parent.parent if cfg_path else Path.cwd()
    return cfg.resolve_paths(Path(base_dir))
