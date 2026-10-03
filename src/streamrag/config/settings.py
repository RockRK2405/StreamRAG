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
    cancel_superseded: Literal["never", "queued_only"] = "queued_only"
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
        return self.model_copy(update={"paths": resolved, "controller": controller, "multi_intent": multi})


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
