"""Shared, expensive grounding resources (built once per index / stack): the index catalog for citations, the
entailment model and the LLM backend. Pipelines and the streaming coordinator create their own engines from them."""

from __future__ import annotations

from dataclasses import dataclass, replace

from streamrag.citations.mapper import ChunkCatalog


@dataclass
class GroundingResources:
    catalog: ChunkCatalog
    nli: object | None
    backend: object | None
    terms_fn: object

    @classmethod
    def build(cls, cfg, bundle, terms_fn, backend="auto") -> "GroundingResources":
        from streamrag.claims.nli import NliModel
        from streamrag.generation.llm import check_loopback, make_backend
        g = cfg.generation
        check_loopback(g.ollama_url)
        nli = NliModel.load(cfg.paths.models_dir, g.nli_model) if g.verifier == "nli" else None
        llm = make_backend(g) if backend == "auto" else backend
        return cls(ChunkCatalog.from_bundle(bundle), nli, llm, terms_fn)

    def with_backend(self, backend) -> "GroundingResources":
        return replace(self, backend=backend)

    def engine(self, cfg, emit=None):
        from streamrag.answer_state.engine import GroundedAnswerEngine
        return GroundedAnswerEngine(cfg.generation, self.terms_fn, self.catalog, self.nli, self.backend, emit)
