"""Cross-intent evidence fusion and reranking (Phase 5): per-intent EvidenceSets -> UnifiedEvidenceSet."""

from streamrag.fusion.engine import EvidenceFusionEngine, IntentEvidence
from streamrag.fusion.models import EvidenceConflict, EvidenceHit, FusedEvidence, IntentCoverage, UnifiedEvidenceSet

__all__ = ["EvidenceConflict", "EvidenceFusionEngine", "EvidenceHit", "FusedEvidence", "IntentCoverage",
           "IntentEvidence", "UnifiedEvidenceSet"]


def make_fusion_engine(cfg, bundle, service=None, reranker=None) -> EvidenceFusionEngine:
    """Fusion over a loaded index: chunk embeddings for cross-intent dense relevance, the service's query embedder,
    and a cross-encoder when one is loaded (or passed)."""
    from streamrag.retrieval.text import load_stopwords
    embedder = getattr(service, "embedder", None) if service is not None else None
    return EvidenceFusionEngine(
        cfg.fusion, chunk_rows={c.chunk_id: n for n, c in enumerate(bundle.chunks)},
        vectors=bundle.dense.matrix if bundle.dense is not None else None,
        embed_fn=(lambda texts: embedder.embed(texts, "query")) if embedder is not None else None,
        reranker=reranker if reranker is not None else getattr(service, "reranker", None),
        stopwords=load_stopwords(cfg.lexical.stopwords))


__all__.append("make_fusion_engine")
