"""Multi-query retrieval (Phase 5): per-intent queries through the Phase 3 retrieval service, and the streaming
coordinator that turns IntentSet deltas into (delta) retrievals and unified evidence."""

from streamrag.multi_retrieval.retriever import MultiQueryRetriever, MultiRetrievalResult, QueryOutcome

__all__ = ["MultiQueryRetriever", "MultiRetrievalResult", "QueryOutcome"]
