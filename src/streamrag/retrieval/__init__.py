"""Retrieval foundation: BM25 + dense + RRF + dedup + optional rerank -> EvidenceSet.

Corpus isolation: this package searches only an index built from the configured corpus directory. It imports
no networking code (enforced by tests/test_isolation.py).
"""

from streamrag.retrieval.service import RetrievalService
from streamrag.retrieval.store import IndexBundle, build_index, load_index, resolve_index

__all__ = ["IndexBundle", "RetrievalService", "build_index", "load_index", "resolve_index"]
