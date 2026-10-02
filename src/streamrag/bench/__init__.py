"""Retrieval benchmark harness (Phase 3). Quality metrics are only meaningful with gold labels authored from
the official corpus; runs on test fixtures are always marked NOT REPORTABLE."""

from streamrag.bench.dataset import load_eval_items
from streamrag.bench.harness import run_benchmark
from streamrag.bench.metrics import mrr_at_k, ndcg_at_k, recall_at_k, success_at_k

__all__ = ["load_eval_items", "mrr_at_k", "ndcg_at_k", "recall_at_k", "run_benchmark", "success_at_k"]
