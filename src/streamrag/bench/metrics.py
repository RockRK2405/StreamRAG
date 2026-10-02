"""Ranking metrics over the returned evidence list.

Each returned evidence item is mapped to a key at the gold level (chunk_id, citation key "Doc §Sec", or
document_id). Metrics are computed over the first k returned *items* (what the caller actually receives):

* Recall@k  = |gold ∩ keys(items[:k])| / |gold|
* Success@k = 1 if any gold key appears in items[:k]
* MRR@k     = 1 / rank of the first item whose key is gold (0 if none within k)
* nDCG@k    = binary-relevance DCG (each gold key credited once, at its first occurrence) / ideal DCG
"""

from __future__ import annotations

import math

from streamrag.models.evidence import Evidence


def item_key(e: Evidence, level: str) -> str:
    if level == "chunk":
        return e.chunk_id
    if level == "document":
        return e.document_id
    return e.citation


def recall_at_k(keys: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(keys[:k])) / len(gold) if gold else 0.0


def success_at_k(keys: list[str], gold: set[str], k: int) -> float:
    return 1.0 if gold & set(keys[:k]) else 0.0


def mrr_at_k(keys: list[str], gold: set[str], k: int) -> float:
    for rank, key in enumerate(keys[:k], start=1):
        if key in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(keys: list[str], gold: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for rank, key in enumerate(keys[:k], start=1):
        if key in gold and key not in seen:
            dcg += 1.0 / math.log2(rank + 1)
            seen.add(key)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
