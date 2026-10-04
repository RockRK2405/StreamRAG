"""Retrieval metrics over the ranked evidence a system hands to its answer stage (section-level keys).

Defined only when the sample has gold evidence (``None`` otherwise - never 0 by default). With ``semantics="any"``
(ambiguous questions: any of several sections answers) recall@k is success@k (1 if any gold key is in the top k).
  Recall@k     |gold ∩ top-k| / |gold|
  Precision@k  |gold ∩ top-k| / k            (k fixed: returning fewer than k items is not rewarded)
  MRR          1 / rank of the first gold key (0 if none in the list)
  nDCG@k       binary-relevance DCG / ideal DCG (each gold key credited once)
"""

from __future__ import annotations

import math

KS = (1, 3, 5, 10)


def _dedup(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(keys))


def recall_at(keys: list[str], gold: list[str], k: int, semantics: str = "all") -> float | None:
    if not gold:
        return None
    top = set(_dedup(keys)[:k])
    if semantics == "any":
        return 1.0 if top & set(gold) else 0.0
    return len(top & set(gold)) / len(set(gold))


def precision_at(keys: list[str], gold: list[str], k: int) -> float | None:
    if not gold:
        return None
    return len(set(_dedup(keys)[:k]) & set(gold)) / k


def mrr(keys: list[str], gold: list[str]) -> float | None:
    if not gold:
        return None
    for r, x in enumerate(_dedup(keys), start=1):
        if x in set(gold):
            return 1.0 / r
    return 0.0


def ndcg_at(keys: list[str], gold: list[str], k: int = 10, semantics: str = "all") -> float | None:
    if not gold:
        return None
    g = set(gold)
    dcg = sum(1.0 / math.log2(r + 1) for r, x in enumerate(_dedup(keys)[:k], start=1) if x in g)
    n_ideal = 1 if semantics == "any" else min(len(g), k)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, n_ideal + 1))
    return min(1.0, dcg / ideal) if ideal else None


def compute(keys: list[str], gold: list[str], semantics: str = "all") -> dict:
    out = {f"recall@{k}": recall_at(keys, gold, k, semantics) for k in KS}
    out.update({f"precision@{k}": precision_at(keys, gold, k) for k in (1, 3, 5)})
    out["mrr"] = mrr(keys, gold)
    out["ndcg@10"] = ndcg_at(keys, gold, 10, semantics)
    return out
