"""Retrieval efficiency, adaptivity, cache and cancellation counters per sample (from the system's own
operation counts - every number is a count of work actually done, never an estimate).

  retrieval_calls     searches executed (one lexical and / or dense search = one call)
  embeddings          query embeddings computed (= dense searches)
  reranker_calls      cross-encoder invocations
  chunks_retrieved    evidence items returned by all searches (sum of k actually used)
  avg_k               mean k per search
  iterations          retrieve-assess rounds (adaptive); 1 per search for fixed systems
  expansions          follow-up searches after the first round (adaptive)
  documents_retrieved distinct documents among the retrieved chunks
  cache_hit           1 if the need was answered from a cache / session reuse without a search
  evidence_reused     evidence items taken from earlier turns of the session (session reuse / Phase 6 reuse)
  llm_calls, prompt_tokens, output_tokens   generation work
"""

from __future__ import annotations

FIELDS = ("retrieval_calls", "embeddings", "lexical_searches", "reranker_calls", "chunks_retrieved", "iterations",
          "expansions", "documents_retrieved", "cache_hit", "evidence_reused", "llm_calls", "prompt_tokens",
          "output_tokens")


def normalize(ops: dict) -> dict:
    out = {k: ops.get(k) for k in FIELDS}
    ks = ops.get("k_values") or []
    out["avg_k"] = (sum(ks) / len(ks)) if ks else None
    return out


def aggregate(rows: list[dict]) -> dict:
    """Totals and per-query means over samples (None-safe)."""
    out = {}
    n = len(rows)
    for k in FIELDS + ("avg_k",):
        vals = [r.get(k) for r in rows if r.get(k) is not None]
        out[f"{k}_total"] = sum(vals) if vals and k != "avg_k" else None
        out[f"{k}_per_query"] = (sum(vals) / n) if vals and n else None
    hits = [r.get("cache_hit") for r in rows if r.get("cache_hit") is not None]
    out["cache_hit_rate"] = (sum(hits) / len(hits)) if hits else None
    return out
