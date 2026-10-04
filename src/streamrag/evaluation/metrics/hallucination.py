"""Hallucination-related metrics - terms kept apart on purpose (docs/evaluation/metrics.md):

  unsupported_claim_rate   claims the evidence does not support (verifier-judged). Not necessarily false: it may be
                           true but unevidenced.
  hallucinated_claim_rate  claims containing a value (number, amount, time, date) that occurs in no evidence item
                           handed to the answer stage and not in the question: content the system invented
                           (model-free check).
  citationless_fact_rate   factual claims (state a value or >= 3 content words, not an abstention) without a citation
  grounding_failure        1 if the answer has >= 1 factual claim that is unsupported or uncited
Abstentions ("the documents do not say ...") are not claims for these rates; wrong answers that *are* supported by
(stale) evidence are generation / evidence errors, not hallucinations.
"""

from __future__ import annotations

import re

VALUE = re.compile(r"\d+(?:[.,:]\d+)?")


def values(text: str) -> set[str]:
    return {v.replace(",", ".") for v in VALUE.findall(text)}


def compute(claims: list[dict], evidence_text: str, question: str) -> dict:
    """claims: [{"text", "factual", "verdict", "cited": bool}]"""
    fact = [c for c in claims if c["factual"]]
    allowed = values(evidence_text) | values(question)
    halluc = [c for c in fact if values(c["text"]) - allowed]
    return {
        "factual_claims": len(fact),
        "unsupported_claim_rate": (sum(c["verdict"] != "SUPPORTED" for c in fact) / len(fact)) if fact else None,
        "hallucinated_claim_rate": (len(halluc) / len(fact)) if fact else None,
        "citationless_fact_rate": (sum(not c["cited"] for c in fact) / len(fact)) if fact else None,
        "grounding_failure": (float(any(c["verdict"] != "SUPPORTED" or not c["cited"] for c in fact))) if fact
        else None,
    }
