"""Citation metrics (claims with their cited evidence; entailment judged by the evaluation instrument).

  citation_precision    citations whose evidence supports (entails) the claim they are attached to / citations
  citation_recall       factual claims with >= 1 supporting citation / factual claims
  citation_completeness gold sections of the expected claims that the answer cites / those sections
  citation_entailment   cited claims whose citations jointly support them / cited claims
  source_validity       citations that resolve to an evidence item the system actually handed to its answer stage /
                        citations (invalid labels, unknown chunks count as invalid)
  position_correct      supporting citations attached to the sentence they support / supporting citations (a
                        citation that supports only a *different* sentence of the answer is misplaced)
"""

from __future__ import annotations


def compute(claims: list[dict], expected_claims: list) -> dict:
    """claims: [{"factual": bool, "citations": [{"key", "valid", "supports_own", "supports_other"}],
    "cited_support": bool}]"""
    cits = [c for cl in claims for c in cl["citations"]]
    factual = [cl for cl in claims if cl["factual"]]
    cited_claims = [cl for cl in claims if cl["citations"]]
    gold_secs = {c.citation for c in expected_claims if c.citation}
    cited_keys = {c["key"] for c in cits if c["valid"]}
    sup = [c for c in cits if c["supports_own"] or c["supports_other"]]
    return {
        "citations": len(cits),
        "citation_precision": (sum(c["supports_own"] for c in cits) / len(cits)) if cits else None,
        "citation_recall": (sum(1 for cl in factual if any(c["supports_own"] for c in cl["citations"]))
                            / len(factual)) if factual else None,
        "citation_completeness": (len(gold_secs & cited_keys) / len(gold_secs)) if gold_secs else None,
        "citation_entailment": (sum(1 for cl in cited_claims if cl["cited_support"]) / len(cited_claims))
        if cited_claims else None,
        "source_validity": (sum(c["valid"] for c in cits) / len(cits)) if cits else None,
        "position_correct": (sum(c["supports_own"] for c in sup) / len(sup)) if sup else None,
    }
