"""Evidence metrics: the evidence set a system hands to its answer stage, against the sample's labels.

  evidence_recall          |gold sections ∩ evidence| / |gold|                                  (gold needed)
  evidence_precision       |gold sections ∩ evidence| / |evidence|                              (gold needed)
  evidence_coverage        expected claims whose key facts all occur in one evidence chunk / expected claims
  unsupported_evidence     evidence items neither in gold nor containing any expected key fact / items
  contradictory_evidence   1 if the evidence contains a stale / forbidden value (``forbidden`` regexes) while the
                           sample is not a CONTRADICTORY one; else 0                       (forbidden needed)
  redundancy               mean pairwise cosine similarity of the evidence vectors (caller passes them)
  source_diversity         distinct documents / items
"""

from __future__ import annotations

import re


def _has_all(text: str, keys: list[str]) -> bool:
    low = text.lower()
    return bool(keys) and all(k.lower() in low for k in keys)


def compute(items: list[dict], gold: list[str], expected_claims: list, forbidden: list[str], sample_type: str,
            vectors=None) -> dict:
    """items: [{"citation", "document_id", "text"}] in rank order."""
    cits = [i["citation"] for i in items]
    out: dict = {"evidence_items": len(items)}
    if gold:
        hit = set(cits) & set(gold)
        out["evidence_recall"] = len(hit) / len(set(gold))
        out["evidence_precision"] = (len(hit) / len(set(cits))) if cits else 0.0
    else:
        out["evidence_recall"] = out["evidence_precision"] = None
    keyed = [c for c in expected_claims if c.key]
    out["evidence_coverage"] = (sum(1 for c in keyed if any(_has_all(i["text"], c.key) for i in items))
                                / len(keyed)) if keyed else None
    if items and (gold or keyed):
        rel = [i for i in items if i["citation"] in set(gold) or any(_has_all(i["text"], c.key) for c in keyed)]
        out["unsupported_evidence"] = 1 - len(rel) / len(items)
    else:
        out["unsupported_evidence"] = None
    if forbidden and sample_type != "CONTRADICTORY":
        out["contradictory_evidence"] = float(any(re.search(p, i["text"], re.I) for i in items for p in forbidden))
    else:
        out["contradictory_evidence"] = None
    out["source_diversity"] = (len({i["document_id"] for i in items}) / len(items)) if items else None
    if vectors is not None and len(vectors) >= 2:
        sims = [float(vectors[a] @ vectors[b]) for a in range(len(vectors)) for b in range(a + 1, len(vectors))]
        out["redundancy"] = sum(sims) / len(sims)
    else:
        out["redundancy"] = None
    return out
