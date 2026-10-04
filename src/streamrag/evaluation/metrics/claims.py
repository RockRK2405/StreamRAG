"""Claim metrics: the factual claims of the final answer, judged by the evaluation instrument.

Instrument: the Phase 7 claim verifier (entailment model nli-deberta-v3-xsmall + rules) applied to every answer
claim against the evidence the system handed to its answer stage. It is the *same* model family the proposed system
uses to filter its own claims, so verifier-judged rates favour systems that verify (stated in every report);
model-free gold-based metrics (claim coverage, generation.py) are reported next to them.

  claim_support_rate     claims SUPPORTED / claims                                   (claims needed)
  unsupported_claim_rate claims UNSUPPORTED or PARTIALLY_SUPPORTED / claims
  contradicted_rate      claims CONTRADICTED by the evidence with no support / claims
  claim_coverage         expected claims stated in the answer (all key strings present) / expected claims
Claim verification *accuracy* needs claims with known labels: measured separately (experiments: verifier_validation).
"""

from __future__ import annotations


def stated(answer: str, claim) -> bool:
    low = answer.lower()
    return bool(claim.key) and all(k.lower() in low for k in claim.key)


def compute(verdicts: list[str], answer: str, expected_claims: list) -> dict:
    n = len(verdicts)
    keyed = [c for c in expected_claims if c.key]
    return {
        "claims": n,
        "claim_support_rate": (sum(v == "SUPPORTED" for v in verdicts) / n) if n else None,
        "unsupported_claim_rate": (sum(v in ("UNSUPPORTED", "PARTIALLY_SUPPORTED") for v in verdicts) / n) if n
        else None,
        "contradicted_claim_rate": (sum(v == "CONTRADICTED" for v in verdicts) / n) if n else None,
        "claim_coverage": (sum(stated(answer, c) for c in keyed) / len(keyed)) if keyed else None,
    }
