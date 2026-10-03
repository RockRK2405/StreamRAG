"""Grounding metrics of one answer version (Phase 7; docs/answer/09).

All values are **verifier-judged**: they say what the claim verifier (entailment model + rules) decided, not what a
human annotator would. Their agreement with labelled data is measured separately (research/phase7: synthetic
perturbations with labels by construction, and a hand-labelled sample). Definitions:

  raw_claims                factual claims extracted from the generator's output, before the policy
  raw_support_rate          raw claims supported by some evidence (SUPPORTED, or supported by one source and disputed
                            by another = evidence CONFLICT) / raw claims                            (CSR, pre-policy)
  raw_unsupported_rate      raw claims UNSUPPORTED or CONTRADICTED without any support / raw claims
  raw_partial_rate          raw claims PARTIALLY_SUPPORTED (some atoms unsupported) / raw claims
  final_claims              factual claims in the final answer (facts + conflict sides)
  final_unsupported_rate    final claims without verified support / final claims (0 by construction; reported to
                            prove it)
  citation_coverage         final claims with >= 1 valid citation / final claims                       (CC)
  citation_precision        valid citations / citations created                                    (supporting)
  evidence_utilization      evidence cited / usable evidence of the answer's needs                     (EU)
  intent_coverage           needs covered by facts or explicit uncertainty / needs                     (IC)
"""

from __future__ import annotations


def _r(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def grounding_metrics(raw_statuses: list[str], answer) -> dict[str, float | None]:
    facts = [c for c in answer.claims if c.kind in ("fact", "conflict")]
    cited = [c for c in facts if c.citation_ids]
    sup = [c for c in facts if c.status in ("SUPPORTED", "CONTRADICTED") and c.evidence_ids]
    valid = answer.citation_report.valid
    used = {c.evidence_id for c in answer.citations.citations if c.status == "valid"}
    total_ev = len(used | set(answer.citation_report.unused_evidence))
    return {
        "raw_claims": float(len(raw_statuses)),
        "raw_support_rate": _r(sum(1 for s in raw_statuses if s in ("SUPPORTED", "CONFLICT")), len(raw_statuses)),
        "raw_unsupported_rate": _r(sum(1 for s in raw_statuses if s in ("UNSUPPORTED", "CONTRADICTED")),
                                   len(raw_statuses)),
        "raw_partial_rate": _r(sum(1 for s in raw_statuses if s == "PARTIALLY_SUPPORTED"), len(raw_statuses)),
        "final_claims": float(len(facts)),
        "final_unsupported_rate": _r(len(facts) - len(sup), len(facts)),
        "citation_coverage": _r(len(cited), len(facts)),
        "citation_precision": _r(valid, answer.citation_report.checked),
        "evidence_utilization": _r(len(used), total_ev),
        "intent_coverage": answer.coverage.intent_coverage,
    }
